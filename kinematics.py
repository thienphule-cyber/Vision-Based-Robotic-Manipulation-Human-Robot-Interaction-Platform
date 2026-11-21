"""
kinematics.py - Kinematics, trajectories and motion planning for VRM-HRI.

Contents
--------
1. Math helpers       : rotation vector <-> matrix, pose error.
2. FK / Jacobian      : forward kinematics wrappers, Jacobian, singularity measures.
3. Inverse kinematics : damped least squares (DLS) with joint limits and random restarts.
4. Servo steps        : single-step resolved-rate control (gesture / GUI teleoperation).
5. Trajectories       : joint-space (quintic / linear) and straight-line Cartesian.
6. Collision checking : lightweight point-sampling check against the Environment.
7. Planning           : RRT-Connect in joint space, path shortcutting, path -> trajectory.

Design notes
------------
* This module only computes. It does NOT enforce safety policy (speed limits imposed by
    the operator, workspace rules, ...). Every command must still go through safety.py.
* Collision checking is approximate: it samples points along the polyline that joins the
    DH frame origins (plus the tool tip) and tests them against the AABBs of the
    environment objects, inflated by a margin. The first link (the base pillar) is skipped
    because the robot base sits on the table. Tune `margin` to cover the real link radius.
* The DH table in modelling.py is nominal, so the results are only as accurate as it is.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from spatialmath import SE3

from modelling import Environment, RobotModel

logger = logging.getLogger(__name__)

PoseLike = Union[SE3, np.ndarray]

# Default joint speed used when a trajectory is requested without any timing information.
DEFAULT_JOINT_SPEED = 1.0  # rad/s
# Peak velocity of a quintic (minimum-jerk) profile is 15/8 of the average velocity.
QUINTIC_PEAK_FACTOR = 15.0 / 8.0


# ----------------------------------------------------------------------------
# 1. Math helpers
# ----------------------------------------------------------------------------

def _as_matrix(T: PoseLike) -> np.ndarray:
    """Convert an SE3 or a 4x4 array to a 4x4 float matrix."""
    arr = np.asarray(T.A if isinstance(T, SE3) else T, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"Pose must be a single SE3 or a 4x4 matrix, got shape {arr.shape}")
    return arr


def skew(v) -> np.ndarray:
    """3x3 skew-symmetric matrix of a 3-vector."""
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_vector(R) -> np.ndarray:
    """Axis * angle (rotation vector) of a rotation matrix, angle in [0, pi]."""
    R = np.asarray(R, dtype=float)
    vec = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = 0.5 * float(np.linalg.norm(vec))      # sin(angle)
    c = 0.5 * (float(np.trace(R)) - 1.0)      # cos(angle)
    angle = math.atan2(s, c)                  # accurate for small angles (unlike arccos)
    if s > 1e-9:
        return angle * vec / (2.0 * s)
    if c > 0.0:                               # angle ~ 0
        return 0.5 * vec
    # angle ~ pi: R + I = 2 n n^T, so any column of (R + I) is proportional to the axis
    k = int(np.argmax(np.diag(R)))
    axis = R[:, k] + np.eye(3)[:, k]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.zeros(3)
    return math.pi * axis / norm


def rotvec_to_matrix(w) -> np.ndarray:
    """Rotation matrix from a rotation vector (Rodrigues formula)."""
    w = np.asarray(w, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w))
    K = skew(w)
    if theta < 1e-12:
        return np.eye(3) + K
    K = K / theta
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def pose_error(T_current: PoseLike, T_goal: PoseLike) -> np.ndarray:
    """6-vector [dx, dy, dz, wx, wy, wz] (world frame) that moves T_current onto T_goal."""
    Tc, Tg = _as_matrix(T_current), _as_matrix(T_goal)
    e_pos = Tg[:3, 3] - Tc[:3, 3]
    e_rot = rotation_vector(Tg[:3, :3] @ Tc[:3, :3].T)
    return np.concatenate([e_pos, e_rot])


def _as_q(q, n: int) -> np.ndarray:
    arr = np.asarray(q, dtype=float).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"Expected {n} joint values, got shape {arr.shape}")
    return arr


# ----------------------------------------------------------------------------
# 2. Forward kinematics, Jacobian, singularity measures
# ----------------------------------------------------------------------------

def fk(model: RobotModel, q) -> SE3:
    """End-effector pose in the world frame."""
    return model.fkine(q)


def jacobian(model: RobotModel, q) -> np.ndarray:
    """6 x n geometric Jacobian in the world frame (rows: vx vy vz wx wy wz)."""
    return np.asarray(model.robot.jacob0(_as_q(q, model.n)), dtype=float)


def min_singular_value(model: RobotModel, q) -> float:
    """Smallest singular value of the Jacobian (0 at a singularity; mixed m / rad units)."""
    return float(np.linalg.svd(jacobian(model, q), compute_uv=False)[-1])


def manipulability(model: RobotModel, q) -> float:
    """Yoshikawa manipulability sqrt(det(J J^T))."""
    J = jacobian(model, q)
    return float(math.sqrt(max(float(np.linalg.det(J @ J.T)), 0.0)))


def is_near_singular(model: RobotModel, q, threshold: float = 0.02) -> bool:
    return min_singular_value(model, q) < threshold


def _dh_params(model: RobotModel) -> List[Tuple[float, float, float, float]]:
    return [(float(l.d), float(l.a), float(l.alpha), float(l.offset)) for l in model.robot.links]


def _dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Standard DH transform Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def _frames_from(params, base: np.ndarray, q: np.ndarray) -> List[np.ndarray]:
    frames = [base]
    T = base
    for (d, a, alpha, offset), qi in zip(params, q):
        T = T @ _dh_matrix(qi + offset, d, a, alpha)
        frames.append(T)
    return frames


def link_frames(model: RobotModel, q) -> List[np.ndarray]:
    """World poses (4x4) of the base frame and of every link frame (n + 1 matrices).

    The tool transform is not included: frames[-1] @ model.tool.A == fkine(q).A.
    """
    q = _as_q(q, model.n)
    return _frames_from(_dh_params(model), np.asarray(model.base.A, dtype=float), q)


# ----------------------------------------------------------------------------
# 3. Inverse kinematics (damped least squares)
# ----------------------------------------------------------------------------

def dls_solve(
    J: np.ndarray,
    e: np.ndarray,
    max_damping: float = 0.1,
    sigma_eps: float = 0.05,
    min_damping: float = 1e-3,
) -> np.ndarray:
    """Damped least-squares solution dq of J dq = e.

    The damping is adaptive (Nakamura & Hanafusa): it is small far from singularities and
    grows towards `max_damping` as the smallest singular value falls below `sigma_eps`.
    """
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    s_min = float(S[-1])
    lam2 = min_damping ** 2
    if s_min < sigma_eps:
        lam2 += (max_damping ** 2) * (1.0 - (s_min / sigma_eps) ** 2)
    return Vt.T @ ((S / (S ** 2 + lam2)) * (U.T @ e))


@dataclass
class IKResult:
    success: bool
    q: np.ndarray
    iterations: int
    pos_error: float                # metres
    rot_error: float                # radians
    restarts_used: int = 0
    message: str = ""


def _ik_attempt(
    model: RobotModel,
    T_goal: np.ndarray,
    q_init: np.ndarray,
    position_only: bool,
    pos_tol: float,
    rot_tol: float,
    max_iters: int,
    max_step: float,
    max_damping: float,
    sigma_eps: float,
):
    q = model.clip(q_init)
    for it in range(max_iters + 1):
        e = pose_error(model.fkine(q).A, T_goal)
        pos_err = float(np.linalg.norm(e[:3]))
        rot_err = float(np.linalg.norm(e[3:]))
        done = pos_err <= pos_tol and (position_only or rot_err <= rot_tol)
        if done or it == max_iters:
            return done, q, it, pos_err, rot_err
        J = jacobian(model, q)
        task = e
        if position_only:
            J, task = J[:3], e[:3]
        dq = dls_solve(J, task, max_damping=max_damping, sigma_eps=sigma_eps)
        biggest = float(np.max(np.abs(dq)))
        if biggest > max_step:
            dq *= max_step / biggest
        q = model.clip(q + dq)
    raise RuntimeError("unreachable")  # pragma: no cover


def ik(
    model: RobotModel,
    T_goal: PoseLike,
    q0=None,
    *,
    position_only: bool = False,
    pos_tol: float = 1e-4,
    rot_tol: float = 1e-3,
    max_iters: int = 200,
    max_step: float = 0.3,
    max_damping: float = 0.1,
    sigma_eps: float = 0.05,
    n_restarts: int = 10,
    seed: int = 0,
) -> IKResult:
    """Numerical IK. The first attempt starts from q0 (default: home), the others from
    random configurations inside the joint limits. The result is always inside the limits.

    If nothing converges, the best attempt is returned with success=False.
    """
    Tg = _as_matrix(T_goal)
    start = model.q_home if q0 is None else model.clip(q0)

    # Quick reject: the target is farther than the arm can ever reach.
    limit = model.max_reach + float(np.linalg.norm(np.asarray(model.tool.t, dtype=float)))
    dist = float(np.linalg.norm(Tg[:3, 3] - np.asarray(model.base.t, dtype=float)))
    if dist > limit:
        e = pose_error(model.fkine(start).A, Tg)
        return IKResult(False, start, 0, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:])),
                        0, f"Target is {dist:.3f} m from the base; max reach is {limit:.3f} m")

    rng = np.random.default_rng(seed)
    best = None
    best_score = float("inf")
    total_iters = 0
    for k in range(n_restarts + 1):
        q_init = start if k == 0 else model.random_q(rng)
        ok, q, iters, pos_err, rot_err = _ik_attempt(
            model, Tg, q_init, position_only, pos_tol, rot_tol, max_iters,
            max_step, max_damping, sigma_eps,
        )
        total_iters += iters
        if ok:
            return IKResult(True, q, total_iters, pos_err, rot_err, k, "converged")
        score = pos_err + (0.0 if position_only else rot_err)
        if score < best_score:
            best_score = score
            best = (q, pos_err, rot_err)
    q, pos_err, rot_err = best
    return IKResult(False, q, total_iters, pos_err, rot_err, n_restarts, "did not converge")


# ----------------------------------------------------------------------------
# 4. Servo steps (gesture / GUI teleoperation)
# ----------------------------------------------------------------------------

def resolved_rate_step(
    model: RobotModel,
    q,
    twist,
    dt: float,
    damping: float = 0.1,
    max_joint_speed: Optional[float] = None,
) -> np.ndarray:
    """One resolved-rate step: q_next = q + J^+ twist * dt (world-frame twist [v; w]).

    If `max_joint_speed` is given, the joint step is scaled so no joint exceeds it.
    The result is clipped into the joint limits.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    q = _as_q(q, model.n)
    twist = np.asarray(twist, dtype=float).reshape(6)
    dq = dls_solve(jacobian(model, q), twist, max_damping=damping) * dt
    if max_joint_speed is not None:
        biggest = float(np.max(np.abs(dq))) / dt
        if biggest > max_joint_speed:
            dq *= max_joint_speed / biggest
    return model.clip(q + dq)


def servo_to_pose(
    model: RobotModel,
    q,
    T_goal: PoseLike,
    dt: float,
    gain: float = 5.0,
    max_lin_speed: float = 0.25,
    max_ang_speed: float = 1.0,
    max_joint_speed: float = 2.0,
    damping: float = 0.1,
) -> Tuple[np.ndarray, float, float]:
    """One proportional servo step towards T_goal with Cartesian and joint speed limits.

    Returns (q_next, position_error, rotation_error); the errors are measured BEFORE the step.
    Call it every control tick. A low-pass filtered gesture target works well as T_goal.
    """
    q = _as_q(q, model.n)
    e = pose_error(model.fkine(q).A, T_goal)
    twist = gain * e
    lin = float(np.linalg.norm(twist[:3]))
    ang = float(np.linalg.norm(twist[3:]))
    if lin > max_lin_speed:
        twist[:3] *= max_lin_speed / lin
    if ang > max_ang_speed:
        twist[3:] *= max_ang_speed / ang
    q_next = resolved_rate_step(model, q, twist, dt, damping, max_joint_speed)
    return q_next, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))


# ----------------------------------------------------------------------------
# 5. Trajectories
# ----------------------------------------------------------------------------

def _blend(method: str, tau: np.ndarray) -> np.ndarray:
    if method == "quintic":
        return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    if method == "linear":
        return tau
    raise ValueError(f"Unknown method '{method}'. Valid: quintic, linear")


def _n_steps(duration: float, dt: float) -> int:
    return max(1, int(math.ceil(duration / dt - 1e-9)))


def min_duration(q0, q1, v_max, method: str = "quintic") -> float:
    """Shortest duration that keeps every joint below v_max (scalar or per-joint, rad/s)."""
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    q1 = np.asarray(q1, dtype=float).reshape(-1)
    v = np.broadcast_to(np.asarray(v_max, dtype=float), q0.shape)
    if np.any(v <= 0):
        raise ValueError("v_max must be positive")
    factor = QUINTIC_PEAK_FACTOR if method == "quintic" else 1.0
    _blend(method, np.zeros(1))  # validates the method name
    return float(factor * np.max(np.abs(q1 - q0) / v))


def joint_trajectory(
    q0,
    q1,
    dt: float = 0.02,
    duration: Optional[float] = None,
    v_max=None,
    method: str = "quintic",
) -> np.ndarray:
    """Joint-space trajectory from q0 to q1, shape (N, n), including both end points.

    * duration given            -> used (but never shorter than v_max allows, if v_max is given)
    * only v_max given          -> shortest duration allowed by v_max
    * neither                   -> v_max = DEFAULT_JOINT_SPEED
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    q1 = np.asarray(q1, dtype=float).reshape(-1)
    if q0.shape != q1.shape:
        raise ValueError("q0 and q1 must have the same length")
    _blend(method, np.zeros(1))  # validates the method name
    if duration is None or v_max is not None:
        t_min = min_duration(q0, q1, DEFAULT_JOINT_SPEED if v_max is None else v_max, method)
        duration = t_min if duration is None else max(duration, t_min)
    if duration < 0:
        raise ValueError("duration must be non-negative")
    n_steps = _n_steps(duration, dt)
    s = _blend(method, np.linspace(0.0, 1.0, n_steps + 1))
    return q0[None, :] + s[:, None] * (q1 - q0)[None, :]


def concat_trajectories(trajs: Sequence[np.ndarray]) -> np.ndarray:
    """Join trajectories end to end, dropping the duplicated first row of later segments."""
    if len(trajs) == 0:
        raise ValueError("Nothing to concatenate")
    parts = [np.asarray(trajs[0], dtype=float)]
    parts += [np.asarray(t, dtype=float)[1:] for t in trajs[1:]]
    return np.vstack(parts)


def path_to_trajectory(
    path: Sequence[np.ndarray],
    dt: float = 0.02,
    v_max=DEFAULT_JOINT_SPEED,
    method: str = "quintic",
) -> np.ndarray:
    """Turn a list of joint waypoints into a trajectory (the robot stops at each waypoint)."""
    if len(path) < 2:
        raise ValueError("A path needs at least two waypoints")
    segments = [joint_trajectory(a, b, dt=dt, v_max=v_max, method=method)
                for a, b in zip(path[:-1], path[1:])]
    return concat_trajectories(segments)


@dataclass
class CartesianResult:
    success: bool
    q_traj: np.ndarray
    failed_index: Optional[int] = None
    message: str = ""


def cartesian_trajectory(
    model: RobotModel,
    T_start: PoseLike,
    T_goal: PoseLike,
    q_seed,
    dt: float = 0.02,
    duration: Optional[float] = None,
    v_lin: float = 0.25,
    v_ang: float = 1.0,
    max_joint_jump: float = 0.5,
    **ik_kwargs,
) -> CartesianResult:
    """Straight-line (position) + SLERP-like (orientation) path, solved with seeded IK.

    Every IK solve starts from the previous solution, so the joint path stays continuous.
    Fails (success=False) if IK does not converge or a joint jumps by more than
    `max_joint_jump` rad between two samples. `q_traj` then holds the samples solved so far.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    T0, T1 = _as_matrix(T_start), _as_matrix(T_goal)
    p0, p1 = T0[:3, 3], T1[:3, 3]
    w = rotation_vector(T1[:3, :3] @ T0[:3, :3].T)
    if duration is None:
        duration = max(
            QUINTIC_PEAK_FACTOR * float(np.linalg.norm(p1 - p0)) / v_lin,
            QUINTIC_PEAK_FACTOR * float(np.linalg.norm(w)) / v_ang,
            dt,
        )
    n_steps = _n_steps(duration, dt)
    s_values = _blend("quintic", np.linspace(0.0, 1.0, n_steps + 1))

    ik_kwargs.setdefault("max_iters", 60)
    ik_kwargs["n_restarts"] = 0  # seeded IK only: random restarts would break continuity

    q_prev = model.clip(q_seed)
    out: List[np.ndarray] = []
    for k, s in enumerate(s_values):
        T = np.eye(4)
        T[:3, :3] = rotvec_to_matrix(s * w) @ T0[:3, :3]
        T[:3, 3] = p0 + s * (p1 - p0)
        res = ik(model, T, q_prev, **ik_kwargs)
        if not res.success:
            return CartesianResult(False, np.array(out).reshape(-1, model.n), k,
                                f"IK failed at sample {k}/{n_steps} ({res.message})")
        if float(np.max(np.abs(res.q - q_prev))) > max_joint_jump:
            return CartesianResult(False, np.array(out).reshape(-1, model.n), k,
                                f"Joint discontinuity at sample {k}/{n_steps}")
        out.append(res.q)
        q_prev = res.q
    return CartesianResult(True, np.array(out), None, "ok")


# ----------------------------------------------------------------------------
# 6. Collision checking (approximate, point sampling)
# ----------------------------------------------------------------------------

class CollisionChecker:
    """Checks robot configurations against the obstacles of an Environment.

    The obstacle boxes are copied when the checker is created; build a new checker if the
    environment or the robot base/tool changes.
    """

    def __init__(
        self,
        model: RobotModel,
        environment: Environment,
        margin: float = 0.02,
        kinds: Sequence[str] = ("table", "obstacle"),
        spacing: float = 0.04,
        skip_links: int = 1,
    ) -> None:
        if spacing <= 0:
            raise ValueError("spacing must be positive")
        self.model = model
        self.margin = float(margin)
        self.spacing = float(spacing)
        self.skip_links = int(skip_links)
        self._params = _dh_params(model)
        self._base = np.asarray(model.base.A, dtype=float)
        self._tool = np.asarray(model.tool.A, dtype=float)
        los, his = [], []
        for obj in environment._objects.values():
            if obj.kind in kinds:
                lo, hi = obj.aabb()
                los.append(lo - self.margin)
                his.append(hi + self.margin)
        self._lo = np.array(los).reshape(-1, 3)
        self._hi = np.array(his).reshape(-1, 3)

    def sample_points(self, q) -> np.ndarray:
        """Points along the robot (skipping the first `skip_links` links), shape (N, 3)."""
        frames = _frames_from(self._params, self._base, _as_q(q, self.model.n))
        origins = [F[:3, 3] for F in frames]
        origins.append((frames[-1] @ self._tool)[:3, 3])  # tool tip
        pts = []
        for i in range(self.skip_links, len(origins) - 1):
            p0, p1 = origins[i], origins[i + 1]
            n = max(1, int(math.ceil(float(np.linalg.norm(p1 - p0)) / self.spacing)))
            t = np.linspace(0.0, 1.0, n + 1)[:, None]
            pts.append(p0 + t * (p1 - p0))
        return np.vstack(pts)

    def in_collision(self, q) -> bool:
        if self._lo.shape[0] == 0:
            return False
        pts = self.sample_points(q)
        inside = np.all(
            (pts[:, None, :] >= self._lo[None, :, :]) & (pts[:, None, :] <= self._hi[None, :, :]),
            axis=2,
        )
        return bool(np.any(inside))

    def edge_free(self, q_a, q_b, resolution: float = 0.1) -> bool:
        """True if the straight joint-space segment q_a -> q_b is collision free."""
        q_a = np.asarray(q_a, dtype=float)
        q_b = np.asarray(q_b, dtype=float)
        n = max(1, int(math.ceil(float(np.max(np.abs(q_b - q_a))) / resolution)))
        for k in range(n + 1):
            if self.in_collision(q_a + (q_b - q_a) * (k / n)):
                return False
        return True


def in_collision(model: RobotModel, environment: Environment, q, **kwargs) -> bool:
    """Convenience one-shot check (builds a CollisionChecker each call)."""
    return CollisionChecker(model, environment, **kwargs).in_collision(q)


# ----------------------------------------------------------------------------
# 7. Planning (RRT-Connect in joint space)
# ----------------------------------------------------------------------------

@dataclass
class PlanResult:
    success: bool
    path: List[np.ndarray] = field(default_factory=list)
    message: str = ""
    iterations: int = 0


class _Tree:
    def __init__(self, root: np.ndarray) -> None:
        self.nodes: List[np.ndarray] = [root]
        self.parents: List[int] = [-1]

    def nearest(self, q: np.ndarray) -> int:
        arr = np.asarray(self.nodes)
        return int(np.argmin(np.sum((arr - q) ** 2, axis=1)))

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(q)
        self.parents.append(parent)
        return len(self.nodes) - 1

    def trace(self, idx: int) -> List[np.ndarray]:
        out = []
        while idx != -1:
            out.append(self.nodes[idx])
            idx = self.parents[idx]
        return out[::-1]


def _extend(tree: _Tree, q_target, step: float, checker: CollisionChecker, resolution: float):
    i = tree.nearest(q_target)
    q_near = tree.nodes[i]
    delta = q_target - q_near
    dist = float(np.linalg.norm(delta))
    if dist < 1e-12:
        return "reached", i
    if dist <= step:
        q_new, status = q_target.copy(), "reached"
    else:
        q_new, status = q_near + delta / dist * step, "advanced"
    if not checker.edge_free(q_near, q_new, resolution):
        return "trapped", -1
    return status, tree.add(q_new, i)


def _connect(tree: _Tree, q_target, step: float, checker: CollisionChecker, resolution: float):
    while True:
        status, idx = _extend(tree, q_target, step, checker, resolution)
        if status == "reached":
            return idx
        if status == "trapped":
            return None


def shortcut_path(
    path: Sequence[np.ndarray],
    checker: CollisionChecker,
    rng: np.random.Generator,
    iters: int = 50,
    resolution: float = 0.1,
) -> List[np.ndarray]:
    """Randomised shortcutting: replace sub-paths by straight segments when they are free."""
    out = [np.asarray(p, dtype=float).copy() for p in path]
    for _ in range(iters):
        if len(out) < 3:
            break
        i = int(rng.integers(0, len(out) - 2))
        j = int(rng.integers(i + 2, len(out)))
        if checker.edge_free(out[i], out[j], resolution):
            out = out[: i + 1] + out[j:]
    return out


def plan_joint_path(
    model: RobotModel,
    environment: Environment,
    q_start,
    q_goal,
    *,
    margin: float = 0.02,
    kinds: Sequence[str] = ("table", "obstacle"),
    max_iters: int = 2000,
    step: float = 0.4,
    resolution: float = 0.1,
    shortcut_iters: int = 50,
    seed: int = 0,
) -> PlanResult:
    """Collision-free joint-space path from q_start to q_goal (list of waypoints)."""
    qs = _as_q(q_start, model.n).copy()
    qg = _as_q(q_goal, model.n).copy()
    if not model.within_limits(qs):
        return PlanResult(False, [], "Start configuration is outside the joint limits")
    if not model.within_limits(qg):
        return PlanResult(False, [], "Goal configuration is outside the joint limits")

    checker = CollisionChecker(model, environment, margin=margin, kinds=kinds)
    if checker.in_collision(qs):
        return PlanResult(False, [], "Start configuration is in collision")
    if checker.in_collision(qg):
        return PlanResult(False, [], "Goal configuration is in collision")
    if checker.edge_free(qs, qg, resolution):
        return PlanResult(True, [qs, qg], "direct path", 0)

    rng = np.random.default_rng(seed)
    tree_a, tree_b = _Tree(qs), _Tree(qg)
    a_is_start = True
    for it in range(1, max_iters + 1):
        q_rand = model.random_q(rng)
        status, idx_a = _extend(tree_a, q_rand, step, checker, resolution)
        if status != "trapped":
            idx_b = _connect(tree_b, tree_a.nodes[idx_a], step, checker, resolution)
            if idx_b is not None:
                path_a, path_b = tree_a.trace(idx_a), tree_b.trace(idx_b)
                if a_is_start:
                    path = path_a + path_b[::-1][1:]
                else:
                    path = path_b + path_a[::-1][1:]
                path = shortcut_path(path, checker, rng, shortcut_iters, resolution)
                return PlanResult(True, path, "path found", it)
        tree_a, tree_b = tree_b, tree_a
        a_is_start = not a_is_start
    return PlanResult(False, [], f"No path found within {max_iters} iterations", max_iters)


def plan_trajectory(
    model: RobotModel,
    environment: Environment,
    q_start,
    q_goal,
    dt: float = 0.02,
    v_max=DEFAULT_JOINT_SPEED,
    **plan_kwargs,
) -> Tuple[PlanResult, Optional[np.ndarray]]:
    """Plan a collision-free path and time-parametrise it. Returns (PlanResult, trajectory|None)."""
    result = plan_joint_path(model, environment, q_start, q_goal, **plan_kwargs)
    if not result.success:
        return result, None
    return result, path_to_trajectory(result.path, dt=dt, v_max=v_max)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    robot = RobotModel()
    q_demo = np.deg2rad([30, 20, 60, 20, 40, 30])
    target = robot.fkine(q_demo)
    sol = ik(robot, target)
    print("IK success:", sol.success, "| pos err:", sol.pos_error, "| rot err:", sol.rot_error)
    print("Min singular value:", min_singular_value(robot, sol.q))