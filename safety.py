"""
safety.py - Safety gate for VRM-HRI.

Every command that comes from a gesture or from the GUI must be validated here before it
reaches simulation.py.

Contents
--------
1. SafetyConfig / SafetyReport : limits and the result of a check.
2. Exact shape distance        : signed distance from points to box / sphere / cylinder.
3. SafetyChecker               : e-stop, joint limits, speed limits, workspace, collision,
                                singularity, trajectory / hand / pose validation and a
                                step limiter.

Collision model (NO AABB)
-------------------------
* The robot is a set of capsules: one per link segment between consecutive DH frame
    origins (the first `skip_segments` segments are skipped because the base pillar sits on
    the table) plus one capsule for the hand along the flange +z axis.
* Obstacles keep their real shape: oriented box (with yaw), sphere, cylinder. The signed
    distance from a point to each shape is computed exactly.
* Points are sampled along every capsule axis. The distance function is 1-Lipschitz, so the
    true minimum distance is at least (sampled minimum - half the sample gap). That slack is
    subtracted, so the check is conservative (it never reports a false "free").
* clearance = distance(axis, obstacle surface) - capsule radius - margin.
    clearance < 0 means collision.

Notes
-----
* Capsule radii, margins and speed limits are nominal values: tune them for the real robot.
* Moving INTO a singularity is blocked; moving away from one is allowed. A singular
    configuration by itself only raises a warning (the all-zero home pose is a wrist
    singularity of the nominal DH table).
* The environment is read live at every check, so objects added or removed later are seen.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from spatialmath import SE3

from kinematics import link_frames, min_singular_value, rotation_vector
from modelling import FINGER_NAMES, GRASP_PRESETS, Environment, RobotModel, WorldObject

logger = logging.getLogger(__name__)

# Violation codes
ESTOP = "estop"
BAD_INPUT = "bad_input"
JOINT_LIMIT = "joint_limit"
JOINT_SPEED = "joint_speed"
EE_SPEED = "ee_speed"
EE_ROT_SPEED = "ee_rot_speed"
WORKSPACE = "workspace"
COLLISION = "collision"
SINGULARITY = "singularity"
UNREACHABLE = "unreachable"
HAND_COMMAND = "hand_command"

_SPEED_TOL = 1e-9  # relative tolerance for speed comparisons


# ----------------------------------------------------------------------------
# 1. Configuration and report
# ----------------------------------------------------------------------------

@dataclass
class SafetyConfig:
    # Speed limits
    joint_speed_limit: Union[float, Sequence[float]] = 2.0   # rad/s (scalar or per joint)
    ee_speed_limit: float = 0.5                              # m/s
    ee_ang_speed_limit: float = 2.0                          # rad/s
    max_dt: float = 0.5                                      # s, largest accepted time step
    # Joint limits
    joint_limit_margin: float = 0.0                          # rad, shrinks the joint range
    # Workspace for the end-effector: ((x, y, z) min, (x, y, z) max) in metres, or None
    workspace: Optional[Tuple[Sequence[float], Sequence[float]]] = (
        (-0.7, -0.7, 0.0),
        (0.7, 0.7, 0.9),
    )
    # Collision model (capsules)
    link_radii: Sequence[float] = (0.07, 0.06, 0.05, 0.05, 0.04, 0.04)  # one per DH segment
    skip_segments: int = 1            # base pillar sits on the table
    margin: float = 0.02              # extra clearance for link capsules [m]
    include_hand: bool = True
    hand_length: float = 0.15         # capsule from the flange along +z [m]
    hand_radius: float = 0.04
    hand_margin: float = 0.0          # small so the hand can still approach the table
    collision_kinds: Tuple[str, ...] = ("table", "obstacle")
    sample_spacing: float = 0.02      # distance between samples along a capsule axis [m]
    step_resolution: float = 0.05     # max joint change between collision samples [rad]
    # Singularity
    singularity_threshold: float = 0.01   # min singular value of J; 0 disables the check


@dataclass
class SafetyViolation:
    code: str
    message: str
    index: Optional[int] = None


@dataclass
class SafetyReport:
    violations: List[SafetyViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def codes(self) -> List[str]:
        return [v.code for v in self.violations]

    def has(self, code: str) -> bool:
        return code in self.codes

    def add(self, code: str, message: str, index: Optional[int] = None) -> None:
        self.violations.append(SafetyViolation(code, message, index))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def merge(self, other: "SafetyReport", index: Optional[int] = None) -> None:
        for v in other.violations:
            self.violations.append(
                SafetyViolation(v.code, v.message, index if v.index is None else v.index)
            )
        self.warnings.extend(other.warnings)

    def summary(self) -> str:
        if self.ok:
            return "OK" + (f" ({len(self.warnings)} warning(s))" if self.warnings else "")
        return "; ".join(
            f"[{v.code}{'' if v.index is None else f'@{v.index}'}] {v.message}"
            for v in self.violations
        )


class SafetyError(RuntimeError):
    """Raised by require_safe() when a report contains violations."""


def require_safe(report: SafetyReport) -> None:
    if not report.ok:
        raise SafetyError(report.summary())


# ----------------------------------------------------------------------------
# 2. Exact shape distance
# ----------------------------------------------------------------------------

def shape_signed_distance(obj: WorldObject, points) -> np.ndarray:
    """Signed distance (negative inside) from points (N, 3) to the real shape of `obj`.

    box: oriented box (rotated by yaw about z); sphere / custom: sphere;
    cylinder: finite cylinder with its axis along the local z axis.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    rel = pts - np.asarray(obj.position, dtype=float)
    if obj.shape in ("sphere", "custom"):
        return np.linalg.norm(rel, axis=1) - obj.dims[0]

    c, s = math.cos(obj.yaw), math.sin(obj.yaw)
    local = np.column_stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1], rel[:, 2]])
    if obj.shape == "box":
        q = np.abs(local) - 0.5 * np.asarray(obj.dims, dtype=float)
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)
    if obj.shape == "cylinder":
        radius, height = obj.dims
        radial = np.hypot(local[:, 0], local[:, 1]) - radius
        axial = np.abs(local[:, 2]) - 0.5 * height
        return np.hypot(np.maximum(radial, 0.0), np.maximum(axial, 0.0)) + np.minimum(
            np.maximum(radial, axial), 0.0
        )
    raise ValueError(f"Unsupported shape '{obj.shape}'")


def _segment_samples(p0, p1, spacing: float) -> Tuple[np.ndarray, float]:
    """Sample points along p0 -> p1. Returns (points, half of the sample gap)."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    length = float(np.linalg.norm(p1 - p0))
    n = max(1, int(math.ceil(length / spacing)))
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return p0 + t * (p1 - p0), 0.5 * length / n


@dataclass
class Capsule:
    name: str
    p0: np.ndarray
    p1: np.ndarray
    radius: float
    margin: float


# ----------------------------------------------------------------------------
# 3. Safety checker
# ----------------------------------------------------------------------------

class SafetyChecker:
    """Validates configurations, steps, trajectories, hand commands and pose targets."""

    def __init__(
        self,
        model: RobotModel,
        environment: Environment,
        config: Optional[SafetyConfig] = None,
    ) -> None:
        self.model = model
        self.environment = environment
        self.config = SafetyConfig() if config is None else config
        self._estop = threading.Event()
        self._estop_reason = ""
        self._validate_config()

    # ---- configuration ----------------------------------------------------
    def _validate_config(self) -> None:
        cfg, n = self.config, self.model.n
        v = np.broadcast_to(np.asarray(cfg.joint_speed_limit, dtype=float), (n,)).copy()
        if not np.all(np.isfinite(v)) or np.any(v <= 0):
            raise ValueError("joint_speed_limit must be positive")
        self._v_limit = v
        for name in ("ee_speed_limit", "ee_ang_speed_limit", "max_dt", "sample_spacing",
                    "step_resolution"):
            if not getattr(cfg, name) > 0:
                raise ValueError(f"{name} must be positive")
        for name in ("margin", "hand_margin", "hand_length", "hand_radius",
                    "joint_limit_margin", "singularity_threshold"):
            if getattr(cfg, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        radii = np.asarray(cfg.link_radii, dtype=float).reshape(-1)
        if radii.shape != (n,) or np.any(radii < 0):
            raise ValueError(f"link_radii needs {n} non-negative values")
        if not 0 <= cfg.skip_segments <= n:
            raise ValueError("skip_segments out of range")
        lo, hi = self.model.qlim
        self._qlo = lo + cfg.joint_limit_margin
        self._qhi = hi - cfg.joint_limit_margin
        if np.any(self._qlo >= self._qhi):
            raise ValueError("joint_limit_margin is larger than the joint range")
        if cfg.workspace is None:
            self._ws = None
        else:
            ws_lo = np.asarray(cfg.workspace[0], dtype=float).reshape(3)
            ws_hi = np.asarray(cfg.workspace[1], dtype=float).reshape(3)
            if np.any(ws_lo >= ws_hi):
                raise ValueError("workspace lower bounds must be below upper bounds")
            self._ws = (ws_lo, ws_hi)

    # ---- emergency stop ----------------------------------------------------
    def estop(self, reason: str = "") -> None:
        """Latch the emergency stop: every check fails until clear_estop() is called."""
        self._estop_reason = reason
        self._estop.set()
        logger.warning("E-STOP engaged%s", f": {reason}" if reason else "")

    def clear_estop(self) -> None:
        self._estop.clear()
        self._estop_reason = ""

    @property
    def estopped(self) -> bool:
        return self._estop.is_set()

    @property
    def estop_reason(self) -> str:
        return self._estop_reason

    def _check_estop(self, report: SafetyReport) -> bool:
        if self.estopped:
            reason = f" ({self._estop_reason})" if self._estop_reason else ""
            report.add(ESTOP, f"Emergency stop is engaged{reason}")
            return True
        return False

    # ---- input parsing -------------------------------------------------------
    def _parse_q(self, q, name: str, report: SafetyReport) -> Optional[np.ndarray]:
        try:
            arr = np.asarray(q, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            report.add(BAD_INPUT, f"{name} is not a numeric vector")
            return None
        if arr.shape != (self.model.n,):
            report.add(BAD_INPUT, f"{name} must have {self.model.n} values, got {arr.shape[0]}")
            return None
        if not np.all(np.isfinite(arr)):
            report.add(BAD_INPUT, f"{name} contains NaN or inf")
            return None
        return arr

    def _check_dt(self, dt, report: SafetyReport) -> bool:
        try:
            value = float(dt)
        except (TypeError, ValueError):
            report.add(BAD_INPUT, "dt is not a number")
            return False
        if not math.isfinite(value) or value <= 0:
            report.add(BAD_INPUT, f"dt must be positive, got {dt}")
            return False
        if value > self.config.max_dt:
            report.add(BAD_INPUT, f"dt {value} s exceeds max_dt {self.config.max_dt} s")
            return False
        return True

    # ---- collision model -------------------------------------------------------
    def robot_capsules(self, q) -> List[Capsule]:
        """Capsules representing the robot (and the hand) at configuration q."""
        cfg = self.config
        frames = link_frames(self.model, q)
        origins = [F[:3, 3] for F in frames]
        caps: List[Capsule] = []
        for i in range(self.model.n):
            if i < cfg.skip_segments:
                continue
            caps.append(Capsule(f"link{i + 1}", origins[i], origins[i + 1],
                                float(cfg.link_radii[i]), cfg.margin))
        if cfg.include_hand:
            T_ee = frames[-1] @ np.asarray(self.model.tool.A, dtype=float)
            p0 = T_ee[:3, 3].copy()
            p1 = p0 + T_ee[:3, 2] * cfg.hand_length
            caps.append(Capsule("hand", p0, p1, cfg.hand_radius, cfg.hand_margin))
        return caps

    def _active_objects(self) -> List[WorldObject]:
        objs: List[WorldObject] = []
        for kind in self.config.collision_kinds:
            objs.extend(self.environment.by_kind(kind))
        return objs

    def collision_clearance(self, q) -> Tuple[float, Optional[str], Optional[str]]:
        """Smallest clearance (metres, margins already subtracted) over all capsule/object
        pairs, with the names of the closest link and object. Negative = collision.
        Returns (inf, None, None) when there is nothing to collide with."""
        objs = self._active_objects()
        if not objs:
            return math.inf, None, None
        worst, worst_link, worst_obj = math.inf, None, None
        for cap in self.robot_capsules(q):
            pts, half_gap = _segment_samples(cap.p0, cap.p1, self.config.sample_spacing)
            for obj in objs:
                d = float(np.min(shape_signed_distance(obj, pts)))
                clearance = d - half_gap - cap.radius - cap.margin
                if clearance < worst:
                    worst, worst_link, worst_obj = clearance, cap.name, obj.name
        return worst, worst_link, worst_obj

    def in_collision(self, q) -> bool:
        return self.collision_clearance(q)[0] < 0.0

    # ---- shared checks -----------------------------------------------------------
    def _check_limits(self, q: np.ndarray, report: SafetyReport) -> None:
        names = self.model.joint_names
        bad = [names[i] for i in range(self.model.n)
            if q[i] < self._qlo[i] - 1e-9 or q[i] > self._qhi[i] + 1e-9]
        if bad:
            report.add(JOINT_LIMIT, f"Joint limit exceeded: {', '.join(bad)}")

    def _check_state(self, q: np.ndarray, report: SafetyReport) -> None:
        """Workspace and collision at one configuration (each reported once per report)."""
        if self._ws is not None and not report.has(WORKSPACE):
            p = self.model.ee_position(q)
            if np.any(p < self._ws[0]) or np.any(p > self._ws[1]):
                report.add(WORKSPACE, f"End-effector {np.round(p, 3).tolist()} is outside the workspace")
        if not report.has(COLLISION):
            clearance, link, obj = self.collision_clearance(q)
            if clearance < 0.0:
                report.add(COLLISION, f"{link} collides with '{obj}' (clearance {clearance:.3f} m)")

    def _ee_speeds(self, qa: np.ndarray, qb: np.ndarray, dt: float) -> Tuple[float, float]:
        Ta, Tb = self.model.fkine(qa).A, self.model.fkine(qb).A
        lin = float(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3])) / dt
        ang = float(np.linalg.norm(rotation_vector(Tb[:3, :3] @ Ta[:3, :3].T))) / dt
        return lin, ang

    # ---- configuration check --------------------------------------------------------
    def check_configuration(self, q) -> SafetyReport:
        """Static check: joint limits, workspace, collision. Singularity only warns."""
        report = SafetyReport()
        blocked = self._check_estop(report)
        qa = self._parse_q(q, "q", report)
        if blocked or qa is None:
            return report
        self._check_limits(qa, report)
        self._check_state(qa, report)
        thr = self.config.singularity_threshold
        if thr > 0:
            sv = min_singular_value(self.model, qa)
            if sv < thr:
                report.warn(f"Configuration is near a singularity (min singular value {sv:.4f})")
        return report

    # ---- step check -------------------------------------------------------------------
    def check_step(self, q_from, q_to, dt) -> SafetyReport:
        """Validate one motion step q_from -> q_to executed in dt seconds."""
        report = SafetyReport()
        blocked = self._check_estop(report)
        qa = self._parse_q(q_from, "q_from", report)
        qb = self._parse_q(q_to, "q_to", report)
        dt_ok = self._check_dt(dt, report)
        if blocked or qa is None or qb is None or not dt_ok:
            return report
        dt = float(dt)
        cfg = self.config
        dq = qb - qa

        over = np.abs(dq) / dt > self._v_limit * (1.0 + _SPEED_TOL) + 1e-12
        if np.any(over):
            names = [self.model.joint_names[i] for i in np.flatnonzero(over)]
            report.add(JOINT_SPEED, f"Joint speed limit exceeded: {', '.join(names)}")

        lin, ang = self._ee_speeds(qa, qb, dt)
        if lin > cfg.ee_speed_limit * (1.0 + _SPEED_TOL):
            report.add(EE_SPEED, f"End-effector speed {lin:.3f} m/s exceeds {cfg.ee_speed_limit} m/s")
        if ang > cfg.ee_ang_speed_limit * (1.0 + _SPEED_TOL):
            report.add(EE_ROT_SPEED,
                    f"End-effector rotation speed {ang:.3f} rad/s exceeds {cfg.ee_ang_speed_limit} rad/s")

        self._check_limits(qb, report)

        n = max(1, int(math.ceil(float(np.max(np.abs(dq))) / cfg.step_resolution)))
        for k in range(1, n + 1):
            self._check_state(qa + dq * (k / n), report)

        thr = cfg.singularity_threshold
        if thr > 0:
            sv_to = min_singular_value(self.model, qb)
            sv_from = min_singular_value(self.model, qa)
            if sv_to < thr and sv_to < sv_from - 1e-12:
                report.add(SINGULARITY,
                        f"Step moves towards a singularity (min singular value {sv_to:.4f})")
        return report

    # ---- trajectory check -----------------------------------------------------------------
    def check_trajectory(self, q_traj, dt, q_start=None, stop_at_first: bool = True) -> SafetyReport:
        """Validate an (N, n) joint trajectory sampled every dt seconds.

        If q_start is given, the jump q_start -> q_traj[0] is checked as a step; otherwise
        q_traj[0] is checked as a static configuration. Violations carry the row index.
        """
        report = SafetyReport()
        if self._check_estop(report):
            return report
        try:
            traj = np.asarray(q_traj, dtype=float)
        except (TypeError, ValueError):
            report.add(BAD_INPUT, "Trajectory is not numeric")
            return report
        if traj.ndim != 2 or traj.shape[1] != self.model.n or traj.shape[0] == 0:
            report.add(BAD_INPUT, f"Trajectory must have shape (N, {self.model.n}) with N >= 1")
            return report
        if not np.all(np.isfinite(traj)):
            report.add(BAD_INPUT, "Trajectory contains NaN or inf")
            return report
        if not self._check_dt(dt, report):
            return report

        first = (self.check_configuration(traj[0]) if q_start is None
                else self.check_step(q_start, traj[0], dt))
        report.merge(first, index=0)
        if not report.ok and stop_at_first:
            return report
        for i in range(1, traj.shape[0]):
            sub = self.check_step(traj[i - 1], traj[i], dt)
            report.merge(sub, index=i)
            if not sub.ok and stop_at_first:
                break
        return report

    # ---- step limiter (gate for gesture / GUI targets) ---------------------------------------
    def limit_step(self, q_from, q_to, dt) -> Tuple[Optional[np.ndarray], SafetyReport]:
        """Turn a requested target into a safe step.

        q_to is clipped into the joint limits and the step is scaled down so the joint and
        end-effector speed limits hold. The result is then validated with check_step().
        Returns (q_safe, report). If the final step is rejected, q_safe is q_from (the robot
        stays put). q_safe is None only if q_from itself is invalid.
        """
        report = SafetyReport()
        blocked = self._check_estop(report)
        qa = self._parse_q(q_from, "q_from", report)
        qb = self._parse_q(q_to, "q_to", report)
        dt_ok = self._check_dt(dt, report)
        if blocked or qa is None or qb is None or not dt_ok:
            return (None if qa is None else qa.copy()), report
        dt = float(dt)

        clipped = np.clip(qb, self._qlo, self._qhi)
        if not np.array_equal(clipped, qb):
            report.warn("q_to was clipped into the joint limits")
        dq = clipped - qa

        scale = 1.0
        peak = float(np.max(np.abs(dq) / (self._v_limit * dt)))
        if peak > 1.0:
            scale = 1.0 / peak
        cand = qa + dq * scale
        for _ in range(5):
            lin, ang = self._ee_speeds(qa, cand, dt)
            f = max(lin / self.config.ee_speed_limit, ang / self.config.ee_ang_speed_limit)
            if f <= 1.0:
                break
            scale *= 0.98 / f
            cand = qa + dq * scale
        if scale < 1.0:
            report.warn(f"Step scaled to {scale * 100:.1f}% to respect the speed limits")

        final = self.check_step(qa, cand, dt)
        report.merge(final)
        if not final.ok:
            return qa.copy(), report
        return cand, report

    # ---- hand command ------------------------------------------------------------------------
    def check_hand_command(
        self,
        grasp: Optional[str] = None,
        flexions: Optional[Mapping[str, float]] = None,
    ) -> SafetyReport:
        """Validate a grasp preset name and/or per-finger flexion fractions in [0, 1]."""
        report = SafetyReport()
        if self._check_estop(report):
            return report
        if grasp is None and flexions is None:
            report.add(HAND_COMMAND, "Empty hand command")
            return report
        if grasp is not None and (not isinstance(grasp, str) or grasp not in GRASP_PRESETS):
            report.add(HAND_COMMAND, f"Unknown grasp '{grasp}'. Valid: {list(GRASP_PRESETS)}")
        if flexions is not None:
            for name, value in dict(flexions).items():
                if name not in FINGER_NAMES:
                    report.add(HAND_COMMAND, f"Unknown finger '{name}'")
                    continue
                try:
                    f = float(value)
                except (TypeError, ValueError):
                    report.add(HAND_COMMAND, f"Flexion of '{name}' is not a number")
                    continue
                if not math.isfinite(f) or not 0.0 <= f <= 1.0:
                    report.add(HAND_COMMAND, f"Flexion of '{name}' must be in [0, 1], got {value}")
        return report

    # ---- pose target -----------------------------------------------------------------------------
    def check_pose_target(self, T) -> SafetyReport:
        """Cheap pre-check of a Cartesian target (SE3 or 4x4): valid pose, reach, workspace.
        Collision is checked later on the joint-space result (check_step / check_trajectory)."""
        report = SafetyReport()
        if self._check_estop(report):
            return report
        try:
            arr = np.asarray(T.A if isinstance(T, SE3) else T, dtype=float)
        except (TypeError, ValueError):
            report.add(BAD_INPUT, "Pose is not numeric")
            return report
        if arr.shape != (4, 4) or not np.all(np.isfinite(arr)):
            report.add(BAD_INPUT, "Pose must be a finite 4x4 matrix or a single SE3")
            return report
        R = arr[:3, :3]
        if (not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or np.linalg.det(R) < 0
                or not np.allclose(arr[3], [0.0, 0.0, 0.0, 1.0])):
            report.add(BAD_INPUT, "Pose is not a valid rigid transform")
            return report

        p = arr[:3, 3]
        reach = self.model.max_reach + float(np.linalg.norm(np.asarray(self.model.tool.t, dtype=float)))
        dist = float(np.linalg.norm(p - np.asarray(self.model.base.t, dtype=float)))
        if dist > reach:
            report.add(UNREACHABLE, f"Target is {dist:.3f} m from the base; max reach is {reach:.3f} m")
        if self._ws is not None and (np.any(p < self._ws[0]) or np.any(p > self._ws[1])):
            report.add(WORKSPACE, f"Target {np.round(p, 3).tolist()} is outside the workspace")
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    robot = RobotModel()
    checker = SafetyChecker(robot, Environment.default_workcell())
    print("Home:", checker.check_configuration(robot.q_home).summary())