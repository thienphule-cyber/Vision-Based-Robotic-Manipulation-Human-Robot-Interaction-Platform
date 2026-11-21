"""
modelling.py - Robot and environment modelling for VRM-HRI.

Contents
--------
1. RobotModel       : computational model of the Mitsubishi RV-2FR (Robotics Toolbox DHRobot).
2. load_visual_model: optional loader for the visual model from ir-support-extra-robots.
3. check_visual_sync: compares visual vs computational model (joint count, limits, FK).
4. Finger / HandModel: 5-finger robotic hand with grasp presets.
5. WorldObject / Environment: tables, obstacles, zones and manipulable objects.
6. SystemModel      : container that ties everything together.

IMPORTANT: the DH table below is a NOMINAL approximation of the RV-2FR.
It must be verified against the manufacturer's data sheet (or against the visual
model) before being trusted for real IK / accuracy-critical work.
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# 1. Computational robot model
# ----------------------------------------------------------------------------

# Standard DH rows (metres / radians). Joint limits in degrees (nominal data sheet values).
RV2FR_DH: List[dict] = [
    dict(d=0.295, a=0.000, alpha=-np.pi / 2, offset=0.0,       qlim_deg=(-240.0, 240.0)),
    dict(d=0.000, a=0.230, alpha=0.0,        offset=-np.pi / 2, qlim_deg=(-115.0, 125.0)),
    dict(d=0.000, a=0.050, alpha=-np.pi / 2, offset=0.0,       qlim_deg=(0.0, 164.0)),
    dict(d=0.270, a=0.000, alpha=np.pi / 2,  offset=0.0,       qlim_deg=(-160.0, 160.0)),
    dict(d=0.000, a=0.000, alpha=-np.pi / 2, offset=0.0,       qlim_deg=(-120.0, 120.0)),
    dict(d=0.070, a=0.000, alpha=0.0,        offset=0.0,       qlim_deg=(-360.0, 360.0)),
]


class RobotModel:
    """Computational (kinematic) model of the RV-2FR built with Robotics Toolbox."""

    def __init__(
        self,
        name: str = "Mitsubishi RV-2FR (computational)",
        dh_table: Optional[Sequence[dict]] = None,
        base: Optional[SE3] = None,
        tool: Optional[SE3] = None,
    ) -> None:
        table = RV2FR_DH if dh_table is None else dh_table
        links = []
        for row in table:
            lo, hi = row["qlim_deg"]
            links.append(
                rtb.RevoluteDH(
                    d=row["d"],
                    a=row["a"],
                    alpha=row["alpha"],
                    offset=row["offset"],
                    qlim=np.deg2rad([lo, hi]),
                )
            )
        self._robot = rtb.DHRobot(links, name=name)
        self._dh_table = list(table)
        self.joint_names: Tuple[str, ...] = tuple(f"J{i + 1}" for i in range(len(links)))
        self.set_base(SE3() if base is None else base)
        self.set_tool(SE3() if tool is None else tool)

    # ---- basic properties -------------------------------------------------
    @property
    def robot(self) -> rtb.DHRobot:
        """Underlying Robotics Toolbox robot (use this for IK, Jacobian, Swift, ...)."""
        return self._robot

    @property
    def n(self) -> int:
        return int(self._robot.n)

    @property
    def qlim(self) -> np.ndarray:
        """Joint limits in radians, shape (2, n): row 0 = lower, row 1 = upper."""
        return np.asarray(self._robot.qlim, dtype=float)

    @property
    def q_home(self) -> np.ndarray:
        """Home configuration (all zeros, clipped into the joint limits)."""
        return self.clip(np.zeros(self.n))

    @property
    def max_reach(self) -> float:
        """Upper bound of the distance from the base origin to the tool origin."""
        return float(sum(abs(r["a"]) + abs(r["d"]) for r in self._dh_table))

    # ---- frames -----------------------------------------------------------
    def set_base(self, base: SE3) -> None:
        self._robot.base = base

    def set_tool(self, tool: SE3) -> None:
        self._robot.tool = tool

    @property
    def base(self) -> SE3:
        return self._robot.base

    @property
    def tool(self) -> SE3:
        return self._robot.tool

    # ---- joint-space helpers ---------------------------------------------
    def _check_q(self, q) -> np.ndarray:
        arr = np.asarray(q, dtype=float).reshape(-1)
        if arr.shape != (self.n,):
            raise ValueError(f"Expected {self.n} joint values, got shape {arr.shape}")
        return arr

    def within_limits(self, q, tol: float = 1e-9) -> bool:
        arr = self._check_q(q)
        lo, hi = self.qlim
        return bool(np.all(arr >= lo - tol) and np.all(arr <= hi + tol))

    def clip(self, q) -> np.ndarray:
        arr = self._check_q(q)
        lo, hi = self.qlim
        return np.clip(arr, lo, hi)

    def random_q(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        lo, hi = self.qlim
        return rng.uniform(lo, hi)

    # ---- kinematics -------------------------------------------------------
    def fkine(self, q) -> SE3:
        return self._robot.fkine(self._check_q(q))

    def ee_position(self, q) -> np.ndarray:
        return np.asarray(self.fkine(q).t, dtype=float)


# ----------------------------------------------------------------------------
# 2. Visual model loading and synchronisation
# ----------------------------------------------------------------------------

# The class name in the project notes is "MitsubishiRV2RF"; the robot is an RV-2FR,
# so both spellings are tried.
VISUAL_CLASS_CANDIDATES = ("MitsubishiRV2FR", "MitsubishiRV2RF")


def load_visual_model(
    module_name: str = "ir_support_extra_robots.robots",
    class_names: Sequence[str] = VISUAL_CLASS_CANDIDATES,
):
    """Load the visual model. Returns None (and logs a warning) if unavailable."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        logger.warning("Visual model package not available: %s", exc)
        return None
    for cname in class_names:
        cls = getattr(module, cname, None)
        if cls is not None:
            try:
                return cls()
            except Exception as exc:  # noqa: BLE001 - third-party constructor
                logger.warning("Could not instantiate %s: %s", cname, exc)
                return None
    logger.warning("None of %s found in %s", list(class_names), module_name)
    return None


@dataclass
class SyncReport:
    ok: bool
    n_joints_match: bool = False
    limits_match: bool = False
    max_position_error: float = float("nan")
    max_rotation_error: float = float("nan")
    messages: List[str] = field(default_factory=list)


def check_visual_sync(
    visual,
    model: RobotModel,
    n_samples: int = 10,
    pos_tol: float = 1e-3,
    rot_tol: float = 1e-2,
    limit_tol: float = 1e-6,
    seed: int = 0,
) -> SyncReport:
    """Compare a visual model against the computational model.

    Checks joint count, joint limits and forward kinematics at random configurations.
    """
    report = SyncReport(ok=False)
    if visual is None:
        report.messages.append("Visual model not loaded.")
        return report

    n_visual = getattr(visual, "n", None)
    report.n_joints_match = n_visual == model.n
    if not report.n_joints_match:
        report.messages.append(f"Joint count mismatch: visual={n_visual}, model={model.n}")
        return report

    try:
        v_qlim = np.asarray(visual.qlim, dtype=float)
        report.limits_match = v_qlim.shape == model.qlim.shape and bool(
            np.allclose(v_qlim, model.qlim, atol=limit_tol)
        )
    except Exception as exc:  # noqa: BLE001
        report.messages.append(f"Could not read visual joint limits: {exc}")
    if not report.limits_match:
        report.messages.append("Joint limits differ between visual and computational model.")

    rng = np.random.default_rng(seed)
    max_pos, max_rot = 0.0, 0.0
    try:
        for _ in range(n_samples):
            q = model.random_q(rng)
            t_v = visual.fkine(q)
            t_m = model.fkine(q)
            max_pos = max(max_pos, float(np.linalg.norm(np.asarray(t_v.t) - np.asarray(t_m.t))))
            r_rel = np.asarray(t_v.R).T @ np.asarray(t_m.R)
            cos_a = np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0)
            max_rot = max(max_rot, float(np.arccos(cos_a)))
    except Exception as exc:  # noqa: BLE001
        report.messages.append(f"FK comparison failed: {exc}")
        return report

    report.max_position_error = max_pos
    report.max_rotation_error = max_rot
    if max_pos > pos_tol:
        report.messages.append(f"FK position error {max_pos:.4f} m exceeds {pos_tol} m")
    if max_rot > rot_tol:
        report.messages.append(f"FK rotation error {max_rot:.4f} rad exceeds {rot_tol} rad")

    report.ok = report.limits_match and max_pos <= pos_tol and max_rot <= rot_tol
    return report


# ----------------------------------------------------------------------------
# 3. Robotic hand (5 fingers, multiple joints per finger)
# ----------------------------------------------------------------------------

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

# name -> (joint names, upper flexion limits [deg], link lengths [m], base offset in palm frame [m])
# Palm frame: x = along the fingers, y = lateral, z = palm normal (fingers curl toward -z).
# Lower flexion limit is 0 for every joint. Abduction/adduction is not modelled.
_FINGER_SPECS = {
    "thumb":  (("cmc", "mcp", "ip"),  (50.0, 60.0, 80.0),  (0.045, 0.032, 0.027), (0.020, -0.040, 0.0)),
    "index":  (("mcp", "pip", "dip"), (90.0, 100.0, 80.0), (0.045, 0.025, 0.020), (0.090, -0.030, 0.0)),
    "middle": (("mcp", "pip", "dip"), (90.0, 100.0, 80.0), (0.050, 0.028, 0.022), (0.095, -0.010, 0.0)),
    "ring":   (("mcp", "pip", "dip"), (90.0, 100.0, 80.0), (0.045, 0.026, 0.020), (0.090, 0.010, 0.0)),
    "pinky":  (("mcp", "pip", "dip"), (90.0, 100.0, 80.0), (0.035, 0.020, 0.018), (0.080, 0.030, 0.0)),
}

# Flexion fraction per finger (0 = fully open, 1 = fully flexed). Missing fingers = 0.
GRASP_PRESETS: Dict[str, Dict[str, float]] = {
    "open":      {},
    "closed":    {n: 1.0 for n in FINGER_NAMES},
    "pinch":     {"thumb": 0.6, "index": 0.6},
    "precision": {"thumb": 0.5, "index": 0.5, "middle": 0.5},
    "power":     {"thumb": 0.7, "index": 0.85, "middle": 0.85, "ring": 0.85, "pinky": 0.85},
}


@dataclass(eq=False)
class Finger:
    """A finger modelled as a planar serial chain of flexion joints."""

    name: str
    joint_names: Tuple[str, ...]
    qlim: np.ndarray            # shape (n_joints, 2), radians
    link_lengths: Tuple[float, ...]
    base_offset: np.ndarray     # shape (3,), in palm frame
    q: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.qlim = np.asarray(self.qlim, dtype=float)
        self.base_offset = np.asarray(self.base_offset, dtype=float).reshape(3)
        n = len(self.joint_names)
        if self.qlim.shape != (n, 2) or len(self.link_lengths) != n:
            raise ValueError(f"Finger '{self.name}': inconsistent joint/limit/link definition")
        self.q = np.clip(np.zeros(n), self.qlim[:, 0], self.qlim[:, 1])

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    def set_q(self, q) -> np.ndarray:
        arr = np.asarray(q, dtype=float).reshape(-1)
        if arr.shape != (self.n_joints,):
            raise ValueError(f"Finger '{self.name}' expects {self.n_joints} joint values")
        self.q = np.clip(arr, self.qlim[:, 0], self.qlim[:, 1])
        return self.q

    def set_flexion(self, fraction: float) -> np.ndarray:
        f = float(np.clip(fraction, 0.0, 1.0))
        lo, hi = self.qlim[:, 0], self.qlim[:, 1]
        return self.set_q(lo + f * (hi - lo))

    @property
    def flexion(self) -> float:
        lo, hi = self.qlim[:, 0], self.qlim[:, 1]
        return float(np.mean((self.q - lo) / (hi - lo)))

    def fingertip_position(self) -> np.ndarray:
        """Fingertip position in the palm frame."""
        theta, x, z = 0.0, 0.0, 0.0
        for q_i, length in zip(self.q, self.link_lengths):
            theta += q_i
            x += length * np.cos(theta)
            z -= length * np.sin(theta)
        return self.base_offset + np.array([x, 0.0, z])


class HandModel:
    """Five-finger robotic hand with per-finger control and grasp presets."""

    def __init__(self) -> None:
        self.fingers: Dict[str, Finger] = {}
        for name in FINGER_NAMES:
            joints, hi_deg, lengths, offset = _FINGER_SPECS[name]
            qlim = np.column_stack([np.zeros(len(joints)), np.deg2rad(hi_deg)])
            self.fingers[name] = Finger(name, joints, qlim, lengths, np.array(offset))

    def finger(self, name: str) -> Finger:
        if name not in self.fingers:
            raise KeyError(f"Unknown finger '{name}'. Valid: {list(self.fingers)}")
        return self.fingers[name]

    @property
    def n_joints(self) -> int:
        return sum(f.n_joints for f in self.fingers.values())

    @property
    def q(self) -> Dict[str, np.ndarray]:
        return {name: f.q.copy() for name, f in self.fingers.items()}

    def set_finger(self, name: str, q) -> np.ndarray:
        return self.finger(name).set_q(q)

    def set_finger_flexion(self, name: str, fraction: float) -> np.ndarray:
        return self.finger(name).set_flexion(fraction)

    def apply_grasp(self, preset: str) -> None:
        if preset not in GRASP_PRESETS:
            raise ValueError(f"Unknown grasp '{preset}'. Valid: {list(GRASP_PRESETS)}")
        spec = GRASP_PRESETS[preset]
        for name, finger in self.fingers.items():
            finger.set_flexion(spec.get(name, 0.0))

    def open_hand(self) -> None:
        self.apply_grasp("open")

    def close_hand(self) -> None:
        self.apply_grasp("closed")

    @staticmethod
    def _as_matrix(T) -> np.ndarray:
        if T is None:
            return np.eye(4)
        if isinstance(T, SE3):
            return np.asarray(T.A, dtype=float)
        arr = np.asarray(T, dtype=float)
        if arr.shape != (4, 4):
            raise ValueError("Mount transform must be an SE3 or a 4x4 matrix")
        return arr

    def fingertip_positions(self, T_mount=None) -> Dict[str, np.ndarray]:
        """Fingertip positions; expressed in the world frame if T_mount (palm pose) is given."""
        T = self._as_matrix(T_mount)
        out = {}
        for name, finger in self.fingers.items():
            p = finger.fingertip_position()
            out[name] = T[:3, :3] @ p + T[:3, 3]
        return out


# ----------------------------------------------------------------------------
# 4. Environment
# ----------------------------------------------------------------------------

OBJECT_KINDS = ("table", "obstacle", "target_zone", "storage_zone", "object")
OBJECT_SHAPES = ("box", "sphere", "cylinder", "custom")
# Number of dimension values expected per shape:
#   box = (x, y, z) full lengths, sphere = (radius,), cylinder = (radius, height),
#   custom = (bounding_radius,)
_SHAPE_DIMS = {"box": 3, "sphere": 1, "cylinder": 2, "custom": 1}


@dataclass(eq=False)
class WorldObject:
    name: str
    kind: str
    shape: str
    position: Sequence[float]
    dims: Sequence[float]
    yaw: float = 0.0
    color: Tuple[float, float, float] = (0.6, 0.6, 0.6)
    graspable: bool = False

    def __post_init__(self) -> None:
        if self.kind not in OBJECT_KINDS:
            raise ValueError(f"Invalid kind '{self.kind}'. Valid: {OBJECT_KINDS}")
        if self.shape not in OBJECT_SHAPES:
            raise ValueError(f"Invalid shape '{self.shape}'. Valid: {OBJECT_SHAPES}")
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.dims = tuple(float(d) for d in self.dims)
        if len(self.dims) != _SHAPE_DIMS[self.shape]:
            raise ValueError(f"Shape '{self.shape}' needs {_SHAPE_DIMS[self.shape]} dimension(s)")
        if any(d <= 0 for d in self.dims):
            raise ValueError("All dimensions must be positive")

    def aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box (min, max) in the world frame (conservative for rotated boxes)."""
        if self.shape == "box":
            lx, ly, lz = self.dims
            c, s = abs(np.cos(self.yaw)), abs(np.sin(self.yaw))
            half = np.array([(c * lx + s * ly) / 2, (s * lx + c * ly) / 2, lz / 2])
        elif self.shape == "cylinder":
            r, h = self.dims
            half = np.array([r, r, h / 2])
        else:  # sphere or custom (bounding radius)
            half = np.full(3, self.dims[0])
        return self.position - half, self.position + half

    def contains_point(self, point, margin: float = 0.0) -> bool:
        p = np.asarray(point, dtype=float).reshape(3)
        lo, hi = self.aabb()
        return bool(np.all(p >= lo - margin) and np.all(p <= hi + margin))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "shape": self.shape,
            "position": self.position.tolist(),
            "dims": list(self.dims),
            "yaw": float(self.yaw),
            "color": [float(c) for c in self.color],
            "graspable": bool(self.graspable),
        }


class Environment:
    """Collection of tables, obstacles, zones and manipulable objects."""

    def __init__(self) -> None:
        self._objects: Dict[str, WorldObject] = {}

    def __len__(self) -> int:
        return len(self._objects)

    def __contains__(self, name: str) -> bool:
        return name in self._objects

    def add(self, obj: WorldObject) -> WorldObject:
        if obj.name in self._objects:
            raise ValueError(f"Object '{obj.name}' already exists")
        self._objects[obj.name] = obj
        return obj

    def remove(self, name: str) -> WorldObject:
        if name not in self._objects:
            raise KeyError(f"Unknown object '{name}'")
        return self._objects.pop(name)

    def get(self, name: str) -> WorldObject:
        if name not in self._objects:
            raise KeyError(f"Unknown object '{name}'")
        return self._objects[name]

    def names(self) -> List[str]:
        return list(self._objects)

    def by_kind(self, kind: str) -> List[WorldObject]:
        return [o for o in self._objects.values() if o.kind == kind]

    def colliding_objects(
        self, point, margin: float = 0.0, kinds: Sequence[str] = ("table", "obstacle")
    ) -> List[str]:
        return [
            o.name
            for o in self._objects.values()
            if o.kind in kinds and o.contains_point(point, margin)
        ]

    def point_in_collision(
        self, point, margin: float = 0.0, kinds: Sequence[str] = ("table", "obstacle")
    ) -> bool:
        return len(self.colliding_objects(point, margin, kinds)) > 0

    def find_zone(self, point, kind: str = "target_zone") -> Optional[WorldObject]:
        for obj in self.by_kind(kind):
            if obj.contains_point(point):
                return obj
        return None

    def to_dict(self) -> dict:
        return {"objects": [o.to_dict() for o in self._objects.values()]}

    @classmethod
    def default_workcell(cls) -> "Environment":
        """Simple workcell: table (top at z = 0), one obstacle, two zones and three objects."""
        env = cls()
        env.add(WorldObject("table", "table", "box", (0.3, 0.0, -0.35), (1.0, 1.0, 0.70),
                            color=(0.5, 0.4, 0.3)))
        env.add(WorldObject("obstacle_1", "obstacle", "box", (0.55, 0.0, 0.1), (0.05, 0.05, 0.2),
                            color=(0.8, 0.2, 0.2)))
        env.add(WorldObject("target_zone", "target_zone", "box", (0.4, -0.25, 0.1), (0.15, 0.15, 0.2),
                            color=(0.2, 0.8, 0.2)))
        env.add(WorldObject("storage_zone", "storage_zone", "box", (0.2, 0.3, 0.1), (0.15, 0.15, 0.2),
                            color=(0.2, 0.2, 0.8)))
        env.add(WorldObject("cube_1", "object", "box", (0.35, 0.2, 0.025), (0.05, 0.05, 0.05),
                            color=(1.0, 0.0, 0.0), graspable=True))
        env.add(WorldObject("sphere_1", "object", "sphere", (0.30, 0.10, 0.025), (0.025,),
                            color=(0.0, 1.0, 0.0), graspable=True))
        env.add(WorldObject("cylinder_1", "object", "cylinder", (0.25, -0.15, 0.04), (0.025, 0.08),
                            color=(0.0, 0.0, 1.0), graspable=True))
        return env


# ----------------------------------------------------------------------------
# 5. System container
# ----------------------------------------------------------------------------

class SystemModel:
    """Holds the robot, hand, environment and (optional) visual model."""

    def __init__(
        self,
        robot: Optional[RobotModel] = None,
        hand: Optional[HandModel] = None,
        environment: Optional[Environment] = None,
        visual=None,
    ) -> None:
        self.robot = robot if robot is not None else RobotModel()
        self.hand = hand if hand is not None else HandModel()
        self.environment = environment if environment is not None else Environment()
        self.visual = visual

    @classmethod
    def build_default(cls, load_visual: bool = True) -> "SystemModel":
        visual = load_visual_model() if load_visual else None
        return cls(RobotModel(), HandModel(), Environment.default_workcell(), visual)

    def check_sync(self, **kwargs) -> SyncReport:
        return check_visual_sync(self.visual, self.robot, **kwargs)

    def summary(self) -> dict:
        return {
            "robot": self.robot.robot.name,
            "n_joints": self.robot.n,
            "visual_loaded": self.visual is not None,
            "hand_joints": self.hand.n_joints,
            "environment_objects": self.environment.names(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    system = SystemModel.build_default()
    print(system.summary())
    print(system.check_sync())