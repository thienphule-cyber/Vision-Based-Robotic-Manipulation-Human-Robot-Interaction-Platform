"""
computer_vision.py - MediaPipe hand tracking and gesture control for VRM-HRI.

Scope: hand tracking and gesture-based robot control only (no object detection, no
object tracking, no camera calibration).

Contents
--------
1. Constants / configs : MediaPipe landmark ids and the tunable configuration objects.
2. HandObservation     : 21 hand landmarks of one frame.
3. Hand features       : per-finger flexion, pinch distance, hand size, palm position.
4. Gestures            : rule-based classifier + debouncer (stable gestures only).
5. Mapping             : hand position in the image -> robot target position (low-pass
                        filtered, with dead zone, clamped to a box around `center`).
6. GestureController   : observation -> GestureCommand (follow / hold / stop, grasp
                        preset, per-finger flexions, target position).
7. HandTracker         : MediaPipe wrapper (legacy `solutions` API or the Tasks API).
8. Camera / pipeline   : OpenCV capture and a background thread that publishes the
                        latest command.

Data flow (outside this module)
-------------------------------
    VisionPipeline.latest().command
        -> target_pose(command, orientation)        (SE3 target)
        -> kinematics.servo_to_pose(...)            (next joint step)
        -> safety.SafetyChecker.limit_step(...)     (the ONLY gate to the simulation)
        -> simulation.Simulation.set_joint_positions(...)
    command.grasp / command.finger_flexions
        -> safety.check_hand_command(...) -> simulation.set_hand_grasp / set_finger_flexion
    command.stop_requested
        -> simulation.stop() and/or safety.estop(...)   (decided by the caller)

Notes
-----
* This module never moves the robot and never enforces safety. It only produces commands.
* Thresholds are heuristics that work for a typical webcam at 40-80 cm. Tune them in
GestureConfig / MappingConfig.
* mediapipe and cv2 are imported lazily, so the module can be imported and tested without
them.
* Handedness labels from MediaPipe assume a mirrored image; they are reported but never
used for decisions.
"""
from __future__ import annotations

import enum
import importlib
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from spatialmath import SE3

from modelling import FINGER_NAMES

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# 1. Constants and configuration
# ----------------------------------------------------------------------------

N_LANDMARKS = 21
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9
PALM_LANDMARKS = (0, 5, 9, 13, 17)

# MediaPipe hand landmark ids: thumb = (CMC, MCP, IP, TIP); other fingers = (MCP, PIP, DIP, TIP).
FINGER_LANDMARKS: Dict[str, Tuple[int, int, int, int]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
_FOUR_FINGERS = ("index", "middle", "ring", "pinky")

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


class Gesture(enum.Enum):
    NONE = "none"        # no hand
    UNKNOWN = "unknown"  # hand present, no known gesture
    OPEN = "open"        # all four fingers extended
    FIST = "fist"        # all four fingers curled
    PINCH = "pinch"      # thumb tip close to index tip
    POINT = "point"      # only the index finger extended
    PEACE = "peace"      # index and middle extended, ring and pinky curled


class Action(enum.Enum):
    NONE = "none"        # no hand
    FOLLOW = "follow"    # robot follows the hand
    HOLD = "hold"        # keep the current robot pose
    STOP = "stop"        # stop request (edge flag in GestureCommand.stop_requested)


GESTURE_ACTIONS: Dict[Gesture, Action] = {
    Gesture.NONE: Action.NONE,
    Gesture.UNKNOWN: Action.HOLD,
    Gesture.OPEN: Action.FOLLOW,
    Gesture.FIST: Action.FOLLOW,
    Gesture.PINCH: Action.FOLLOW,
    Gesture.POINT: Action.HOLD,
    Gesture.PEACE: Action.STOP,
}

# Grasp presets (names from modelling.GRASP_PRESETS) sent to the robotic hand while following.
GESTURE_GRASP: Dict[Gesture, str] = {
    Gesture.OPEN: "open",
    Gesture.FIST: "closed",
    Gesture.PINCH: "pinch",
}


@dataclass
class TrackerConfig:
    max_hands: int = 1
    detection_confidence: float = 0.6
    tracking_confidence: float = 0.5
    min_score: float = 0.6            # detections with a lower handedness score are dropped
    model_complexity: int = 0         # legacy API only (0 = fast, 1 = accurate)
    model_path: Optional[str] = None  # Tasks API: path to hand_landmarker.task

    def __post_init__(self) -> None:
        if self.max_hands < 1:
            raise ValueError("max_hands must be >= 1")
        for name in ("detection_confidence", "tracking_confidence", "min_score"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass
class GestureConfig:
    # Joint angle -> flexion fraction: angle >= open -> 0, angle <= closed -> 1.
    open_angle_deg: float = 165.0
    closed_angle_deg: float = 60.0
    thumb_open_angle_deg: float = 160.0
    thumb_closed_angle_deg: float = 110.0
    extended_below: float = 0.35      # flexion below this = finger extended
    curled_above: float = 0.65        # flexion above this = finger curled
    pinch_on: float = 0.25            # thumb-index distance / hand size to enter PINCH
    pinch_off: float = 0.40           # ... to leave PINCH (hysteresis)
    frames_to_switch: int = 4         # consecutive frames needed to change gesture
    lost_frames: int = 8              # consecutive frames without a hand to drop to NONE
    min_hand_size: float = 0.03       # wrist-to-middle-MCP length (image height units)

    def __post_init__(self) -> None:
        if self.open_angle_deg <= self.closed_angle_deg:
            raise ValueError("open_angle_deg must be larger than closed_angle_deg")
        if self.thumb_open_angle_deg <= self.thumb_closed_angle_deg:
            raise ValueError("thumb_open_angle_deg must be larger than thumb_closed_angle_deg")
        if not 0.0 <= self.extended_below < self.curled_above <= 1.0:
            raise ValueError("need 0 <= extended_below < curled_above <= 1")
        if not 0.0 < self.pinch_on < self.pinch_off:
            raise ValueError("need 0 < pinch_on < pinch_off")
        if self.frames_to_switch < 1 or self.lost_frames < 1:
            raise ValueError("frames_to_switch and lost_frames must be >= 1")
        if self.min_hand_size < 0:
            raise ValueError("min_hand_size must be non-negative")


@dataclass
class MappingConfig:
    """Absolute mapping from the hand in the image to a robot position (world frame).

    image x (left-right)  -> robot y   (user's right hand movement = +y unless invert_lateral)
    image y (up-down)     -> robot z   (hand up = +z)
    apparent hand size    -> robot x   (hand closer to the camera = further forward)
    """
    center: Tuple[float, float, float] = (0.35, 0.0, 0.35)  # target for a hand at the image centre
    span: Tuple[float, float, float] = (0.12, 0.25, 0.20)   # half range along x, y, z [m]
    margin: float = 0.15              # image border fraction that is not used
    size_ref: float = 0.18            # hand size (image height units) at the neutral distance
    depth_range: float = 0.5          # relative size change that gives the full x span
    invert_lateral: bool = False
    pos_tau: float = 0.15             # low-pass time constant of the target position [s]
    flex_tau: float = 0.05            # low-pass time constant of the finger flexions [s]
    deadzone: float = 0.003           # target changes smaller than this are ignored [m]
    max_dt: float = 0.5               # largest time step accepted by the filters [s]

    def __post_init__(self) -> None:
        if len(self.center) != 3 or len(self.span) != 3:
            raise ValueError("center and span need 3 values")
        if any(s < 0 for s in self.span):
            raise ValueError("span must be non-negative")
        if not 0.0 <= self.margin < 0.5:
            raise ValueError("margin must be in [0, 0.5)")
        if self.size_ref <= 0 or self.depth_range <= 0:
            raise ValueError("size_ref and depth_range must be positive")
        if self.pos_tau < 0 or self.flex_tau < 0 or self.deadzone < 0:
            raise ValueError("pos_tau, flex_tau and deadzone must be non-negative")
        if self.max_dt <= 0:
            raise ValueError("max_dt must be positive")


# ----------------------------------------------------------------------------
# 2. Observation
# ----------------------------------------------------------------------------

@dataclass
class HandObservation:
    """One detected hand. `landmarks` are MediaPipe image-normalised coordinates (21, 3):
    x in [0, 1] of the width, y in [0, 1] of the height (downwards), z relative depth with
    roughly the same scale as x."""
    landmarks: np.ndarray
    aspect: float = 1.0       # image width / height
    label: str = ""
    score: float = 1.0
    timestamp: float = 0.0

    def metric(self) -> np.ndarray:
        """Landmarks with isotropic scale (units of image height), for angles and distances."""
        pts = np.asarray(self.landmarks, dtype=float).copy()
        pts[:, 0] *= self.aspect
        pts[:, 2] *= self.aspect
        return pts


def make_observation(
    landmarks,
    *,
    aspect: float = 1.0,
    label: str = "",
    score: float = 1.0,
    timestamp: float = 0.0,
) -> HandObservation:
    arr = np.asarray(landmarks, dtype=float)
    if arr.shape != (N_LANDMARKS, 3):
        raise ValueError(f"Expected landmarks of shape ({N_LANDMARKS}, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Landmarks contain NaN or inf")
    if not aspect > 0:
        raise ValueError("aspect must be positive")
    return HandObservation(arr.copy(), float(aspect), str(label), float(score), float(timestamp))


def landmarks_to_array(landmarks) -> np.ndarray:
    """Convert an iterable of objects with .x .y .z (MediaPipe landmarks) to an (N, 3) array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=float)


# ----------------------------------------------------------------------------
# 3. Hand features
# ----------------------------------------------------------------------------

def joint_angle_deg(a, b, c) -> float:
    """Interior angle at b between the segments b->a and b->c, in degrees (180 = straight)."""
    u = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return 180.0
    cos = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def finger_flexion(pts: np.ndarray, finger: str, config: Optional[GestureConfig] = None) -> float:
    """Flexion fraction in [0, 1] (0 = straight, 1 = fully curled) from isotropic landmarks.

    Fingers use the angle at the PIP joint (MCP, PIP, TIP); the thumb uses the angle at
    the IP joint (MCP, IP, TIP).
    """
    cfg = GestureConfig() if config is None else config
    ids = FINGER_LANDMARKS[finger]
    if finger == "thumb":
        a, b, c = ids[1], ids[2], ids[3]
        open_deg, closed_deg = cfg.thumb_open_angle_deg, cfg.thumb_closed_angle_deg
    else:
        a, b, c = ids[0], ids[1], ids[3]
        open_deg, closed_deg = cfg.open_angle_deg, cfg.closed_angle_deg
    angle = joint_angle_deg(pts[a], pts[b], pts[c])
    return float(np.clip((open_deg - angle) / (open_deg - closed_deg), 0.0, 1.0))


@dataclass
class HandFeatures:
    flexions: Dict[str, float]    # finger name -> flexion fraction
    pinch_distance: float         # thumb tip - index tip distance / hand size
    hand_size: float              # wrist - middle MCP length (image height units)
    palm_uv: np.ndarray           # palm centre in normalised image coordinates (x, y)


def extract_features(
    obs: HandObservation, config: Optional[GestureConfig] = None
) -> Optional[HandFeatures]:
    """Features of one observation, or None if the hand is degenerate / too small."""
    cfg = GestureConfig() if config is None else config
    pts = obs.metric()
    hand_size = float(np.linalg.norm(pts[WRIST] - pts[MIDDLE_MCP]))
    if not math.isfinite(hand_size) or hand_size < max(cfg.min_hand_size, 1e-9):
        return None
    flexions = {name: finger_flexion(pts, name, cfg) for name in FINGER_NAMES}
    pinch = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP])) / hand_size
    palm_uv = np.asarray(obs.landmarks, dtype=float)[list(PALM_LANDMARKS), :2].mean(axis=0)
    return HandFeatures(flexions, pinch, hand_size, palm_uv)


# ----------------------------------------------------------------------------
# 4. Gesture classification and debouncing
# ----------------------------------------------------------------------------

class GestureClassifier:
    """Rule-based classifier (single frame, with hysteresis on PINCH)."""

    def __init__(self, config: Optional[GestureConfig] = None) -> None:
        self.config = GestureConfig() if config is None else config
        self._pinching = False

    def reset(self) -> None:
        self._pinching = False

    def classify(self, features: HandFeatures) -> Gesture:
        cfg = self.config
        threshold = cfg.pinch_off if self._pinching else cfg.pinch_on
        self._pinching = features.pinch_distance < threshold
        if self._pinching:
            return Gesture.PINCH

        ext = {n: features.flexions[n] < cfg.extended_below for n in _FOUR_FINGERS}
        cur = {n: features.flexions[n] > cfg.curled_above for n in _FOUR_FINGERS}
        if all(ext.values()):
            return Gesture.OPEN
        if all(cur.values()):
            return Gesture.FIST
        if ext["index"] and cur["middle"] and cur["ring"] and cur["pinky"]:
            return Gesture.POINT
        if ext["index"] and ext["middle"] and cur["ring"] and cur["pinky"]:
            return Gesture.PEACE
        return Gesture.UNKNOWN


class GestureDebouncer:
    """Turns raw per-frame gestures into a stable gesture.

    A new gesture must be seen on `frames_to_switch` consecutive frames; losing the hand
    (raw NONE) must last `lost_frames` consecutive frames.
    """

    def __init__(self, frames_to_switch: int = 4, lost_frames: int = 8) -> None:
        if frames_to_switch < 1 or lost_frames < 1:
            raise ValueError("frames_to_switch and lost_frames must be >= 1")
        self._frames = frames_to_switch
        self._lost = lost_frames
        self.reset()

    def reset(self) -> None:
        self._stable = Gesture.NONE
        self._candidate = Gesture.NONE
        self._count = 0

    @property
    def stable(self) -> Gesture:
        return self._stable

    def update(self, raw: Gesture) -> Gesture:
        if raw == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = raw, 1
        needed = self._lost if raw is Gesture.NONE else self._frames
        if self._candidate != self._stable and self._count >= needed:
            self._stable = self._candidate
        return self._stable


# ----------------------------------------------------------------------------
# 5. Filtering and mapping
# ----------------------------------------------------------------------------

class LowPassFilter:
    """First-order exponential filter with a time constant `tau` (seconds).

    The first sample initialises the state. tau <= 0 disables filtering.
    """

    def __init__(self, tau: float) -> None:
        if tau < 0:
            raise ValueError("tau must be non-negative")
        self.tau = float(tau)
        self._state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._state = None

    @property
    def value(self) -> Optional[np.ndarray]:
        return None if self._state is None else self._state.copy()

    def update(self, x, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._state is None or self.tau <= 0:
            self._state = x.copy()
        elif dt > 0:
            alpha = 1.0 - math.exp(-dt / self.tau)
            self._state = self._state + alpha * (x - self._state)
        return self._state.copy()


def map_hand_to_position(
    uv, hand_size: float, config: Optional[MappingConfig] = None
) -> np.ndarray:
    """Robot target position (world frame, metres) for a hand at image position `uv`.

    The result always lies inside the box center +/- span.
    """
    cfg = MappingConfig() if config is None else config
    half = 0.5 - cfg.margin
    u_n = float(np.clip((float(uv[0]) - 0.5) / half, -1.0, 1.0))
    v_n = float(np.clip((float(uv[1]) - 0.5) / half, -1.0, 1.0))
    d_n = float(np.clip((hand_size / cfg.size_ref - 1.0) / cfg.depth_range, -1.0, 1.0))
    lateral = u_n if cfg.invert_lateral else -u_n
    cx, cy, cz = cfg.center
    sx, sy, sz = cfg.span
    return np.array([cx + sx * d_n, cy + sy * lateral, cz - sz * v_n])


def target_pose(command: "GestureCommand", orientation: Optional[SE3]) -> Optional[SE3]:
    """SE3 target from a FOLLOW command and a fixed tool orientation (rotation of `orientation`).

    Returns None when the command carries no target or no orientation is given.
    """
    if command.target_position is None or orientation is None:
        return None
    return SE3.Rt(np.asarray(orientation.R, dtype=float), np.asarray(command.target_position, dtype=float))


# ----------------------------------------------------------------------------
# 6. Gesture controller
# ----------------------------------------------------------------------------

@dataclass
class GestureCommand:
    timestamp: float
    hand_detected: bool
    raw_gesture: Gesture                       # this frame, before debouncing
    gesture: Gesture                           # stable gesture
    action: Action
    target_position: Optional[np.ndarray] = None            # only when action is FOLLOW
    grasp: Optional[str] = None                             # GRASP_PRESETS name, only when FOLLOW
    finger_flexions: Optional[Dict[str, float]] = None      # smoothed, whenever a hand is seen
    stop_requested: bool = False               # True only on the frame a stable PEACE starts
    confidence: float = 0.0

    @property
    def tracking(self) -> bool:
        return self.action is Action.FOLLOW


class GestureController:
    """Observation -> GestureCommand. Not thread safe: call it from one thread."""

    def __init__(
        self,
        gesture_config: Optional[GestureConfig] = None,
        mapping_config: Optional[MappingConfig] = None,
    ) -> None:
        self.gesture_config = GestureConfig() if gesture_config is None else gesture_config
        self.mapping = MappingConfig() if mapping_config is None else mapping_config
        self._classifier = GestureClassifier(self.gesture_config)
        self._debouncer = GestureDebouncer(
            self.gesture_config.frames_to_switch, self.gesture_config.lost_frames
        )
        self._pos_filter = LowPassFilter(self.mapping.pos_tau)
        self._flex_filter = LowPassFilter(self.mapping.flex_tau)
        self._last_t: Optional[float] = None
        self._last_target: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._classifier.reset()
        self._debouncer.reset()
        self._reset_filters()
        self._last_t = None

    def _reset_filters(self) -> None:
        self._pos_filter.reset()
        self._flex_filter.reset()
        self._last_target = None

    def update(self, obs: Optional[HandObservation], t: Optional[float] = None) -> GestureCommand:
        cfg = self.mapping
        now = time.monotonic() if t is None else float(t)
        dt = 0.0 if self._last_t is None else min(max(now - self._last_t, 0.0), cfg.max_dt)
        self._last_t = now

        features = None if obs is None else extract_features(obs, self.gesture_config)
        if features is None:
            self._classifier.reset()
            raw = Gesture.NONE
        else:
            raw = self._classifier.classify(features)

        previous = self._debouncer.stable
        stable = self._debouncer.update(raw)
        if features is None and stable is Gesture.NONE:
            self._reset_filters()  # the hand is really gone: re-acquire from scratch

        flexions: Optional[Dict[str, float]] = None
        position: Optional[np.ndarray] = None
        if features is not None:
            target = map_hand_to_position(features.palm_uv, features.hand_size, cfg)
            smooth = self._pos_filter.update(target, dt)
            if self._last_target is not None and np.linalg.norm(smooth - self._last_target) < cfg.deadzone:
                smooth = self._last_target.copy()
            else:
                self._last_target = smooth.copy()
            position = smooth
            vec = self._flex_filter.update([features.flexions[n] for n in FINGER_NAMES], dt)
            flexions = {n: float(v) for n, v in zip(FINGER_NAMES, vec)}

        action = GESTURE_ACTIONS[stable]
        if features is None and action is not Action.NONE:
            action = Action.HOLD  # short dropout: keep the robot where it is
        follow = action is Action.FOLLOW and position is not None

        return GestureCommand(
            timestamp=now,
            hand_detected=features is not None,
            raw_gesture=raw,
            gesture=stable,
            action=action,
            target_position=position.copy() if follow else None,
            grasp=GESTURE_GRASP.get(stable) if follow else None,
            finger_flexions=flexions,
            stop_requested=stable is Gesture.PEACE and previous is not Gesture.PEACE,
            confidence=float(obs.score) if (obs is not None and features is not None) else 0.0,
        )


# ----------------------------------------------------------------------------
# 7. MediaPipe tracker
# ----------------------------------------------------------------------------

@dataclass
class RawDetection:
    landmarks: np.ndarray   # (21, 3) image-normalised
    label: str
    score: float


class _LegacyBackend:
    """mediapipe.solutions.hands (bundled models, no model file needed)."""

    def __init__(self, mp, cfg: TrackerConfig) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg.max_hands,
            model_complexity=cfg.model_complexity,
            min_detection_confidence=cfg.detection_confidence,
            min_tracking_confidence=cfg.tracking_confidence,
        )

    def detect(self, rgb: np.ndarray, timestamp_s: float) -> List[RawDetection]:
        result = self._hands.process(rgb)
        out: List[RawDetection] = []
        if result.multi_hand_landmarks:
            handed = result.multi_handedness or []
            for i, lms in enumerate(result.multi_hand_landmarks):
                label, score = "", 1.0
                if i < len(handed):
                    cls = handed[i].classification[0]
                    label, score = str(cls.label), float(cls.score)
                out.append(RawDetection(landmarks_to_array(lms.landmark), label, score))
        return out

    def close(self) -> None:
        self._hands.close()


class _TasksBackend:
    """mediapipe.tasks HandLandmarker (needs the hand_landmarker.task model file)."""

    def __init__(self, mp, cfg: TrackerConfig) -> None:
        if not cfg.model_path:
            raise RuntimeError("TrackerConfig.model_path (hand_landmarker.task) is required")
        if not Path(cfg.model_path).is_file():
            raise FileNotFoundError(f"Hand landmarker model not found: {cfg.model_path}")
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(cfg.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=cfg.detection_confidence,
            min_hand_presence_confidence=cfg.detection_confidence,
            min_tracking_confidence=cfg.tracking_confidence,
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ms = -1

    def detect(self, rgb: np.ndarray, timestamp_s: float) -> List[RawDetection]:
        ts = int(timestamp_s * 1000.0)
        if ts <= self._last_ms:  # VIDEO mode needs strictly increasing timestamps
            ts = self._last_ms + 1
        self._last_ms = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)
        out: List[RawDetection] = []
        for i, lms in enumerate(result.hand_landmarks):
            label, score = "", 1.0
            if i < len(result.handedness) and result.handedness[i]:
                cat = result.handedness[i][0]
                label, score = str(cat.category_name), float(cat.score)
            out.append(RawDetection(landmarks_to_array(lms), label, score))
        return out

    def close(self) -> None:
        self._landmarker.close()


class HandTracker:
    """Detects the most confident hand in a BGR frame.

    `backend` can be injected (tests); otherwise MediaPipe is opened lazily on the first
    `process()` call, in the calling thread. Without `config.model_path` the legacy
    `mp.solutions.hands` API is used when available; with a model path the Tasks API is used.
    """

    def __init__(self, config: Optional[TrackerConfig] = None, backend=None) -> None:
        self.config = TrackerConfig() if config is None else config
        self._backend = backend

    def open(self) -> None:
        if self._backend is not None:
            return
        try:
            mp = importlib.import_module("mediapipe")
        except ImportError as exc:
            raise ImportError("mediapipe is not installed (pip install mediapipe)") from exc
        solutions = getattr(mp, "solutions", None)
        has_legacy = getattr(solutions, "hands", None) is not None
        if self.config.model_path is None and has_legacy:
            self._backend = _LegacyBackend(mp, self.config)
        elif self.config.model_path is not None:
            self._backend = _TasksBackend(mp, self.config)
        else:
            raise RuntimeError(
                "This mediapipe build has no legacy 'solutions' API. Download "
                "hand_landmarker.task and pass TrackerConfig(model_path=...)."
            )

    def close(self) -> None:
        if self._backend is not None:
            try:
                self._backend.close()
            finally:
                self._backend = None

    def __enter__(self) -> "HandTracker":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def process(self, frame_bgr, timestamp: Optional[float] = None) -> Optional[HandObservation]:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError(f"Expected a BGR frame of shape (H, W, 3), got {frame.shape}")
        self.open()
        t = time.monotonic() if timestamp is None else float(timestamp)
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        detections = [d for d in self._backend.detect(rgb, t) if d.score >= self.config.min_score]
        if not detections:
            return None
        best = max(detections, key=lambda d: d.score)
        return make_observation(
            best.landmarks, aspect=w / h, label=best.label, score=best.score, timestamp=t
        )


# ----------------------------------------------------------------------------
# 8. Camera, overlay and pipeline thread
# ----------------------------------------------------------------------------

class Camera:
    """Minimal OpenCV webcam wrapper (cv2 imported lazily)."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self.index, self.width, self.height = index, width, height
        self._cap = None

    def open(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open camera {self.index} (check camera permission)")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def draw_overlay(
    frame_bgr: np.ndarray,
    obs: Optional[HandObservation],
    command: Optional[GestureCommand] = None,
    mirror: bool = False,
) -> np.ndarray:
    """Return a copy of the frame with the hand skeleton and the gesture text drawn on it."""
    import cv2

    out = np.ascontiguousarray(frame_bgr[:, ::-1]) if mirror else np.array(frame_bgr, copy=True)
    h, w = out.shape[:2]
    if obs is not None:
        xy = np.asarray(obs.landmarks, dtype=float)[:, :2].copy()
        if mirror:
            xy[:, 0] = 1.0 - xy[:, 0]
        px = [(int(x * w), int(y * h)) for x, y in xy]
        for a, b in HAND_CONNECTIONS:
            cv2.line(out, px[a], px[b], (0, 255, 0), 2)
        for p in px:
            cv2.circle(out, p, 3, (0, 0, 255), -1)
    if command is not None:
        text = f"{command.gesture.value} | {command.action.value}"
        cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


@dataclass
class VisionOutput:
    command: GestureCommand
    observation: Optional[HandObservation]
    fps: float
    frame_id: int
    timestamp: float


class VisionPipeline:
    """Runs frame source -> tracker -> controller in a background thread.

    Other threads read the newest result with latest() / latest_frame(); nothing here blocks
    the simulation or the GUI. `frame_source` is any callable returning a BGR frame or None.
    """

    def __init__(
        self,
        tracker,
        controller: GestureController,
        frame_source: Callable[[], Optional[np.ndarray]],
        *,
        fps_limit: float = 30.0,
        annotate: bool = False,
        mirror_preview: bool = True,
        clock: Callable[[], float] = time.monotonic,
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        if fps_limit <= 0:
            raise ValueError("fps_limit must be positive")
        self.tracker = tracker
        self.controller = controller
        self._source = frame_source
        self._period = 1.0 / fps_limit
        self.annotate = annotate
        self.mirror_preview = mirror_preview
        self._clock = clock
        self._on_close = on_close

        self._lock = threading.Lock()
        self._latest: Optional[VisionOutput] = None
        self._frame: Optional[np.ndarray] = None
        self._error: Optional[str] = None
        self._fps = 0.0
        self._last_t: Optional[float] = None
        self._frame_id = 0
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def from_camera(
        cls,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        tracker_config: Optional[TrackerConfig] = None,
        gesture_config: Optional[GestureConfig] = None,
        mapping_config: Optional[MappingConfig] = None,
        **kwargs,
    ) -> "VisionPipeline":
        camera = Camera(camera_index, width, height)
        camera.open()
        return cls(
            HandTracker(tracker_config),
            GestureController(gesture_config, mapping_config),
            camera.read,
            on_close=camera.release,
            **kwargs,
        )

    # ---- one iteration -----------------------------------------------------
    def step(self) -> Optional[VisionOutput]:
        """Grab one frame and process it. Returns None if no frame is available."""
        frame = self._source()
        if frame is None:
            return None
        t = self._clock()
        obs = self.tracker.process(frame, t)
        cmd = self.controller.update(obs, t)
        with self._lock:
            if self._last_t is not None and t > self._last_t:
                inst = 1.0 / (t - self._last_t)
                self._fps = inst if self._fps == 0.0 else 0.9 * self._fps + 0.1 * inst
            self._last_t = t
            self._frame_id += 1
            out = VisionOutput(cmd, obs, self._fps, self._frame_id, t)
            self._latest = out
        if self.annotate:
            try:
                annotated = draw_overlay(frame, obs, cmd, mirror=self.mirror_preview)
                with self._lock:
                    self._frame = annotated
            except Exception as exc:  # noqa: BLE001 - e.g. cv2 missing
                logger.warning("Preview annotation disabled: %s", exc)
                self.annotate = False
        return out

    # ---- thread control ------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_evt.is_set():
            started = time.monotonic()
            try:
                out = self.step()
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                logger.exception("Vision step failed")
                with self._lock:
                    self._error = str(exc)
                out = None
            if out is None:
                self._stop_evt.wait(0.01)
            else:
                self._stop_evt.wait(max(0.0, self._period - (time.monotonic() - started)))

    def start(self) -> None:
        if self.running:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="vision-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def close(self) -> None:
        self.stop()
        if hasattr(self.tracker, "close"):
            self.tracker.close()
        if self._on_close is not None:
            self._on_close()

    # ---- results ------------------------------------------------------------------
    def latest(self) -> Optional[VisionOutput]:
        with self._lock:
            return self._latest

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------

def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="VRM-HRI hand tracking and gesture demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--preview", action="store_true", help="show an OpenCV preview window")
    parser.add_argument("--model-path", default=None, help="hand_landmarker.task (Tasks API)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pipeline = VisionPipeline.from_camera(
        args.camera, tracker_config=TrackerConfig(model_path=args.model_path), annotate=args.preview
    )
    pipeline.start()
    last_print = 0.0
    try:
        while True:
            if args.preview:
                import cv2

                frame = pipeline.latest_frame()
                if frame is not None:
                    cv2.imshow("VRM-HRI vision (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.05)
            out = pipeline.latest()
            if out is not None and time.monotonic() - last_print > 0.25:
                last_print = time.monotonic()
                cmd = out.command
                pos = None if cmd.target_position is None else np.round(cmd.target_position, 3).tolist()
                print(f"{cmd.gesture.value:8s} {cmd.action.value:7s} grasp={cmd.grasp} "
                    f"target={pos} fps={out.fps:.1f}")
            if pipeline.error:
                print("Pipeline error:", pipeline.error)
                break
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        pipeline.close()
        if args.preview:
            import cv2

            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()