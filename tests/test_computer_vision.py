"""Unit tests for computer_vision.py (run from the project root:
pytest tests/test_computer_vision.py -v). No webcam is needed: hands are synthetic."""
import itertools
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from spatialmath import SE3

from computer_vision import (
    FINGER_LANDMARKS,
    GESTURE_GRASP,
    Action,
    Gesture,
    GestureClassifier,
    GestureConfig,
    GestureController,
    GestureDebouncer,
    HandFeatures,
    HandTracker,
    LowPassFilter,
    MappingConfig,
    RawDetection,
    TrackerConfig,
    VisionOutput,
    VisionPipeline,
    draw_overlay,
    extract_features,
    finger_flexion,
    joint_angle_deg,
    landmarks_to_array,
    make_observation,
    map_hand_to_position,
    target_pose,
)
from modelling import FINGER_NAMES, GRASP_PRESETS

FOUR = ("index", "middle", "ring", "pinky")

# ----------------------------------------------------------------------------
# Synthetic hand (image-normalised coordinates, aspect 1)
# Palm: wrist (0.5, 0.8), middle MCP (0.5, 0.6) -> hand size 0.2.
# Fingers point up (-y) and curl towards +z.
# ----------------------------------------------------------------------------
_FINGER_BASES = {
    "index": ((0.43, 0.61), (0.09, 0.05, 0.04)),
    "middle": ((0.50, 0.60), (0.10, 0.055, 0.045)),
    "ring": ((0.56, 0.61), (0.09, 0.05, 0.04)),
    "pinky": ((0.62, 0.64), (0.07, 0.04, 0.035)),
}


def synthetic_landmarks(flex=None, pinch=False, offset=(0.0, 0.0), scale=1.0):
    f = {n: 0.0 for n in FOUR}
    f.update(flex or {})
    pts = np.zeros((21, 3))
    pts[0] = (0.5, 0.8, 0.0)
    pts[1] = (0.46, 0.76, 0.0)
    pts[2] = (0.41, 0.71, 0.0)
    pts[3] = (0.36, 0.66, 0.0)
    pts[4] = (0.31, 0.61, 0.0)  # straight thumb
    for name, (base, lengths) in _FINGER_BASES.items():
        bends = (f[name] * 80.0, f[name] * 90.0, f[name] * 70.0)
        ids = FINGER_LANDMARKS[name]
        p = np.array([base[0], base[1], 0.0])
        pts[ids[0]] = p
        theta = 0.0
        for k, (bend, length) in enumerate(zip(bends, lengths)):
            theta += math.radians(bend)
            p = p + length * np.array([0.0, -math.cos(theta), math.sin(theta)])
            pts[ids[k + 1]] = p
    if pinch:
        pts[4] = pts[8] + np.array([0.005, 0.0, 0.0])
    if scale != 1.0:
        pts = pts[0] + (pts - pts[0]) * scale
    pts[:, 0] += offset[0]
    pts[:, 1] += offset[1]
    return pts


OPEN = synthetic_landmarks()
FIST = synthetic_landmarks({n: 1.0 for n in FOUR})
POINT = synthetic_landmarks({"middle": 1.0, "ring": 1.0, "pinky": 1.0})
PEACE = synthetic_landmarks({"ring": 1.0, "pinky": 1.0})
PINCH = synthetic_landmarks(pinch=True)


def features_of(lm, cfg=None):
    return extract_features(make_observation(lm), cfg)


def feat(flex=None, pinch=1.0):
    f = {n: 0.0 for n in FINGER_NAMES}
    f.update(flex or {})
    return HandFeatures(f, pinch, 0.2, np.array([0.5, 0.5]))


# ----------------------------------------------------------------------------
# Observation
# ----------------------------------------------------------------------------
def test_make_observation_validates_input():
    obs = make_observation(OPEN, aspect=2.0, label="Right", score=0.9, timestamp=1.5)
    assert obs.landmarks.shape == (21, 3) and obs.label == "Right" and obs.timestamp == 1.5
    with pytest.raises(ValueError):
        make_observation(np.zeros((20, 3)))
    bad = OPEN.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        make_observation(bad)
    with pytest.raises(ValueError):
        make_observation(OPEN, aspect=0.0)


def test_metric_scales_x_and_z_by_aspect():
    lm = np.ones((21, 3))
    metric = make_observation(lm, aspect=2.0).metric()
    assert np.allclose(metric[:, 0], 2.0) and np.allclose(metric[:, 1], 1.0)
    assert np.allclose(metric[:, 2], 2.0)


def test_landmarks_to_array():
    lms = [SimpleNamespace(x=i * 0.01, y=i * 0.02, z=i * 0.03) for i in range(21)]
    arr = landmarks_to_array(lms)
    assert arr.shape == (21, 3)
    assert np.allclose(arr[5], [0.05, 0.10, 0.15])


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def test_joint_angle():
    assert joint_angle_deg([1, 0, 0], [0, 0, 0], [-1, 0, 0]) == pytest.approx(180.0)
    assert joint_angle_deg([1, 0, 0], [0, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert joint_angle_deg([0, 0, 0], [0, 0, 0], [0, 1, 0]) == pytest.approx(180.0)  # degenerate


def test_open_hand_features():
    f = features_of(OPEN)
    assert f.hand_size == pytest.approx(0.2)
    assert all(v < 0.05 for v in f.flexions.values())
    assert f.pinch_distance > 0.8
    assert np.allclose(f.palm_uv, [0.522, 0.652], atol=1e-9)


def test_fist_features():
    f = features_of(FIST)
    assert all(f.flexions[n] > 0.95 for n in FOUR)
    assert f.pinch_distance > 0.4


def test_pinch_distance_is_small_when_pinching():
    assert features_of(PINCH).pinch_distance < 0.1


def test_flexion_is_between_open_and_closed_for_half_curl():
    pts = make_observation(synthetic_landmarks({"index": 0.5})).metric()
    assert 0.35 < finger_flexion(pts, "index") < 0.65


def test_tiny_hand_is_rejected():
    assert features_of(synthetic_landmarks(scale=0.05)) is None


# ----------------------------------------------------------------------------
# Classifier
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "lm, expected",
    [
        (OPEN, Gesture.OPEN),
        (FIST, Gesture.FIST),
        (POINT, Gesture.POINT),
        (PEACE, Gesture.PEACE),
        (PINCH, Gesture.PINCH),
        (synthetic_landmarks({"index": 0.5}), Gesture.UNKNOWN),
    ],
)
def test_classify_synthetic_hands(lm, expected):
    assert GestureClassifier().classify(features_of(lm)) == expected


def test_pinch_has_hysteresis():
    clf = GestureClassifier()
    assert clf.classify(feat(pinch=0.30)) == Gesture.OPEN       # above pinch_on, not pinching yet
    assert clf.classify(feat(pinch=0.10)) == Gesture.PINCH
    assert clf.classify(feat(pinch=0.30)) == Gesture.PINCH      # between on and off: stays
    assert clf.classify(feat(pinch=0.50)) == Gesture.OPEN       # above pinch_off: released
    clf.classify(feat(pinch=0.10))
    clf.reset()
    assert clf.classify(feat(pinch=0.30)) == Gesture.OPEN


# ----------------------------------------------------------------------------
# Debouncer and filter
# ----------------------------------------------------------------------------
def test_debouncer_needs_consecutive_frames():
    deb = GestureDebouncer(frames_to_switch=3, lost_frames=4)
    seq = [Gesture.OPEN, Gesture.OPEN, Gesture.FIST, Gesture.OPEN, Gesture.OPEN, Gesture.OPEN]
    assert [deb.update(g) for g in seq] == [Gesture.NONE] * 5 + [Gesture.OPEN]


def test_debouncer_loses_the_hand_slowly():
    deb = GestureDebouncer(frames_to_switch=1, lost_frames=4)
    assert deb.update(Gesture.OPEN) == Gesture.OPEN
    assert [deb.update(Gesture.NONE) for _ in range(3)] == [Gesture.OPEN] * 3
    assert deb.update(Gesture.NONE) == Gesture.NONE
    deb.reset()
    assert deb.stable == Gesture.NONE


def test_debouncer_invalid_arguments():
    with pytest.raises(ValueError):
        GestureDebouncer(0, 1)


def test_low_pass_filter():
    flt = LowPassFilter(tau=0.1)
    assert np.allclose(flt.update([1.0, 2.0], dt=0.1), [1.0, 2.0])   # first sample initialises
    out = flt.update([2.0, 4.0], dt=0.1)
    alpha = 1.0 - math.exp(-1.0)
    assert np.allclose(out, [1.0 + alpha, 2.0 + 2 * alpha])
    for _ in range(100):
        out = flt.update([2.0, 4.0], dt=0.1)
    assert np.allclose(out, [2.0, 4.0], atol=1e-6)
    flt.reset()
    assert flt.value is None


def test_low_pass_filter_zero_tau_is_passthrough():
    flt = LowPassFilter(tau=0.0)
    flt.update([0.0], 0.1)
    assert np.allclose(flt.update([5.0], 0.1), [5.0])
    with pytest.raises(ValueError):
        LowPassFilter(-1.0)


# ----------------------------------------------------------------------------
# Mapping
# ----------------------------------------------------------------------------
def test_mapping_centre_and_extremes():
    cfg = MappingConfig()
    cx, cy, cz = cfg.center
    sx, sy, sz = cfg.span
    half = 0.5 - cfg.margin
    assert np.allclose(map_hand_to_position([0.5, 0.5], cfg.size_ref, cfg), cfg.center)
    # Hand moved to the right edge of the active region -> y decreases (invert_lateral False).
    assert np.allclose(map_hand_to_position([0.5 + half, 0.5], cfg.size_ref, cfg), [cx, cy - sy, cz])
    # Hand moved up in the image -> z increases.
    assert np.allclose(map_hand_to_position([0.5, 0.5 - half], cfg.size_ref, cfg), [cx, cy, cz + sz])
    # Hand closer to the camera -> x increases.
    near = cfg.size_ref * (1.0 + cfg.depth_range)
    assert np.allclose(map_hand_to_position([0.5, 0.5], near, cfg), [cx + sx, cy, cz])


def test_mapping_invert_lateral():
    cfg = MappingConfig(invert_lateral=True)
    half = 0.5 - cfg.margin
    p = map_hand_to_position([0.5 + half, 0.5], cfg.size_ref, cfg)
    assert p[1] == pytest.approx(cfg.center[1] + cfg.span[1])


def test_mapping_is_clamped():
    cfg = MappingConfig()
    c = np.array(cfg.center)
    s = np.array(cfg.span)
    for uv, size in (([5.0, -5.0], 10.0), ([-5.0, 5.0], 0.001)):
        p = map_hand_to_position(uv, size, cfg)
        assert np.all(p >= c - s - 1e-12) and np.all(p <= c + s + 1e-12)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MappingConfig(margin=0.6),
        lambda: MappingConfig(size_ref=0.0),
        lambda: MappingConfig(span=(-0.1, 0.1, 0.1)),
        lambda: MappingConfig(pos_tau=-1.0),
        lambda: GestureConfig(open_angle_deg=50.0),
        lambda: GestureConfig(extended_below=0.8),
        lambda: GestureConfig(pinch_on=0.5, pinch_off=0.4),
        lambda: GestureConfig(frames_to_switch=0),
        lambda: TrackerConfig(max_hands=0),
        lambda: TrackerConfig(min_score=1.5),
    ],
)
def test_invalid_configs_raise(factory):
    with pytest.raises(ValueError):
        factory()


def test_target_pose():
    cmd_pos = np.array([0.3, 0.1, 0.4])
    cmd = SimpleNamespace(target_position=cmd_pos)
    T = target_pose(cmd, SE3.Rx(math.pi))
    assert np.allclose(T.t, cmd_pos)
    assert np.allclose(np.asarray(T.R), np.asarray(SE3.Rx(math.pi).R))
    assert target_pose(SimpleNamespace(target_position=None), SE3()) is None
    assert target_pose(cmd, None) is None


# ----------------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------------
class Runner:
    def __init__(self, **gesture_kwargs):
        cfg = GestureConfig(frames_to_switch=3, lost_frames=4, **gesture_kwargs)
        self.ctrl = GestureController(cfg, MappingConfig())
        self.t = 0.0

    def feed(self, lm, n=1):
        out = []
        for _ in range(n):
            obs = None if lm is None else make_observation(lm, timestamp=self.t)
            out.append(self.ctrl.update(obs, self.t))
            self.t += 0.033
        return out


def expected_position(lm):
    f = features_of(lm)
    return map_hand_to_position(f.palm_uv, f.hand_size, MappingConfig())


def test_follow_starts_after_debounce():
    cmds = Runner().feed(OPEN, 3)
    assert not cmds[0].tracking and not cmds[1].tracking
    last = cmds[2]
    assert last.tracking and last.gesture == Gesture.OPEN and last.action == Action.FOLLOW
    assert last.grasp == "open"
    assert last.target_position.shape == (3,)
    assert not last.stop_requested


def test_static_hand_target_equals_mapped_position():
    cmds = Runner().feed(OPEN, 6)
    assert np.allclose(cmds[-1].target_position, expected_position(OPEN), atol=1e-9)


def test_grasp_follows_gesture():
    assert Runner().feed(FIST, 3)[-1].grasp == "closed"
    assert Runner().feed(PINCH, 3)[-1].grasp == "pinch"
    assert set(GESTURE_GRASP.values()) <= set(GRASP_PRESETS)


def test_point_holds_the_robot():
    last = Runner().feed(POINT, 3)[-1]
    assert last.gesture == Gesture.POINT and last.action == Action.HOLD
    assert last.target_position is None and last.grasp is None
    assert last.finger_flexions is not None


def test_peace_requests_stop_once():
    cmds = Runner().feed(PEACE, 6)
    assert cmds[2].action == Action.STOP and cmds[2].stop_requested
    assert not any(c.stop_requested for c in cmds[3:])
    assert all(c.target_position is None for c in cmds)


def test_gesture_switch_requires_consecutive_frames():
    r = Runner()
    r.feed(OPEN, 3)
    cmds = r.feed(FIST, 3)
    assert cmds[0].gesture == Gesture.OPEN and cmds[1].gesture == Gesture.OPEN
    assert cmds[2].gesture == Gesture.FIST


def test_dropout_holds_then_releases_and_reacquires():
    r = Runner()
    r.feed(OPEN, 3)
    short = r.feed(None, 3)
    assert all(c.action == Action.HOLD and c.gesture == Gesture.OPEN for c in short)
    assert all(c.target_position is None and not c.hand_detected for c in short)
    gone = r.feed(None, 1)[0]
    assert gone.gesture == Gesture.NONE and gone.action == Action.NONE

    moved = synthetic_landmarks(offset=(0.1, -0.05))
    last = r.feed(moved, 3)[-1]
    assert last.tracking
    assert np.allclose(last.target_position, expected_position(moved), atol=1e-9)


def test_target_is_smoothed():
    r = Runner()
    r.feed(OPEN, 5)
    start = expected_position(OPEN)
    moved = synthetic_landmarks(offset=(0.1, 0.0))
    goal = expected_position(moved)
    cmd = r.feed(moved, 1)[0]
    d_goal = np.linalg.norm(cmd.target_position - goal)
    assert 0.0 < d_goal < np.linalg.norm(start - goal)


def test_finger_flexions_in_command():
    fist = Runner().feed(FIST, 3)[-1]
    assert fist.finger_flexions["index"] > 0.9
    open_ = Runner().feed(OPEN, 3)[-1]
    assert open_.finger_flexions["index"] < 0.1
    assert set(open_.finger_flexions) == set(FINGER_NAMES)


def test_no_hand_command():
    cmd = Runner().feed(None, 1)[0]
    assert cmd.action == Action.NONE and not cmd.hand_detected
    assert cmd.finger_flexions is None and cmd.target_position is None


def test_tiny_hand_is_treated_as_no_hand():
    cmd = Runner().feed(synthetic_landmarks(scale=0.05), 1)[0]
    assert not cmd.hand_detected


def test_controller_reset():
    r = Runner()
    r.feed(OPEN, 3)
    r.ctrl.reset()
    assert not r.ctrl.update(make_observation(OPEN), 100.0).tracking  # needs the debounce again


# ----------------------------------------------------------------------------
# Tracker (fake backend)
# ----------------------------------------------------------------------------
class FakeBackend:
    def __init__(self, detections):
        self.detections = detections
        self.last_rgb = None
        self.closed = False

    def detect(self, rgb, timestamp_s):
        self.last_rgb = rgb
        return self.detections

    def close(self):
        self.closed = True


def blank_frame():
    return np.zeros((48, 64, 3), dtype=np.uint8)


def test_tracker_converts_bgr_to_rgb_and_sets_aspect():
    backend = FakeBackend([RawDetection(OPEN, "Right", 0.9)])
    tracker = HandTracker(backend=backend)
    frame = blank_frame()
    frame[..., 0] = 255  # blue channel in BGR
    obs = tracker.process(frame, timestamp=1.0)
    assert np.all(backend.last_rgb[..., 2] == 255) and np.all(backend.last_rgb[..., 0] == 0)
    assert obs.aspect == pytest.approx(64 / 48)
    assert obs.label == "Right" and obs.timestamp == 1.0


def test_tracker_picks_best_detection():
    other = synthetic_landmarks(offset=(0.1, 0.0))
    backend = FakeBackend([RawDetection(OPEN, "Left", 0.7), RawDetection(other, "Right", 0.9)])
    obs = HandTracker(backend=backend).process(blank_frame(), 0.0)
    assert obs.label == "Right"
    assert np.allclose(obs.landmarks, other)


def test_tracker_filters_low_scores_and_empty_results():
    low = HandTracker(TrackerConfig(min_score=0.8), backend=FakeBackend([RawDetection(OPEN, "L", 0.5)]))
    assert low.process(blank_frame(), 0.0) is None
    assert HandTracker(backend=FakeBackend([])).process(blank_frame(), 0.0) is None


def test_tracker_rejects_invalid_frames():
    tracker = HandTracker(backend=FakeBackend([]))
    for bad in (np.zeros((10, 10)), np.zeros((10, 10, 4)), np.zeros((0, 10, 3))):
        with pytest.raises(ValueError):
            tracker.process(bad, 0.0)


def test_tracker_close_closes_backend():
    backend = FakeBackend([])
    tracker = HandTracker(backend=backend)
    tracker.close()
    assert backend.closed


def test_real_mediapipe_blank_frame_returns_none():
    pytest.importorskip("mediapipe")
    tracker = HandTracker()
    try:
        tracker.open()
    except (ImportError, RuntimeError, FileNotFoundError, AttributeError) as exc:
        pytest.skip(f"MediaPipe backend not available: {exc}")
    try:
        assert tracker.process(np.zeros((240, 320, 3), dtype=np.uint8)) is None
    finally:
        tracker.close()


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
class FakeTracker:
    def __init__(self, lm=None, exc=None):
        self.lm, self.exc, self.closed = lm, exc, False

    def process(self, frame, t):
        if self.exc is not None:
            raise self.exc
        return None if self.lm is None else make_observation(self.lm, timestamp=t)

    def close(self):
        self.closed = True


def make_pipeline(tracker, source=None, **kwargs):
    source = (lambda: blank_frame()) if source is None else source
    return VisionPipeline(tracker, GestureController(), source, fps_limit=1000.0, **kwargs)


def test_step_publishes_latest_and_fps():
    ticks = itertools.count(0.0, 0.05)
    pipe = make_pipeline(FakeTracker(OPEN), clock=lambda: next(ticks))
    out1 = pipe.step()
    assert isinstance(out1, VisionOutput) and out1.frame_id == 1
    assert out1.command.hand_detected and out1.observation is not None
    assert pipe.latest() is out1
    out2 = pipe.step()
    assert out2.frame_id == 2 and out2.fps == pytest.approx(20.0)


def test_step_without_frame_returns_none():
    pipe = make_pipeline(FakeTracker(OPEN), source=lambda: None)
    assert pipe.step() is None and pipe.latest() is None


def test_pipeline_thread_start_and_stop():
    pipe = make_pipeline(FakeTracker(OPEN))
    pipe.start()
    try:
        deadline = time.time() + 3.0
        while pipe.latest() is None and time.time() < deadline:
            time.sleep(0.01)
        assert pipe.latest() is not None
        assert pipe.running
    finally:
        pipe.stop()
    assert not pipe.running


def test_pipeline_survives_tracker_errors():
    pipe = make_pipeline(FakeTracker(exc=RuntimeError("boom")))
    pipe.start()
    try:
        deadline = time.time() + 3.0
        while pipe.error is None and time.time() < deadline:
            time.sleep(0.01)
        assert pipe.error is not None and "boom" in pipe.error
        assert pipe.running
    finally:
        pipe.stop()


def test_pipeline_close_releases_resources():
    tracker = FakeTracker(OPEN)
    released = []
    pipe = make_pipeline(tracker, on_close=lambda: released.append(True))
    pipe.close()
    assert tracker.closed and released == [True]


def test_pipeline_annotation():
    pytest.importorskip("cv2")
    pipe = make_pipeline(FakeTracker(OPEN), annotate=True)
    pipe.step()
    frame = pipe.latest_frame()
    assert frame is not None and frame.shape == (48, 64, 3)


def test_draw_overlay_keeps_frame_shape():
    pytest.importorskip("cv2")
    obs = make_observation(OPEN)
    frame = blank_frame()
    out = draw_overlay(frame, obs, None, mirror=True)
    assert out.shape == frame.shape
    assert out.any()            # something was drawn
    assert not frame.any()      # the input frame is untouched