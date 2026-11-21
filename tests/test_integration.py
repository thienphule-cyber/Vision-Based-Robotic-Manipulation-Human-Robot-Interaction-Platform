"""Integration tests (run from the project root: pytest tests/test_integration.py -v).

Everything is real (safety, kinematics, gui, computer_vision, the controller) except the Swift
simulation, which is replaced by FakeSim so no window or camera is needed.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest
from spatialmath import SE3

import main
from computer_vision import (
    Action,
    Gesture,
    GestureCommand,
    GestureConfig,
    GestureController,
    MappingConfig,
    VisionOutput,
    VisionPipeline,
)
from gui import Command, GuiServer, RobotHub
from main import AppConfig, Application, MotionController, parse_args, pose_from_params
from modelling import FINGER_NAMES, Environment, SystemModel, WorldObject
from safety import SafetyChecker
from test_computer_vision import OPEN, PEACE, FakeTracker, blank_frame

GOAL_DEG = [20, 0, 0, 0, 0, 0]      # rotate J1 only: a safe, easy motion starting from home
GOAL_Q = np.deg2rad(GOAL_DEG)
Q_OFF = np.deg2rad([0, 0, 0, 0, 30, 0])  # tilted wrist: a pose that is not singular


# ----------------------------------------------------------------------------
# Test doubles and fixtures
# ----------------------------------------------------------------------------
class FakeSim:
    """Implements the part of simulation.Simulation that main.py uses, without Swift."""

    def __init__(self, system):
        self.system = system
        self.model = system.robot
        self.hand = system.hand
        self._q = self.model.q_home.copy()
        self.state = SimpleNamespace(value="idle")
        self.frames = []
        self._recording = False
        self.launched = self.loaded = self.closed = False
        self.stop_calls = self.pause_calls = self.resume_calls = self.reset_calls = 0
        self.execute_calls = self.executed_rows = self.set_joint_calls = 0
        self.paused = False
        self.playback_calls = []
        self.after_execute = None
        self.fail_next = False

    @property
    def q(self):
        return self._q.copy()

    def launch(self):
        self.launched = True

    def load_scene(self):
        self.loaded = True

    def print_status(self):
        pass

    def close(self):
        self.closed = True

    def _capture(self):
        if self._recording:
            self.frames.append({"t": float(len(self.frames)), "q": self._q.copy()})

    def set_joint_positions(self, q, clip=True, render=True):
        self._q = self.model.clip(q) if clip else np.asarray(q, dtype=float).copy()
        self.set_joint_calls += 1
        self._capture()

    def execute_trajectory(self, q_traj, dt=0.02):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("boom")
        self.execute_calls += 1
        for q in np.asarray(q_traj, dtype=float):
            self._q = q.copy()
            self.executed_rows += 1
            self._capture()
        if self.after_execute is not None:
            self.after_execute(self.execute_calls)
        return True

    def set_hand_grasp(self, preset, render=True):
        self.hand.apply_grasp(preset)

    def set_finger_flexion(self, name, fraction, render=True):
        self.hand.set_finger_flexion(name, fraction)

    def ee_pose(self):
        return self.model.fkine(self._q)

    def pause(self):
        self.pause_calls += 1
        self.paused = True

    def resume(self):
        self.resume_calls += 1
        self.paused = False

    def stop(self):
        self.stop_calls += 1
        self.paused = False

    def reset(self):
        self.reset_calls += 1
        self.hand.open_hand()
        self._q = self.model.q_home.copy()

    def start_recording(self, clear=True):
        if clear:
            self.frames = []
        self._recording = True

    def stop_recording(self):
        self._recording = False
        return self.frames

    def clear_recording(self):
        self.frames = []

    def playback(self, frames=None, speed=1.0):
        self.playback_calls.append(speed)
        return True


class FakeVision:
    def __init__(self):
        self.output = None
        self.frame = None
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def latest(self):
        return self.output

    def latest_frame(self):
        return self.frame


def make_output(frame_id, action=Action.FOLLOW, gesture=Gesture.OPEN, target=None, grasp=None,
                stop=False, flexions=None):
    t = frame_id / 30.0
    cmd = GestureCommand(
        timestamp=t,
        hand_detected=True,
        raw_gesture=gesture,
        gesture=gesture,
        action=action,
        target_position=None if target is None else np.asarray(target, dtype=float),
        grasp=grasp,
        finger_flexions=flexions,
        stop_requested=stop,
    )
    return VisionOutput(cmd, None, 30.0, frame_id, t)


def make_rig(vision=None, vision_factory=None, config=None):
    cfg = AppConfig() if config is None else config
    system = SystemModel.build_default(load_visual=False)
    sim = FakeSim(system)
    safety = SafetyChecker(system.robot, system.environment)
    hub = RobotHub(system.robot, safety)
    controller = MotionController(system, sim, safety, hub, cfg, vision=vision,
                                vision_factory=vision_factory)
    return SimpleNamespace(system=system, model=system.robot, sim=sim, safety=safety, hub=hub,
                        controller=controller, config=cfg)


@pytest.fixture
def rig():
    return make_rig()


def run_command(rig, data):
    """GUI path: hub.submit -> supervisor -> worker (run synchronously)."""
    result = rig.hub.submit(data)
    rig.controller.process(rig.hub.poll_commands())
    rig.controller.run_pending()
    return result


def messages(rig):
    return [e["message"] for e in rig.hub.log_since(0)[0]]


def wait_for(app, predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.step()
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ----------------------------------------------------------------------------
# Configuration and helpers
# ----------------------------------------------------------------------------
def test_parse_args():
    cfg = parse_args(["--port", "9001", "--no-browser", "--vision", "--camera", "2",
                    "--no-visual", "--no-hand", "--headless"])
    assert cfg.port == 9001 and not cfg.open_browser and cfg.vision and cfg.camera == 2
    assert not cfg.use_visual and not cfg.show_hand and cfg.headless and cfg.show_axes
    defaults = parse_args([])
    assert defaults.host == "127.0.0.1" and defaults.open_browser and not defaults.vision


@pytest.mark.parametrize(
    "kwargs",
    [dict(dt=0.0), dict(joint_speed=-1.0), dict(chunk_size=0), dict(speed_retries=-1),
    dict(plan_margin=-0.1), dict(port=70000), dict(state_hz=0.0)],
)
def test_invalid_app_config_raises(kwargs):
    with pytest.raises(ValueError):
        AppConfig(**kwargs)


def test_pose_from_params_keeps_or_sets_orientation():
    current = SE3.Rx(0.3) * SE3.Rz(0.2)
    kept = pose_from_params([0.1, 0.2, 0.3], None, current)
    assert np.allclose(kept.t, [0.1, 0.2, 0.3])
    assert np.allclose(np.asarray(kept.R), np.asarray(current.R))
    rpy = [10.0, 20.0, 30.0]
    explicit = pose_from_params([0, 0, 0], rpy, current)
    expected = SE3.Rz(30, unit="deg") * SE3.Ry(20, unit="deg") * SE3.Rx(10, unit="deg")
    assert np.allclose(np.asarray(explicit.R), np.asarray(expected.R))


# ----------------------------------------------------------------------------
# GUI command -> safety -> simulation
# ----------------------------------------------------------------------------
def test_set_joints_moves_the_robot_in_chunks(rig):
    assert run_command(rig, {"type": "set_joints", "q_deg": GOAL_DEG}).ok
    assert np.allclose(rig.sim.q, GOAL_Q, atol=1e-9)
    assert rig.sim.execute_calls >= 2           # executed in several safety-checked chunks
    assert any("set_joints completed" in m for m in messages(rig))
    assert not rig.controller.busy


def test_move_to_reaches_the_pose(rig):
    pose = rig.model.fkine(GOAL_Q)
    rpy = np.asarray(pose.rpy(unit="deg")).tolist()
    run_command(rig, {"type": "move_to", "position": np.asarray(pose.t).tolist(), "rpy_deg": rpy})
    assert np.linalg.norm(rig.model.ee_position(rig.sim.q) - np.asarray(pose.t)) < 2e-3


def test_controller_rechecks_an_unreachable_pose(rig):
    cmd = Command("move_to", {"position": [0.69, 0.69, 0.85], "rpy_deg": None})
    rig.controller.process([cmd])
    rig.controller.run_pending()
    assert np.allclose(rig.sim.q, 0.0)
    assert any("move_to rejected" in m for m in messages(rig))


def test_goal_in_collision_is_rejected(rig):
    tip = rig.safety.robot_capsules(GOAL_Q)[-1].p1
    rig.system.environment.add(WorldObject("block", "obstacle", "box", tuple(tip), (0.1, 0.1, 0.1)))
    # The hub pre-check already refuses it ...
    assert not rig.hub.submit({"type": "set_joints", "q_deg": GOAL_DEG}).ok
    # ... and the controller refuses it again if the command bypasses the hub.
    rig.controller.process([Command("set_joints", {"q_deg": GOAL_DEG})])
    rig.controller.run_pending()
    assert np.allclose(rig.sim.q, 0.0) and rig.sim.execute_calls == 0
    assert any("collision" in m.lower() for m in messages(rig))


def test_precondition_off_pose_is_safe(rig):
    assert rig.safety.check_configuration(Q_OFF).ok


def test_moving_into_a_singularity_is_rejected(rig):
    rig.sim.set_joint_positions(Q_OFF)
    run_command(rig, {"type": "set_joints", "q_deg": [0, 0, 0, 0, 0, 0]})
    assert np.allclose(rig.sim.q, Q_OFF)
    assert any("singular" in m.lower() for m in messages(rig))


def test_home_is_allowed_despite_the_singularity(rig):
    rig.sim.set_joint_positions(Q_OFF)
    run_command(rig, {"type": "home"})
    assert np.allclose(rig.sim.q, 0.0, atol=1e-9)
    assert any("home completed" in m for m in messages(rig))


def test_grasp_and_finger_commands(rig):
    run_command(rig, {"type": "grasp", "preset": "pinch"})
    assert rig.system.hand.finger("thumb").flexion == pytest.approx(0.6)
    run_command(rig, {"type": "finger", "finger": "index", "flexion": 0.5})
    assert rig.system.hand.finger("index").flexion == pytest.approx(0.5)


def test_invalid_hand_command_is_rejected_by_the_controller(rig):
    rig.controller.process([Command("grasp", {"preset": "wave"})])
    rig.controller.run_pending()
    assert all(f.flexion == pytest.approx(0.0) for f in rig.system.hand.fingers.values())
    assert any("grasp rejected" in m for m in messages(rig))


def test_recording_and_playback(rig):
    run_command(rig, {"type": "playback"})                    # nothing recorded yet
    assert rig.sim.playback_calls == []
    assert any("Nothing recorded" in m for m in messages(rig))

    run_command(rig, {"type": "start_recording"})
    run_command(rig, {"type": "set_joints", "q_deg": GOAL_DEG})
    run_command(rig, {"type": "stop_recording"})
    assert len(rig.sim.frames) > 10 and not rig.sim._recording
    run_command(rig, {"type": "playback", "speed": 2.0})
    assert rig.sim.playback_calls == [2.0]
    run_command(rig, {"type": "clear_recording"})
    assert rig.sim.frames == []


def test_reset_restores_home_and_stops_motion(rig):
    run_command(rig, {"type": "set_joints", "q_deg": GOAL_DEG})
    run_command(rig, {"type": "grasp", "preset": "closed"})
    run_command(rig, {"type": "reset"})
    assert rig.sim.reset_calls == 1 and rig.sim.stop_calls >= 1
    assert np.allclose(rig.sim.q, 0.0)
    assert not rig.controller.gesture_enabled


def test_a_failing_handler_does_not_break_the_controller(rig):
    rig.sim.fail_next = True
    run_command(rig, {"type": "set_joints", "q_deg": GOAL_DEG})
    assert any("failed" in m for m in messages(rig))
    run_command(rig, {"type": "set_joints", "q_deg": GOAL_DEG})
    assert np.allclose(rig.sim.q, GOAL_Q, atol=1e-9)


# ----------------------------------------------------------------------------
# Stop, e-stop, pause
# ----------------------------------------------------------------------------
def test_stop_cancels_queued_jobs(rig):
    rig.hub.submit({"type": "set_joints", "q_deg": GOAL_DEG})
    rig.hub.submit({"type": "set_joints", "q_deg": [-20, 0, 0, 0, 0, 0]})
    rig.controller.process(rig.hub.poll_commands())       # queued as jobs, not executed yet
    assert rig.controller.busy
    rig.hub.submit({"type": "stop"})
    rig.controller.process(rig.hub.poll_commands())
    assert rig.controller.run_pending() == 0
    assert not rig.controller.busy
    assert np.allclose(rig.sim.q, 0.0) and rig.sim.stop_calls == 1


def test_estop_flow(rig):
    assert rig.hub.submit({"type": "estop"}).ok
    rig.controller.process(rig.hub.poll_commands())
    assert rig.safety.estopped and rig.sim.stop_calls >= 1
    assert not rig.controller.gesture_enabled
    assert not rig.hub.submit({"type": "set_joints", "q_deg": GOAL_DEG}).ok

    # A motion command that bypasses the hub is still ignored by the controller.
    rig.controller.process([Command("set_joints", {"q_deg": GOAL_DEG})])
    assert rig.controller.run_pending() == 0 and np.allclose(rig.sim.q, 0.0)

    assert rig.hub.submit({"type": "clear_estop"}).ok
    rig.controller.process(rig.hub.poll_commands())
    assert not rig.safety.estopped
    run_command(rig, {"type": "set_joints", "q_deg": GOAL_DEG})
    assert np.allclose(rig.sim.q, GOAL_Q, atol=1e-9)


def test_estop_in_the_middle_of_a_trajectory_stops_at_the_next_chunk():
    rig = make_rig(config=AppConfig(chunk_size=10))
    rig.sim.after_execute = lambda n: rig.safety.estop("test") if n == 1 else None
    rig.controller.process([Command("set_joints", {"q_deg": GOAL_DEG})])
    rig.controller.run_pending()
    assert rig.sim.execute_calls == 1 and rig.sim.executed_rows == 10
    assert not np.allclose(rig.sim.q, GOAL_Q)
    assert any("interrupted" in m for m in messages(rig))


def test_pause_and_resume_are_immediate(rig):
    rig.controller.process([Command("pause")])
    assert rig.sim.paused and rig.sim.pause_calls == 1
    rig.controller.process([Command("resume")])
    assert not rig.sim.paused and rig.sim.resume_calls == 1


def test_resume_is_refused_during_estop(rig):
    rig.controller.process([Command("pause")])
    rig.safety.estop("test")
    rig.controller.process([Command("resume")])
    assert rig.sim.resume_calls == 0 and rig.sim.paused


# ----------------------------------------------------------------------------
# Gesture control
# ----------------------------------------------------------------------------
def test_gesture_needs_vision(rig):
    run_command(rig, {"type": "gesture_control", "enabled": True})
    assert not rig.controller.gesture_enabled
    assert any("not available" in m for m in messages(rig))


def test_gesture_starts_vision_on_demand_and_releases_it():
    vision = FakeVision()
    rig = make_rig(vision_factory=lambda: vision)
    run_command(rig, {"type": "gesture_control", "enabled": True})
    assert rig.controller.gesture_enabled and vision.started and rig.controller.vision is vision
    run_command(rig, {"type": "gesture_control", "enabled": False})
    assert not rig.controller.gesture_enabled and vision.closed and rig.controller.vision is None


def test_gesture_cannot_be_enabled_during_estop():
    rig = make_rig(vision=FakeVision())
    rig.safety.estop("test")
    run_command(rig, {"type": "gesture_control", "enabled": True})
    assert not rig.controller.gesture_enabled


def test_gesture_follow_moves_towards_the_target_and_sets_the_grasp():
    vision = FakeVision()
    rig = make_rig(vision=vision)
    rig.sim.set_joint_positions(Q_OFF)
    rig.controller.process([Command("gesture_control", {"enabled": True})])
    assert rig.controller.gesture_enabled

    ee0 = rig.model.ee_position(rig.sim.q)
    target = ee0 + np.array([0.0, 0.02, 0.0])
    vision.output = make_output(1, target=target, flexions={n: 1.0 for n in FINGER_NAMES})
    assert rig.controller.gesture_tick()
    ee1 = rig.model.ee_position(rig.sim.q)
    assert np.linalg.norm(ee1 - target) < np.linalg.norm(ee0 - target)
    assert rig.system.hand.finger("index").flexion == pytest.approx(1.0)
    assert not rig.controller.gesture_tick()      # same frame: nothing to do


def test_gesture_steps_are_speed_limited():
    vision = FakeVision()
    rig = make_rig(vision=vision)
    rig.sim.set_joint_positions(Q_OFF)
    rig.controller.process([Command("gesture_control", {"enabled": True})])
    ee0 = rig.model.ee_position(rig.sim.q)
    vision.output = make_output(1, target=ee0 + np.array([0.0, 0.3, 0.0]))
    rig.controller.gesture_tick()
    moved = np.linalg.norm(rig.model.ee_position(rig.sim.q) - ee0)
    assert 0.0 < moved <= 0.5 / 30.0 + 1e-9


def test_gesture_hold_does_not_move_the_robot():
    vision = FakeVision()
    rig = make_rig(vision=vision)
    rig.sim.set_joint_positions(Q_OFF)
    rig.controller.process([Command("gesture_control", {"enabled": True})])
    vision.output = make_output(1, action=Action.HOLD, gesture=Gesture.POINT)
    assert rig.controller.gesture_tick()
    assert np.allclose(rig.sim.q, Q_OFF)


def test_stop_gesture_stops_motion_but_keeps_gesture_control_enabled():
    vision = FakeVision()
    rig = make_rig(vision=vision)
    rig.sim.set_joint_positions(Q_OFF)
    rig.controller.process([Command("gesture_control", {"enabled": True})])
    vision.output = make_output(1, action=Action.STOP, gesture=Gesture.PEACE, stop=True)
    assert rig.controller.gesture_tick()
    assert rig.controller.gesture_enabled and rig.sim.stop_calls >= 1
    assert np.allclose(rig.sim.q, Q_OFF)

    # The robot follows the hand again as soon as a follow gesture comes back.
    ee0 = rig.model.ee_position(rig.sim.q)
    target = ee0 + np.array([0.0, 0.02, 0.0])
    vision.output = make_output(2, target=target)
    assert rig.controller.gesture_tick()
    assert np.linalg.norm(rig.model.ee_position(rig.sim.q) - target) < np.linalg.norm(ee0 - target)


def test_gesture_is_blocked_while_paused_and_during_estop_and_resumes_after():
    vision = FakeVision()
    rig = make_rig(vision=vision)
    rig.sim.set_joint_positions(Q_OFF)
    rig.controller.process([Command("gesture_control", {"enabled": True})])
    target = rig.model.ee_position(rig.sim.q) + np.array([0.0, 0.02, 0.0])
    vision.output = make_output(1, target=target)

    rig.controller.process([Command("pause")])
    assert not rig.controller.gesture_tick()
    assert np.allclose(rig.sim.q, Q_OFF)

    rig.controller.process([Command("resume")])
    rig.controller.process([Command("estop")])
    assert not rig.controller.gesture_tick()      # nothing moves during the e-stop ...
    assert rig.controller.gesture_enabled         # ... but the user's setting is kept
    assert np.allclose(rig.sim.q, Q_OFF)

    rig.controller.process([Command("clear_estop")])
    assert rig.controller.gesture_tick()
    assert not np.allclose(rig.sim.q, Q_OFF)


def test_stop_estop_and_reset_do_not_switch_gesture_control_off():
    rig = make_rig(vision=FakeVision())
    rig.controller.process([Command("gesture_control", {"enabled": True})])
    for name in ("stop", "estop", "clear_estop", "reset"):
        rig.controller.process([Command(name)])
        rig.controller.run_pending()
        assert rig.controller.gesture_enabled, name
    rig.controller.process([Command("gesture_control", {"enabled": False})])
    assert not rig.controller.gesture_enabled


def test_vision_pipeline_drives_the_robot_and_the_peace_gesture_stops_it():
    """Synthetic hand -> VisionPipeline -> GestureController -> safety -> simulation."""
    tracker = FakeTracker(OPEN)
    pipeline = VisionPipeline(
        tracker,
        GestureController(GestureConfig(frames_to_switch=2, lost_frames=3), MappingConfig()),
        blank_frame,
        fps_limit=1000.0,
    )
    rig = make_rig(vision=pipeline)
    rig.sim.set_joint_positions(Q_OFF)
    rig.controller.process([Command("gesture_control", {"enabled": True})])
    assert rig.controller.gesture_enabled

    start = rig.model.ee_position(rig.sim.q)
    for _ in range(6):
        pipeline.step()
        rig.controller.gesture_tick()
    assert np.linalg.norm(rig.model.ee_position(rig.sim.q) - start) > 1e-4
    assert rig.safety.check_configuration(rig.sim.q).ok

    tracker.lm = PEACE
    for _ in range(4):
        pipeline.step()
        rig.controller.gesture_tick()
    assert rig.controller.gesture_enabled      # stays on until the user switches it off
    assert rig.sim.stop_calls >= 1
    frozen = rig.sim.q
    for _ in range(3):
        pipeline.step()
        rig.controller.gesture_tick()
    assert np.allclose(rig.sim.q, frozen)      # PEACE holds the robot


# ----------------------------------------------------------------------------
# Application (supervisor loop, worker thread, dashboard)
# ----------------------------------------------------------------------------
def make_app(rig, server=None, **config_kwargs):
    config = AppConfig(open_browser=False, **config_kwargs)
    return Application(config, rig.sim, rig.hub, rig.controller, server)


def test_application_runs_commands_on_the_worker_thread(rig):
    app = make_app(rig)
    app.start()
    try:
        assert rig.sim.launched and rig.sim.loaded and rig.controller.worker_alive
        assert rig.hub.submit({"type": "set_joints", "q_deg": GOAL_DEG}).ok
        assert wait_for(app, lambda: np.allclose(rig.sim.q, GOAL_Q, atol=1e-6))
        app.publish_now()
        snap = rig.hub.snapshot()
        assert snap["q_deg"][0] == pytest.approx(20.0, abs=1e-3)
        assert snap["sim_state"] == "idle" and snap["busy"] is False
        assert snap["vision"]["gesture"] == "none"
    finally:
        app.shutdown()
    assert rig.sim.closed and not rig.controller.worker_alive
    app.shutdown()  # idempotent


def test_application_publishes_the_camera_preview(rig, monkeypatch):
    vision = FakeVision()
    vision.frame = np.zeros((4, 4, 3), dtype=np.uint8)
    rig.controller.vision = vision
    monkeypatch.setattr(main, "encode_jpeg", lambda frame: b"jpeg")
    app = make_app(rig)
    app.publish_now()
    assert rig.hub.latest_frame() == b"jpeg"
    assert rig.hub.snapshot()["camera"] is True


def test_application_survives_a_preview_encoding_error(rig, monkeypatch):
    vision = FakeVision()
    vision.frame = np.zeros((4, 4, 3), dtype=np.uint8)
    rig.controller.vision = vision

    def broken(frame):
        raise RuntimeError("no cv2")

    monkeypatch.setattr(main, "encode_jpeg", broken)
    app = make_app(rig)
    app.publish_now()
    assert rig.hub.latest_frame() is None
    assert any("preview disabled" in m for m in messages(rig))


def test_application_shutdown_releases_vision(rig):
    vision = FakeVision()
    rig.controller.vision = vision
    app = make_app(rig)
    app.start()
    assert vision.started
    app.shutdown()
    assert vision.closed and rig.controller.vision is None


def test_http_command_reaches_the_robot_end_to_end(rig):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    server = GuiServer(rig.hub, port=0, log_level="error")
    app = make_app(rig, server)
    app.start()
    try:
        body = json.dumps({"type": "set_joints", "q_deg": GOAL_DEG, "id": "e2e"}).encode()
        request = urllib.request.Request(
            server.url + "/api/command", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
            assert json.loads(response.read())["ok"] is True
        assert wait_for(app, lambda: np.allclose(rig.sim.q, GOAL_Q, atol=1e-6))
        app.publish_now()
        with urllib.request.urlopen(server.url + "/api/state", timeout=5) as response:
            state = json.loads(response.read())
        assert state["q_deg"][0] == pytest.approx(20.0, abs=1e-3)
    finally:
        app.shutdown()
    assert not server.running

def test_find_free_port_skips_busy_ports():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        found = main.find_free_port("127.0.0.1", port)
        assert found != port and found > 0


def test_find_free_port_with_zero_lets_the_os_choose():
    assert main.find_free_port("127.0.0.1", 0) > 0


def test_fingers_mirror_the_hand_even_when_the_robot_holds():
    vision = FakeVision()
    rig = make_rig(vision=vision)
    rig.controller.process([Command("gesture_control", {"enabled": True})])
    flex = {"thumb": 0.2, "index": 0.8, "middle": 0.5, "ring": 0.1, "pinky": 0.9}
    vision.output = make_output(1, action=Action.HOLD, gesture=Gesture.UNKNOWN, flexions=flex)
    assert rig.controller.gesture_tick()
    for name, value in flex.items():
        assert rig.system.hand.finger(name).flexion == pytest.approx(value)

    vision.output = make_output(2, action=Action.HOLD, gesture=Gesture.UNKNOWN,
                                flexions={**flex, "index": 0.2})
    rig.controller.gesture_tick()
    assert rig.system.hand.finger("index").flexion == pytest.approx(0.2)
    assert rig.system.hand.finger("middle").flexion == pytest.approx(0.5)