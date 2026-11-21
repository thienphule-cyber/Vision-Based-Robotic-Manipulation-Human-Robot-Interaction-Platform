"""Unit tests for gui.py (run from the project root: pytest tests/test_gui.py -v).

The hub / command tests need no web stack. The HTTP and WebSocket tests need
fastapi + httpx and the real-server test needs fastapi + uvicorn; they are skipped otherwise.
"""
import dataclasses
import enum
import json
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from gui import (
    DASHBOARD_HTML,
    Command,
    CommandError,
    CommandQueue,
    GuiServer,
    RobotHub,
    build_sim_state,
    build_vision_state,
    create_app,
    jsonable,
    origin_allowed,
    parse_command,
)
from modelling import FINGER_NAMES, GRASP_PRESETS, Environment, RobotModel, WorldObject
from safety import SafetyChecker, SafetyConfig


@pytest.fixture
def model():
    return RobotModel()


@pytest.fixture
def hub(model):
    return RobotHub(model)


def make_safety(model, env=None, **overrides):
    cfg = dict(workspace=None, singularity_threshold=0.0)
    cfg.update(overrides)
    return SafetyChecker(model, Environment() if env is None else env, SafetyConfig(**cfg))


# ----------------------------------------------------------------------------
# jsonable
# ----------------------------------------------------------------------------
class Colour(enum.Enum):
    RED = "red"


@dataclasses.dataclass
class Point:
    x: float
    y: float


def test_jsonable_conversions():
    data = {
        "arr": np.array([1.0, np.nan, 3.0]),
        "f": np.float64(1.5),
        "i": np.int64(3),
        "b": np.bool_(True),
        "inf": float("inf"),
        "enum": Colour.RED,
        "dc": Point(1.0, 2.0),
        "tuple": (1, 2),
        1: "key",
    }
    out = jsonable(data)
    assert out["arr"] == [1.0, None, 3.0]
    assert out["f"] == 1.5 and out["i"] == 3 and out["b"] is True
    assert out["inf"] is None
    assert out["enum"] == "red"
    assert out["dc"] == {"x": 1.0, "y": 2.0}
    assert out["tuple"] == [1, 2]
    assert out["1"] == "key"
    json.dumps(out)


# ----------------------------------------------------------------------------
# parse_command
# ----------------------------------------------------------------------------
def test_parse_valid_commands():
    assert parse_command({"type": "home", "id": 7}).id == "7"
    c = parse_command({"type": "set_joints", "q_deg": [0, 1, 2, 3, 4, 5]})
    assert c.params["q_deg"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    c = parse_command({"type": "move_to", "position": [0.3, 0.0, 0.4]})
    assert c.params["rpy_deg"] is None
    c = parse_command({"type": "move_to", "position": [0.3, 0, 0.4], "rpy_deg": [0, 90, 0]})
    assert c.params["rpy_deg"] == [0.0, 90.0, 0.0]
    assert parse_command({"type": "grasp", "preset": "pinch"}).params == {"preset": "pinch"}
    assert parse_command({"type": "finger", "finger": "index", "flexion": 0.5}).params["flexion"] == 0.5
    assert parse_command({"type": "playback"}).params["speed"] == 1.0
    assert parse_command({"type": "gesture_control", "enabled": False}).params["enabled"] is False


@pytest.mark.parametrize(
    "data",
    [
        "home",
        ["home"],
        {},
        {"type": "explode"},
        {"type": 5},
        {"type": "home", "id": [1]},
        {"type": "home", "id": True},
        {"type": "home", "id": "x" * 100},
        {"type": "set_joints"},
        {"type": "set_joints", "q_deg": [0, 0, 0]},
        {"type": "set_joints", "q_deg": [0, 0, 0, 0, 0, float("nan")]},
        {"type": "set_joints", "q_deg": [0, 0, 0, 0, 0, True]},
        {"type": "set_joints", "q_deg": [0, 0, 0, 0, 0, "1"]},
        {"type": "set_joints", "q_deg": [0, 0, 0, 0, 0, 1e6]},
        {"type": "move_to", "position": [1, 2]},
        {"type": "move_to", "position": [100, 0, 0]},
        {"type": "move_to", "position": [0, 0, 0], "rpy_deg": [0, 0]},
        {"type": "grasp", "preset": "wave"},
        {"type": "grasp"},
        {"type": "finger", "finger": "tail", "flexion": 0.5},
        {"type": "finger", "finger": "index", "flexion": 1.5},
        {"type": "finger", "finger": "index"},
        {"type": "playback", "speed": 0},
        {"type": "playback", "speed": 99},
        {"type": "gesture_control", "enabled": "yes"},
    ],
)
def test_parse_invalid_commands(data):
    with pytest.raises(CommandError):
        parse_command(data)


# ----------------------------------------------------------------------------
# CommandQueue
# ----------------------------------------------------------------------------
def test_queue_returns_urgent_first():
    q = CommandQueue()
    q.put(Command("home"))
    q.put(Command("pause"))
    q.put(Command("resume"))
    assert [c.type for c in q.drain()] == ["pause", "home", "resume"]
    assert len(q) == 0


def test_queue_stop_and_estop_flush_pending_commands():
    q = CommandQueue()
    q.put(Command("home"))
    q.put(Command("reset"))
    q.put(Command("stop"))
    assert [c.type for c in q.drain()] == ["stop"]
    q.put(Command("home"))
    q.put(Command("estop"))
    assert [c.type for c in q.drain()] == ["estop"]


def test_queue_is_bounded_but_accepts_urgent():
    q = CommandQueue(maxsize=2)
    assert q.put(Command("home")) and q.put(Command("home"))
    assert not q.put(Command("home"))
    assert q.put(Command("pause"))
    assert len(q) == 3  # urgent commands live in a separate lane: 2 normal + 1 urgent
    with pytest.raises(ValueError):
        CommandQueue(0)


# ----------------------------------------------------------------------------
# RobotHub: submit
# ----------------------------------------------------------------------------
def test_submit_valid_command_is_queued(hub):
    result = hub.submit({"type": "home", "id": "c1"})
    assert result.ok and result.command_id == "c1" and result.status == "ok"
    cmds = hub.poll_commands()
    assert [c.type for c in cmds] == ["home"] and cmds[0].id == "c1"
    assert hub.poll_commands() == []


def test_submit_invalid_command_is_not_queued(hub):
    result = hub.submit({"type": "explode"})
    assert not result.ok and result.status == "invalid"
    assert hub.poll_commands() == []


def test_submit_rejects_joint_limit_violation(hub):
    result = hub.submit({"type": "set_joints", "q_deg": [0, 0, -10, 0, 0, 0]})  # J3 lower limit is 0
    assert not result.ok and result.status == "rejected"
    assert "J3" in result.message
    assert hub.poll_commands() == []


def test_submit_accepts_joint_limits_exactly(hub, model):
    lo, hi = np.rad2deg(model.qlim)
    assert hub.submit({"type": "set_joints", "q_deg": hi.tolist()}).ok
    assert hub.submit({"type": "set_joints", "q_deg": lo.tolist()}).ok


def test_submit_when_queue_is_full(model):
    hub = RobotHub(model, max_queue=2)
    assert hub.submit({"type": "home"}).ok and hub.submit({"type": "home"}).ok
    result = hub.submit({"type": "home"})
    assert not result.ok and "full" in result.message.lower()
    assert hub.submit({"type": "stop"}).ok


def test_stop_command_discards_pending_commands(hub):
    hub.submit({"type": "home"})
    hub.submit({"type": "set_joints", "q_deg": [0, 0, 0, 0, 0, 0]})
    hub.submit({"type": "stop"})
    assert [c.type for c in hub.poll_commands()] == ["stop"]


def test_pause_is_polled_before_normal_commands(hub):
    hub.submit({"type": "home"})
    hub.submit({"type": "pause"})
    assert [c.type for c in hub.poll_commands()] == ["pause", "home"]


# ----------------------------------------------------------------------------
# RobotHub: safety integration
# ----------------------------------------------------------------------------
def test_estop_latches_safety_immediately(model):
    safety = make_safety(model)
    hub = RobotHub(model, safety)
    assert hub.submit({"type": "estop"}).ok
    assert safety.estopped and hub.estopped and hub.snapshot()["estop"] is True
    assert [c.type for c in hub.poll_commands()] == ["estop"]

    result = hub.submit({"type": "home"})
    assert not result.ok and result.status == "rejected"
    assert hub.submit({"type": "pause"}).ok  # non-motion commands still work

    assert hub.submit({"type": "clear_estop"}).ok
    assert safety.estopped  # only the control loop clears the safety latch
    safety.clear_estop()
    assert hub.submit({"type": "home"}).ok


def test_estop_without_safety_object(hub):
    assert hub.submit({"type": "estop"}).ok
    assert hub.estopped and hub.estop_reason
    assert not hub.submit({"type": "home"}).ok
    assert hub.submit({"type": "clear_estop"}).ok
    assert not hub.estopped
    assert hub.submit({"type": "home"}).ok


def test_set_joints_collision_is_rejected_with_safety(model):
    probe = make_safety(model)
    tip = probe.robot_capsules(model.q_home)[-1].p1
    env = Environment()
    env.add(WorldObject("o", "obstacle", "box", tuple(tip), (0.1, 0.1, 0.1)))
    hub = RobotHub(model, make_safety(model, env))
    result = hub.submit({"type": "set_joints", "q_deg": [0, 0, 0, 0, 0, 0]})
    assert not result.ok and result.status == "rejected"
    assert "collision" in result.message.lower()
    assert hub.poll_commands() == []


def test_set_joints_ok_with_safety(model):
    hub = RobotHub(model, SafetyChecker(model, Environment.default_workcell()))
    result = hub.submit({"type": "set_joints", "q_deg": [0, 0, 0, 0, 0, 0]})
    assert result.ok, result.message
    assert isinstance(result.warnings, list)


def test_move_to_is_prechecked_with_safety(model):
    hub = RobotHub(model, SafetyChecker(model, Environment()))
    assert hub.submit({"type": "move_to", "position": [0.3, 0.0, 0.4]}).ok
    far = hub.submit({"type": "move_to", "position": [1.5, 0.0, 0.4]})
    assert not far.ok and far.status == "rejected"


def test_hand_commands_are_prechecked_with_safety(model):
    hub = RobotHub(model, make_safety(model))
    assert hub.submit({"type": "grasp", "preset": "pinch"}).ok
    assert hub.submit({"type": "finger", "finger": "index", "flexion": 0.5}).ok


# ----------------------------------------------------------------------------
# RobotHub: state, log, config, frame
# ----------------------------------------------------------------------------
def test_snapshot_merges_state_and_is_json_serialisable(hub):
    hub.publish_state(q_deg=np.array([1.0, 2.0]), sim_state="idle")
    hub.publish_state({"extra": 1}, sim_state="running")
    hub.client_connected()
    snap = hub.snapshot()
    assert snap["q_deg"] == [1.0, 2.0] and snap["sim_state"] == "running" and snap["extra"] == 1
    assert snap["estop"] is False and snap["queue_depth"] == 0 and snap["clients"] == 1
    assert snap["camera"] is False and "timestamp" in snap
    json.dumps(snap)
    hub.client_disconnected()
    hub.client_disconnected()
    assert hub.clients == 0


def test_log_sequence_and_size_limit(model):
    hub = RobotHub(model, log_size=3)
    for i in range(5):
        hub.log("info", f"m{i}")
    entries, seq = hub.log_since(0)
    assert [e["message"] for e in entries] == ["m2", "m3", "m4"]
    assert seq == 5
    assert hub.log_since(seq)[0] == []
    hub.log("bogus-level", "x")
    assert hub.log_since(seq)[0][0]["level"] == "info"


def test_config_describes_the_robot(hub, model):
    cfg = hub.config()
    assert len(cfg["joints"]) == 6
    lo, hi = np.rad2deg(model.qlim)
    assert cfg["joints"][2]["min_deg"] == pytest.approx(lo[2])
    assert cfg["joints"][0]["max_deg"] == pytest.approx(hi[0])
    assert cfg["fingers"] == list(FINGER_NAMES)
    assert cfg["grasp_presets"] == list(GRASP_PRESETS)
    json.dumps(cfg)


def test_camera_frame(hub):
    assert hub.latest_frame() is None
    hub.publish_frame(b"\xff\xd8jpeg")
    assert hub.latest_frame() == b"\xff\xd8jpeg"
    assert hub.snapshot()["camera"] is True
    with pytest.raises(TypeError):
        hub.publish_frame("not bytes")


# ----------------------------------------------------------------------------
# State builders
# ----------------------------------------------------------------------------
def test_build_sim_state(model):
    q = np.deg2rad([10, 20, 30, 0, 40, 0])
    fake = SimpleNamespace(
        q=q,
        ee_pose=lambda: model.fkine(q),
        hand=SimpleNamespace(fingers={n: SimpleNamespace(flexion=0.25) for n in FINGER_NAMES}),
        state=SimpleNamespace(value="idle"),
        frames=[1, 2, 3],
        _recording=True,
    )
    state = build_sim_state(fake)
    assert np.allclose(state["q_deg"], [10, 20, 30, 0, 40, 0])
    assert len(state["ee"]["position"]) == 3 and len(state["ee"]["rpy_deg"]) == 3
    assert state["hand"]["thumb"] == 0.25 and state["sim_state"] == "idle"
    assert state["recording"] == {"recording": True, "frames": 3}
    json.dumps(jsonable(state))


def test_build_vision_state():
    assert build_vision_state(None)["gesture"] == "none"
    out = SimpleNamespace(
        command=SimpleNamespace(
            hand_detected=True,
            gesture=SimpleNamespace(value="fist"),
            action=SimpleNamespace(value="follow"),
        ),
        fps=29.5,
    )
    state = build_vision_state(out, enabled=True)
    assert state == {"enabled": True, "hand_detected": True, "gesture": "fist",
                     "action": "follow", "fps": 29.5}


# ----------------------------------------------------------------------------
# Origin check and dashboard page
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "origin, expected",
    [
        (None, True),
        ("", True),
        ("http://localhost:8000", True),
        ("http://127.0.0.1:5000", True),
        ("http://[::1]:8000", True),
        ("http://evil.example", False),
        ("http://localhost.evil.example", False),
        ("null", False),
    ],
)
def test_origin_allowed(origin, expected):
    assert origin_allowed(origin) is expected


def test_origin_allowed_with_custom_hosts():
    assert origin_allowed("http://192.168.1.5:8000", ("192.168.1.5",))
    assert not origin_allowed("http://localhost:8000", ("192.168.1.5",))


def test_dashboard_html_has_the_main_elements():
    for needle in ("WebSocket", 'id="estop"', 'id="joints"', 'id="log"', "/ws", "set_joints"):
        assert needle in DASHBOARD_HTML
    assert "localStorage" not in DASHBOARD_HTML


def test_create_app_rejects_bad_rate(hub):
    with pytest.raises(ValueError):
        create_app(hub, push_hz=0)


# ----------------------------------------------------------------------------
# HTTP and WebSocket API (fastapi + httpx)
# ----------------------------------------------------------------------------
@pytest.fixture
def api(model):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    hub = RobotHub(model)
    app = create_app(hub, push_hz=100.0)
    with TestClient(app) as client:
        yield hub, client


def recv_until(ws, msg_type, limit=300):
    for _ in range(limit):
        msg = ws.receive_json()
        if msg["type"] == msg_type:
            return msg
    raise AssertionError(f"No '{msg_type}' message received")


def test_http_index_health_config_state(api):
    hub, client = api
    page = client.get("/")
    assert page.status_code == 200 and "VRM-HRI" in page.text and "/ws" in page.text
    assert client.get("/api/health").json() == {"ok": True}
    assert len(client.get("/api/config").json()["joints"]) == 6
    hub.publish_state(sim_state="idle")
    assert client.get("/api/state").json()["sim_state"] == "idle"


def test_http_camera_endpoint(api):
    hub, client = api
    assert client.get("/api/camera.jpg").status_code == 404
    hub.publish_frame(b"\xff\xd8fake")
    response = client.get("/api/camera.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"\xff\xd8fake"


def test_http_command_ok_invalid_and_rejected(api):
    hub, client = api
    ok = client.post("/api/command", json={"type": "home", "id": "a"})
    assert ok.status_code == 200 and ok.json()["ok"] is True
    assert [c.id for c in hub.poll_commands()] == ["a"]

    bad = client.post("/api/command", json={"type": "explode"})
    assert bad.status_code == 400 and bad.json()["status"] == "invalid"

    rejected = client.post("/api/command", json={"type": "set_joints", "q_deg": [0, 0, -10, 0, 0, 0]})
    assert rejected.status_code == 409 and rejected.json()["status"] == "rejected"


def test_http_command_protections(api):
    hub, client = api
    not_json = client.post("/api/command", content='{"type": "home"}',
                           headers={"content-type": "text/plain"})
    assert not_json.status_code == 415
    cross_site = client.post("/api/command", json={"type": "home"},
                             headers={"origin": "http://evil.example"})
    assert cross_site.status_code == 403
    broken = client.post("/api/command", content="{not json",
                         headers={"content-type": "application/json"})
    assert broken.status_code == 400
    assert hub.poll_commands() == []


def test_websocket_initial_messages_and_state_push(api):
    hub, client = api
    hub.publish_state(q_deg=[1, 2, 3, 4, 5, 6])
    hub.log("info", "hello")
    with client.websocket_connect("/ws") as ws:
        first = [ws.receive_json() for _ in range(3)]
        assert [m["type"] for m in first] == ["config", "log", "state"]
        assert len(first[0]["config"]["joints"]) == 6
        assert any(e["message"] == "hello" for e in first[1]["entries"])
        assert first[2]["state"]["q_deg"] == [1, 2, 3, 4, 5, 6]
        assert hub.clients == 1


def test_websocket_command_ack(api):
    hub, client = api
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "home", "id": "c1"})
        ack = recv_until(ws, "ack")
        assert ack["ok"] is True and ack["id"] == "c1"
        ws.send_json({"type": "explode", "id": "c2"})
        assert recv_until(ws, "ack")["ok"] is False
        ws.send_text("this is not json")
        bad = recv_until(ws, "ack")
        assert bad["ok"] is False and bad["status"] == "invalid"
    assert [c.type for c in hub.poll_commands()] == ["home"]


def test_websocket_receives_new_log_entries(api):
    hub, client = api
    with client.websocket_connect("/ws") as ws:
        hub.log("warning", "something happened")
        deadline = 300
        found = False
        for _ in range(deadline):
            msg = ws.receive_json()
            if msg["type"] == "log" and any(e["message"] == "something happened" for e in msg["entries"]):
                found = True
                break
        assert found


def test_websocket_rejects_foreign_origin(api):
    from starlette.websockets import WebSocketDisconnect

    _, client = api
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}):
            pass


# ----------------------------------------------------------------------------
# Real server (uvicorn in a background thread)
# ----------------------------------------------------------------------------
def test_gui_server_serves_the_dashboard(hub):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    server = GuiServer(hub, port=0, log_level="error")
    server.start()
    try:
        assert server.running and server.port > 0
        with urllib.request.urlopen(server.url + "/api/health", timeout=5) as response:
            assert json.loads(response.read())["ok"] is True
        with urllib.request.urlopen(server.url + "/", timeout=5) as response:
            assert b"VRM-HRI" in response.read()
    finally:
        server.stop()
    assert not server.running