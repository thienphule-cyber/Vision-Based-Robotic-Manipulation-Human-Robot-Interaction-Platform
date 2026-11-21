"""
gui.py - Web dashboard (FastAPI + WebSocket) for VRM-HRI.

Running `python gui.py` starts the dashboard with a tiny fake robot so the page can be tried
without Swift, MuJoCo or a camera. In the real application main.py owns a RobotHub and a
GuiServer.

Contents
--------
1. Helpers        : JSON conversion, command parsing / validation.
2. CommandQueue   : thread-safe queue (urgent commands first).
3. RobotHub       : the bridge between the GUI thread and the control loop.
4. State builders : build_sim_state() / build_vision_state() for main.py.
5. Dashboard HTML : HTML, CSS and JS in one string.
6. Server         : create_app() (FastAPI), GuiServer (uvicorn in a background thread).

Architecture
------------
    browser <--WebSocket / REST--> FastAPI (GUI thread)
                                        |  hub.submit(): validate, pre-check with safety.py, enqueue
                                        v
                                RobotHub.queue
                                        |  hub.poll_commands()     (control loop in main.py)
                                        v
            kinematics (IK / planning) -> safety.limit_step / check_trajectory -> simulation

* The GUI never moves the robot. Every executed motion must still go through safety.py in
the control loop. The pre-checks in RobotHub.submit() only give fast feedback.
* E-STOP is the exception to "queue everything": hub.submit({"type": "estop"}) latches
safety.estop() immediately, in the GUI thread, then also queues the command so the control
loop can stop the simulation. Clearing the e-stop is done by the control loop
(safety.clear_estop()) after it handles the "clear_estop" command.
* "stop" and "estop" discard all pending non-urgent commands.
* While a trajectory is running in the control loop, urgent commands (stop / pause / estop)
must still be polled: run trajectories in a worker thread or poll inside the loop.
* The server binds to 127.0.0.1 by default. POST requests must be application/json and the
Origin of browser requests must be a local host, so other web pages cannot drive the robot.

Wire format (client -> server): a JSON object {"type": ..., "id": optional, ...params}
    set_joints       q_deg: [n numbers]
    move_to          position: [x, y, z] (m), rpy_deg: [r, p, y] (optional)
    grasp            preset: one of GRASP_PRESETS
    finger           finger: finger name, flexion: 0..1
    playback         speed: (0, 10], default 1
    gesture_control  enabled: bool
    home, reset, pause, resume, stop, estop, clear_estop,
    start_recording, stop_recording, clear_recording      (no parameters)
Server -> client messages: config, log, state, ack ("type" is the message kind).
"""
from __future__ import annotations

import asyncio
import collections
import dataclasses
import enum
import logging
import math
import socket
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
from spatialmath import SE3

try:
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse, Response
except ImportError:  # the dashboard needs fastapi; the rest of the module does not
    FastAPI = Request = WebSocket = WebSocketDisconnect = None
    HTMLResponse = JSONResponse = Response = None

from modelling import FINGER_NAMES, GRASP_PRESETS, RobotModel

logger = logging.getLogger(__name__)

LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

SIMPLE_COMMANDS = frozenset({
    "home", "reset", "pause", "resume", "stop", "estop", "clear_estop",
    "start_recording", "stop_recording", "clear_recording",
})
COMMAND_TYPES = SIMPLE_COMMANDS | {"set_joints", "move_to", "grasp", "finger", "playback", "gesture_control"}
# Commands that move the robot or the hand: rejected while the e-stop is engaged.
MOTION_COMMANDS = frozenset({"set_joints", "move_to", "home", "reset", "grasp", "finger", "playback", "resume"})
URGENT_COMMANDS = frozenset({"estop", "stop", "pause"})
# These urgent commands also discard everything that is still waiting in the queue.
FLUSHING_COMMANDS = frozenset({"estop", "stop"})


# ----------------------------------------------------------------------------
# 1. Helpers
# ----------------------------------------------------------------------------

def jsonable(obj: Any) -> Any:
    """Convert numpy arrays / scalars, enums, dataclasses and non-finite floats to JSON types."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, enum.Enum):
        return jsonable(obj.value)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(dataclasses.asdict(obj))
    return str(obj)


class CommandError(ValueError):
    """Raised when a client command is malformed."""


@dataclass
class Command:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    source: str = "gui"
    timestamp: float = field(default_factory=time.time)

    @property
    def urgent(self) -> bool:
        return self.type in URGENT_COMMANDS

    def to_dict(self) -> dict:
        return {"type": self.type, "id": self.id, "source": self.source,
                "timestamp": self.timestamp, **jsonable(self.params)}


@dataclass
class CommandResult:
    ok: bool
    message: str = ""
    command_id: Optional[str] = None
    status: str = "ok"                 # "ok" | "invalid" | "rejected"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.command_id, "ok": self.ok, "status": self.status,
                "message": self.message, "warnings": list(self.warnings)}


def _number(value, name: str, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommandError(f"{name} must be a number")
    v = float(value)
    if not math.isfinite(v):
        raise CommandError(f"{name} must be finite")
    if lo is not None and v < lo:
        raise CommandError(f"{name} must be >= {lo}")
    if hi is not None and v > hi:
        raise CommandError(f"{name} must be <= {hi}")
    return v


def _vector(value, n: int, name: str, bound: float) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise CommandError(f"{name} must be a list of {n} numbers")
    return [_number(v, f"{name}[{i}]", -bound, bound) for i, v in enumerate(value)]


def parse_command(data, n_joints: int = 6) -> Command:
    """Validate the shape and ranges of a client command. Raises CommandError."""
    if not isinstance(data, Mapping):
        raise CommandError("Command must be a JSON object")
    ctype = data.get("type")
    if not isinstance(ctype, str) or ctype not in COMMAND_TYPES:
        raise CommandError(f"Unknown command type: {ctype!r}")
    cid = data.get("id")
    if cid is not None:
        if isinstance(cid, bool) or not isinstance(cid, (str, int)):
            raise CommandError("id must be a string or an integer")
        cid = str(cid)
        if len(cid) > 64:
            raise CommandError("id is too long")

    params: Dict[str, Any] = {}
    if ctype == "set_joints":
        params["q_deg"] = _vector(data.get("q_deg"), n_joints, "q_deg", 720.0)
    elif ctype == "move_to":
        params["position"] = _vector(data.get("position"), 3, "position", 5.0)
        rpy = data.get("rpy_deg")
        params["rpy_deg"] = None if rpy is None else _vector(rpy, 3, "rpy_deg", 360.0)
    elif ctype == "grasp":
        preset = data.get("preset")
        if not isinstance(preset, str) or preset not in GRASP_PRESETS:
            raise CommandError(f"Unknown grasp {preset!r}. Valid: {list(GRASP_PRESETS)}")
        params["preset"] = preset
    elif ctype == "finger":
        name = data.get("finger")
        if not isinstance(name, str) or name not in FINGER_NAMES:
            raise CommandError(f"Unknown finger {name!r}. Valid: {list(FINGER_NAMES)}")
        params["finger"] = name
        params["flexion"] = _number(data.get("flexion"), "flexion", 0.0, 1.0)
    elif ctype == "playback":
        speed = 1.0 if data.get("speed") is None else _number(data.get("speed"), "speed")
        if not 0.0 < speed <= 10.0:
            raise CommandError("speed must be in (0, 10]")
        params["speed"] = speed
    elif ctype == "gesture_control":
        enabled = data.get("enabled")
        if not isinstance(enabled, bool):
            raise CommandError("enabled must be true or false")
        params["enabled"] = enabled
    return Command(ctype, params, cid)


# ----------------------------------------------------------------------------
# 2. Command queue
# ----------------------------------------------------------------------------

class CommandQueue:
    """Thread-safe queue. Urgent commands are returned first. Normal commands are bounded."""

    def __init__(self, maxsize: int = 100) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self._lock = threading.Lock()
        self._urgent: Deque[Command] = collections.deque()
        self._normal: Deque[Command] = collections.deque()

    def put(self, cmd: Command) -> bool:
        """Returns False if a normal command was dropped because the queue is full."""
        with self._lock:
            if cmd.urgent:
                if cmd.type in FLUSHING_COMMANDS:
                    self._normal.clear()
                self._urgent.append(cmd)
                return True
            if len(self._normal) >= self.maxsize:
                return False
            self._normal.append(cmd)
            return True

    def drain(self) -> List[Command]:
        with self._lock:
            items = list(self._urgent) + list(self._normal)
            self._urgent.clear()
            self._normal.clear()
            return items

    def __len__(self) -> int:
        with self._lock:
            return len(self._urgent) + len(self._normal)


# ----------------------------------------------------------------------------
# 3. RobotHub
# ----------------------------------------------------------------------------

class RobotHub:
    """Shared state, event log and command queue between the GUI and the control loop.

    `safety` (a safety.SafetyChecker) is optional but recommended: with it, commands are
    pre-checked (joint limits, workspace, collision, hand commands) and the e-stop is shared.

    The state is a free-form dict filled with publish_state(). The dashboard reads these keys
    when present: sim_state, q_deg, ee{position, rpy_deg}, hand{finger: flexion},
    recording{recording, frames}, vision{enabled, hand_detected, gesture, action, fps}.
    """

    def __init__(
        self,
        model: Optional[RobotModel] = None,
        safety=None,
        max_queue: int = 100,
        log_size: int = 200,
    ) -> None:
        self.model = RobotModel() if model is None else model
        self.safety = safety
        self._queue = CommandQueue(max_queue)
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {}
        self._log: Deque[dict] = collections.deque(maxlen=log_size)
        self._log_seq = 0
        self._frame: Optional[bytes] = None
        self._clients = 0
        self._estop = False
        self._estop_reason = ""

    @property
    def n_joints(self) -> int:
        return int(self.model.n)

    # ---- e-stop -----------------------------------------------------------
    @property
    def estopped(self) -> bool:
        if self.safety is not None:
            return bool(self.safety.estopped)
        return self._estop

    @property
    def estop_reason(self) -> str:
        if self.safety is not None:
            return str(self.safety.estop_reason)
        return self._estop_reason

    def _engage_estop(self) -> None:
        if self.safety is not None:
            self.safety.estop("GUI emergency stop")
        else:
            self._estop = True
            self._estop_reason = "GUI emergency stop"
        self.log("error", "Emergency stop engaged from the GUI")

    # ---- configuration for the dashboard ------------------------------------------
    def config(self) -> dict:
        lo, hi = np.rad2deg(self.model.qlim)
        workspace = getattr(getattr(self.safety, "config", None), "workspace", None)
        return jsonable({
            "version": 1,
            "robot": self.model.robot.name,
            "joints": [
                {"name": name, "min_deg": float(lo[i]), "max_deg": float(hi[i])}
                for i, name in enumerate(self.model.joint_names)
            ],
            "fingers": list(FINGER_NAMES),
            "grasp_presets": list(GRASP_PRESETS),
            "workspace": workspace,
        })

    # ---- state ------------------------------------------------------------------------
    def publish_state(self, fields: Optional[Mapping[str, Any]] = None, **kwargs) -> None:
        with self._lock:
            if fields:
                self._state.update(fields)
            self._state.update(kwargs)

    def snapshot(self) -> dict:
        with self._lock:
            state = dict(self._state)
            state["camera"] = self._frame is not None
            clients = self._clients
        state["timestamp"] = time.time()
        state["estop"] = self.estopped
        state["estop_reason"] = self.estop_reason
        state["queue_depth"] = len(self._queue)
        state["clients"] = clients
        return jsonable(state)

    # ---- camera frame --------------------------------------------------------------------
    def publish_frame(self, jpeg: bytes) -> None:
        if not isinstance(jpeg, (bytes, bytearray)):
            raise TypeError("publish_frame expects JPEG bytes")
        with self._lock:
            self._frame = bytes(jpeg)

    def latest_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._frame

    # ---- event log -------------------------------------------------------------------------
    def log(self, level: str, message: str) -> None:
        if level not in ("info", "warning", "error"):
            level = "info"
        with self._lock:
            self._log_seq += 1
            self._log.append({"seq": self._log_seq, "t": time.time(), "level": level,
                            "message": str(message)})

    def log_since(self, seq: int) -> Tuple[List[dict], int]:
        """Log entries with seq > `seq`, and the newest seq."""
        with self._lock:
            return [dict(e) for e in self._log if e["seq"] > seq], self._log_seq

    # ---- clients ---------------------------------------------------------------------------------
    @property
    def clients(self) -> int:
        with self._lock:
            return self._clients

    def client_connected(self) -> None:
        with self._lock:
            self._clients += 1

    def client_disconnected(self) -> None:
        with self._lock:
            self._clients = max(0, self._clients - 1)

    # ---- commands ------------------------------------------------------------------------------------
    def poll_commands(self) -> List[Command]:
        """Called by the control loop: returns and clears the pending commands (urgent first)."""
        return self._queue.drain()

    def submit(self, data) -> CommandResult:
        """Validate, pre-check and enqueue a client command. Thread safe."""
        try:
            cmd = parse_command(data, self.n_joints)
        except CommandError as exc:
            return CommandResult(False, str(exc), None, "invalid")

        if self.estopped and cmd.type in MOTION_COMMANDS:
            return self._reject(cmd, "Emergency stop is engaged; clear it first")
        ok, message, warnings = self._precheck(cmd)
        if not ok:
            return self._reject(cmd, message)

        if cmd.type == "estop":
            self._engage_estop()
        elif cmd.type == "clear_estop" and self.safety is None:
            self._estop = False
            self._estop_reason = ""
        if not self._queue.put(cmd):
            return self._reject(cmd, "Command queue is full")
        self.log("info", f"Command accepted: {cmd.type}")
        return CommandResult(True, "queued", cmd.id, "ok", warnings)

    def _reject(self, cmd: Command, message: str) -> CommandResult:
        self.log("warning", f"Command rejected ({cmd.type}): {message}")
        return CommandResult(False, message, cmd.id, "rejected")

    def _precheck(self, cmd: Command) -> Tuple[bool, str, List[str]]:
        p = cmd.params
        warnings: List[str] = []
        if cmd.type == "set_joints":
            q = np.deg2rad(p["q_deg"])
            lo, hi = self.model.qlim
            bad = [self.model.joint_names[i] for i in range(self.model.n)
                if q[i] < lo[i] - 1e-9 or q[i] > hi[i] + 1e-9]
            if bad:
                return False, f"Joint limit exceeded: {', '.join(bad)}", warnings
            if self.safety is not None:
                report = self.safety.check_configuration(q)
                warnings = list(report.warnings)
                if not report.ok:
                    return False, report.summary(), warnings
        elif cmd.type == "grasp" and self.safety is not None:
            report = self.safety.check_hand_command(grasp=p["preset"])
            if not report.ok:
                return False, report.summary(), warnings
        elif cmd.type == "finger" and self.safety is not None:
            report = self.safety.check_hand_command(flexions={p["finger"]: p["flexion"]})
            if not report.ok:
                return False, report.summary(), warnings
        elif cmd.type == "move_to" and self.safety is not None:
            r, pitch, y = (p["rpy_deg"] or [0.0, 0.0, 0.0])
            pose = (SE3(*p["position"]) * SE3.Rz(y, unit="deg")
                    * SE3.Ry(pitch, unit="deg") * SE3.Rx(r, unit="deg"))
            report = self.safety.check_pose_target(pose)
            warnings = list(report.warnings)
            if not report.ok:
                return False, report.summary(), warnings
        return True, "", warnings


# ----------------------------------------------------------------------------
# 4. State builders (duck-typed, used by main.py)
# ----------------------------------------------------------------------------

def build_sim_state(sim) -> dict:
    """State fields from a simulation.Simulation (call from the thread that owns it)."""
    T = sim.ee_pose()
    return {
        "q_deg": np.rad2deg(sim.q),
        "ee": {"position": np.asarray(T.t, dtype=float),
            "rpy_deg": np.asarray(T.rpy(unit="deg"), dtype=float)},
        "hand": {name: finger.flexion for name, finger in sim.hand.fingers.items()},
        "sim_state": sim.state.value,
        "recording": {"recording": bool(getattr(sim, "_recording", False)),
                    "frames": len(sim.frames)},
    }


def build_vision_state(output, enabled: bool = True) -> dict:
    """Vision fields from a computer_vision.VisionOutput (or None)."""
    if output is None:
        return {"enabled": enabled, "hand_detected": False, "gesture": "none",
                "action": "none", "fps": 0.0}
    cmd = output.command
    return {"enabled": enabled, "hand_detected": bool(cmd.hand_detected),
            "gesture": cmd.gesture.value, "action": cmd.action.value, "fps": float(output.fps)}


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 70) -> Optional[bytes]:
    """Encode a BGR frame as JPEG (cv2 imported lazily). Returns None on failure."""
    import cv2

    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return bytes(buf) if ok else None


# ----------------------------------------------------------------------------
# 5. Dashboard (HTML + CSS + JS in one file)
# ----------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VRM-HRI Dashboard</title>
<style>
:root{--bg:#f4f5f7;--panel:#ffffff;--text:#1d2330;--muted:#6b7280;--border:#d9dde5;--accent:#2563eb;--danger:#dc2626;--ok:#16a34a;--warn:#d97706}
@media (prefers-color-scheme: dark){:root{--bg:#12151c;--panel:#1b2030;--text:#e6e9f0;--muted:#98a2b3;--border:#2b3347;--accent:#60a5fa;--danger:#f87171;--ok:#4ade80;--warn:#fbbf24}}
*{box-sizing:border-box}
body{margin:0;font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--text)}
header{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:5}
h1{font-size:17px;margin:0 8px 0 0}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:0 0 10px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;padding:12px}
section{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px}
section.wide{grid-column:1/-1}
.badge{padding:2px 9px;border-radius:99px;font-size:12px;border:1px solid var(--border);color:var(--muted)}
.badge.ok{color:var(--ok);border-color:var(--ok)}
.badge.bad{color:var(--danger);border-color:var(--danger)}
.badge.warn{color:var(--warn);border-color:var(--warn)}
.hidden{display:none}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--border);background:var(--panel);color:var(--text);cursor:pointer}
button:hover{border-color:var(--accent)}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.danger{background:var(--danger);border-color:var(--danger);color:#fff;font-weight:700}
button.big{padding:8px 18px}
.row{display:grid;grid-template-columns:58px 1fr 120px;gap:8px;align-items:center;margin:5px 0}
.row input[type=range]{width:100%}
.row .val{font-variant-numeric:tabular-nums;color:var(--muted);text-align:right;font-size:12px}
.btns{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}
.kv{display:flex;justify-content:space-between;margin:3px 0}
.kv span{color:var(--muted)}
.kv b{font-variant-numeric:tabular-nums;font-weight:600}
.form{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:8px 0}
.form input,.inline input[type=number]{width:100%;padding:5px;border:1px solid var(--border);border-radius:6px;background:var(--bg);color:var(--text)}
.inline{display:flex;gap:8px;align-items:center}
.inline input[type=number]{width:70px}
#log{max-height:240px;overflow:auto;font:12px/1.5 ui-monospace,Menlo,Consolas,monospace}
.log.warning{color:var(--warn)}
.log.error{color:var(--danger)}
#cam{max-width:100%;border-radius:6px;border:1px solid var(--border);display:none;margin-top:8px}
.toast{position:fixed;right:14px;bottom:14px;padding:8px 14px;border-radius:6px;color:#fff;opacity:0;transition:opacity .2s;pointer-events:none;max-width:70vw}
.toast.show{opacity:1}
.toast.ok{background:var(--ok)}
.toast.bad{background:var(--danger)}
</style>
</head>
<body>
<header>
  <h1>VRM-HRI</h1>
  <span id="conn" class="badge bad">disconnected</span>
  <span id="simstate" class="badge">-</span>
  <span id="estopstate" class="badge bad hidden">E-STOP ENGAGED</span>
  <span style="flex:1"></span>
  <button id="clear-estop">Clear E-STOP</button>
  <button id="estop" class="danger big">E-STOP (Esc)</button>
</header>
<main>
  <section>
    <h2>Joints (deg)</h2>
    <div id="joints"></div>
    <div class="btns"><button id="send-joints" class="primary">Send joint targets</button></div>
  </section>

  <section>
    <h2>End-effector</h2>
    <div class="kv"><span>Position [m]</span><b id="ee-pos">-</b></div>
    <div class="kv"><span>RPY [deg]</span><b id="ee-rpy">-</b></div>
    <div class="form">
      <input id="mx" type="number" step="0.01" placeholder="x [m]">
      <input id="my" type="number" step="0.01" placeholder="y [m]">
      <input id="mz" type="number" step="0.01" placeholder="z [m]">
      <input id="mr" type="number" step="1" placeholder="roll [deg]">
      <input id="mp" type="number" step="1" placeholder="pitch [deg]">
      <input id="myw" type="number" step="1" placeholder="yaw [deg]">
    </div>
    <div class="btns"><button id="move-btn" class="primary">Move to pose</button></div>
    <div class="kv"><span>Orientation</span><b>leave roll/pitch/yaw empty to keep the current one</b></div>
  </section>

  <section>
    <h2>Hand</h2>
    <div id="grasps" class="btns"></div>
    <div id="fingers"></div>
  </section>

  <section>
    <h2>Control</h2>
    <div class="btns">
      <button data-cmd="home">Home</button>
      <button data-cmd="reset">Reset</button>
      <button data-cmd="pause">Pause</button>
      <button data-cmd="resume">Resume</button>
      <button data-cmd="stop">Stop</button>
    </div>
    <div class="kv"><span>Queued commands</span><b id="queue">0</b></div>
    <div class="kv"><span>Connected clients</span><b id="clients">0</b></div>
  </section>

  <section>
    <h2>Recording</h2>
    <div class="btns">
      <button data-cmd="start_recording">Start</button>
      <button data-cmd="stop_recording">Stop</button>
      <button data-cmd="clear_recording">Clear</button>
    </div>
    <div class="inline">
      <button id="playback-btn">Playback</button>
      <label>speed <input id="speed" type="number" min="0.1" max="10" step="0.1" value="1"></label>
    </div>
    <div class="kv"><span>Recording</span><b id="rec-state">-</b></div>
    <div class="kv"><span>Frames</span><b id="rec-frames">0</b></div>
  </section>

  <section>
    <h2>Gesture control</h2>
    <label class="inline"><input id="gesture-toggle" type="checkbox"> Enable gesture control</label>
    <div class="kv"><span>Hand detected</span><b id="g-hand">-</b></div>
    <div class="kv"><span>Gesture</span><b id="g-gesture">-</b></div>
    <div class="kv"><span>Action</span><b id="g-action">-</b></div>
    <div class="kv"><span>Vision FPS</span><b id="g-fps">-</b></div>
    <img id="cam" alt="camera preview">
  </section>

  <section class="wide">
    <h2>Event log</h2>
    <div id="log"></div>
  </section>
</main>
<div id="toast" class="toast"></div>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
let ws = null, config = null, nextId = 1, retry = 500;
let lastTouch = 0, camOn = false;
let gesturePending = null;
const jointInputs = [], jointTarget = [], jointActual = [], fingerInputs = {}, fingerVal = {};

function fmt(v, d) { return (typeof v === "number" && isFinite(v)) ? v.toFixed(d) : "-"; }
function setConn(on) {
  const s = $("conn");
  s.textContent = on ? "connected" : "disconnected";
  s.className = "badge " + (on ? "ok" : "bad");
}
function toast(msg, ok) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast show " + (ok ? "ok" : "bad");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.className = "toast"; }, 3500);
}
function send(type, params) {
  if (!ws || ws.readyState !== 1) { toast("Not connected", false); return; }
  ws.send(JSON.stringify(Object.assign({ type: type, id: String(nextId++) }, params || {})));
}
function touch() { lastTouch = Date.now(); }
function idle() { return Date.now() - lastTouch > 1500; }

function buildUi(cfg) {
  const joints = $("joints");
  joints.replaceChildren();
  jointInputs.length = 0; jointTarget.length = 0; jointActual.length = 0;
  cfg.joints.forEach((j, i) => {
    const row = el("div", "row");
    row.appendChild(el("span", "", j.name));
    const input = document.createElement("input");
    input.type = "range"; input.min = j.min_deg; input.max = j.max_deg; input.step = 0.5; input.value = 0;
    const val = el("span", "val", "");
    const target = el("span", "", "0.0");
    const actual = el("span", "", "-");
    val.append(target, document.createTextNode(" / "), actual);
    input.addEventListener("input", () => { target.textContent = fmt(parseFloat(input.value), 1); touch(); });
    input.addEventListener("change", () => { touch(); sendJoints(); });
    row.append(input, val);
    joints.appendChild(row);
    jointInputs.push(input); jointTarget.push(target); jointActual.push(actual);
  });
  const grasps = $("grasps");
  grasps.replaceChildren();
  cfg.grasp_presets.forEach((name) => {
    const b = el("button", "", name);
    b.addEventListener("click", () => send("grasp", { preset: name }));
    grasps.appendChild(b);
  });
  const fingers = $("fingers");
  fingers.replaceChildren();
  cfg.fingers.forEach((name) => {
    const row = el("div", "row");
    row.appendChild(el("span", "", name));
    const input = document.createElement("input");
    input.type = "range"; input.min = 0; input.max = 1; input.step = 0.05; input.value = 0;
    const val = el("span", "val", "0.00");
    input.addEventListener("input", () => { val.textContent = fmt(parseFloat(input.value), 2); touch(); });
    input.addEventListener("change", () => { touch(); send("finger", { finger: name, flexion: parseFloat(input.value) }); });
    row.append(input, val);
    fingers.appendChild(row);
    fingerInputs[name] = input; fingerVal[name] = val;
  });
}

function sendJoints() { send("set_joints", { q_deg: jointInputs.map((i) => parseFloat(i.value)) }); }

function render(s) {
  const sim = $("simstate");
  sim.textContent = s.sim_state || "-";
  $("estopstate").classList.toggle("hidden", !s.estop);
  $("estopstate").textContent = "E-STOP ENGAGED" + (s.estop_reason ? " (" + s.estop_reason + ")" : "");
  if (Array.isArray(s.q_deg)) {
    s.q_deg.forEach((v, i) => {
      if (jointActual[i]) jointActual[i].textContent = fmt(v, 1);
      if (jointInputs[i] && idle()) { jointInputs[i].value = v; jointTarget[i].textContent = fmt(v, 1); }
    });
  }
  if (s.ee) {
    $("ee-pos").textContent = (s.ee.position || []).map((v) => fmt(v, 3)).join(", ") || "-";
    $("ee-rpy").textContent = (s.ee.rpy_deg || []).map((v) => fmt(v, 1)).join(", ") || "-";
  }
  if (s.hand) {
    Object.keys(s.hand).forEach((name) => {
      if (fingerInputs[name] && idle()) { fingerInputs[name].value = s.hand[name]; fingerVal[name].textContent = fmt(s.hand[name], 2); }
    });
  }
  if (s.recording) {
    $("rec-state").textContent = s.recording.recording ? "recording" : "idle";
    $("rec-frames").textContent = String(s.recording.frames || 0);
  }
  if (s.vision) {
    const want = (gesturePending && Date.now() < gesturePending.until) ? gesturePending.value : null;
    if (want === null || s.vision.enabled === want) {
        gesturePending = null;
        $("gesture-toggle").checked = !!s.vision.enabled;
    }
    $("g-hand").textContent = s.vision.hand_detected ? "yes" : "no";
    $("g-gesture").textContent = s.vision.gesture || "-";
    $("g-action").textContent = s.vision.action || "-";
    $("g-fps").textContent = fmt(s.vision.fps, 1);
  }
  $("queue").textContent = String(s.queue_depth || 0);
  $("clients").textContent = String(s.clients || 0);
  camOn = !!s.camera;
  if (!camOn) $("cam").style.display = "none";
}

function appendLog(entries) {
  const box = $("log");
  entries.forEach((e) => {
    const line = el("div", "log " + e.level, new Date(e.t * 1000).toLocaleTimeString() + "  " + e.message);
    box.prepend(line);
  });
  while (box.children.length > 200) box.removeChild(box.lastChild);
}

function handle(msg) {
  if (msg.type === "config") { config = msg.config; buildUi(config); }
  else if (msg.type === "state") { render(msg.state); }
  else if (msg.type === "log") { appendLog(msg.entries || []); }
  else if (msg.type === "ack") {
    if (!msg.ok) toast(msg.message || "Command rejected", false);
    else if (msg.warnings && msg.warnings.length) toast(msg.warnings[0], true);
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(proto + "://" + location.host + "/ws");
  ws.onopen = () => { setConn(true); retry = 500; };
  ws.onclose = () => { setConn(false); setTimeout(connect, retry); retry = Math.min(retry * 2, 5000); };
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }
    handle(msg);
  };
}

$("estop").addEventListener("click", () => send("estop"));
$("clear-estop").addEventListener("click", () => send("clear_estop"));
$("send-joints").addEventListener("click", sendJoints);
$("playback-btn").addEventListener("click", () => send("playback", { speed: parseFloat($("speed").value) || 1 }));
$("gesture-toggle").addEventListener("change", (e) => {
  gesturePending = { value: e.target.checked, until: Date.now() + 8000 };
  send("gesture_control", { enabled: e.target.checked });
});
document.querySelectorAll("[data-cmd]").forEach((b) => b.addEventListener("click", () => send(b.dataset.cmd)));
$("move-btn").addEventListener("click", () => {
  const num = (id) => parseFloat($(id).value);
  const pos = [num("mx"), num("my"), num("mz")];
  if (pos.some((v) => !isFinite(v))) { toast("Enter x, y and z", false); return; }
  const rpy = [num("mr"), num("mp"), num("myw")];
  const params = { position: pos };
  if (rpy.every((v) => isFinite(v))) params.rpy_deg = rpy;
  send("move_to", params);
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") send("estop"); });
setInterval(() => {
  if (!camOn) return;
  const cam = $("cam");
  cam.onload = () => { cam.style.display = "block"; };
  cam.onerror = () => { cam.style.display = "none"; };
  cam.src = "/api/camera.jpg?t=" + Date.now();
}, 150);
connect();
</script>
</body>
</html>
"""


# ----------------------------------------------------------------------------
# 6. Server
# ----------------------------------------------------------------------------

def origin_allowed(origin: Optional[str], allowed_hosts=LOCAL_HOSTS) -> bool:
    """True if there is no Origin header (non-browser client) or it points to an allowed host."""
    if not origin:
        return True
    if origin == "null":
        return False
    try:
        host = urlparse(origin).hostname
    except ValueError:
        return False
    return host is not None and host.lower() in {h.lower() for h in allowed_hosts}


def create_app(hub: RobotHub, push_hz: float = 10.0, allowed_hosts=LOCAL_HOSTS):
    """Build the FastAPI application for a RobotHub."""
    if push_hz <= 0:
        raise ValueError("push_hz must be positive")
    if FastAPI is None:
        raise ImportError('fastapi is not installed (pip install fastapi "uvicorn[standard]")')

    period = 1.0 / push_hz
    app = FastAPI(title="VRM-HRI dashboard", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(DASHBOARD_HTML)

    @app.get("/api/health")
    async def health():
        return {"ok": True}

    @app.get("/api/config")
    async def get_config():
        return hub.config()

    @app.get("/api/state")
    async def get_state():
        return hub.snapshot()

    @app.get("/api/camera.jpg")
    async def camera():
        frame = hub.latest_frame()
        if frame is None:
            return Response(status_code=404)
        return Response(frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/command")
    async def post_command(request: Request):
        if not origin_allowed(request.headers.get("origin"), allowed_hosts):
            return JSONResponse({"ok": False, "message": "Origin not allowed"}, status_code=403)
        if not request.headers.get("content-type", "").lower().startswith("application/json"):
            return JSONResponse({"ok": False, "message": "Content-Type must be application/json"},
                                status_code=415)
        try:
            data = await request.json()
        except Exception:  # noqa: BLE001 - any parse failure is a client error
            return JSONResponse({"ok": False, "status": "invalid", "message": "Invalid JSON"},
                                status_code=400)
        result = hub.submit(data)
        code = 200 if result.ok else (400 if result.status == "invalid" else 409)
        return JSONResponse(result.to_dict(), status_code=code)

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        if not origin_allowed(websocket.headers.get("origin"), allowed_hosts):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        hub.client_connected()
        send_lock = asyncio.Lock()

        async def send(obj: dict) -> None:
            async with send_lock:
                await websocket.send_json(obj)

        async def pusher() -> None:
            try:
                entries, seq = hub.log_since(0)
                await send({"type": "log", "entries": entries})
                while True:
                    await send({"type": "state", "state": hub.snapshot()})
                    entries, seq = hub.log_since(seq)
                    if entries:
                        await send({"type": "log", "entries": entries})
                    await asyncio.sleep(period)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - socket closed while pushing
                return

        task = None
        try:
            await send({"type": "config", "config": hub.config()})
            task = asyncio.create_task(pusher())
            while True:
                try:
                    data = await websocket.receive_json()
                except WebSocketDisconnect:
                    raise
                except ValueError:
                    await send({"type": "ack", "id": None, "ok": False, "status": "invalid",
                                "message": "Invalid JSON", "warnings": []})
                    continue
                result = hub.submit(data)
                await send({"type": "ack", **result.to_dict()})
        except WebSocketDisconnect:
            pass
        finally:
            hub.client_disconnected()
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    return app


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


class GuiServer:
    """Runs the dashboard (uvicorn) in a background daemon thread."""

    def __init__(
        self,
        hub: RobotHub,
        host: str = "127.0.0.1",
        port: int = 8000,
        push_hz: float = 10.0,
        log_level: str = "warning",
        allowed_hosts=LOCAL_HOSTS,
    ) -> None:
        self.hub = hub
        self.host = host
        self._port = int(port)
        self.log_level = log_level
        self.app = create_app(hub, push_hz, allowed_hosts)
        self._server = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        shown = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{shown}:{self._port}"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, wait: bool = True, timeout: float = 5.0) -> None:
        if self.running:
            return
        try:
            import uvicorn
        except ImportError as exc:
            raise ImportError('uvicorn is not installed (pip install "uvicorn[standard]")') from exc
        if self._port == 0:
            self._port = _free_port(self.host)
        config = uvicorn.Config(self.app, host=self.host, port=self._port,
                                log_level=self.log_level, lifespan="off")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="gui-server", daemon=True)
        self._thread.start()
        if wait:
            deadline = time.monotonic() + timeout
            while not self._server.started and self._thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.02)
            if not self._server.started:
                self.stop()
                raise RuntimeError(f"GUI server did not start on port {self._port} (port in use?)")

    def stop(self, timeout: float = 5.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def open_browser(self) -> bool:
        return webbrowser.open(self.url)


# ----------------------------------------------------------------------------
# Demo: the dashboard with a tiny fake robot
# ----------------------------------------------------------------------------

def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="VRM-HRI dashboard demo (fake robot)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    hub = RobotHub(RobotModel())
    server = GuiServer(hub, host=args.host, port=args.port)
    server.start()
    print(f"Dashboard: {server.url}  (Ctrl+C to quit)")
    if not args.no_browser:
        server.open_browser()

    model = hub.model
    q = np.zeros(model.n)
    target = q.copy()
    hand = {name: 0.0 for name in FINGER_NAMES}
    try:
        while True:
            for cmd in hub.poll_commands():
                if cmd.type == "set_joints":
                    target = np.deg2rad(cmd.params["q_deg"])
                elif cmd.type in ("home", "reset"):
                    target = np.zeros(model.n)
                elif cmd.type == "grasp":
                    spec = GRASP_PRESETS[cmd.params["preset"]]
                    hand = {name: spec.get(name, 0.0) for name in FINGER_NAMES}
                elif cmd.type == "finger":
                    hand[cmd.params["finger"]] = cmd.params["flexion"]
                elif cmd.type in ("stop", "estop"):
                    target = q.copy()
                hub.log("info", f"Demo executed: {cmd.type}")
            if not hub.estopped:
                q = q + 0.15 * (target - q)
            T = model.fkine(q)
            hub.publish_state(
                q_deg=np.rad2deg(q),
                ee={"position": np.asarray(T.t), "rpy_deg": np.asarray(T.rpy(unit="deg"))},
                hand=hand,
                sim_state="demo",
            )
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.stop()


if __name__ == "__main__":
    main()