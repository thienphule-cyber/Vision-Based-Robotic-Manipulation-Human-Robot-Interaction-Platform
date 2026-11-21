"""
main.py - Application entry point for VRM-HRI.

    python main.py                 # Swift window + dashboard (http://127.0.0.1:8000)
    python main.py --vision        # also start the webcam / gesture pipeline at launch

Contents
--------
1. AppConfig / parse_args : configuration and command line.
2. pose_from_params       : pose target from GUI parameters.
3. MotionController       : turns GUI / gesture commands into validated robot motion.
4. Application            : wiring (Swift, safety, vision, dashboard) and the supervisor loop.

Threads
-------
* main thread   : supervisor loop (Application.step): polls the GUI command queue, handles
                urgent commands immediately (stop, pause, resume, e-stop, clear e-stop,
                gesture on/off) and publishes state to the dashboard.
* motion-worker : the ONLY thread that renders in the simulation. It runs queued jobs one at a
                time (trajectories, hand commands, recording, ...) and, when idle, one
                gesture servo step per new camera result.
* vision thread : camera + MediaPipe (computer_vision.VisionPipeline), started on demand.
* gui-server    : uvicorn / FastAPI (gui.GuiServer).

Safety layering (see safety.py)
-------------------------------
* Planned motion  : kinematics.plan_joint_path -> time-parametrise -> safety.check_trajectory
                    -> executed in short chunks. Before every chunk the worker re-checks the
                    e-stop latch and the cancel counter, so stop / e-stop take effect within
                    one chunk even though Simulation.execute_trajectory blocks.
* Gesture motion  : kinematics.servo_to_pose -> safety.limit_step -> simulation.
* Hand commands   : safety.check_hand_command -> simulation.
* Playback replays states that were validated when they were recorded; it is blocked while
the e-stop is engaged and aborted by stop / e-stop.
* Gesture control stays enabled until the user switches it off. stop, e-stop, reset and the
PEACE gesture only stop the current motion; while the e-stop is engaged nothing moves, and
following resumes when the e-stop is cleared / a follow gesture is shown again.
* "home" may end in the all-zero pose, which is a wrist singularity of the nominal DH table.
Moving INTO a singularity is normally blocked by safety.py; for the home command only,
SINGULARITY violations are tolerated (every other violation still blocks the motion).
"""
from __future__ import annotations

import argparse
import collections
import logging
import signal
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Sequence

import numpy as np
from spatialmath import SE3

from computer_vision import Action, TrackerConfig, VisionPipeline, target_pose
from gui import (
    Command,
    GuiServer,
    RobotHub,
    build_sim_state,
    build_vision_state,
    encode_jpeg,
)
from kinematics import ik, path_to_trajectory, plan_joint_path, servo_to_pose
from modelling import SystemModel
from safety import EE_ROT_SPEED, EE_SPEED, JOINT_SPEED, SINGULARITY, SafetyChecker

logger = logging.getLogger(__name__)

SPEED_CODES = frozenset({JOINT_SPEED, EE_SPEED, EE_ROT_SPEED})


# ----------------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------------

@dataclass
class AppConfig:
    # Motion
    dt: float = 0.02                  # trajectory time step [s]
    joint_speed: float = 0.6          # peak joint speed of planned trajectories [rad/s]
    speed_retries: int = 3            # halve the speed up to this many times if safety objects
    chunk_size: int = 25              # trajectory rows executed between two safety re-checks
    plan_margin: float = 0.06         # obstacle inflation used by the RRT planner [m]
    plan_iters: int = 2000
    ik_restarts: int = 10
    seed: int = 0
    # Gesture control
    gesture_dt: float = 1.0 / 30.0    # servo time step for the first frame after enabling [s]
    # Supervisor loop
    state_hz: float = 10.0
    poll_hz: float = 50.0
    # Dashboard
    host: str = "127.0.0.1"
    port: int = 8000
    open_browser: bool = True
    # Simulation
    use_visual: bool = True
    show_hand: bool = True
    show_axes: bool = True
    headless: bool = False
    browser: Optional[str] = None
    # Vision
    vision: bool = False              # start the camera at launch (otherwise on demand)
    camera: int = 0
    model_path: Optional[str] = None  # hand_landmarker.task (MediaPipe Tasks API)
    verbose: bool = False

    def __post_init__(self) -> None:
        for name in ("dt", "joint_speed", "gesture_dt", "state_hz", "poll_hz"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if self.chunk_size < 1 or self.plan_iters < 1:
            raise ValueError("chunk_size and plan_iters must be >= 1")
        if self.speed_retries < 0 or self.ik_restarts < 0:
            raise ValueError("speed_retries and ik_restarts must be >= 0")
        if self.plan_margin < 0:
            raise ValueError("plan_margin must be non-negative")
        if not 0 <= self.port <= 65535:
            raise ValueError("port must be in [0, 65535]")


def parse_args(argv: Optional[Sequence[str]] = None) -> AppConfig:
    parser = argparse.ArgumentParser(description="VRM-HRI: vision-based manipulation platform")
    parser.add_argument("--host", default="127.0.0.1", help="dashboard host")
    parser.add_argument("--port", type=int, default=8000, help="dashboard port")
    parser.add_argument("--no-browser", action="store_true", help="do not open the dashboard")
    parser.add_argument("--no-visual", action="store_true", help="show the DH model instead of the visual model")
    parser.add_argument("--no-hand", action="store_true", help="hide the robotic hand")
    parser.add_argument("--no-axes", action="store_true", help="hide coordinate frames")
    parser.add_argument("--headless", action="store_true", help="launch Swift without a browser")
    parser.add_argument("--browser", default=None, help="browser used by Swift")
    parser.add_argument("--vision", action="store_true", help="start the camera at launch")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--model-path", default=None, help="hand_landmarker.task (Tasks API)")
    parser.add_argument("-v", "--verbose", action="store_true")
    a = parser.parse_args(argv)
    return AppConfig(
        host=a.host, port=a.port, open_browser=not a.no_browser,
        use_visual=not a.no_visual, show_hand=not a.no_hand, show_axes=not a.no_axes,
        headless=a.headless, browser=a.browser,
        vision=a.vision, camera=a.camera, model_path=a.model_path, verbose=a.verbose,
    )


# ----------------------------------------------------------------------------
# 2. Helpers
# ----------------------------------------------------------------------------

def pose_from_params(position, rpy_deg, current: SE3) -> SE3:
    """Pose target from GUI parameters. rpy_deg = [roll, pitch, yaw] with R = Rz Ry Rx;
    None keeps the orientation of `current`."""
    p = np.asarray(position, dtype=float).reshape(3)
    if rpy_deg is None:
        R = np.asarray(current.R, dtype=float)
    else:
        roll, pitch, yaw = (float(v) for v in rpy_deg)
        R = np.asarray(
            (SE3.Rz(yaw, unit="deg") * SE3.Ry(pitch, unit="deg") * SE3.Rx(roll, unit="deg")).R,
            dtype=float,
        )
    return SE3.Rt(R, p)


def find_free_port(host: str, preferred: int, attempts: int = 50) -> int:
    """First bindable port in [preferred, preferred + attempts).

    Falls back to a port chosen by the OS if all of them are busy (or if preferred is 0).
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    if preferred > 0:
        for port in range(preferred, min(preferred + attempts, 65536)):
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind((host, port))
                except OSError:
                    continue
                return port
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


@dataclass
class _Job:
    cmd: Command
    gen: int


# ----------------------------------------------------------------------------
# 3. Motion controller
# ----------------------------------------------------------------------------

class MotionController:
    """Executes GUI / gesture commands on the simulation, always through safety.py.

    `process()` is called by the supervisor thread; the worker thread (start_worker) runs the
    queued jobs. Tests can skip the thread and call run_pending() / gesture_tick() directly.
    """

    IMMEDIATE = frozenset({"estop", "stop", "pause", "resume", "clear_estop", "gesture_control"})
    MOTION = frozenset({"set_joints", "move_to", "home", "reset", "grasp", "finger", "playback"})

    def __init__(
        self,
        system: SystemModel,
        sim,
        safety: SafetyChecker,
        hub: RobotHub,
        config: Optional[AppConfig] = None,
        vision=None,
        vision_factory: Optional[Callable[[], object]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.system = system
        self.model = system.robot
        self.sim = sim
        self.safety = safety
        self.hub = hub
        self.config = AppConfig() if config is None else config
        self.vision = vision
        self._vision_factory = vision_factory
        self._lazy_vision = False
        self._clock = clock

        self._lock = threading.RLock()
        self._jobs: Deque[_Job] = collections.deque()
        self._current: Optional[_Job] = None
        self._gen = 0
        self._wake = threading.Event()
        self._worker_stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

        self.gesture_enabled = False
        self._orientation: Optional[SE3] = None
        self._last_frame_id: Optional[int] = None
        self._last_ts: Optional[float] = None
        self._last_grasp: Optional[str] = None
        self._last_flex: Optional[dict] = None
        self._paused = False
        self._throttle: dict = {}

        self._handlers = {
            "set_joints": self._do_set_joints,
            "move_to": self._do_move_to,
            "home": self._do_home,
            "reset": self._do_reset,
            "grasp": self._do_grasp,
            "finger": self._do_finger,
            "playback": self._do_playback,
            "start_recording": self._do_start_recording,
            "stop_recording": self._do_stop_recording,
            "clear_recording": self._do_clear_recording,
        }

    # ---- status ----------------------------------------------------------------
    @property
    def busy(self) -> bool:
        with self._lock:
            return self._current is not None or len(self._jobs) > 0

    @property
    def worker_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _log(self, level: str, message: str) -> None:
        logger.log({"info": logging.INFO, "warning": logging.WARNING}.get(level, logging.ERROR), message)
        self.hub.log(level, message)

    def _log_throttled(self, key: str, level: str, message: str, interval: float = 2.0) -> None:
        now = self._clock()
        if now - self._throttle.get(key, -1e9) >= interval:
            self._throttle[key] = now
            self._log(level, message)

    # ---- command intake (supervisor thread) -----------------------------------------
    def process(self, commands: Sequence[Command]) -> None:
        for cmd in commands:
            try:
                self.dispatch(cmd)
            except Exception as exc:  # noqa: BLE001 - never let one command kill the loop
                logger.exception("Command %s failed", cmd.type)
                self._log("error", f"{cmd.type} failed: {exc}")

    def dispatch(self, cmd: Command) -> None:
        if cmd.type in self.IMMEDIATE:
            self._immediate(cmd)
            return
        if cmd.type in self.MOTION and self.safety.estopped:
            self._log("warning", f"{cmd.type} ignored: emergency stop is engaged")
            return
        if cmd.type == "reset":
            self._cancel("reset")
            self.sim.stop()
        self._enqueue(cmd)

    def _enqueue(self, cmd: Command) -> None:
        with self._lock:
            self._jobs.append(_Job(cmd, self._gen))
        self._wake.set()

    def _cancel(self, reason: str) -> None:
        with self._lock:
            self._gen += 1
            dropped = len(self._jobs)
            self._jobs.clear()
        if dropped:
            self._log("info", f"Cancelled {dropped} pending command(s) ({reason})")

    def _cancelled(self, gen: int) -> bool:
        return gen != self._gen or self.safety.estopped

    def _immediate(self, cmd: Command) -> None:
        t = cmd.type
        if t == "estop":
            if not self.safety.estopped:
                self.safety.estop("emergency stop")
            self._cancel("emergency stop")
            self.sim.stop()
            self._paused = False
            self._log("error", "Emergency stop: motion aborted")
        elif t == "stop":
            self._cancel("stop")
            self.sim.stop()
            self._paused = False
            self._log("info", "Stop: motion aborted")
        elif t == "pause":
            self.sim.pause()
            self._paused = True
            self._log("info", "Paused")
        elif t == "resume":
            if self.safety.estopped:
                self._log("warning", "resume ignored: emergency stop is engaged")
            else:
                self.sim.resume()
                self._paused = False
                self._log("info", "Resumed")
        elif t == "clear_estop":
            self.safety.clear_estop()
            self._log("info", "Emergency stop cleared")
        elif t == "gesture_control":
            self._set_gesture(bool(cmd.params["enabled"]), "requested")

    # ---- gesture control ----------------------------------------------------------------
    def _ensure_vision(self) -> bool:
        if self.vision is not None:
            return True
        if self._vision_factory is None:
            self._log("error", "Gesture control is not available: vision is not configured")
            return False
        try:
            vision = self._vision_factory()
            vision.start()
        except Exception as exc:  # noqa: BLE001 - no camera, no mediapipe, no permission, ...
            self._log("error", f"Gesture control is not available: {exc}")
            return False
        self.vision = vision
        self._lazy_vision = True
        return True

    def release_vision(self) -> None:
        vision, self.vision, self._lazy_vision = self.vision, None, False
        if vision is not None and hasattr(vision, "close"):
            try:
                vision.close()
            except Exception:  # noqa: BLE001
                logger.exception("Closing the vision pipeline failed")

    def _set_gesture(self, enabled: bool, reason: str = "") -> None:
        if enabled:
            if self.safety.estopped:
                self._log("warning", "Cannot enable gesture control: emergency stop is engaged")
                return
            if not self._ensure_vision():
                return
            self._orientation = self.model.fkine(self.sim.q)  # keep the current tool orientation
            latest = self.vision.latest()
            self._last_frame_id = None if latest is None else latest.frame_id
            self._last_ts = None
            self._last_grasp = None
            self._last_flex = None
            if not self.gesture_enabled:
                self._log("info", "Gesture control enabled")
            self.gesture_enabled = True
        else:
            was = self.gesture_enabled
            self.gesture_enabled = False
            if self._lazy_vision:
                self.release_vision()
            if was:
                self._log("info", f"Gesture control disabled ({reason})" if reason else "Gesture control disabled")

    def _mirror_fingers(self, cmd) -> None:
        """Copy the user's per-finger flexion onto the simulated hand (changes > 0.02 only)."""
        flex = cmd.finger_flexions
        if not cmd.hand_detected or not flex:
            return
        if not self.safety.check_hand_command(flexions=flex).ok:
            return
        last = self._last_flex
        if last is not None and all(abs(flex[n] - last.get(n, 0.0)) < 0.02 for n in flex):
            return
        names = list(flex)
        for i, name in enumerate(names):
            self.sim.set_finger_flexion(name, flex[name], render=(i == len(names) - 1))
        self._last_flex = dict(flex)
    
    def gesture_tick(self) -> bool:
        """One gesture servo step if there is a new camera result. Returns True if it consumed
        a result (or changed state), False if there was nothing to do."""
        vision = self.vision
        if not self.gesture_enabled or vision is None:
            return False
        if self.safety.estopped:
            return False  # nothing moves during the e-stop; the user's setting is kept
        if self._paused:
            return False
        out = vision.latest()
        if out is None or out.frame_id == self._last_frame_id:
            return False
        self._last_frame_id = out.frame_id

        cfg = self.config
        ts = float(out.timestamp)
        dt = cfg.gesture_dt if self._last_ts is None else min(max(ts - self._last_ts, 0.005), 0.2)
        self._last_ts = ts

        cmd = out.command
        if cmd.hand_detected:
            self._mirror_fingers(cmd)
        if cmd.stop_requested:
            self._log("info", "Stop gesture detected: motion stopped, gesture control stays on")
            self._cancel("stop gesture")
            self.sim.stop()
            return True
        if cmd.action is not Action.FOLLOW or cmd.target_position is None:
            return True

        target = target_pose(cmd, self._orientation)
        if target is not None:
            q = self.sim.q
            q_next, _, _ = servo_to_pose(self.model, q, target, dt)
            q_safe, report = self.safety.limit_step(q, q_next, dt)
            if not report.ok:
                self._log_throttled("gesture-step", "warning", f"Gesture step rejected: {report.summary()}")
            if q_safe is not None and not np.allclose(q_safe, q, atol=1e-12):
                self.sim.set_joint_positions(q_safe)
        return True

    # ---- job execution (worker thread, or tests) --------------------------------------------
    def run_pending(self, max_jobs: Optional[int] = None) -> int:
        """Run queued jobs one after another. Returns how many jobs were taken."""
        count = 0
        while max_jobs is None or count < max_jobs:
            with self._lock:
                if not self._jobs:
                    break
                job = self._jobs.popleft()
                self._current = job
            try:
                if job.gen != self._gen:
                    self._log("info", f"{job.cmd.type} dropped (cancelled)")
                else:
                    self._execute(job)
            finally:
                with self._lock:
                    self._current = None
            count += 1
        return count

    def _execute(self, job: _Job) -> None:
        handler = self._handlers.get(job.cmd.type)
        if handler is None:
            self._log("warning", f"Unsupported command: {job.cmd.type}")
            return
        try:
            handler(job.cmd.params, job.gen)
        except Exception as exc:  # noqa: BLE001 - keep the worker alive
            logger.exception("%s failed", job.cmd.type)
            self._log("error", f"{job.cmd.type} failed: {exc}")

    # ---- worker thread ---------------------------------------------------------------------------
    def start_worker(self) -> None:
        if self.worker_alive:
            return
        self._worker_stop.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="motion-worker", daemon=True)
        self._worker.start()

    def stop_worker(self, timeout: float = 5.0) -> None:
        self._worker_stop.set()
        self._cancel("shutdown")
        try:
            self.sim.stop()
        except Exception:  # noqa: BLE001
            logger.exception("sim.stop() failed during shutdown")
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)
            self._worker = None

    def _worker_loop(self) -> None:
        while not self._worker_stop.is_set():
            try:
                if self.run_pending(max_jobs=1):
                    continue
                if self.gesture_tick():
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.exception("Worker iteration failed")
                self._log("error", f"Worker error: {exc}")
            self._wake.wait(0.005)
            self._wake.clear()

    # ---- planned motion ------------------------------------------------------------------------------
    def _plan_validated(self, q0, q_goal, tolerate=frozenset()):
        """Plan q0 -> q_goal and validate it with safety.py. If only speed limits are violated
        the trajectory is re-timed at half speed (up to `speed_retries` times).
        Returns (trajectory, "") or (None, reason)."""
        cfg = self.config
        plan = plan_joint_path(
            self.model, self.system.environment, q0, q_goal,
            margin=cfg.plan_margin, max_iters=cfg.plan_iters, seed=cfg.seed,
        )
        if not plan.success:
            return None, plan.message
        v_max = cfg.joint_speed
        for _ in range(cfg.speed_retries + 1):
            traj = path_to_trajectory(plan.path, dt=cfg.dt, v_max=v_max)
            report = self.safety.check_trajectory(
                traj, cfg.dt, q_start=q0, stop_at_first=not tolerate
            )
            blocking = [v for v in report.violations if v.code not in tolerate]
            if not blocking:
                return traj, ""
            if all(v.code in SPEED_CODES for v in blocking):
                v_max *= 0.5
                continue
            return None, "; ".join(f"[{v.code}] {v.message}" for v in blocking[:3])
        return None, "speed limits could not be met"

    def _run_trajectory(self, traj: np.ndarray, gen: int) -> bool:
        """Execute in chunks; between chunks the e-stop latch and the cancel counter are checked."""
        step = self.config.chunk_size
        for i in range(0, len(traj), step):
            if self._cancelled(gen):
                return False
            if not self.sim.execute_trajectory(traj[i:i + step], dt=self.config.dt):
                return False
        return True

    def _move_joints(self, q_goal, gen: int, label: str, tolerate=frozenset()) -> bool:
        q_goal = np.asarray(q_goal, dtype=float)
        q0 = self.sim.q
        goal_report = self.safety.check_configuration(q_goal)
        if not goal_report.ok:
            self._log("warning", f"{label} rejected: {goal_report.summary()}")
            return False
        traj, message = self._plan_validated(q0, q_goal, tolerate)
        if traj is None:
            self._log("warning", f"{label} rejected: {message}")
            return False
        if self._cancelled(gen):
            self._log("info", f"{label} cancelled")
            return False
        ok = self._run_trajectory(traj, gen)
        self._log("info" if ok else "warning", f"{label} {'completed' if ok else 'interrupted'}")
        return ok

    # ---- command handlers ----------------------------------------------------------------------------------
    def _do_set_joints(self, p: dict, gen: int) -> None:
        self._move_joints(np.deg2rad(p["q_deg"]), gen, "set_joints")

    def _do_home(self, p: dict, gen: int) -> None:
        self._move_joints(self.model.q_home, gen, "home", tolerate=frozenset({SINGULARITY}))

    def _do_move_to(self, p: dict, gen: int) -> None:
        q0 = self.sim.q
        pose = pose_from_params(p["position"], p.get("rpy_deg"), self.model.fkine(q0))
        report = self.safety.check_pose_target(pose)
        if not report.ok:
            self._log("warning", f"move_to rejected: {report.summary()}")
            return
        res = ik(self.model, pose, q0, n_restarts=self.config.ik_restarts, seed=self.config.seed)
        if not res.success:
            self._log("warning",
                      f"move_to rejected: IK failed ({res.message}, position error {res.pos_error * 1000:.1f} mm)")
            return
        self._move_joints(res.q, gen, "move_to")

    def _do_reset(self, p: dict, gen: int) -> None:
        self.sim.reset()
        self._last_grasp = None
        self._last_flex = None
        self._log("info", "Simulation reset")

    def _do_grasp(self, p: dict, gen: int) -> None:
        report = self.safety.check_hand_command(grasp=p["preset"])
        if not report.ok:
            self._log("warning", f"grasp rejected: {report.summary()}")
            return
        self.sim.set_hand_grasp(p["preset"])
        self._last_grasp = p["preset"]
        self._last_flex = None
        self._log("info", f"Grasp: {p['preset']}")

    def _do_finger(self, p: dict, gen: int) -> None:
        report = self.safety.check_hand_command(flexions={p["finger"]: p["flexion"]})
        if not report.ok:
            self._log("warning", f"finger rejected: {report.summary()}")
            return
        self.sim.set_finger_flexion(p["finger"], p["flexion"])
        self._last_grasp = None
        self._last_flex = None

    def _do_playback(self, p: dict, gen: int) -> None:
        if not self.sim.frames:
            self._log("warning", "Nothing recorded to play back")
            return
        ok = self.sim.playback(speed=p["speed"])
        self._log("info" if ok else "warning", f"Playback {'finished' if ok else 'interrupted'}")

    def _do_start_recording(self, p: dict, gen: int) -> None:
        self.sim.start_recording()
        self._log("info", "Recording started")

    def _do_stop_recording(self, p: dict, gen: int) -> None:
        frames = self.sim.stop_recording()
        self._log("info", f"Recording stopped ({len(frames)} frames)")

    def _do_clear_recording(self, p: dict, gen: int) -> None:
        self.sim.clear_recording()
        self._log("info", "Recording cleared")


# ----------------------------------------------------------------------------
# 4. Application
# ----------------------------------------------------------------------------

def make_vision(config: AppConfig) -> VisionPipeline:
    """Camera + MediaPipe pipeline with an annotated preview for the dashboard."""
    return VisionPipeline.from_camera(
        config.camera, tracker_config=TrackerConfig(model_path=config.model_path), annotate=True
    )


class Application:
    """Owns the supervisor loop and the lifecycle of simulation, worker, vision and dashboard."""

    def __init__(
        self,
        config: AppConfig,
        sim,
        hub: RobotHub,
        controller: MotionController,
        server: Optional[GuiServer] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.sim = sim
        self.hub = hub
        self.controller = controller
        self.server = server
        self._clock = clock
        self._started = False
        self._closed = False
        self._stop_evt = threading.Event()
        self._last_pub: Optional[float] = None
        self._last_frame = None
        self._frames_ok = True

    @classmethod
    def from_config(cls, config: AppConfig) -> "Application":
        from simulation import Simulation  # imported lazily: pulls in Swift

        system = SystemModel.build_default(load_visual=config.use_visual)
        sim = Simulation(
            system,
            use_visual=config.use_visual,
            show_hand=config.show_hand,
            show_axes=config.show_axes,
            headless=config.headless,
            browser=config.browser,
        )
        safety = SafetyChecker(system.robot, system.environment)
        hub = RobotHub(system.robot, safety)

        def factory():
            return make_vision(config)

        vision = None
        if config.vision:
            try:
                vision = factory()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Vision unavailable: %s", exc)
                hub.log("warning", f"Vision unavailable: {exc}")
        controller = MotionController(system, sim, safety, hub, config, vision=vision,
                                    vision_factory=factory)
        port = find_free_port(config.host, config.port)
        if port != config.port:
            message = f"Port {config.port} is in use; the dashboard uses port {port}"
            logger.warning(message)
            hub.log("warning", message)
        server = GuiServer(hub, host=config.host, port=port)
        return cls(config, sim, hub, controller, server)

    # ---- lifecycle --------------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self.sim.launch()
        self.sim.load_scene()
        self.sim.print_status()
        vision = self.controller.vision
        if vision is not None:
            vision.start()
        self.controller.start_worker()
        if self.server is not None:
            self.server.start()
            print(f"Dashboard: {self.server.url}")
            if self.config.open_browser:
                self.server.open_browser()
        self.hub.log("info", "System ready")
        self._started = True

    def request_shutdown(self) -> None:
        self._stop_evt.set()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_evt.set()
        steps: List[Callable[[], None]] = [
            self.controller.stop_worker,
            (self.server.stop if self.server is not None else (lambda: None)),
            self.controller.release_vision,
            self.sim.close,
        ]
        for step in steps:
            try:
                step()
            except Exception:  # noqa: BLE001 - keep shutting down
                logger.exception("Shutdown step failed")

    # ---- supervisor loop ----------------------------------------------------------------
    def step(self) -> None:
        """One supervisor iteration: execute GUI commands, publish state when due."""
        self.controller.process(self.hub.poll_commands())
        now = self._clock()
        if self._last_pub is None or now - self._last_pub >= 1.0 / self.config.state_hz:
            self.publish_now()

    def publish_now(self) -> None:
        self._last_pub = self._clock()
        try:
            state = build_sim_state(self.sim)
            vision = self.controller.vision
            output = vision.latest() if vision is not None else None
            state["vision"] = build_vision_state(output, self.controller.gesture_enabled)
            state["busy"] = self.controller.busy
            self.hub.publish_state(state)
            self._publish_frame(vision)
        except Exception:  # noqa: BLE001 - publishing must never stop the loop
            logger.exception("Publishing state failed")

    def _publish_frame(self, vision) -> None:
        if vision is None or not self._frames_ok:
            return
        frame = vision.latest_frame()
        if frame is None or frame is self._last_frame:
            return
        self._last_frame = frame
        try:
            jpeg = encode_jpeg(frame)
        except Exception as exc:  # noqa: BLE001 - e.g. cv2 missing
            self._frames_ok = False
            self.hub.log("warning", f"Camera preview disabled: {exc}")
            return
        if jpeg:
            self.hub.publish_frame(jpeg)

    def run(self) -> None:
        period = 1.0 / self.config.poll_hz
        try:
            self.start()
            print("Running. Press Ctrl+C to quit.")
            while not self._stop_evt.is_set():
                t0 = self._clock()
                self.step()
                self._stop_evt.wait(max(0.0, period - (self._clock() - t0)))
        except KeyboardInterrupt:
            print("\nShutting down.")
        finally:
            self.shutdown()


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if config.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    app = Application.from_config(config)
    try:
        signal.signal(signal.SIGTERM, lambda *_: app.request_shutdown())
    except ValueError:  # not in the main thread
        pass
    app.run()


if __name__ == "__main__":
    main()