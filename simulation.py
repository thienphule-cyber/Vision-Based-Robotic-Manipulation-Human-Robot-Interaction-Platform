"""
simulation.py - Visual simulation (Swift) for VRM-HRI.

Running `python simulation.py` opens the Swift window and shows everything the
visual simulation contains, with NO decision logic (no planning, no gestures, no
pick-and-place logic):

  * Mitsubishi RV-2FR 3D robot (visual model from ir-support-extra-robots,
    falls back to the DH computational model if the visual model is missing)
  * Joint position updates
  * End-effector pose (coordinate frame in the scene + printed pose)
  * Five-finger robotic hand mounted on the end-effector
  * Environment: table, obstacles, target/storage zones, cubes, spheres, cylinders
  * Trajectory playback (plays a given joint trajectory; does NOT generate one)
  * Reset / pause / resume / stop
  * Recording and playback of the simulation

Other modules (kinematics, safety, gui, ...) are expected to call the public API
of the `Simulation` class. Safety validation belongs in safety.py, not here.

Usage:
    python simulation.py              # static scene, Ctrl+C to quit
    python simulation.py --demo       # joint sweep + hand grasps + record/playback
    python simulation.py --no-visual  # show the DH model instead of the visual model
"""
from __future__ import annotations

import argparse
import enum
import logging
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import spatialgeometry as sg
from spatialmath import SE3
from swift import Swift

from modelling import Finger, HandModel, SystemModel

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Constants and small helpers
# ----------------------------------------------------------------------------

SKIN = (0.90, 0.75, 0.62)
JOINT_COLOR = (0.55, 0.40, 0.30)
ZONE_ALPHA = 0.25

# Palm frame (modelling.py): x = along fingers, y = lateral, z = palm normal.
# Mapping the palm frame onto the end-effector (flange) frame so that the fingers
# point along the flange +z axis: x_palm = z_ee, y_palm = y_ee, z_palm = -x_ee.
HAND_MOUNT = SE3(np.array([
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0,  0.0, 0.0],
    [1.0, 0.0,  0.0, 0.0],
    [0.0, 0.0,  0.0, 1.0],
]))

PALM_SIZE = (0.09, 0.09, 0.02)
PALM_CENTER = (0.045, -0.005, 0.0)
FINGER_RADIUS = 0.006
JOINT_RADIUS = 0.008


class SimState(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


def finger_points(finger: Finger) -> List[np.ndarray]:
    """Joint positions of a finger in the palm frame (base, joints..., tip)."""
    p = np.asarray(finger.base_offset, dtype=float).copy()
    pts = [p.copy()]
    theta = 0.0
    for q_i, length in zip(finger.q, finger.link_lengths):
        theta += q_i
        p = p + np.array([length * np.cos(theta), 0.0, -length * np.sin(theta)])
        pts.append(p.copy())
    return pts


def segment_transform(p0: np.ndarray, p1: np.ndarray) -> SE3:
    """Pose of a z-axis aligned cylinder spanning p0 -> p1 (origin at the midpoint)."""
    d = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    length = np.linalg.norm(d)
    center = (np.asarray(p0, dtype=float) + np.asarray(p1, dtype=float)) / 2.0
    if length < 1e-12:
        return SE3(*center)
    z = d / length
    v = np.cross([0.0, 0.0, 1.0], z)
    c = z[2]
    if np.linalg.norm(v) < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center
    return SE3(T)


# ----------------------------------------------------------------------------
# Robotic hand visual
# ----------------------------------------------------------------------------

class HandVisual:
    """Draws a HandModel as a palm box, phalanx cylinders and joint spheres."""

    def __init__(self, hand: HandModel, env: Swift) -> None:
        self.hand = hand
        self.palm = sg.Cuboid(scale=list(PALM_SIZE), color=SKIN)
        env.add(self.palm)

        self.segments: Dict[str, List[sg.Cylinder]] = {}
        self.joints: Dict[str, List[sg.Sphere]] = {}
        for name, finger in hand.fingers.items():
            segs = []
            for length in finger.link_lengths:
                cyl = sg.Cylinder(radius=FINGER_RADIUS, length=float(length), color=SKIN)
                env.add(cyl)
                segs.append(cyl)
            spheres = []
            for _ in range(finger.n_joints + 1):
                sph = sg.Sphere(radius=JOINT_RADIUS, color=JOINT_COLOR)
                env.add(sph)
                spheres.append(sph)
            self.segments[name] = segs
            self.joints[name] = spheres

    def update(self, T_mount: SE3) -> None:
        """Move all hand parts. T_mount is the world pose of the palm frame."""
        T = T_mount.A
        self.palm.T = T_mount * SE3(*PALM_CENTER)
        for name, finger in self.hand.fingers.items():
            pts = [T[:3, :3] @ p + T[:3, 3] for p in finger_points(finger)]
            for i, seg in enumerate(self.segments[name]):
                seg.T = segment_transform(pts[i], pts[i + 1])
            for i, sph in enumerate(self.joints[name]):
                sph.T = SE3(*pts[i])


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------

class Simulation:
    """Swift-based visual simulation of the VRM-HRI workcell (no decision logic)."""

    def __init__(
        self,
        system: Optional[SystemModel] = None,
        use_visual: bool = True,
        show_hand: bool = True,
        show_axes: bool = True,
        headless: bool = False,
        browser: Optional[str] = None,
    ) -> None:
        self.system = system if system is not None else SystemModel.build_default(load_visual=use_visual)
        self.use_visual = use_visual
        self.show_hand = show_hand
        self.show_axes = show_axes
        self.headless = headless
        self.browser = browser

        self.env: Optional[Swift] = None
        self._display_robot = None
        self._hand_visual: Optional[HandVisual] = None
        self._ee_axes: Optional[sg.Axes] = None
        self._shapes: Dict[str, object] = {}
        self._initial_T: Dict[str, np.ndarray] = {}
        self._object_T: Dict[str, np.ndarray] = {}

        self._q = self.model.q_home.copy()

        # Run control
        self._state = SimState.IDLE
        self._busy = False
        self._stop_evt = threading.Event()
        self._run_evt = threading.Event()
        self._run_evt.set()

        # Recording
        self._recording = False
        self._playing = False
        self._rec_t0 = 0.0
        self._frames: List[dict] = []

    # ---- convenient accessors -------------------------------------------
    @property
    def model(self):
        return self.system.robot

    @property
    def hand(self) -> HandModel:
        return self.system.hand

    @property
    def environment(self):
        return self.system.environment

    @property
    def state(self) -> SimState:
        return self._state

    @property
    def q(self) -> np.ndarray:
        return self._q.copy()

    @property
    def frames(self) -> List[dict]:
        return self._frames

    # ---- lifecycle --------------------------------------------------------
    def launch(self) -> None:
        """Start Swift (opens the 3D view in the browser) and set the camera."""
        self.env = Swift()
        try:
            self.env.launch(realtime=False, headless=self.headless, browser=self.browser)
        except TypeError:  # older/newer Swift signature
            self.env.launch(realtime=False)
        try:
            self.env.set_camera_pose([1.3, -1.3, 0.9], [0.3, 0.0, 0.3])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not set camera pose: %s", exc)

    def load_scene(self) -> None:
        """Add robot, hand, end-effector frame and environment to Swift."""
        if self.env is None:
            raise RuntimeError("Call launch() before load_scene()")
        self._add_robot()
        self._add_environment()
        if self.show_axes:
            self.env.add(sg.Axes(0.25, base=SE3()))
            if not np.allclose(self.model.base.A, np.eye(4)):
                self.env.add(sg.Axes(0.15, base=self.model.base))
            self._ee_axes = sg.Axes(0.08, base=SE3())
            self.env.add(self._ee_axes)
        if self.show_hand:
            self._hand_visual = HandVisual(self.hand, self.env)
        self.set_joint_positions(self.model.q_home, render=False)
        self.render()

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Swift close failed: %s", exc)

    def hold(self) -> None:
        """Keep the process alive so the Swift window stays open (Ctrl+C to quit)."""
        while True:
            time.sleep(0.2)

    # ---- scene construction -----------------------------------------------
    def _add_robot(self) -> None:
        visual = self.system.visual if self.use_visual else None
        if visual is not None:
            try:
                try:
                    visual.base = self.model.base  # keep frames aligned with the model
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not set visual model base: %s", exc)
                if hasattr(visual, "add_to_env"):
                    visual.add_to_env(self.env)
                else:
                    self.env.add(visual)
                self._display_robot = visual
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Visual model failed to load into Swift (%s); using DH model.", exc)
        self.env.add(self.model.robot)
        self._display_robot = self.model.robot

    def _add_environment(self) -> None:
        for name in self.environment.names():
            obj = self.environment.get(name)
            alpha = ZONE_ALPHA if obj.kind in ("target_zone", "storage_zone") else 1.0
            color = (*obj.color, alpha)
            pose = SE3(*obj.position) * SE3.Rz(obj.yaw)
            if obj.shape == "box":
                shape = sg.Cuboid(scale=list(obj.dims), base=pose, color=color)
            elif obj.shape == "sphere":
                shape = sg.Sphere(radius=obj.dims[0], base=pose, color=color)
            elif obj.shape == "cylinder":
                # Cylinder axis is the local z axis
                shape = sg.Cylinder(radius=obj.dims[0], length=obj.dims[1], base=pose, color=color)
            else:  # custom: drawn as its bounding sphere
                shape = sg.Sphere(radius=obj.dims[0], base=pose, color=color)
            self.env.add(shape)
            self._shapes[name] = shape
            self._initial_T[name] = pose.A.copy()
            self._object_T[name] = pose.A.copy()

    # ---- state setters (visual only) -------------------------------------
    def set_joint_positions(self, q, clip: bool = True, render: bool = True) -> None:
        """Update the robot joints. `clip` only keeps the picture inside joint limits;
        real validation is the job of safety.py."""
        arr = np.asarray(q, dtype=float).reshape(-1)
        if arr.shape != (self.model.n,):
            raise ValueError(f"Expected {self.model.n} joint values, got {arr.shape}")
        if clip:
            arr = self.model.clip(arr)
        self._q = arr
        if self._display_robot is not None:
            self._display_robot.q = arr
        if render:
            self.render()

    def set_hand_grasp(self, preset: str, render: bool = True) -> None:
        self.hand.apply_grasp(preset)
        if render:
            self.render()

    def set_finger_flexion(self, name: str, fraction: float, render: bool = True) -> None:
        self.hand.set_finger_flexion(name, fraction)
        if render:
            self.render()

    def set_object_pose(self, name: str, pose: SE3, render: bool = True) -> None:
        if name not in self._shapes:
            raise KeyError(f"Unknown object '{name}'")
        self._shapes[name].T = pose
        self._object_T[name] = pose.A.copy()
        if render:
            self.render()

    # ---- end-effector pose --------------------------------------------------
    def ee_pose(self) -> SE3:
        """End-effector pose of the robot that is actually displayed (visual model if loaded)."""
        robot = self._display_robot if self._display_robot is not None else self.model.robot
        try:
            return robot.fkine(self._q)
        except Exception:  # fall back to the computational model
            return self.model.fkine(self._q)

    def print_ee_pose(self) -> None:
        T = self.ee_pose()
        print("End-effector position [m]    :", np.round(T.t, 4))
        print("End-effector RPY [deg]       :", np.round(T.rpy(unit="deg"), 2))

    # ---- rendering -----------------------------------------------------------
    def _sync_overlays(self) -> None:
        ee = self.ee_pose()
        if self._ee_axes is not None:
            self._ee_axes.T = ee
        if self._hand_visual is not None:
            self._hand_visual.update(ee * HAND_MOUNT)

    def render(self, dt: float = 0.02) -> None:
        """Push the current state to Swift (and capture a frame if recording)."""
        self._sync_overlays()
        if self._recording and not self._playing:
            self._capture()
        if self.env is not None:
            self.env.step(dt)

    # ---- run control ----------------------------------------------------------
    def pause(self) -> None:
        self._run_evt.clear()
        self._state = SimState.PAUSED

    def resume(self) -> None:
        self._run_evt.set()
        self._state = SimState.RUNNING if self._busy else SimState.IDLE

    def stop(self) -> None:
        """Abort a running trajectory or playback."""
        self._stop_evt.set()
        self._run_evt.set()
        self._state = SimState.STOPPED

    def reset(self) -> None:
        """Stop, then restore home pose, open hand and initial object poses."""
        self.stop()
        t0 = time.perf_counter()
        while self._busy and time.perf_counter() - t0 < 1.0:
            time.sleep(0.01)
        self._stop_evt.clear()
        self._run_evt.set()
        self.hand.open_hand()
        for name, T in self._initial_T.items():
            self._shapes[name].T = SE3(T)
            self._object_T[name] = T.copy()
        self.set_joint_positions(self.model.q_home, render=False)
        self._state = SimState.IDLE
        self.render()

    def _wait_if_paused(self) -> float:
        """Block while paused. Returns the paused time, or -1.0 if stop was requested."""
        t0 = time.perf_counter()
        while not self._run_evt.wait(0.05):
            if self._stop_evt.is_set():
                return -1.0
        if self._stop_evt.is_set():
            return -1.0
        return time.perf_counter() - t0

    @staticmethod
    def _sleep_until(deadline: float) -> None:
        delay = deadline - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    # ---- trajectory execution (plays a given trajectory, does not plan) -----
    def execute_trajectory(self, q_traj, dt: float = 0.02) -> bool:
        """Play an (N, n_joints) joint trajectory. Returns True if it ran to the end."""
        traj = np.asarray(q_traj, dtype=float)
        if traj.ndim != 2 or traj.shape[1] != self.model.n:
            raise ValueError(f"Trajectory must have shape (N, {self.model.n})")
        self._stop_evt.clear()
        self._busy = True
        self._state = SimState.RUNNING
        completed = True
        try:
            next_t = time.perf_counter()
            for q in traj:
                paused = self._wait_if_paused()
                if paused < 0:
                    completed = False
                    break
                next_t += paused
                self.set_joint_positions(q, render=False)
                self.render(dt)
                next_t += dt
                self._sleep_until(next_t)
        finally:
            self._busy = False
            if self._state is not SimState.STOPPED:
                self._state = SimState.IDLE
        return completed

    # ---- recording and playback ------------------------------------------------
    def start_recording(self, clear: bool = True) -> None:
        if clear:
            self._frames = []
        self._rec_t0 = time.perf_counter()
        self._recording = True
        self._capture()

    def stop_recording(self) -> List[dict]:
        self._recording = False
        return self._frames

    def clear_recording(self) -> None:
        self._frames = []

    def _capture(self) -> None:
        self._frames.append({
            "t": time.perf_counter() - self._rec_t0,
            "q": self._q.copy(),
            "hand": self.hand.q,
            "objects": {n: T.copy() for n, T in self._object_T.items()},
        })

    def _apply_frame(self, frame: dict) -> None:
        self.set_joint_positions(frame["q"], clip=False, render=False)
        for name, q in frame["hand"].items():
            self.hand.set_finger(name, q)
        for name, T in frame["objects"].items():
            if name in self._shapes:
                self._shapes[name].T = SE3(T)
                self._object_T[name] = T.copy()

    def playback(self, frames: Optional[List[dict]] = None, speed: float = 1.0) -> bool:
        """Replay recorded frames. Returns True if playback ran to the end."""
        data = self._frames if frames is None else frames
        if not data:
            logger.warning("Nothing to play back.")
            return False
        if speed <= 0:
            raise ValueError("speed must be positive")
        self._stop_evt.clear()
        self._busy = True
        self._playing = True
        self._state = SimState.RUNNING
        completed = True
        try:
            start = time.perf_counter()
            for frame in data:
                paused = self._wait_if_paused()
                if paused < 0:
                    completed = False
                    break
                start += paused
                self._sleep_until(start + frame["t"] / speed)
                self._apply_frame(frame)
                self.render(0.02)
        finally:
            self._playing = False
            self._busy = False
            if self._state is not SimState.STOPPED:
                self._state = SimState.IDLE
        return completed

    # ---- console summary ---------------------------------------------------------
    def print_status(self) -> None:
        print("=" * 60)
        print("VRM-HRI visual simulation")
        print("=" * 60)
        for key, value in self.system.summary().items():
            print(f"{key:>22}: {value}")
        print(f"{'displayed robot':>22}: "
            f"{'visual model' if self._display_robot is self.system.visual else 'DH computational model'}")
        if self.system.visual is not None:
            report = self.system.check_sync()
            print(f"{'visual/model sync':>22}: {'OK' if report.ok else 'MISMATCH'}")
            for msg in report.messages:
                print(f"{'':>22}  - {msg}")
        self.print_ee_pose()
        print("=" * 60)


# ----------------------------------------------------------------------------
# Demo (optional): shows that joints, hand, recording and playback all update
# ----------------------------------------------------------------------------

def run_demo(sim: Simulation) -> None:
    model = sim.model
    q0 = model.q_home
    q1 = model.clip(np.deg2rad([45, 20, 60, 30, 40, 90]))
    q2 = model.clip(np.deg2rad([-45, 40, 90, -30, 60, -90]))

    sim.start_recording()
    for a, b in ((q0, q1), (q1, q2), (q2, q0)):
        sim.execute_trajectory(np.linspace(a, b, 100), dt=0.02)
    for grasp in ("pinch", "precision", "power", "closed", "open"):
        sim.set_hand_grasp(grasp)
        time.sleep(0.6)
    sim.stop_recording()

    time.sleep(1.0)
    sim.reset()
    sim.playback(speed=2.0)
    sim.print_ee_pose()


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="VRM-HRI visual simulation (Swift)")
    parser.add_argument("--demo", action="store_true", help="run a short animated demo")
    parser.add_argument("--no-visual", action="store_true",
                        help="skip ir-support-extra-robots and show the DH model")
    parser.add_argument("--no-hand", action="store_true", help="hide the robotic hand")
    parser.add_argument("--no-axes", action="store_true", help="hide coordinate frames")
    parser.add_argument("--headless", action="store_true", help="launch Swift without a browser")
    parser.add_argument("--browser", default=None, help="browser to open Swift in")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    system = SystemModel.build_default(load_visual=not args.no_visual)
    sim = Simulation(
        system,
        use_visual=not args.no_visual,
        show_hand=not args.no_hand,
        show_axes=not args.no_axes,
        headless=args.headless,
        browser=args.browser,
    )
    try:
        sim.launch()
        sim.load_scene()
        sim.print_status()
        if args.demo:
            run_demo(sim)
        print("Simulation running. Press Ctrl+C to quit.")
        sim.hold()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        sim.close()


if __name__ == "__main__":
    main()