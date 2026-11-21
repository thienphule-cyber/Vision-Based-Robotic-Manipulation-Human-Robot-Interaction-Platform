# Vision-Based Robotic Manipulation & Human-Robot Interaction Platform (VRM-HRI)

A Python simulation platform for a **Mitsubishi RV-2FR** (6-DOF) industrial robot with a five-finger
robotic hand. The robot can be driven from a browser dashboard or by **hand gestures** captured with
a webcam (MediaPipe). Every command, whatever its source, passes through a central safety layer
before it reaches the 3D simulation.

> **Simulation only.** This project is not a certified safety system and must not be used to control
> a real robot without an independent safety assessment.

## Features

- 6-DOF RV-2FR robot: computational model (Robotics Toolbox) plus a visual model
  (`ir-support-extra-robots`) displayed in Swift. If the visual model is missing, the DH model is shown.
- Five-finger robotic hand with grasp presets (`open`, `closed`, `pinch`, `precision`, `power`) and
  per-finger control.
- Kinematics: forward kinematics, Jacobian, damped least-squares IK, resolved-rate servoing,
  joint-space and Cartesian trajectories, RRT-Connect planning.
- Safety gate for all commands: joint limits, workspace box, speed limits, singularity check,
  capsule-based collision checking (no AABB), emergency stop.
- Hand tracking and gesture control with MediaPipe: the robot follows the hand position and the
  simulated fingers mirror the user's fingers.
- Web dashboard (FastAPI + WebSocket): joint sliders, move-to-pose, hand control, record / playback,
  live camera preview, event log and E-STOP. Swift is **not** embedded in the dashboard.

## Architecture

```
 browser dashboard ──WebSocket / REST──▶ gui.py (FastAPI, own thread)
                                              │ validated commands
 webcam ─▶ computer_vision.py ─▶ gesture commands
                                              ▼
                                   main.py  (MotionController)
                        kinematics.py: IK, planning, servoing
                                              ▼
                              safety.py  (single gate for every command)
                                              ▼
                              simulation.py  (Swift 3D window)
```

| File | Role |
|------|------|
| `main.py` | Entry point: wiring, supervisor loop, motion worker thread |
| `modelling.py` | Robot (DH model), visual-model loader and sync check, hand, environment |
| `kinematics.py` | FK, IK, Jacobian, trajectories, collision sampling, RRT-Connect |
| `computer_vision.py` | MediaPipe hand tracking, gesture classification, hand-to-robot mapping |
| `simulation.py` | Swift scene, trajectory execution, record / playback |
| `safety.py` | Safety checks and the step limiter |
| `gui.py` | Dashboard (HTML / CSS / JS in one file) and its server |
| `tests/` | Unit and integration tests |

Threads: the main thread runs the supervisor loop (urgent commands such as stop and E-STOP are
handled immediately); `motion-worker` is the only thread that renders in the simulation; the vision
pipeline and the GUI server have their own threads.

## Requirements

- Python 3.12 (developed on macOS, Apple Silicon)
- A webcam (only for gesture control)

```bash
python3.12 -m venv venv
source venv/bin/activate

pip install roboticstoolbox-python spatialmath-python swift-sim spatialgeometry numpy
pip install mediapipe opencv-python
pip install fastapi "uvicorn[standard]"
pip install pytest httpx          # tests only
```

The visual robot model comes from the `ir-support-extra-robots` package; install it as described by
its own documentation. Without it, the application falls back to the DH model.

## Usage

```bash
python main.py                    # Swift window + dashboard
python main.py --vision           # also open the camera at start-up
python main.py --no-visual        # show the DH model instead of the visual model
```

The terminal prints the dashboard address (default `http://127.0.0.1:8000`). If the port is busy,
the next free port is used automatically.

| Option | Meaning |
|--------|---------|
| `--host`, `--port` | Dashboard address (default `127.0.0.1:8000`) |
| `--no-browser` | Do not open the dashboard automatically |
| `--no-visual` | Show the DH model instead of the visual model |
| `--no-hand`, `--no-axes` | Hide the robotic hand / coordinate frames |
| `--headless`, `--browser` | Swift without a browser / choose the browser for Swift |
| `--vision`, `--camera N` | Open the camera at start-up / camera index |
| `--model-path FILE` | `hand_landmarker.task` (only for MediaPipe builds without the legacy API) |
| `-v` | Verbose logging |

Each module can also be tried on its own:

```bash
python simulation.py --demo       # animated Swift demo
python gui.py                     # dashboard with a fake robot (no Swift, no camera)
python computer_vision.py --preview   # webcam + gesture preview window
```

## Dashboard

- **Joints**: sliders per joint, limits taken from the robot model.
- **End-effector**: live pose and a *Move to pose* form (position and optional roll / pitch / yaw).
- **Hand**: grasp presets and one slider per finger.
- **Control**: Home, Reset, Pause, Resume, Stop.
- **Recording**: start / stop / clear and playback with a speed factor.
- **Gesture control**: tick box, live gesture status and camera preview.
- **E-STOP** (button or `Esc`) and **Clear E-STOP**, plus an event log.

## Gesture control

Tick *Enable gesture control* (or start with `--vision`). The setting stays on until you untick it.

| Gesture | Effect |
|---------|--------|
| Open hand | Robot follows the hand, hand `open` |
| Fist | Robot follows the hand, hand `closed` |
| Pinch (thumb + index) | Robot follows the hand, hand `pinch` |
| Point (index only) | Robot holds its pose |
| Peace (index + middle) | Stops the current motion; following resumes with the next follow gesture |

Hand position maps to the end-effector position (left-right to `y`, up-down to `z`, distance to the
camera to `x`) inside a box around `MappingConfig.center`; the tool orientation is kept from the moment
gesture control is enabled. The simulated fingers mirror the flexion of the user's fingers, even
while the robot holds its pose. A gesture must be seen on several consecutive frames before it is
accepted, and the target is low-pass filtered.

## Safety model

All commands from the GUI, gestures and planners go through `safety.py`:

- joint limits, workspace box and speed limits (joint and end-effector);
- collision checking with **capsules** for the links and the hand against the **true shapes** of
  obstacles (oriented boxes, spheres, cylinders); axis-aligned bounding boxes are not used;
- moving *into* a singularity is blocked (the `home` command is the documented exception);
- an E-STOP latch that blocks every motion until cleared; the dashboard pre-checks commands for fast
  feedback, but the control loop validates again before executing;
- trajectories are executed in short chunks so Stop and E-STOP act within one chunk.

## Tests

```bash
pytest -v
```

`tests/test_integration.py` replaces the Swift simulation with a fake, so no window or camera is
needed. Tests that depend on optional packages (MediaPipe, OpenCV, FastAPI, uvicorn) skip themselves
when the package is missing.

## Project structure

```
.
├── main.py
├── modelling.py
├── kinematics.py
├── computer_vision.py
├── simulation.py
├── safety.py
├── gui.py
├── pytest.ini
├── tests/
│   ├── test_modelling.py
│   ├── test_kinematics.py
│   ├── test_safety.py
│   ├── test_computer_vision.py
│   ├── test_gui.py
│   └── test_integration.py
├── README.md
├── LICENSE.md
├── references.md
└── .gitignore
```

## Known limitations

- The RV-2FR DH table in `modelling.py` is **nominal**. The startup report can show
  `visual/model sync: MISMATCH` when it differs from the visual model; verify the table against the
  manufacturer's data sheet before trusting IK or collision results.
- Collision checking is approximate (sampled capsules); radii and margins in `SafetyConfig` are
  nominal values.
- Gesture thresholds are heuristics for a typical webcam at 40-80 cm and may need tuning
  (`GestureConfig`, `MappingConfig`).
- The home pose (all joints zero) is a wrist singularity of the nominal DH model.

## Troubleshooting

- **Camera does not open (macOS):** allow camera access for your terminal / IDE in
  *System Settings > Privacy & Security > Camera*. The reason is shown in the dashboard event log.
- **Port already in use:** the application picks the next free port; check the printed address.
- **MediaPipe has no `solutions` API:** download `hand_landmarker.task` and run with
  `--model-path path/to/hand_landmarker.task`.

## License

Released under the MIT License, see [LICENSE.md](LICENSE.md). Third-party packages keep their own
licenses.

## References

See [references.md](references.md).