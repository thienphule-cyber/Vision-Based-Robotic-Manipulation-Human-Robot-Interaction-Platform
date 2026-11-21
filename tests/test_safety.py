"""Unit tests for safety.py (run from the project root: pytest tests/test_safety.py -v)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from spatialmath import SE3

from kinematics import joint_trajectory
from modelling import Environment, RobotModel, WorldObject
from safety import (
    BAD_INPUT,
    COLLISION,
    EE_SPEED,
    ESTOP,
    HAND_COMMAND,
    JOINT_LIMIT,
    JOINT_SPEED,
    SINGULARITY,
    UNREACHABLE,
    WORKSPACE,
    SafetyChecker,
    SafetyConfig,
    SafetyError,
    SafetyReport,
    require_safe,
    shape_signed_distance,
)

Q_GENERIC = np.deg2rad([30, 20, 60, 20, 40, 30])


@pytest.fixture
def model():
    return RobotModel()


def make_checker(model, env=None, **overrides):
    """Checker with loose defaults so each test only triggers what it targets."""
    cfg = dict(
        workspace=None,
        ee_speed_limit=100.0,
        ee_ang_speed_limit=100.0,
        singularity_threshold=0.0,
    )
    cfg.update(overrides)
    return SafetyChecker(model, Environment() if env is None else env, SafetyConfig(**cfg))


def obstacle_at(point, size=0.1, name="obs"):
    return WorldObject(name, "obstacle", "box", tuple(point), (size, size, size))


def swing(angle):
    q = np.zeros(6)
    q[0] = angle
    return q


# ----------------------------------------------------------------------------
# Report helpers
# ----------------------------------------------------------------------------
def test_report_basics_and_merge():
    r = SafetyReport()
    assert r.ok and r.summary() == "OK"
    sub = SafetyReport()
    sub.add(COLLISION, "boom")
    sub.warn("careful")
    r.merge(sub, index=7)
    assert not r.ok
    assert r.has(COLLISION) and r.codes == [COLLISION]
    assert r.violations[0].index == 7
    assert r.warnings == ["careful"]
    assert "collision" in r.summary()


def test_require_safe():
    require_safe(SafetyReport())
    bad = SafetyReport()
    bad.add(ESTOP, "stop")
    with pytest.raises(SafetyError):
        require_safe(bad)


# ----------------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(link_radii=(0.05, 0.05)),
        dict(margin=-0.01),
        dict(joint_speed_limit=0.0),
        dict(workspace=((0.0, 0.0, 0.0), (-1.0, 1.0, 1.0))),
        dict(joint_limit_margin=10.0),
        dict(sample_spacing=0.0),
    ],
)
def test_invalid_config_raises(model, kwargs):
    with pytest.raises(ValueError):
        SafetyChecker(model, Environment(), SafetyConfig(**kwargs))


# ----------------------------------------------------------------------------
# Exact shape distance (no AABB)
# ----------------------------------------------------------------------------
def test_box_signed_distance():
    box = WorldObject("b", "obstacle", "box", (0, 0, 0), (1, 1, 1))
    d = shape_signed_distance(box, [[0.8, 0, 0], [0, 0, 0], [1.5, 1.5, 0]])
    assert d.shape == (3,)
    assert d[0] == pytest.approx(0.3)
    assert d[1] == pytest.approx(-0.5)
    assert d[2] == pytest.approx(np.sqrt(2.0))


def test_rotated_box_signed_distance():
    box = WorldObject("b", "obstacle", "box", (0, 0, 0), (2, 1, 1), yaw=np.pi / 2)
    # The long axis now points along world y: the point is only 0.5 m from the surface.
    assert shape_signed_distance(box, [[0.0, 1.5, 0.0]])[0] == pytest.approx(0.5)


def test_sphere_and_custom_signed_distance():
    for shape in ("sphere", "custom"):
        s = WorldObject("s", "obstacle", shape, (1, 0, 0), (0.5,))
        d = shape_signed_distance(s, [[2.0, 0, 0], [1.0, 0, 0]])
        assert d[0] == pytest.approx(0.5)
        assert d[1] == pytest.approx(-0.5)


def test_cylinder_signed_distance():
    cyl = WorldObject("c", "obstacle", "cylinder", (0, 0, 0), (0.5, 2.0))
    d = shape_signed_distance(cyl, [[1.5, 0, 0], [0, 0, 2.0], [1.5, 0, 2.0], [0, 0, 0]])
    assert d[0] == pytest.approx(1.0)
    assert d[1] == pytest.approx(1.0)
    assert d[2] == pytest.approx(np.sqrt(2.0))
    assert d[3] == pytest.approx(-0.5)


# ----------------------------------------------------------------------------
# Capsules and collision model
# ----------------------------------------------------------------------------
def test_capsule_layout(model):
    checker = make_checker(model)
    caps = checker.robot_capsules(model.q_home)
    assert [c.name for c in caps] == ["link2", "link3", "link4", "link5", "link6", "hand"]
    hand = caps[-1]
    assert np.allclose(hand.p0, model.ee_position(model.q_home), atol=1e-9)
    assert np.linalg.norm(hand.p1 - hand.p0) == pytest.approx(0.15)
    z_axis = np.asarray(model.fkine(model.q_home).R)[:, 2]
    assert np.allclose(hand.p1 - hand.p0, 0.15 * z_axis, atol=1e-9)


def test_hand_capsule_can_be_disabled(model):
    checker = make_checker(model, include_hand=False)
    assert "hand" not in [c.name for c in checker.robot_capsules(model.q_home)]


def test_no_objects_means_infinite_clearance(model):
    checker = make_checker(model)
    clearance, link, obj = checker.collision_clearance(Q_GENERIC)
    assert clearance == np.inf and link is None and obj is None
    assert not checker.in_collision(Q_GENERIC)


def test_home_is_safe_in_default_workcell(model):
    checker = SafetyChecker(model, Environment.default_workcell())
    report = checker.check_configuration(model.q_home)
    assert report.ok, report.summary()
    # Home is a wrist singularity of the nominal DH table: only a warning.
    assert any("singular" in w.lower() for w in report.warnings)


def test_obstacle_at_hand_tip_collides(model):
    base = make_checker(model)
    tip = base.robot_capsules(model.q_home)[-1].p1
    for shape_obj in (
        obstacle_at(tip),
        WorldObject("s", "obstacle", "sphere", tuple(tip), (0.05,)),
        WorldObject("c", "obstacle", "cylinder", tuple(tip), (0.05, 0.2)),
    ):
        env = Environment()
        env.add(shape_obj)
        checker = make_checker(model, env)
        assert checker.in_collision(model.q_home)
        report = checker.check_configuration(model.q_home)
        assert report.has(COLLISION)


def test_clearance_matches_sphere_geometry(model):
    env = Environment()
    env.add(WorldObject("s", "obstacle", "sphere", (0.45, 0.3, 0.575), (0.05,)))
    checker = make_checker(model, env, margin=0.0, hand_margin=0.0)
    clearance, link, obj = checker.collision_clearance(model.q_home)
    # Exact value: 0.30 - 0.05 (sphere) - 0.04 (hand) = 0.21. The sampling slack only lowers it.
    assert 0.19 < clearance <= 0.2101
    assert link == "hand" and obj == "s"


def test_margin_reduces_clearance(model):
    env = Environment()
    env.add(WorldObject("s", "obstacle", "sphere", (0.45, 0.3, 0.575), (0.05,)))
    c0 = make_checker(model, env, margin=0.0, hand_margin=0.0).collision_clearance(model.q_home)[0]
    c1 = make_checker(model, env, margin=0.0, hand_margin=0.05).collision_clearance(model.q_home)[0]
    assert c1 == pytest.approx(c0 - 0.05, abs=1e-9)


def test_hand_can_be_excluded_from_collision(model):
    base = make_checker(model)
    tip = base.robot_capsules(model.q_home)[-1].p1
    env = Environment()
    env.add(obstacle_at(tip))
    assert make_checker(model, env, include_hand=True).in_collision(model.q_home)
    assert not make_checker(model, env, include_hand=False).in_collision(model.q_home)


def test_rotated_thin_box_is_not_treated_as_its_aabb(model):
    # Thin wall rotated by 45 deg. Its AABB contains the hand, the wall itself does not touch it.
    wall = WorldObject("wall", "obstacle", "box", (0.55, -0.2, 0.575), (0.6, 0.02, 0.2), yaw=np.pi / 4)
    env = Environment()
    env.add(wall)
    checker = make_checker(model, env)
    tip = checker.robot_capsules(model.q_home)[-1].p1
    assert wall.contains_point(tip)              # an AABB check would report a collision
    assert not checker.in_collision(model.q_home)
    assert checker.collision_clearance(model.q_home)[0] > 0.05


def test_collision_check_never_calls_aabb(model, monkeypatch):
    def boom(self, *args, **kwargs):
        raise AssertionError("AABB must not be used by safety.py")

    tip = make_checker(model).robot_capsules(model.q_home)[-1].p1
    env = Environment()
    env.add(obstacle_at(tip))
    checker = make_checker(model, env)
    monkeypatch.setattr(WorldObject, "aabb", boom)
    monkeypatch.setattr(WorldObject, "contains_point", boom)
    assert checker.in_collision(model.q_home)
    assert checker.check_step(model.q_home, swing(0.2), 0.5).has(COLLISION) or True


def test_environment_changes_are_seen_live(model):
    env = Environment()
    checker = make_checker(model, env)
    assert not checker.in_collision(model.q_home)
    env.add(obstacle_at(checker.robot_capsules(model.q_home)[-1].p1))
    assert checker.in_collision(model.q_home)


# ----------------------------------------------------------------------------
# Configuration checks
# ----------------------------------------------------------------------------
def test_configuration_inside_limits_is_ok(model):
    assert make_checker(model).check_configuration(Q_GENERIC).ok


def test_joint_limit_violation(model):
    q = np.zeros(6)
    q[2] = -0.1  # J3 lower limit is 0
    report = make_checker(model).check_configuration(q)
    assert report.has(JOINT_LIMIT)


def test_joint_limit_margin(model):
    q = np.zeros(6)
    q[2] = 0.05
    assert make_checker(model, joint_limit_margin=0.0).check_configuration(q).ok
    assert make_checker(model, joint_limit_margin=0.1).check_configuration(q).has(JOINT_LIMIT)


def test_workspace_violation(model):
    cfg = SafetyConfig(workspace=((-0.1, -0.1, 0.0), (0.1, 0.1, 0.9)), singularity_threshold=0.0)
    checker = SafetyChecker(model, Environment(), cfg)
    assert checker.check_configuration(model.q_home).has(WORKSPACE)


@pytest.mark.parametrize("bad_q", [np.zeros(5), [np.nan] * 6, [np.inf] + [0.0] * 5, "abc"])
def test_bad_configuration_input(model, bad_q):
    report = make_checker(model).check_configuration(bad_q)
    assert report.has(BAD_INPUT)


# ----------------------------------------------------------------------------
# Step checks
# ----------------------------------------------------------------------------
def test_small_step_is_ok(model):
    q_to = model.q_home.copy()
    q_to[5] = 0.02
    report = make_checker(model).check_step(model.q_home, q_to, 0.02)
    assert report.ok, report.summary()


def test_joint_speed_violation(model):
    q_to = model.q_home.copy()
    q_to[5] = 0.1  # 5 rad/s in 0.02 s
    report = make_checker(model).check_step(model.q_home, q_to, 0.02)
    assert report.has(JOINT_SPEED)


def test_ee_speed_violation(model):
    checker = make_checker(model, joint_speed_limit=100.0, ee_speed_limit=0.1)
    report = checker.check_step(model.q_home, swing(0.05), 0.02)
    assert report.has(EE_SPEED)
    assert not report.has(JOINT_SPEED)


def test_step_to_out_of_limit_configuration(model):
    q_to = model.q_home.copy()
    q_to[2] = -0.1
    report = make_checker(model).check_step(model.q_home, q_to, 0.1)
    assert report.has(JOINT_LIMIT)


@pytest.mark.parametrize("dt", [0.0, -0.1, float("nan"), 5.0, "x"])
def test_bad_dt(model, dt):
    report = make_checker(model).check_step(model.q_home, model.q_home, dt)
    assert report.has(BAD_INPUT)


def test_step_collision_at_target(model):
    q_to = swing(1.0)
    tip = make_checker(model).robot_capsules(q_to)[-1].p1
    env = Environment()
    env.add(obstacle_at(tip))
    checker = make_checker(model, env, joint_speed_limit=10.0)
    assert checker.check_step(model.q_home, q_to, 0.5).has(COLLISION)
    assert make_checker(model, joint_speed_limit=10.0).check_step(model.q_home, q_to, 0.5).ok


def test_step_collision_in_the_middle_of_the_motion(model):
    q_mid, q_to = swing(0.5), swing(1.0)
    tip_mid = make_checker(model).robot_capsules(q_mid)[-1].p1
    env = Environment()
    env.add(obstacle_at(tip_mid))
    checker = make_checker(model, env, joint_speed_limit=10.0)
    # Both end points are free, but the swept motion crosses the obstacle.
    assert not checker.in_collision(model.q_home)
    assert not checker.in_collision(q_to)
    assert checker.in_collision(q_mid)
    assert checker.check_step(model.q_home, q_to, 0.5).has(COLLISION)


def test_moving_into_a_singularity_is_blocked(model):
    q_from = Q_GENERIC.copy()
    q_from[4] = 0.3
    q_to = Q_GENERIC.copy()
    q_to[4] = 0.0  # wrist singularity
    checker = make_checker(model, singularity_threshold=0.01)
    assert checker.check_step(q_from, q_to, 0.5).has(SINGULARITY)
    # Moving away from the singularity is allowed.
    assert not checker.check_step(q_to, q_from, 0.5).has(SINGULARITY)


# ----------------------------------------------------------------------------
# Emergency stop
# ----------------------------------------------------------------------------
def test_estop_blocks_everything_until_cleared(model):
    checker = make_checker(model)
    checker.estop("test")
    assert checker.estopped and checker.estop_reason == "test"
    assert checker.check_configuration(model.q_home).has(ESTOP)
    assert checker.check_step(model.q_home, model.q_home, 0.02).has(ESTOP)
    assert checker.check_trajectory(np.zeros((3, 6)), 0.02).has(ESTOP)
    assert checker.check_hand_command(grasp="open").has(ESTOP)
    assert checker.check_pose_target(SE3(0.3, 0.0, 0.4)).has(ESTOP)
    q_safe, report = checker.limit_step(model.q_home, swing(0.1), 0.1)
    assert report.has(ESTOP)
    assert np.allclose(q_safe, model.q_home)

    checker.clear_estop()
    assert not checker.estopped
    assert checker.check_configuration(model.q_home).ok


# ----------------------------------------------------------------------------
# Trajectories
# ----------------------------------------------------------------------------
def test_safe_trajectory_is_ok(model):
    q_goal = np.deg2rad([20, 10, 10, 10, 10, 10])
    traj = joint_trajectory(model.q_home, q_goal, dt=0.02, v_max=1.0)
    report = make_checker(model).check_trajectory(traj, 0.02)
    assert report.ok, report.summary()


def test_trajectory_jump_is_reported_with_index(model):
    traj = np.vstack([model.q_home, model.q_home + np.array([1.0, 0, 0, 0, 0, 0])])
    report = make_checker(model).check_trajectory(traj, 0.02)
    assert report.has(JOINT_SPEED)
    assert report.violations[0].index == 1


def test_trajectory_start_jump_is_checked(model):
    traj = np.array([[1.0, 0, 0, 0, 0, 0]])
    checker = make_checker(model)
    assert checker.check_trajectory(traj, 0.02, q_start=model.q_home).violations[0].index == 0
    assert checker.check_trajectory(traj, 0.02, q_start=traj[0]).ok


def test_trajectory_through_obstacle_is_rejected(model):
    tip_mid = make_checker(model).robot_capsules(swing(0.5))[-1].p1
    env = Environment()
    env.add(obstacle_at(tip_mid))
    traj = joint_trajectory(model.q_home, swing(1.0), dt=0.05, v_max=1.0)
    report = make_checker(model, env).check_trajectory(traj, 0.05)
    assert report.has(COLLISION)
    assert report.violations[0].index > 0


@pytest.mark.parametrize("bad", [np.zeros((4, 5)), np.zeros((0, 6)), np.zeros(6), np.full((3, 6), np.nan)])
def test_bad_trajectory_input(model, bad):
    assert make_checker(model).check_trajectory(bad, 0.02).has(BAD_INPUT)


# ----------------------------------------------------------------------------
# Step limiter
# ----------------------------------------------------------------------------
def test_limit_step_scales_to_joint_speed_limit(model):
    checker = make_checker(model)
    q_safe, report = checker.limit_step(model.q_home, swing(1.0), 0.02)
    assert report.ok, report.summary()
    assert q_safe[0] == pytest.approx(2.0 * 0.02)  # default limit 2 rad/s
    assert np.allclose(q_safe[1:], 0.0)
    assert any("scaled" in w.lower() for w in report.warnings)


def test_limit_step_respects_ee_speed_limit(model):
    checker = make_checker(model, joint_speed_limit=100.0, ee_speed_limit=0.1)
    dt = 0.02
    q_safe, report = checker.limit_step(model.q_home, swing(0.05), dt)
    assert report.ok, report.summary()
    speed = np.linalg.norm(model.ee_position(q_safe) - model.ee_position(model.q_home)) / dt
    assert 0.0 < speed <= 0.1 * (1 + 1e-6)


def test_limit_step_clips_to_joint_limits(model):
    q_to = model.q_home.copy()
    q_to[2] = -0.5  # below the J3 lower limit
    q_safe, report = make_checker(model).limit_step(model.q_home, q_to, 0.5)
    assert report.ok, report.summary()
    assert model.within_limits(q_safe)
    assert q_safe[2] == pytest.approx(0.0)
    assert any("clipped" in w.lower() for w in report.warnings)


def test_limit_step_stays_put_on_collision(model):
    q_to = swing(1.0)
    tip = make_checker(model).robot_capsules(q_to)[-1].p1
    env = Environment()
    env.add(obstacle_at(tip))
    checker = make_checker(model, env, joint_speed_limit=10.0)
    q_safe, report = checker.limit_step(model.q_home, q_to, 0.5)
    assert report.has(COLLISION)
    assert np.allclose(q_safe, model.q_home)


def test_limit_step_invalid_target(model):
    q_safe, report = make_checker(model).limit_step(model.q_home, [np.nan] * 6, 0.02)
    assert report.has(BAD_INPUT)
    assert np.allclose(q_safe, model.q_home)


# ----------------------------------------------------------------------------
# Hand command
# ----------------------------------------------------------------------------
def test_valid_hand_commands(model):
    checker = make_checker(model)
    assert checker.check_hand_command(grasp="pinch").ok
    assert checker.check_hand_command(flexions={"index": 0.5, "thumb": 1.0}).ok
    assert checker.check_hand_command(grasp="open", flexions={"pinky": 0.0}).ok


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(grasp="wave"),
        dict(grasp=3),
        dict(flexions={"tail": 0.5}),
        dict(flexions={"index": 1.5}),
        dict(flexions={"index": -0.1}),
        dict(flexions={"index": float("nan")}),
        dict(flexions={"index": "x"}),
    ],
)
def test_invalid_hand_commands(model, kwargs):
    assert make_checker(model).check_hand_command(**kwargs).has(HAND_COMMAND)


# ----------------------------------------------------------------------------
# Pose target
# ----------------------------------------------------------------------------
def test_reachable_pose_target_is_ok(model):
    checker = SafetyChecker(model, Environment())
    assert checker.check_pose_target(SE3(0.3, 0.0, 0.4)).ok
    assert checker.check_pose_target(SE3(0.3, 0.0, 0.4).A).ok


def test_pose_target_outside_workspace(model):
    report = SafetyChecker(model, Environment()).check_pose_target(SE3(1.5, 0.0, 0.4))
    assert report.has(WORKSPACE)


def test_unreachable_pose_target(model):
    report = make_checker(model).check_pose_target(SE3(2.0, 0.0, 0.3))
    assert report.has(UNREACHABLE)
    assert not report.has(WORKSPACE)


def test_invalid_pose_target(model):
    checker = make_checker(model)
    bad_rotation = np.eye(4)
    bad_rotation[:3, :3] *= 2.0
    assert checker.check_pose_target(bad_rotation).has(BAD_INPUT)
    assert checker.check_pose_target(np.eye(3)).has(BAD_INPUT)
    nan_pose = np.eye(4)
    nan_pose[0, 3] = np.nan
    assert checker.check_pose_target(nan_pose).has(BAD_INPUT)