"""Unit tests for kinematics.py (run from the project root: pytest tests/test_kinematics.py -v)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from spatialmath import SE3

from kinematics import (
    CollisionChecker,
    cartesian_trajectory,
    concat_trajectories,
    dls_solve,
    ik,
    is_near_singular,
    jacobian,
    joint_trajectory,
    link_frames,
    manipulability,
    min_duration,
    min_singular_value,
    path_to_trajectory,
    plan_joint_path,
    plan_trajectory,
    pose_error,
    resolved_rate_step,
    rotation_vector,
    rotvec_to_matrix,
    servo_to_pose,
)
from modelling import Environment, RobotModel, WorldObject

# A generic, well-conditioned configuration inside all joint limits.
Q_GENERIC = np.deg2rad([30, 20, 60, 20, 40, 30])


@pytest.fixture
def model():
    return RobotModel()


# ----------------------------------------------------------------------------
# Math helpers
# ----------------------------------------------------------------------------
def test_rotation_vector_identity():
    assert np.allclose(rotation_vector(np.eye(3)), 0.0)


def test_rotation_vector_about_z():
    assert np.allclose(rotation_vector(SE3.Rz(0.5).R), [0.0, 0.0, 0.5], atol=1e-9)


def test_rotation_vector_at_pi():
    w = rotation_vector(SE3.Rx(np.pi).R)
    assert np.linalg.norm(w) == pytest.approx(np.pi, abs=1e-6)


def test_rotation_vector_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        w = axis * rng.uniform(0.01, 3.0)
        assert np.allclose(rotation_vector(rotvec_to_matrix(w)), w, atol=1e-8)


def test_rotvec_to_matrix_is_a_rotation():
    R = rotvec_to_matrix([0.3, -0.2, 0.5])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_pose_error_zero_for_same_pose():
    T = SE3(0.1, 0.2, 0.3) * SE3.Rz(0.4)
    assert np.allclose(pose_error(T, T), 0.0, atol=1e-12)


def test_pose_error_translation_only():
    e = pose_error(SE3(0.0, 0.0, 0.0), SE3(0.1, -0.2, 0.3))
    assert np.allclose(e, [0.1, -0.2, 0.3, 0.0, 0.0, 0.0], atol=1e-12)


# ----------------------------------------------------------------------------
# Frames, Jacobian, singularities
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("use_offsets", [False, True])
def test_link_frames_match_fkine(use_offsets):
    if use_offsets:
        m = RobotModel(base=SE3(0.1, 0.2, 0.3) * SE3.Rz(0.4), tool=SE3.Tz(0.1))
    else:
        m = RobotModel()
    rng = np.random.default_rng(1)
    for _ in range(3):
        q = m.random_q(rng)
        frames = link_frames(m, q)
        assert len(frames) == m.n + 1
        assert np.allclose(frames[-1] @ np.asarray(m.tool.A), m.fkine(q).A, atol=1e-9)


def _fd_jacobian(model, q, h=1e-6):
    J = np.zeros((6, model.n))
    for i in range(model.n):
        dq = np.zeros(model.n)
        dq[i] = h
        Tp, Tm = model.fkine(q + dq).A, model.fkine(q - dq).A
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
        J[3:, i] = rotation_vector(Tp[:3, :3] @ Tm[:3, :3].T) / (2 * h)
    return J


def test_jacobian_shape(model):
    assert jacobian(model, Q_GENERIC).shape == (6, 6)


def test_jacobian_matches_finite_differences(model):
    for q in (Q_GENERIC, model.random_q(np.random.default_rng(2))):
        assert np.allclose(jacobian(model, q), _fd_jacobian(model, q), atol=1e-5)


def test_jacobian_wrong_size_raises(model):
    with pytest.raises(ValueError):
        jacobian(model, np.zeros(5))


def test_generic_configuration_is_not_singular(model):
    assert manipulability(model, Q_GENERIC) > 0.0
    assert min_singular_value(model, Q_GENERIC) > 1e-4
    assert not is_near_singular(model, Q_GENERIC, threshold=1e-4)


def test_wrist_singularity_is_detected(model):
    q = Q_GENERIC.copy()
    q[4] = 0.0  # J4 and J6 axes become collinear
    assert min_singular_value(model, q) < 1e-6
    assert is_near_singular(model, q, threshold=1e-4)


def test_dls_matches_inverse_far_from_singularity():
    J = np.diag([1.0, 2.0, 3.0])
    e = np.array([0.3, -0.4, 0.9])
    assert np.allclose(dls_solve(J, e), np.linalg.solve(J, e), atol=1e-4)


def test_dls_output_is_bounded_near_singularity():
    J = np.diag([1.0, 1.0, 1e-9])
    dq = dls_solve(J, np.array([0.0, 0.0, 1.0]))
    assert np.linalg.norm(dq) < 1.0


# ----------------------------------------------------------------------------
# Inverse kinematics
# ----------------------------------------------------------------------------
def test_ik_recovers_reachable_pose(model):
    T = model.fkine(Q_GENERIC)
    q0 = model.clip(Q_GENERIC + np.deg2rad([8, -6, 7, -9, 6, 8]))
    res = ik(model, T, q0)
    assert res.success
    err = pose_error(model.fkine(res.q).A, T.A)
    assert np.linalg.norm(err[:3]) < 1e-3
    assert np.linalg.norm(err[3:]) < 1e-2
    assert model.within_limits(res.q)


def test_ik_random_reachable_targets(model):
    rng = np.random.default_rng(3)
    n_ok, n_total = 0, 8
    for _ in range(n_total):
        q_true = model.random_q(rng)
        T = model.fkine(q_true)
        res = ik(model, T, model.q_home, n_restarts=20, max_iters=300, seed=1)
        assert model.within_limits(res.q)
        if res.success:
            n_ok += 1
            err = pose_error(model.fkine(res.q).A, T.A)
            assert np.linalg.norm(err[:3]) < 1e-3
    assert n_ok >= 5


def test_ik_position_only(model):
    target = model.fkine(Q_GENERIC)
    q0 = model.clip(Q_GENERIC + np.deg2rad([10, 5, -8, 12, -6, 5]))
    res = ik(model, target, q0, position_only=True)
    assert res.success
    assert np.linalg.norm(model.ee_position(res.q) - target.t) < 1e-3


def test_ik_unreachable_target_fails_gracefully(model):
    res = ik(model, SE3(5.0, 0.0, 0.5), n_restarts=2)
    assert not res.success
    assert model.within_limits(res.q)
    assert res.message


def test_ik_wrong_start_size_raises(model):
    with pytest.raises(ValueError):
        ik(model, SE3(0.3, 0.0, 0.3), q0=np.zeros(5))


# ----------------------------------------------------------------------------
# Servo steps
# ----------------------------------------------------------------------------
def test_resolved_rate_zero_twist_keeps_q(model):
    q = resolved_rate_step(model, Q_GENERIC, np.zeros(6), dt=0.02)
    assert np.allclose(q, Q_GENERIC)


def test_resolved_rate_respects_joint_speed_limit(model):
    dt = 0.02
    q = resolved_rate_step(model, Q_GENERIC, [5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dt, max_joint_speed=0.5)
    assert np.max(np.abs(q - Q_GENERIC)) / dt <= 0.5 + 1e-9
    assert model.within_limits(q)


def test_servo_to_pose_converges(model):
    q_goal = Q_GENERIC + np.deg2rad([10, -5, 5, 5, 5, 10])
    T_goal = model.fkine(q_goal)
    q = Q_GENERIC.copy()
    first_pos_err = None
    for _ in range(400):
        q, pos_err, rot_err = servo_to_pose(model, q, T_goal, dt=0.02)
        if first_pos_err is None:
            first_pos_err = pos_err
    err = pose_error(model.fkine(q).A, T_goal.A)
    assert np.linalg.norm(err[:3]) < first_pos_err
    assert np.linalg.norm(err[:3]) < 5e-3
    assert np.linalg.norm(err[3:]) < 5e-2
    assert model.within_limits(q)


# ----------------------------------------------------------------------------
# Joint trajectories
# ----------------------------------------------------------------------------
def test_joint_trajectory_endpoints_and_shape():
    q0, q1 = np.zeros(6), np.deg2rad([10, 20, 30, 40, 50, 60])
    traj = joint_trajectory(q0, q1, dt=0.1, duration=1.0)
    assert traj.shape == (11, 6)
    assert np.allclose(traj[0], q0) and np.allclose(traj[-1], q1)


def test_quintic_starts_and_ends_at_rest():
    q0, q1 = np.zeros(6), np.deg2rad([10, 20, 30, 40, 50, 60])
    traj = joint_trajectory(q0, q1, dt=0.02, duration=1.0)
    total = np.linalg.norm(q1 - q0)
    assert np.linalg.norm(traj[1] - traj[0]) < 1e-3 * total
    assert np.linalg.norm(traj[-1] - traj[-2]) < 1e-3 * total


def test_linear_trajectory_has_constant_velocity():
    traj = joint_trajectory(np.zeros(6), np.ones(6), dt=0.1, duration=1.0, method="linear")
    steps = np.diff(traj, axis=0)
    assert np.allclose(steps, steps[0])


def test_joint_trajectory_respects_v_max():
    q0, q1 = np.zeros(6), np.deg2rad([90, -60, 30, 120, 10, 45])
    dt, v_max = 0.02, 0.8
    traj = joint_trajectory(q0, q1, dt=dt, v_max=v_max)
    assert np.max(np.abs(np.diff(traj, axis=0))) / dt <= v_max + 1e-9


def test_joint_trajectory_duration_is_raised_to_v_max_limit():
    q0, q1 = np.zeros(6), np.array([2.0, 0, 0, 0, 0, 0])
    traj = joint_trajectory(q0, q1, dt=0.1, duration=0.1, v_max=1.0)
    assert len(traj) > 2  # one step would need a far higher speed


def test_zero_length_move_is_valid():
    traj = joint_trajectory(np.zeros(6), np.zeros(6), dt=0.02)
    assert np.allclose(traj, 0.0)


def test_min_duration_values():
    q0, q1 = np.zeros(6), np.array([1.0, 0, 0, 0, 0, 0])
    assert min_duration(q0, q1, 2.0, "quintic") == pytest.approx(1.875 / 2.0)
    assert min_duration(q0, q1, 2.0, "linear") == pytest.approx(0.5)


def test_joint_trajectory_invalid_arguments():
    with pytest.raises(ValueError):
        joint_trajectory(np.zeros(6), np.ones(6), dt=0.0)
    with pytest.raises(ValueError):
        joint_trajectory(np.zeros(6), np.ones(5))
    with pytest.raises(ValueError):
        joint_trajectory(np.zeros(6), np.ones(6), method="cubic")
    with pytest.raises(ValueError):
        joint_trajectory(np.zeros(6), np.ones(6), v_max=-1.0)


def test_concat_and_path_to_trajectory():
    a, b, c = np.zeros(6), np.full(6, 0.3), np.full(6, -0.2)
    traj = path_to_trajectory([a, b, c], dt=0.05, v_max=1.0)
    assert np.allclose(traj[0], a) and np.allclose(traj[-1], c)
    assert np.min(np.linalg.norm(traj - b, axis=1)) < 1e-9
    seg1 = joint_trajectory(a, b, dt=0.05, v_max=1.0)
    seg2 = joint_trajectory(b, c, dt=0.05, v_max=1.0)
    assert len(traj) == len(seg1) + len(seg2) - 1
    assert len(concat_trajectories([seg1])) == len(seg1)


def test_path_to_trajectory_needs_two_waypoints():
    with pytest.raises(ValueError):
        path_to_trajectory([np.zeros(6)])


# ----------------------------------------------------------------------------
# Cartesian trajectories
# ----------------------------------------------------------------------------
def test_cartesian_trajectory_is_straight_and_reaches_goal(model):
    q_a = Q_GENERIC
    q_b = Q_GENERIC + np.deg2rad([10, -5, 5, 5, 5, 10])
    T0, T1 = model.fkine(q_a), model.fkine(q_b)
    res = cartesian_trajectory(model, T0, T1, q_a, dt=0.05)
    assert res.success, res.message

    pts = np.array([model.ee_position(q) for q in res.q_traj])
    p0, p1 = np.asarray(T0.t), np.asarray(T1.t)
    d = p1 - p0
    rel = pts - p0
    proj = rel @ d / float(d @ d)
    deviation = np.linalg.norm(rel - np.outer(proj, d), axis=1)
    assert deviation.max() < 1e-3
    assert np.linalg.norm(pts[0] - p0) < 1e-3
    assert np.linalg.norm(pts[-1] - p1) < 1e-3
    assert np.max(np.abs(np.diff(res.q_traj, axis=0))) < 0.5
    assert all(model.within_limits(q) for q in res.q_traj)

    err = pose_error(model.fkine(res.q_traj[-1]).A, T1.A)
    assert np.linalg.norm(err[3:]) < 1e-2


def test_cartesian_trajectory_reports_failure(model):
    res = cartesian_trajectory(model, model.fkine(Q_GENERIC), SE3(5.0, 0.0, 0.5), Q_GENERIC, dt=0.1)
    assert not res.success
    assert res.failed_index is not None
    assert res.message


# ----------------------------------------------------------------------------
# Collision checking
# ----------------------------------------------------------------------------
def test_empty_environment_has_no_collision(model):
    checker = CollisionChecker(model, Environment())
    assert not checker.in_collision(Q_GENERIC)


def test_obstacle_at_tool_tip_is_detected(model):
    env = Environment()
    tip = model.ee_position(Q_GENERIC)
    env.add(WorldObject("o", "obstacle", "box", tip, (0.1, 0.1, 0.1)))
    checker = CollisionChecker(model, env, margin=0.0)
    assert checker.in_collision(Q_GENERIC)


def test_zones_and_objects_are_ignored_by_default(model):
    env = Environment()
    tip = model.ee_position(Q_GENERIC)
    env.add(WorldObject("z", "target_zone", "box", tip, (0.1, 0.1, 0.1)))
    env.add(WorldObject("c", "object", "box", tip, (0.1, 0.1, 0.1)))
    assert not CollisionChecker(model, env, margin=0.0).in_collision(Q_GENERIC)


def test_base_pillar_is_not_flagged_against_the_table(model):
    checker = CollisionChecker(model, Environment.default_workcell())
    assert not checker.in_collision(model.q_home)


def test_sample_points_shape(model):
    pts = CollisionChecker(model, Environment()).sample_points(Q_GENERIC)
    assert pts.ndim == 2 and pts.shape[1] == 3 and len(pts) > model.n


# ----------------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------------
def _edges_are_free(checker, path):
    return all(checker.edge_free(a, b) for a, b in zip(path[:-1], path[1:]))


def test_plan_in_free_space_is_direct(model):
    q_goal = Q_GENERIC + np.deg2rad([20, 5, 10, 10, 5, 10])
    res = plan_joint_path(model, Environment(), Q_GENERIC, q_goal)
    assert res.success
    assert len(res.path) == 2
    assert np.allclose(res.path[0], Q_GENERIC) and np.allclose(res.path[-1], q_goal)


def test_plan_rejects_start_in_collision(model):
    env = Environment()
    env.add(WorldObject("o", "obstacle", "box", model.ee_position(Q_GENERIC), (0.1, 0.1, 0.1)))
    q_goal = Q_GENERIC + np.deg2rad([90, 0, 0, 0, 0, 0])
    res = plan_joint_path(model, env, Q_GENERIC, q_goal, margin=0.0)
    assert not res.success
    assert "start" in res.message.lower()


def test_plan_rejects_configuration_outside_limits(model):
    q_bad = Q_GENERIC.copy()
    q_bad[2] = -0.5  # J3 lower limit is 0
    res = plan_joint_path(model, Environment(), Q_GENERIC, q_bad)
    assert not res.success
    assert "limit" in res.message.lower()


def test_plan_goes_around_an_obstacle(model):
    q_start = Q_GENERIC
    q_goal = Q_GENERIC + np.deg2rad([40, -10, 10, 20, 10, 30])
    # edge_free() checks the straight segment at n samples (default resolution 0.1 rad).
    n = int(np.ceil(np.max(np.abs(q_goal - q_start)) / 0.1))

    probe = CollisionChecker(model, Environment(), margin=0.0)
    scenario = None
    for k in sorted(range(1, n), key=lambda k: abs(k - n / 2)):
        q_k = q_start + (q_goal - q_start) * (k / n)
        tip = probe.sample_points(q_k)[-1]  # last sample point = tool tip
        env = Environment()
        env.add(WorldObject("block", "obstacle", "box", tuple(tip), (0.06, 0.06, 0.06)))
        checker = CollisionChecker(model, env, margin=0.0)
        if not checker.in_collision(q_start) and not checker.in_collision(q_goal):
            scenario = (env, checker)
            break
    assert scenario is not None, "no obstacle placement blocks only the direct edge"
    env, checker = scenario
    assert not checker.edge_free(q_start, q_goal)

    res = plan_joint_path(model, env, q_start, q_goal, margin=0.0, max_iters=3000, seed=0)
    assert res.success, res.message
    assert np.allclose(res.path[0], q_start) and np.allclose(res.path[-1], q_goal)
    assert all(model.within_limits(q) for q in res.path)
    assert _edges_are_free(checker, res.path)


def test_plan_trajectory_returns_trajectory(model):
    q_goal = Q_GENERIC + np.deg2rad([20, 5, 10, 10, 5, 10])
    res, traj = plan_trajectory(model, Environment(), Q_GENERIC, q_goal, dt=0.05, v_max=1.0)
    assert res.success
    assert traj is not None and traj.shape[1] == model.n
    assert np.allclose(traj[0], Q_GENERIC) and np.allclose(traj[-1], q_goal)


def test_plan_trajectory_returns_none_on_failure(model):
    q_bad = Q_GENERIC.copy()
    q_bad[2] = -0.5
    res, traj = plan_trajectory(model, Environment(), Q_GENERIC, q_bad)
    assert not res.success and traj is None