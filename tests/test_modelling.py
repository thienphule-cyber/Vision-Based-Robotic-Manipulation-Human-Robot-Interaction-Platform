"""Unit tests for modelling.py (run from the project root: pytest tests/test_modelling.py -v)."""
import copy
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from spatialmath import SE3

import modelling
from modelling import (
    FINGER_NAMES,
    GRASP_PRESETS,
    RV2FR_DH,
    Environment,
    HandModel,
    RobotModel,
    SystemModel,
    WorldObject,
    check_visual_sync,
    load_visual_model,
)


# ----------------------------------------------------------------------------
# Robot model
# ----------------------------------------------------------------------------
@pytest.fixture
def model():
    return RobotModel()


def test_robot_has_six_joints(model):
    assert model.n == 6
    assert model.joint_names == ("J1", "J2", "J3", "J4", "J5", "J6")


def test_joint_limits_match_nominal_table(model):
    expected = np.deg2rad(np.array([row["qlim_deg"] for row in RV2FR_DH])).T
    assert model.qlim.shape == (2, 6)
    assert np.allclose(model.qlim, expected)


def test_home_is_within_limits(model):
    assert model.within_limits(model.q_home)


def test_within_limits_and_clip(model):
    q = np.zeros(6)
    q[2] = -0.1  # J3 lower limit is 0
    assert not model.within_limits(q)
    clipped = model.clip(q)
    assert model.within_limits(clipped)
    assert clipped[2] == pytest.approx(0.0)


def test_wrong_joint_count_raises(model):
    with pytest.raises(ValueError):
        model.fkine(np.zeros(5))
    with pytest.raises(ValueError):
        model.clip(np.zeros(7))


def test_fkine_returns_valid_se3(model):
    T = model.fkine(model.q_home)
    assert isinstance(T, SE3)
    R = np.asarray(T.R)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_random_q_is_within_limits_and_reproducible(model):
    q1 = model.random_q(np.random.default_rng(1))
    q2 = model.random_q(np.random.default_rng(1))
    assert model.within_limits(q1)
    assert np.allclose(q1, q2)


def test_end_effector_never_exceeds_max_reach(model):
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = model.random_q(rng)
        assert np.linalg.norm(model.ee_position(q)) <= model.max_reach + 1e-9


def test_base_translation_shifts_end_effector():
    q = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.1])
    p0 = RobotModel().ee_position(q)
    p1 = RobotModel(base=SE3(0.1, 0.2, 0.3)).ee_position(q)
    assert np.allclose(p1 - p0, [0.1, 0.2, 0.3], atol=1e-9)


def test_tool_offset_moves_end_effector():
    q = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.1])
    p0 = RobotModel().ee_position(q)
    p1 = RobotModel(tool=SE3.Tz(0.1)).ee_position(q)
    assert np.linalg.norm(p1 - p0) == pytest.approx(0.1, abs=1e-9)


# ----------------------------------------------------------------------------
# Visual model loading and synchronisation
# ----------------------------------------------------------------------------
def test_sync_ok_for_identical_models(model):
    visual = RobotModel().robot
    report = check_visual_sync(visual, model)
    assert report.ok
    assert report.n_joints_match and report.limits_match
    assert report.max_position_error < 1e-9


def test_sync_detects_missing_visual(model):
    report = check_visual_sync(None, model)
    assert not report.ok
    assert report.messages


def test_sync_detects_joint_count_mismatch(model):
    visual = RobotModel(dh_table=RV2FR_DH[:5]).robot
    report = check_visual_sync(visual, model)
    assert not report.ok
    assert not report.n_joints_match


def test_sync_detects_limit_mismatch(model):
    table = copy.deepcopy(RV2FR_DH)
    table[0]["qlim_deg"] = (-100.0, 100.0)
    report = check_visual_sync(RobotModel(dh_table=table).robot, model)
    assert not report.ok
    assert not report.limits_match


def test_sync_detects_base_frame_mismatch(model):
    visual = RobotModel(base=SE3.Tx(0.1)).robot
    report = check_visual_sync(visual, model)
    assert not report.ok
    assert report.max_position_error == pytest.approx(0.1, abs=1e-6)


def test_load_visual_model_returns_none_when_package_missing(monkeypatch):
    def fake_import(name):
        raise ImportError("not installed")

    monkeypatch.setattr(modelling.importlib, "import_module", fake_import)
    assert load_visual_model() is None


def test_load_visual_model_uses_first_available_class(monkeypatch):
    class FakeRobot:
        n = 6

    fake_module = types.SimpleNamespace(MitsubishiRV2RF=FakeRobot)
    monkeypatch.setattr(modelling.importlib, "import_module", lambda name: fake_module)
    assert isinstance(load_visual_model(), FakeRobot)


def test_real_visual_model_if_installed():
    pytest.importorskip("ir_support_extra_robots")
    visual = load_visual_model()
    if visual is None:
        pytest.skip("Visual model class could not be instantiated")
    assert visual.n == 6


# ----------------------------------------------------------------------------
# Robotic hand
# ----------------------------------------------------------------------------
@pytest.fixture
def hand():
    return HandModel()


def test_hand_has_five_fingers_with_multiple_joints(hand):
    assert tuple(hand.fingers) == FINGER_NAMES
    assert all(f.n_joints >= 3 for f in hand.fingers.values())
    assert hand.n_joints == 15


def test_hand_starts_open(hand):
    assert all(f.flexion == pytest.approx(0.0) for f in hand.fingers.values())


def test_set_finger_clips_to_limits(hand):
    q = hand.set_finger("index", [10.0, 10.0, 10.0])
    assert np.allclose(q, hand.finger("index").qlim[:, 1])
    q = hand.set_finger("index", [-1.0, -1.0, -1.0])
    assert np.allclose(q, 0.0)


def test_set_finger_wrong_size_raises(hand):
    with pytest.raises(ValueError):
        hand.set_finger("index", [0.0, 0.0])


def test_unknown_finger_raises(hand):
    with pytest.raises(KeyError):
        hand.finger("tail")


def test_flexion_fraction_roundtrip(hand):
    hand.set_finger_flexion("middle", 0.5)
    assert hand.finger("middle").flexion == pytest.approx(0.5)
    hand.set_finger_flexion("middle", 5.0)  # clipped to 1.0
    assert hand.finger("middle").flexion == pytest.approx(1.0)


@pytest.mark.parametrize("preset", list(GRASP_PRESETS))
def test_grasp_presets_apply_expected_flexion(hand, preset):
    hand.apply_grasp(preset)
    for name, finger in hand.fingers.items():
        assert finger.flexion == pytest.approx(GRASP_PRESETS[preset].get(name, 0.0))


def test_unknown_grasp_raises(hand):
    with pytest.raises(ValueError):
        hand.apply_grasp("wave")


def test_closed_hand_fingertips_are_closer_to_finger_base(hand):
    hand.open_hand()
    open_dist = {n: np.linalg.norm(p - hand.finger(n).base_offset)
                 for n, p in hand.fingertip_positions().items()}
    hand.close_hand()
    closed_dist = {n: np.linalg.norm(p - hand.finger(n).base_offset)
                   for n, p in hand.fingertip_positions().items()}
    for name in FINGER_NAMES:
        assert closed_dist[name] < open_dist[name]


def test_open_finger_length_equals_sum_of_links(hand):
    hand.open_hand()
    for name, tip in hand.fingertip_positions().items():
        finger = hand.finger(name)
        assert np.linalg.norm(tip - finger.base_offset) == pytest.approx(sum(finger.link_lengths))


def test_fingertips_follow_mount_transform(hand):
    local = hand.fingertip_positions()
    T = SE3(1.0, 2.0, 3.0)
    world = hand.fingertip_positions(T)
    for name in FINGER_NAMES:
        assert np.allclose(world[name] - local[name], [1.0, 2.0, 3.0])


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
def test_add_get_remove_object():
    env = Environment()
    env.add(WorldObject("c", "object", "box", (0, 0, 0), (0.1, 0.1, 0.1)))
    assert "c" in env and len(env) == 1
    assert env.get("c").name == "c"
    env.remove("c")
    assert "c" not in env


def test_duplicate_name_raises():
    env = Environment()
    env.add(WorldObject("c", "object", "sphere", (0, 0, 0), (0.1,)))
    with pytest.raises(ValueError):
        env.add(WorldObject("c", "object", "sphere", (1, 0, 0), (0.1,)))


def test_unknown_object_raises():
    env = Environment()
    with pytest.raises(KeyError):
        env.get("nope")
    with pytest.raises(KeyError):
        env.remove("nope")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kind="bad", shape="box", dims=(1, 1, 1)),
        dict(kind="object", shape="pyramid", dims=(1, 1, 1)),
        dict(kind="object", shape="box", dims=(1, 1)),
        dict(kind="object", shape="sphere", dims=(-0.1,)),
    ],
)
def test_invalid_world_object_raises(kwargs):
    with pytest.raises(ValueError):
        WorldObject("x", position=(0, 0, 0), **kwargs)


def test_aabb_for_each_shape():
    box = WorldObject("b", "object", "box", (1, 2, 3), (0.2, 0.4, 0.6))
    lo, hi = box.aabb()
    assert np.allclose(hi - lo, [0.2, 0.4, 0.6])

    sph = WorldObject("s", "object", "sphere", (0, 0, 0), (0.5,))
    lo, hi = sph.aabb()
    assert np.allclose(hi - lo, [1.0, 1.0, 1.0])

    cyl = WorldObject("c", "object", "cylinder", (0, 0, 0), (0.5, 2.0))
    lo, hi = cyl.aabb()
    assert np.allclose(hi - lo, [1.0, 1.0, 2.0])


def test_aabb_accounts_for_yaw():
    box = WorldObject("b", "object", "box", (0, 0, 0), (0.2, 0.1, 0.1), yaw=np.pi / 2)
    lo, hi = box.aabb()
    assert np.allclose(hi - lo, [0.1, 0.2, 0.1], atol=1e-9)


def test_contains_point_with_margin():
    box = WorldObject("b", "obstacle", "box", (0, 0, 0), (1, 1, 1))
    assert box.contains_point((0.4, 0.0, 0.0))
    assert not box.contains_point((0.6, 0.0, 0.0))
    assert box.contains_point((0.6, 0.0, 0.0), margin=0.2)


def test_default_workcell_contents():
    env = Environment.default_workcell()
    assert len(env.by_kind("table")) == 1
    assert len(env.by_kind("obstacle")) >= 1
    assert len(env.by_kind("target_zone")) == 1
    assert len(env.by_kind("storage_zone")) == 1
    shapes = {o.shape for o in env.by_kind("object")}
    assert {"box", "sphere", "cylinder"} <= shapes
    assert all(o.graspable for o in env.by_kind("object"))


def test_point_collision_with_table_and_obstacle():
    env = Environment.default_workcell()
    assert env.point_in_collision((0.3, 0.0, -0.01))              # inside the table
    assert "obstacle_1" in env.colliding_objects((0.55, 0.0, 0.1))
    assert not env.point_in_collision((0.3, 0.0, 0.5))            # free space above the table


def test_zones_are_not_treated_as_collisions():
    env = Environment.default_workcell()
    assert not env.point_in_collision((0.4, -0.25, 0.1))


def test_find_zone():
    env = Environment.default_workcell()
    zone = env.find_zone((0.4, -0.25, 0.1), kind="target_zone")
    assert zone is not None and zone.name == "target_zone"
    assert env.find_zone((0.35, 0.2, 0.025), kind="target_zone") is None


def test_environment_to_dict_is_json_serialisable():
    data = Environment.default_workcell().to_dict()
    assert len(data["objects"]) == 7
    json.dumps(data)


# ----------------------------------------------------------------------------
# System container
# ----------------------------------------------------------------------------
def test_build_default_without_visual():
    system = SystemModel.build_default(load_visual=False)
    assert system.visual is None
    summary = system.summary()
    assert summary["n_joints"] == 6
    assert summary["hand_joints"] == 15
    assert summary["visual_loaded"] is False
    assert "table" in summary["environment_objects"]


def test_check_sync_without_visual_is_not_ok():
    system = SystemModel.build_default(load_visual=False)
    assert not system.check_sync().ok