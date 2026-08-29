"""FK/IK round-trip and gait sanity checks. Runs without ROS."""
import json, os, math, sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
PARAMS = os.path.abspath(os.path.join(HERE, '..', '..',
                                      'smalldog_description', 'robot_params.json'))

from smalldog_walker.leg_kinematics import LegKinematics
from smalldog_walker.gait import TrotGait


def params():
    return json.load(open(PARAMS))


def make_leg(leg='fl'):
    p = params()
    mm = 1e-3
    dy, dz = p['hip_to_pitch_mm'][leg][1] * mm, p['hip_to_pitch_mm'][leg][2] * mm
    return LegKinematics(dy, dz, p['l_thigh_mm'] * mm, p['l_shin_mm'] * mm)


@pytest.mark.parametrize('leg', ['fl', 'fr', 'rl', 'rr'])
def test_fk_ik_roundtrip(leg):
    k = make_leg(leg)
    for q1 in (-0.4, 0.0, 0.4):
        for q2 in (-0.6, -0.35, 0.2):
            for q3 in (0.4, 0.75, 1.2):
                p = k.fk((q1, q2, q3))
                r = k.ik(p, prev=(q1, q2, q3))
                back = k.fk(r)
                assert max(abs(a - b) for a, b in zip(p, back)) < 1e-6


def test_stance_matches_generated_height():
    p = params()
    k = make_leg('fl')
    q = (0.0, p['stance_rad']['pitch'], p['stance_rad']['knee'])
    foot = k.fk(q)
    z = -(foot[2]) + p['hip_xyz_mm'][ 'fl'][2] * 1e-3
    expected = p['stance_base_height_m'] - p['foot_r_mm'] * 1e-3 - 0.003
    assert abs(z - expected) < 1e-6


def test_gait_stays_in_joint_limits():
    p = params()
    g = TrotGait(p)
    lim = p['joint_limits_rad']
    dt = 0.01
    for i in range(400):
        q = g.joint_targets(dt, 0.25, 0.05, 0.8)
        for name, v in zip(g.joint_names, q):
            kind = name.split('_')[1]
            assert abs(v) <= lim[kind] + 1e-6, f'{name}={v:.3f} exceeds {lim[kind]}'


def test_gait_pushes_backwards_when_walking_forward():
    """during stance the foot must travel backwards in the body frame"""
    g = TrotGait(params())
    dt = 0.005
    for _ in range(60):
        g.foot_targets(dt, 0.2, 0.0, 0.0)
    prev = g.foot_targets(dt, 0.2, 0.0, 0.0)
    stance_deltas = []
    for _ in range(40):
        cur = g.foot_targets(dt, 0.2, 0.0, 0.0)
        for leg in g.legs:
            if abs(cur[leg][2] - (-g.body_height)) < 1e-9:      # on the ground
                stance_deltas.append(cur[leg][0] - prev[leg][0])
        prev = cur
    assert stance_deltas
    assert sum(stance_deltas) / len(stance_deltas) < 0


# --- heading hold -------------------------------------------------------------------
def quat_yaw(a):
    """body quaternion for a yaw of `a` radians, level."""
    return (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))


def walking_gait(yaw=0.0, wzg=0.0):
    """a gait that has been walking long enough for `_moving` to be up, fed a heading."""
    g = TrotGait(params())
    for _ in range(200):
        g.feedback(quat=quat_yaw(yaw), gyro=(0.0, 0.0, wzg))
        g.foot_targets(0.01, 0.20, 0.0, 0.0)
    return g


def test_heading_hold_turns_back_towards_the_reference():
    # latched straight ahead, then knocked 15 deg to the left: the correction must be
    # negative (turn right).  Getting this sign backwards is a divergent loop, and the
    # equivalent mistake in the roll term of _level put the robot on its back in a second.
    g = walking_gait()
    assert g._yaw_ref is not None
    g.feedback(quat=quat_yaw(g._yaw_ref + math.radians(15)))
    assert g._heading(0.0, True) < 0.0
    g.feedback(quat=quat_yaw(g._yaw_ref - math.radians(15)))
    assert g._heading(0.0, True) > 0.0


def test_heading_hold_takes_the_short_way_round():
    # a reference just past +180 and a body just short of -180 are 10 deg apart, not 350
    g = walking_gait()
    g._yaw_ref = math.radians(175)
    g.feedback(quat=quat_yaw(math.radians(-175)))
    c = g._heading(0.0, True)
    assert c < 0.0 and abs(c) < g.yaw_kp * math.radians(20)


def test_heading_hold_lets_a_commanded_turn_through():
    g = walking_gait()
    g.feedback(quat=quat_yaw(g._yaw_ref + math.radians(15)))
    assert g._heading(1.0, True) == 0.0          # operator is turning: do not fight it
    assert g._yaw_ref is None                    # ... and the reference is dropped
    assert g._heading(0.0, False) == 0.0         # stale feedback: blind gait


def test_heading_hold_is_clamped():
    # a large error must not turn into a large turn command
    g = walking_gait()
    g._yaw_ref = math.radians(170)
    g.feedback(quat=quat_yaw(0.0), gyro=(0.0, 0.0, 3.0))
    assert abs(g._heading(0.0, True)) <= g.yaw_max + 1e-9
