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
