"""
rewards.py — the reward terms, and the weight on each.

Every term here is either a thing we want (velocity tracking) or a thing the
robot cannot afford (torque, joint speed past the servo's limit, a foot skidding
under load). The weights are the part of this tree with the least evidence behind
them: they are the published quadruped defaults, rescaled for a 2.5 kg robot on a
4.71 rad/s servo. They are guesses, they are labelled as guesses, and unlike the
model parameters there is no bench that can ever measure them — the only honest
way to change one is to run eval.py before and after and read the numbers.

Why there are two tracking shapes and not one
---------------------------------------------
`tracking_*` go through exp(-err^2/sigma), which is bounded, forgiving, and the
published default.  It is also FLAT AT ZERO by construction: d/dx exp(-x^2/s) =
-2x/s * exp(...), which vanishes as the error does, so a small persistent bias
costs almost nothing no matter what sigma is.  Measured on b9c7a73's policy at
its 0.055 rad/s yaw bias: 0.0096 of reward per step, 4.8 points of a ~720
episode, 0.67 %.  Narrowing sigma does not fix the shape -- 0.25 -> 0.15 buys
1.65x of gradient there, 0.25 -> 0.10 buys 2.5x, and both still go to zero.

That is how this robot has now failed twice.  9c2d5cc crabbed at +0.088 m/s
sideways for 3 % of return; eff9916 gave the policy a signal for that, it fixed
it, and the bias moved to yaw where it cost 0.67 %.  A bounded reward will keep
paying for whichever axis is cheapest.

`bias_lin` is L1 on the linear error.  |x| has constant gradient everywhere
including at zero, which is the property exp lacks, and it worked: body-frame
slip went 317 -> 211 mm across the run that introduced it.

`bias_ang` was the same idea on yaw and it FAILED, in a way worth keeping written
down.  Measured on the two recorded policies, 10 s rollouts, commanded straight:

    yaw drift (mean w)            obsA -0.055     bias +0.098    true ratio 1.78x
    yaw swing (sd w)                   0.520           0.441
    |w| instantaneous                  0.4034          0.3558         1.13x
    |integrated heading error|         0.2933          0.5184         1.77x

The instantaneous penalty is not merely insensitive to the drift.  It is
ANTI-correlated with it: obsA drifts half as much and scores WORSE, because its
gait swings the body harder and |w| is dominated by the swing, not the offset.
A policy could lower that penalty by drifting more as long as it swung less, and
that is what happened -- yaw went -0.055 -> +0.098 rad/s and the heading
contribution to sideways travel went 1583 -> 2142 mm over 10 s.

The lesson is about what the term measures, not about L1.  Drift is a
LOW-FREQUENCY property, and a penalty on an instantaneous quantity whose swing is
4-5x the offset spends its gradient on the swing.  `heading_drift` integrates the
rate error instead -- which IS the heading error, exactly, with no filter to tune
-- and reproduces the true drift ratio to 1.77x against 1.78x.  Integrating also
removes the wrap-around a quaternion heading would need, since the reference
heading over an episode with a turn command runs to 10 rad.

Heading is privileged and that is allowed here: the reward is computed in the
simulator.  It must never reach the observation, where yaw is not measurable --
walk.py's docstring is the authority on that.

Two of them are NOT free choices and must not be tuned away:

  joint_vel     the CAD reports joint_velocity_limit = 4.71 rad/s, which is the
                servo's no-load speed at 12 V. A policy that commands past it is
                writing cheques the hardware cannot cash, and in sim it simply
                gets them — actuator.py's back-EMF makes the torque fall off but
                MuJoCo will still integrate whatever the leg's momentum does.
  joint_limit   the SOFT limits. rl/CLAUDE.md: the three ladders mean three
                different things. Hitting the hard ROM limit in sim is a
                simulated part collision; hitting it on the robot is a real one.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class Weights:
    # what we want
    tracking_lin_vel: float = 1.5
    # 0.8 -> 1.5: yaw is one of three commanded axes and was weighted half of the
    # other two, for no reason this tree can point at.
    tracking_ang_vel: float = 1.5
    # the unsaturated half of tracking; see the docstring
    bias_lin: float = -0.2
    # accumulated heading error. Replaces bias_ang, which was measured to point
    # the wrong way. The quantity is a LEAKY integral (walk.py, HEADING_TAU) and
    # runs 0.17 for a policy drifting like eff9916 and 0.26 for one drifting like
    # 73e25d3, so -0.20 costs them ~2.4 % and ~3.6 % of an episode against ~0 for
    # one that holds its heading.
    heading_drift: float = -0.20
    # what a trot must not do
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = -5.0
    base_height: float = -1.0
    # what the hardware cannot afford
    torque: float = -2.0e-4
    action_rate: float = -1.0e-2
    joint_vel: float = -1.0
    joint_limit: float = -1.0
    # gait shaping
    feet_air_time: float = 0.2
    foot_slip: float = -0.1
    stand_still: float = -0.5
    # the end
    termination: float = -1.0

    # tracking sharpness: exp(-err^2 / sigma). Not a weight — a width. 0.25 puts
    # the reward at 1/e when the tracking error is 0.5 m/s, which on a robot
    # whose top command is 0.8 m/s is a forgiving but not meaningless target.
    tracking_sigma: float = 0.25
    # Angular gets its own, because it was sharing a width with a quantity in
    # different units: yaw commands span +-1.0 rad/s. 0.15 is a modest tightening
    # and is NOT the fix — bias_ang is. Both are here because they act in
    # different places: the width where the error is large, the L1 where it is
    # small.
    tracking_sigma_ang: float = 0.15

    def asdict(self) -> dict:
        widths = ("tracking_sigma", "tracking_sigma_ang")
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)
                if f.name not in widths}


def terms(*, cmd, lin_vel_b, ang_vel_b, gravity_b, base_z, stance_z,
          qpos_j, qvel_j, tau, action, last_action, vel_limit, soft_lo, soft_hi,
          air_time, first_contact, foot_vel_xy, in_contact, heading_err,
          done, dt) -> dict:
    """Every reward term, unweighted. Keys match Weights' field names.

    All velocities are in the BODY frame, which is the frame the command is given
    in and the only frame the robot can measure itself in.
    """
    cmd_xy, cmd_yaw = cmd[:2], cmd[2]
    still = jnp.linalg.norm(cmd) < 0.05          # "stand" is a command, not an absence

    lin_err = jnp.sum((cmd_xy - lin_vel_b[:2]) ** 2)
    ang_err = (cmd_yaw - ang_vel_b[2]) ** 2

    # past the servo's no-load speed. Hinge, not quadratic-everywhere: below the
    # limit there is nothing to discourage.
    over_speed = jnp.clip(jnp.abs(qvel_j) - vel_limit, 0.0, None)
    # past the soft limits, same shape.
    over_lo = jnp.clip(soft_lo - qpos_j, 0.0, None)
    over_hi = jnp.clip(qpos_j - soft_hi, 0.0, None)

    return {
        "tracking_lin_vel": lin_err,          # weighted through exp() below
        "tracking_ang_vel": ang_err,
        # The same two errors, L1 and unsaturated. Not a duplicate: these carry
        # gradient where the exp terms have none, which is exactly at the small
        # persistent offsets that have twice survived a whole training run.
        "bias_lin": jnp.sum(jnp.abs(cmd_xy - lin_vel_b[:2])),
        # How far off the commanded heading the robot has drifted over the last
        # few seconds. The env carries the leaky integral; this only reads it.
        "heading_drift": jnp.abs(heading_err),
        "lin_vel_z": lin_vel_b[2] ** 2,
        "ang_vel_xy": jnp.sum(ang_vel_b[:2] ** 2),
        "orientation": jnp.sum(gravity_b[:2] ** 2),
        "base_height": (base_z - stance_z) ** 2,
        "torque": jnp.sum(tau ** 2),
        "action_rate": jnp.sum((action - last_action) ** 2),
        "joint_vel": jnp.sum(over_speed ** 2),
        "joint_limit": jnp.sum(over_lo ** 2 + over_hi ** 2),
        # air time is only worth anything at the moment a foot lands: rewarding it
        # continuously pays a robot for holding a leg up forever.
        "feet_air_time": jnp.sum((air_time - 0.2) * first_contact),
        "foot_slip": jnp.sum(foot_vel_xy ** 2 * in_contact[:, None]),
        # standing still is a skill, and without this the policy fidgets in place
        # because fidgeting is free under a velocity-tracking reward at zero.
        "stand_still": jnp.where(still, jnp.sum(jnp.abs(action)), 0.0),
        "termination": done,
    }


def total(unweighted: dict, w: Weights) -> tuple[jax.Array, dict]:
    """Weighted sum. The two tracking terms go through exp() first.

    The rest are penalties and enter linearly, so a term that blows up can
    dominate — which is intentional and is how a policy learns that the servo's
    speed limit is not negotiable, but it does mean a NaN anywhere shows up as a
    reward of -inf rather than as a wrong gait. eval.py reports the terms
    separately for exactly that reason.
    """
    r = dict(unweighted)
    r["tracking_lin_vel"] = jnp.exp(-r["tracking_lin_vel"] / w.tracking_sigma)
    r["tracking_ang_vel"] = jnp.exp(-r["tracking_ang_vel"] / w.tracking_sigma_ang)
    weights = w.asdict()
    scaled = {k: weights[k] * v for k, v in r.items()}
    return sum(scaled.values()), scaled
