"""
rewards.py — the reward terms, and the weight on each.

Every term here is either a thing we want (velocity tracking) or a thing the
robot cannot afford (torque, joint speed past the servo's limit, a foot skidding
under load). The weights are the part of this tree with the least evidence behind
them: they are the published quadruped defaults, rescaled for a 2.5 kg robot on a
4.7 rad/s servo. They are guesses, they are labelled as guesses, and unlike the
model parameters there is no bench that can ever measure them — the only honest
way to change one is to run eval.py before and after and read the numbers.

Two of them are NOT free choices and must not be tuned away:

  joint_vel     the CAD reports joint_velocity_limit = 4.7 rad/s, which is the
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
    tracking_ang_vel: float = 0.8
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

    def asdict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)
                if f.name != "tracking_sigma"}


def terms(*, cmd, lin_vel_b, ang_vel_b, gravity_b, base_z, stance_z,
          qpos_j, qvel_j, tau, action, last_action, vel_limit, soft_lo, soft_hi,
          air_time, first_contact, foot_vel_xy, in_contact, done, dt) -> dict:
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
    r["tracking_ang_vel"] = jnp.exp(-r["tracking_ang_vel"] / w.tracking_sigma)
    weights = w.asdict()
    scaled = {k: weights[k] * v for k, v in r.items()}
    return sum(scaled.values()), scaled
