"""
walk.py — the training environment: velocity-command walking for the 12-DOF
SmallDog, in MJX.

    from env import Walk
    env = Walk(terrain=False, n_boxes=24)

What the policy sees, and why it is only this
---------------------------------------------
The observation is restricted to what the robot can actually measure at 50 Hz
over its own bus and off one IMU. Every entry below exists on the hardware:

    projected gravity   3   from the IMU. NOT the quaternion: the robot has no
                            magnetometer (the MJCF says so — imu_quat is marked
                            "sim only"), so yaw is not observable and must not
                            appear anywhere in the observation. A policy that
                            learns to use absolute yaw learns something that will
                            be a slow drift on the real robot.
    gyro                3   angular velocity, straight off the BMI088
    accelerometer       3   proper acceleration, the BMI088's other half
    joint position     12   relative to the CAD stance, as the bus reports it
    joint velocity     12   differenced on the bus, hence the scaling below
    last action        12   what we asked for last tick; the robot knows this
    command             3   vx, vy, yaw rate — what we are asking of it
                       --
                       48   x OBS_HIST frames = the observation

Why the accelerometer, and why a stack of frames
------------------------------------------------
Both exist to answer one question the policy could not previously ask: how fast
is the body moving? Linear velocity is not measurable on this robot and so is
not in the frame -- but it is not unknowable either, and the first policy that
walked showed the cost of pretending otherwise. Measured on that policy: a
persistent +0.088 m/s of body-frame lateral velocity under a straight-ahead
command, 72 % of a 1229 mm sideways excursion over 10 s. A drift the controller
has no signal for is a drift it cannot correct, however hard the reward pushes;
all PPO can do without one is find a feed-forward gait that happens to average
out, and it did not.

The accelerometer is the direct signal and the robot already carries it -- the
BMI088 is six axes, and only three of them were being read. It is not clean:
`checks/imu_placement.py` measures that a board offset by r from the site reads
w x (w x r) + a x r on top of gravity, which is 25 degrees of apparent tilt at
the mount 1046e06 put in the CAD, correlated with the policy's own actions. That
is an argument for handing the network the raw channel and a history to
difference it against, not for integrating it by hand into a number that would
then have to be reproduced on the robot.

OBS_HIST frames at 50 Hz span 100 ms, which is one full cycle of the gait the
first policy learned. Leg odometry lives in that window: a stance leg's joint
velocities carry the body's, and the frame stack is what lets a feed-forward MLP
see which legs those are.

Deliberately still absent: base linear velocity itself (privileged -- no
estimator exists, and one would have to be written twice, here and on the
robot), foot contact booleans (the touch sensors exist in sim and the robot has
none; `smalldog_walker/contact.py` infers it from servo load and that inference
is not free), absolute height, and anything about the terrain. The rewards may
use all of those — the reward is computed in the simulator, at training time,
where privileged information is free. The observation may not.

The mount is no longer the open question it was. `checks/imu_placement.py`
measured the old `imu` site at the base_link origin, where no board physically
fits, and rl/CLAUDE.md required the mount to reach the CAD before the observation
was frozen; 1046e06 put it there and both exporters read it. It has MOVED since,
which is the point of not writing the number here any more: the CAD put the
battery in a case, the IMU went from under the deck to on top of it, and the site
went 23.4 -> 31.0 mm (3d/CLAUDE.md, 2026-09-09). `model.build()` prints the live
`site_pos` in its build notes, and train_ppo.py logs those, so the height a run
was trained at is in the run's own record rather than in a docstring that goes
stale in silence. This env reads whatever site the generated model calls `imu`;
when that site moves, the policy is retrained, not patched. That matters more now
than it did, because the accelerometer is in the observation and it is the
channel the offset corrupts.

The servo is in the loop, not around it
---------------------------------------
The action is a joint TARGET, exactly as it is on the robot: the policy writes a
position to the bus and the servo's own loop decides what current flows. So the
torque is recomputed from `actuator.py` at every 1 ms physics step, not held for
the whole 20 ms control tick. Holding it would make the servo a torque source and
delete the two things that make it a servo — the back-EMF speed limit and the
falling torque ceiling as the pack drains.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

import actuator
import model as model_mod
from env import rewards as rw

CTRL_HZ = 50.0                 # robot/README.md, "The 50 Hz budget"
ACTION_SCALE = 0.35            # rad per unit action, before the soft-limit clip

# Observation scalings. Not tuning: they put each block within about +-1 so the
# running normaliser starts from something sane rather than learning the scale.
OBS_SCALE_GYRO = 0.25
OBS_SCALE_QVEL = 0.10
# Gravity alone is 9.81 and imu_placement.py's artefact adds up to 9 more, so
# 0.1 puts the channel in the same O(1) band as the rest of the frame.
OBS_SCALE_ACCEL = 0.10

# Frames of history in the observation, newest first. 5 at 50 Hz is 100 ms.
OBS_HIST = 5

# The frame, term by term, in assemble_obs's own order. Written as a sum and not
# as a number because the number was hardcoded once and outlived the observation
# it described: `observation_size` said 45 while reset returned 240, which builds
# a network with the wrong input width and loses whatever the run cost. The
# assert in assemble_obs is the other half — this constant may not drift from the
# concatenate below without the first frame of the first episode saying so.
OBS_FRAME = 3 + 3 + 3 + 12 + 12 + 12 + 3
OBS_SIZE = OBS_FRAME * OBS_HIST

# Time constant of the heading-error integral the reward reads (rewards.py,
# heading_drift). It LEAKS, and the leak was once the only thing that made the
# quantity safe.
#
# brax's AutoResetWrapper.step resets `pipeline_state` and `obs` and NOTHING
# ELSE: a custom field in `info` survives the episode boundary. Every other
# field here was either overwritten every step (foot_xy, last_action) or clamped
# by contact (air_time), so none of them noticed. A plain integral would not
# have been: it would accumulate across every episode in a rollout and grow
# without bound, which is exactly what it did -- ep_len fell to 397 against the
# 465-480 of every previous run.
#
# Resetting it on this env's own `done` was not enough either. Truncation at
# episode_length happens in EpisodeWrapper, OUTSIDE this step(), and with ep_len
# near 480 of 500 truncation is the COMMON ending, not the rare one.
#
# `_new_episode` below now sees truncation too — EpisodeWrapper writes
# `info["episode_done"]` and this env reads it — so the integral IS zeroed at
# every boundary and the leak is no longer load-bearing. It stays anyway, and on
# its own merits: drift is a LOW-FREQUENCY property (rewards.py, bias_ang), and
# a leak with tau = 5 s is the low-pass that makes it one. Measured cost of the
# choice, on the two recorded policies: a plain integral separates their drift
# 1.77x against a true 1.78x, and this separates it 1.55x, flat in tau from 2 s
# to 10 s. Keeping 14 % of fidelity on the table for a quantity that cannot be
# broken by wrapper semantics was the right trade when the wrapper was the only
# defence, and it is a cheap one now that it is the second.
HEADING_TAU = 5.0


def rotate_inv(q, v, xp=jnp):
    """Rotate v from the world frame into the body frame given body quat q."""
    qw = q[0]
    ax = xp.stack([-q[1], -q[2], -q[3]])            # conjugate
    t = 2.0 * xp.cross(ax, v)
    return v + qw * t + xp.cross(ax, t)


def assemble_obs(*, quat, gyro, accel, qpos_j, qvel_j, stance_j, last_action,
                 command, xp=jnp):
    """ONE FRAME of the observation, from arrays either engine can produce.

    One frame, not the observation: the policy is fed OBS_HIST of these, and
    stacking them is the caller's job because only the caller has the buffer.
    `stack_obs` below is that stacking, stated once so the two callers agree.

    This exists as one function and not two because eval.py's sim-to-sim pass
    steps VANILLA MuJoCo rather than MJX, and an observation that is assembled
    differently there is not a sim-to-sim test — it is a test of two different
    policies. Same reasoning as actuator.py's xp threading: one statement of the
    thing, two backends.
    """
    gravity_b = rotate_inv(quat, xp.asarray([0.0, 0.0, -1.0]), xp)
    obs = xp.concatenate([
        gravity_b,
        gyro * OBS_SCALE_GYRO,
        accel * OBS_SCALE_ACCEL,
        qpos_j - stance_j,
        qvel_j * OBS_SCALE_QVEL,
        last_action,
        command,
    ])
    assert obs.shape[-1] == OBS_FRAME, (
        f"frame is {obs.shape[-1]}, OBS_FRAME says {OBS_FRAME} — one of the two "
        f"moved without the other")
    return obs, gravity_b


def stack_obs(hist, frame, xp=jnp):
    """Push one frame into the history and flatten it into an observation.

    Newest first, so index 0 is always now and the network never has to learn
    which end is which. Returns (observation, new history) — the history is the
    thing the caller has to carry, in `info` under MJX and in a local under
    vanilla MuJoCo.

    On the first frame of an episode there is no past, and the buffer is filled
    with copies of the present rather than with zeros: a zero frame is a robot
    reporting no gravity, which is a state that never occurs and which the
    normaliser would then have to make room for.
    """
    hist = xp.concatenate([frame[None], hist[:-1]])
    return hist.reshape(-1), hist


def init_hist(frame, xp=jnp):
    """The history at reset: OBS_HIST copies of the first frame."""
    return xp.repeat(frame[None], OBS_HIST, axis=0)


def params_obs_width(params):
    """The observation width a saved policy was trained for, or None.

    brax's observation normaliser carries one running mean per element, so the
    shape of that mean IS the width the network expects. Worth reading before
    the first forward pass: a checkpoint from before an observation change
    otherwise fails as a contracting-dimension mismatch several frames deep in
    XLA, which does not say `this policy is older than this observation`.

    Returns None rather than raising if the structure is not what we expect —
    a width check has no business being the thing that breaks a rollout.
    """
    try:
        import numpy as _np
        return int(_np.asarray(params[0].mean).shape[-1])
    except Exception:
        return None


def check_obs_width(params, expected, where=""):
    """Raise with a sentence if a checkpoint predates the current observation."""
    got = params_obs_width(params)
    if got is not None and got != expected:
        raise SystemExit(
            f"\n!! {where}: this checkpoint was trained on a {got}-element "
            f"observation and\n"
            f"!! the environment now builds {expected}. It cannot be loaded, and "
            f"resizing it\n"
            f"!! would not be the same policy. Check out the commit the run "
            f"belongs to, or\n"
            f"!! retrain. rl/env/walk.py's docstring lists what the frame "
            f"contains now.")


def _rotate_inv(q, v):
    return rotate_inv(q, v, jnp)


@dataclasses.dataclass(frozen=True)
class Commands:
    """The velocity commands the policy is asked to track, in the body frame.

    Ranges are what the hardware can plausibly do, not what the sim can: the
    analytic trot in ros2/ makes 0.20 m/s and the sim reaches 0.78 m in a 5 s trot
    on this box. Asking for 2 m/s would train a policy to fall over quickly.

    The top of vx is above what the actuator can deliver, and this is measured,
    not suspected. Commanding a trained policy 0.2 / 0.4 / 0.6 / 0.8 m/s on the
    CPU engine, nominal servo, it answers 0.236 / 0.432 / 0.426 / 0.423 -- pinned
    from 0.4 upward. At the pin the 95th percentile joint speed was 4.51 rad/s
    and did not move between the 0.6 and the 0.8 command. The robot is not
    failing to learn 0.8 m/s, it is geared out of it.

    So roughly (0.8 - 0.43) / 1.2 = 31 % of sampled vx commands are unreachable,
    and about 0.057 m/s of the ~0.184 m/s track_err a run reports is arithmetic
    rather than skill.

    What that 4.51 is 96 % of has since changed twice, and the two numbers now
    disagree, which is the reason to re-measure rather than to narrow this range
    from the paragraph above. params/st3215.json is a FIT (44 runs, three
    voltages) and its derived free speed at 12 V is 5.899 rad/s, so the law would
    let the joint run half as fast again as that p95. The HARDWARE does not:
    robot/bench/noload_speed.py read 3.86 rad/s on a free hub on 2026-09-11, and
    it is a firmware plateau rather than d*U/k_e, which is why the law overshoots
    it and why rewards.py's joint_vel penalty is the only thing holding the
    policy under the real ceiling (PLAN.md 3c). So the pin above sat at 117 % of
    what the servo can actually do and 76 % of what the sim would allow.

    Re-run tools/ceiling.py against the current model before touching this range.
    The 4.51 was measured on a pre-fit actuator and every constant under it has
    moved since; a range narrowed against a stale pin is a range narrowed against
    a servo that no longer exists in either direction.
    """
    vx: tuple = (-0.4, 0.8)
    vy: tuple = (-0.3, 0.3)
    yaw: tuple = (-1.0, 1.0)
    stand_fraction: float = 0.15      # of episodes commanded to stand still

    def sample(self, rng):
        k1, k2, k3, k4 = jax.random.split(rng, 4)
        c = jnp.array([
            jax.random.uniform(k1, (), minval=self.vx[0], maxval=self.vx[1]),
            jax.random.uniform(k2, (), minval=self.vy[0], maxval=self.vy[1]),
            jax.random.uniform(k3, (), minval=self.yaw[0], maxval=self.yaw[1]),
        ])
        stand = jax.random.uniform(k4, ()) < self.stand_fraction
        return jnp.where(stand, jnp.zeros(3), c)


class Walk(PipelineEnv):
    def __init__(self, terrain: bool = False, n_boxes: int = 0,
                 weights: rw.Weights | None = None, commands: Commands | None = None,
                 action_scale: float = ACTION_SCALE, ctrl_hz: float = CTRL_HZ,
                 push: bool = True, obs_noise: float = 0.02, **kw):
        mj_model, self.build_notes = model_mod.build(terrain=terrain, n_boxes=n_boxes, **kw)
        self.mj_model = mj_model
        self.n_boxes = n_boxes
        self.box_geoms = model_mod.box_geom_ids(mj_model, n_boxes) if n_boxes else []
        P = model_mod.robot_params()
        self.P = P

        sys = mjcf.load_model(mj_model)
        n_frames = int(round((1.0 / ctrl_hz) / float(sys.opt.timestep)))
        super().__init__(sys=sys, backend="mjx", n_frames=n_frames)

        self._w = weights or rw.Weights()
        self._cmd = commands or Commands()
        self._action_scale = action_scale
        self._push = push
        self._obs_noise = obs_noise

        qadr, vadr, act = model_mod.joint_order(mj_model, P)
        self._qadr, self._vadr, self._act = jnp.array(qadr), jnp.array(vadr), jnp.array(act)
        # Public, like box_geoms, and for the same reason: env/randomize.py needs
        # to know which twelve of the eighteen dofs have a rotor behind them, and
        # joint_order() is the only thing that does.
        self.joint_dofs = vadr
        lo, hi = model_mod.limits(P, soft=True)
        self._soft_lo, self._soft_hi = jnp.array(lo), jnp.array(hi)
        self._vel_limit = float(P["joint_velocity_limit"])
        self._stance_z = float(P["stance_base_height_m"])

        q0 = model_mod.stance_qpos(mj_model, P)
        self._init_q = jnp.array(q0)
        self._stance_j = jnp.array(q0[qadr])

        self._p0 = actuator.load(quiet=True)
        self._ranges = model_mod.domain_ranges()

        def sensor_adr(name):
            i = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            assert i >= 0, f"sensor {name} missing"
            return int(mj_model.sensor_adr[i]), int(mj_model.sensor_dim[i])

        self._s_quat = sensor_adr("imu_quat")
        self._s_gyro = sensor_adr("imu_gyro")
        # Verified to agree with vanilla MuJoCo to 6e-4 on an identical state:
        # MJX implements this sensor, and the sim-to-sim pass stays a test of
        # the physics rather than of two different observations.
        self._s_accel = sensor_adr("imu_accel")
        self._s_touch = [sensor_adr(f"{leg}_contact") for leg in P["legs"]]
        # Foot-site height at the CAD stance, i.e. the height a foot sits at when
        # it is on flat ground. Subtracting it turns site z into clearance.
        # Measured off the model rather than assumed: the site is not the contact
        # sphere's lowest point and the offset is not the sphere radius.
        _d = mujoco.MjData(mj_model)
        _d.qpos[:] = q0
        mujoco.mj_forward(mj_model, _d)
        self._foot_z0 = jnp.array([
            _d.site_xpos[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE,
                                           f"{leg}_foot_site")][2]
            for leg in P["legs"]])

        self._foot_site = jnp.array([
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site")
            for leg in P["legs"]])

        # One control tick is the delay quantum: the bus either delivers this
        # tick's target or the last one. What sets the odds is MEASURED, and has
        # been since 2026-09-07 -- params/bus_timing.json, written by
        # robot/bench/bus_probe.py over twelve servos. model.domain_ranges()
        # reads the composite out of it: sync_read p50 3.24 + sync_write p50 0.01
        # = 3.25 ms at the median, and their p95s = 8.90 ms at the tail, of a
        # 20 ms tick. The comment this replaces said the file "does not exist",
        # and the range it justified -- a whole flat control period -- both
        # overstated the tail and, truncated to whole ticks, produced a delay of
        # zero on every draw. See model.delay_ticks for how a fraction of a tick
        # becomes a whole one.
        #
        # n_delay is the buffer depth, so it has to cover the WORST band, not the
        # median: one tick per whole tick of latency, plus the one this tick's
        # own target occupies.
        self._delay_s = tuple(self._ranges["bus"]["delay_s_abs"]["range"])
        self._n_delay = 1 + int(np.ceil(self._delay_s[1] * ctrl_hz))

    # ------------------------------------------------------------------ obs
    def _sensor(self, ps, adr_dim):
        adr, dim = adr_dim
        return jax.lax.dynamic_slice(ps.sensordata, (adr,), (dim,))

    def _frame(self, ps, info, rng):
        """One noised frame. The noise goes on the FRAME, before it enters the
        history, because that is where it is on the robot: a stored reading is a
        reading that was already noisy, not one that gets noisy later."""
        quat = self._sensor(ps, self._s_quat)
        gyro = self._sensor(ps, self._s_gyro)
        accel = self._sensor(ps, self._s_accel)
        frame, gravity_b = assemble_obs(
            quat=quat, gyro=gyro, accel=accel, qpos_j=ps.qpos[self._qadr],
            qvel_j=ps.qvel[self._vadr], stance_j=self._stance_j,
            last_action=info["last_action"], command=info["command"], xp=jnp)
        noise = jax.random.uniform(rng, frame.shape, minval=-1.0, maxval=1.0)
        return frame + noise * self._obs_noise, gravity_b, gyro

    # --------------------------------------------------------------- torque
    def _params(self, info):
        """A Params whose fields are this environment's draw. One copy of the law:
        these feed actuator.py's own functions with xp=jnp.

        `J_m` is absent on purpose and keeps its nominal value here. It is
        inertia; MuJoCo owns the mass matrix, no function in actuator.py's array
        API reads it, and it is randomised as `dof_armature` in env/randomize.py
        where the physics can see it. A draw placed here moved nothing at all.
        """
        return dataclasses.replace(
            self._p0,
            k_u=info["k_u"], k_e=info["k_e"], R=info["R"],
            tau_c=info["tau_c"], b_v=info["b_v"], mu_load=info["mu_load"],
            kp=info["kp"],
            deadband=info["deadband"], punch=info["punch"])

    # ---------------------------------------------------------------- reset
    def reset(self, rng: jax.Array) -> State:
        rng, k_cmd, k_q, k_v, k_a, k_obs, k_push = jax.random.split(rng, 7)

        q = self._init_q.at[self._qadr].add(
            jax.random.uniform(k_q, (12,), minval=-0.05, maxval=0.05))
        qd = jnp.zeros(self.sys.nv).at[:6].set(
            jax.random.uniform(k_v, (6,), minval=-0.05, maxval=0.05))
        ps = self.pipeline_init(q, qd)

        info = {"rng": rng, **self._episode_fields(ps, k_a)}
        frame, _, _ = self._frame(ps, info, k_obs)
        hist = init_hist(frame, xp=jnp)
        obs = hist.reshape(-1)
        info["obs_hist"] = hist
        metrics = {k: jnp.zeros(()) for k in self._w.asdict()}
        metrics.update({"vx_body_per_step": jnp.zeros(()),
                        "track_err_xy_per_step": jnp.zeros(())})
        return State(ps, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def _episode_fields(self, ps, rng) -> dict:
        """Everything one EPISODE owns, for a robot standing at `ps`.

        One function and two callers — `reset` and `_new_episode` below — because
        the second of those did not exist for the whole of step 4, and what went
        wrong is exactly what happens when a reset lives in only one of the two
        places an episode can begin. See `_new_episode`.

        `obs_hist` is deliberately NOT here: it needs a frame, a frame needs the
        command, and the command is drawn in this function. The caller stacks it.
        """
        k_cmd, k_a, k_push = jax.random.split(rng, 3)
        gap_lo, gap_hi = self._ranges["push"]["interval_s_abs"]["range"]
        return {
            "command": self._cmd.sample(k_cmd),
            "last_action": jnp.zeros(12),
            "action_buf": jnp.zeros((self._n_delay, 12)),
            "air_time": jnp.zeros(4),
            # leaky integral of (yaw rate - commanded yaw rate): the heading
            # error the last few seconds have accumulated. Reward-side only —
            # yaw stays out of the observation. See HEADING_TAU.
            "heading_err": jnp.zeros(()),
            "foot_xy": ps.site_xpos[self._foot_site][:, :2],
            "step": jnp.array(0, jnp.int32),
            "next_push": jax.random.uniform(k_push, (), minval=gap_lo, maxval=gap_hi),
            **self._sample_episode(k_a),
        }

    def _sample_episode(self, rng):
        """The servo, the pack and the bus, drawn once per episode.

        Per joint where the spread is per-servo — twelve motors out of one bag —
        and per robot where it is not: one pack, one bus. WHICH fields are drawn
        and from which range is `model.EPISODE_DRAW`, walked here with jax keys
        and in `model.sample_actuator_params` with a numpy Generator, so the
        training draw and the draw checks/check_model.py probes cannot be two
        different draws again. The ranges and the evidence behind each are in
        params/domain_rand.json.
        """
        keys = jax.random.split(rng, len(model_mod.EPISODE_DRAW) + 1)

        def uniform(i, lo, hi, shape):
            return jax.random.uniform(keys[i], shape, minval=lo, maxval=hi)

        d = model_mod.sample_episode(uniform, self._ranges, self._p0)
        # Seconds of latency into whole ticks of action buffer. The fractional
        # tick becomes the probability of the extra one — model.delay_ticks says
        # why, and what the arithmetic that shipped before it did instead.
        d["delay"] = model_mod.delay_ticks(
            d.pop("delay_s"), jax.random.uniform(keys[-1], ()), CTRL_HZ,
            xp=jnp).astype(jnp.int32)
        return d

    def _new_episode(self, info, ps, rng):
        """Re-draw everything an episode owns, where the last step ended one.

        THE EPISODE BOUNDARY IS NOT IN THIS FILE, and that is the whole problem
        this function exists for. brax's `AutoResetWrapper.step` restores
        `pipeline_state` and `obs` and nothing else, and `ppo.train` calls
        `env.reset` ONCE for the entire run (`num_resets_per_eval` is 0). So
        every field of `info` was drawn once, at step zero, and then held for
        60 M steps: `command` meant ~15 % of environments stood still forever and
        never learned to walk, the servo/pack/bus draw was per ENVIRONMENT rather
        than per episode — the opposite of what env/randomize.py's docstring says
        the split is — `obs_hist` carried four frames of the previous, fallen
        robot into the first frame of each new episode, and `foot_xy` differenced
        the new pose against the old one and charged the foot_slip reward for a
        metre of skid that never happened.

        `EpisodeWrapper` writes `info["episode_done"]` at the end of its own step
        (brax/envs/wrappers/training.py), which is 1 exactly when the previous
        step ended the episode — TRUNCATION INCLUDED, which is the case this env
        cannot see for itself and the common one at ep_len 480 of 500. That is
        the gate. Callers that step the env without those wrappers — eval.py,
        replay.py, the battery — have no such key and get no resampling, which is
        what their held-command premise requires.

        Written as `jnp.where` over every field rather than a `lax.cond`, because
        under `jax.vmap` the gate is a traced per-environment scalar: it is data,
        not control flow, and the step has to stay one jitted graph.

        One thing it cannot fix, and it is worth knowing: AutoReset also replaces
        `obs` with `first_obs`, so the FIRST action of a new episode was computed
        from an observation carrying the previous episode's command. One step in
        five hundred, and unfixable from inside the env — the wrapper owns that
        substitution. The history re-seeded here is the honest one from the step
        after it.
        """
        fresh = info["episode_done"] > 0.5
        k_ep, k_obs = jax.random.split(rng, 2)
        fields = self._episode_fields(ps, k_ep)
        out = dict(info)
        for k, v in fields.items():
            out[k] = jnp.where(fresh, v, info[k])
        # The history is re-seeded from the RESTORED pose and the NEW command,
        # the same way reset() seeds it: OBS_HIST copies of one frame. Zeros
        # would be a robot reporting no gravity; the old frames would be the
        # robot that just fell over.
        frame, _, _ = self._frame(ps, out, k_obs)
        out["obs_hist"] = jnp.where(fresh, init_hist(frame, xp=jnp),
                                    info["obs_hist"])
        return out

    # ----------------------------------------------------------------- step
    def step(self, state: State, action: jax.Array) -> State:
        info = dict(state.info)
        rng, k_obs, k_push, k_gap, k_dir, k_new = jax.random.split(info["rng"], 6)
        info["rng"] = rng

        # An episode may have ended on the step before this one. If it did, the
        # wrappers have already put the reset pose back under us and left every
        # custom field of `info` belonging to the robot that fell. See
        # _new_episode — and note the key: it is present only under
        # EpisodeWrapper, which is what makes this a no-op for eval.py.
        if "episode_done" in info:
            info = self._new_episode(info, state.pipeline_state, k_new)

        # The bus delivers this tick's target or the previous one. Not noise —
        # a latency, and one the FTDI adapter alone can spend (robot/README.md).
        buf = jnp.concatenate([action[None], info["action_buf"][:-1]])
        info["action_buf"] = buf
        applied = buf[jnp.clip(info["delay"], 0, self._n_delay - 1)]

        target = jnp.clip(self._stance_j + applied * self._action_scale,
                          self._soft_lo, self._soft_hi)
        p = self._params(info)
        u_bat, sag = info["u_bat"], info["sag"]

        def one(ps, _):
            q = ps.qpos[self._qadr]
            w = ps.qvel[self._vadr]
            # One pack and one harness, so the sag is applied to the SUMMED
            # current of all twelve — actuator.bus_torque is that whole chain,
            # stated once and shared with eval.py's CPU pass and with
            # check_model.py's probe that every drawn parameter is consumed.
            tau = actuator.bus_torque(p, target - q, w, u_bat, sag, xp=jnp)
            return self._pipeline.step(self.sys, ps, tau, self._debug), tau

        ps, taus = jax.lax.scan(one, state.pipeline_state, (), self._n_frames)
        tau = taus[-1]

        # A shove, on a schedule sampled per episode. Not a model of anything —
        # a way to stop the policy learning a gait that only works from rest.
        if self._push:
            t = info["step"] / CTRL_HZ
            due = t >= info["next_push"]
            lo, hi = self._ranges["push"]["vel_m_s_abs"]["range"]
            ang = jax.random.uniform(k_dir, (), maxval=2 * jnp.pi)
            mag = jax.random.uniform(k_push, (), minval=lo, maxval=hi) * due
            ps = ps.replace(qvel=ps.qvel.at[0:2].add(
                mag * jnp.array([jnp.cos(ang), jnp.sin(ang)])))
            # k_gap, NOT k_push. Two uniforms off one key are the SAME uniform,
            # so the gap to the next shove was a deterministic function of this
            # shove's strength — gap = 3 + 6*mag/0.7 exactly, i.e. the hardest
            # shove was always followed by the longest wait and a gentle one by
            # the shortest. A schedule that anti-correlates with the disturbance
            # is a schedule a policy can learn instead of a disturbance it has to
            # survive.
            gap_lo, gap_hi = self._ranges["push"]["interval_s_abs"]["range"]
            info["next_push"] = jnp.where(
                due, t + jax.random.uniform(k_gap, (), minval=gap_lo, maxval=gap_hi),
                info["next_push"])

        # ---- what happened
        quat = self._sensor(ps, self._s_quat)
        gyro = self._sensor(ps, self._s_gyro)
        gravity_b = _rotate_inv(quat, jnp.array([0.0, 0.0, -1.0]))
        lin_vel_b = _rotate_inv(quat, ps.qvel[0:3])
        touch = jnp.array([self._sensor(ps, a)[0] for a in self._s_touch])
        in_contact = touch > 1.0                       # N; a foot carrying weight

        foot_xy = ps.site_xpos[self._foot_site][:, :2]
        foot_h = ps.site_xpos[self._foot_site][:, 2] - self._foot_z0
        foot_vel_xy = (foot_xy - info["foot_xy"]) / self.dt
        info["foot_xy"] = foot_xy

        # Integrated before the reward reads it, so step k is charged for the
        # error it has actually accumulated including this step. The -h/tau term
        # is the leak; HEADING_TAU explains why it is not optional.
        h = info["heading_err"]
        info["heading_err"] = h + self.dt * (
            (gyro[2] - info["command"][2]) - h / HEADING_TAU)

        first_contact = (info["air_time"] > 0.0) & in_contact
        air_time = info["air_time"]
        info["air_time"] = jnp.where(in_contact, 0.0, air_time + self.dt)

        upright = -gravity_b[2]                        # 1 level, 0 on its side
        done = jnp.where((upright < 0.4) | (ps.qpos[2] < 0.10), 1.0, 0.0)
        done = jnp.where(jnp.isnan(ps.qpos).any() | jnp.isnan(ps.qvel).any(), 1.0, done)

        unweighted = rw.terms(
            cmd=info["command"], lin_vel_b=lin_vel_b, ang_vel_b=gyro,
            gravity_b=gravity_b, base_z=ps.qpos[2], stance_z=self._stance_z,
            qpos_j=ps.qpos[self._qadr], qvel_j=ps.qvel[self._vadr], tau=tau,
            action=action, last_action=info["last_action"],
            vel_limit=self._vel_limit, soft_lo=self._soft_lo, soft_hi=self._soft_hi,
            air_time=air_time, first_contact=first_contact.astype(jnp.float32),
            foot_vel_xy=foot_vel_xy, foot_h=foot_h,
            in_contact=in_contact.astype(jnp.float32),
            heading_err=info["heading_err"], done=done, dt=self.dt)
        reward, scaled = rw.total(unweighted, self._w)
        reward = jnp.nan_to_num(reward)

        info["last_action"] = action
        info["step"] = info["step"] + 1
        frame, _, _ = self._frame(ps, info, k_obs)
        obs, hist = stack_obs(info["obs_hist"], jnp.nan_to_num(frame), xp=jnp)
        info["obs_hist"] = hist

        # Start from what is already there, not from a fresh dict: brax's episode
        # wrappers add their own keys (`reward`) to state.metrics, and lax.scan
        # requires the carry's pytree structure to be identical in and out.
        metrics = dict(state.metrics)
        metrics.update(scaled)
        # Metrics are averaged across environments by brax's evaluator, so one
        # NaN anywhere makes the whole reported number NaN and hides which term
        # produced it. The reward and the observation are already guarded above.
        # The `_per_step` suffix is load-bearing, not decoration. brax's
        # EvalWrapper accumulates every metric as a running SUM over the episode
        # and its Evaluator divides by the episode length ONLY for names ending
        # in `per_step` (brax/training/acting.py). A metric named `track_err_xy`
        # is reported as the sum of 200 per-step errors and reads as a blown-up
        # simulation when it is nothing of the kind. Summing a POSITION over
        # steps, as an earlier `travelled_x` here did, is meaningless at any
        # scaling — so the forward progress metric is a velocity, which means
        # something once divided.
        # Belt as well as braces, and there are three of them now: _new_episode
        # zeroes this at every boundary including truncation, the leak erases
        # anything that somehow crosses one, and this catches the terminations
        # this env can see for itself one step earlier than the wrapper reports
        # them.
        info["heading_err"] = jnp.where(done > 0.5, 0.0, info["heading_err"])

        metrics["vx_body_per_step"] = jnp.nan_to_num(lin_vel_b[0])
        metrics["track_err_xy_per_step"] = jnp.nan_to_num(
            jnp.linalg.norm(info["command"][:2] - lin_vel_b[:2]))
        metrics = {k: jnp.nan_to_num(v) for k, v in metrics.items()}
        return state.replace(pipeline_state=ps, obs=obs, reward=reward,
                             done=done, metrics=metrics, info=info)

    @property
    def observation_size(self) -> int:
        # Derived, not asserted from memory. brax's own implementation calls
        # reset() to find this out; that is correct but costs a pipeline_init
        # every time a network is built, and three scripts build one.
        return OBS_SIZE

    @property
    def action_size(self) -> int:
        return 12
