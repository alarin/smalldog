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
was frozen; 1046e06 put it there and both exporters read it, so the site is now
at [0, 0, 0.0234] — between the pack top at 21.4 mm and the deck underside at 25,
where the BMI088 actually sits. This env still reads whatever site the generated
model calls `imu`; when that site moves, the policy is retrained, not patched.
That matters more now than it did, because the accelerometer is in the
observation and it is the channel the offset corrupts.

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
        self._foot_site = jnp.array([
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site")
            for leg in P["legs"]])

        # One control tick is the delay quantum: the bus either delivers this
        # tick's target or the last one. params/bus_timing.json does not exist,
        # so 0..20 ms is one whole tick of honest ignorance.
        self._n_delay = 2

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
        these feed actuator.py's own functions with xp=jnp."""
        return dataclasses.replace(
            self._p0,
            k_u=info["k_u"], k_e=info["k_e"], R=info["R"], J_m=info["J_m"],
            tau_c=info["tau_c"], b_v=info["b_v"], kp=info["kp"],
            deadband=info["deadband"], punch=info["punch"])

    def _torque(self, p, target, q, w, u_bat, sag):
        d = actuator.duty(p, target - q, w, xp=jnp)
        # The pack sags under the current the servos are drawing. One pass, the
        # same order actuator.simulate() does it in — except that the current
        # here is the sum over all twelve, because there is one pack and one
        # harness, and that is the load case robot/README.md's 50 Hz budget is
        # about.
        #
        # The clamp is not cosmetic, and the arithmetic is worth writing down
        # because it is reachable rather than hypothetical: at duty 1 and a
        # joint running backwards at 10 rad/s, i = (12 + 3.06*10)/3.33 = 12.8 A
        # per servo, and twelve of those into the original 0.18 ohm range is a
        # 27.7 V sag on a 12 V pack. `volt` would go NEGATIVE, k_u*duty*volt
        # would flip sign, and the torque would drive the joint harder in the
        # direction it was already going — positive feedback. A discharged pack
        # delivers less voltage; it never delivers negative voltage. (The range
        # has since been narrowed to 0-0.06 ohm as well, which makes the clamp
        # unreachable in normal operation. Both, not either: a floor that is
        # only satisfied by accident is not a floor.)
        i = (d * u_bat - p.k_e * w) / p.R
        volt = jnp.clip(u_bat - sag * jnp.sum(jnp.abs(i)), 0.0, u_bat)
        return actuator.motor_torque(p, d * volt, w, xp=jnp)

    # ---------------------------------------------------------------- reset
    def reset(self, rng: jax.Array) -> State:
        rng, k_cmd, k_q, k_v, k_a, k_obs, k_push = jax.random.split(rng, 7)

        q = self._init_q.at[self._qadr].add(
            jax.random.uniform(k_q, (12,), minval=-0.05, maxval=0.05))
        qd = jnp.zeros(self.sys.nv).at[:6].set(
            jax.random.uniform(k_v, (6,), minval=-0.05, maxval=0.05))
        ps = self.pipeline_init(q, qd)

        draw = self._sample_episode(k_a)
        info = {
            "rng": rng,
            "command": self._cmd.sample(k_cmd),
            "last_action": jnp.zeros(12),
            "action_buf": jnp.zeros((self._n_delay, 12)),
            "air_time": jnp.zeros(4),
            "foot_xy": ps.site_xpos[self._foot_site][:, :2],
            "step": jnp.array(0, jnp.int32),
            "next_push": jax.random.uniform(
                k_push, (), minval=self._ranges["push"]["interval_s_abs"]["range"][0],
                maxval=self._ranges["push"]["interval_s_abs"]["range"][1]),
            **draw,
        }
        frame, _, _ = self._frame(ps, info, k_obs)
        hist = init_hist(frame, xp=jnp)
        obs = hist.reshape(-1)
        info["obs_hist"] = hist
        metrics = {k: jnp.zeros(()) for k in self._w.asdict()}
        metrics.update({"vx_body_per_step": jnp.zeros(()),
                        "track_err_xy_per_step": jnp.zeros(())})
        return State(ps, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def _sample_episode(self, rng):
        """The servo, the pack and the bus, drawn once per episode.

        Per joint where the spread is per-servo — twelve motors out of one bag —
        and per robot where it is not: one pack, one bus. The ranges and the
        evidence behind each are in params/domain_rand.json.
        """
        R = self._ranges
        A, S, B = R["actuator"], R["supply"], R["bus"]
        keys = jax.random.split(rng, 12)

        def per_joint(key, name, nominal):
            lo, hi = A[name]["range"]
            return jax.random.uniform(key, (12,), minval=lo, maxval=hi) * nominal

        def scalar(key, d, name):
            lo, hi = d[name]["range"]
            return jax.random.uniform(key, (), minval=lo, maxval=hi)

        def per_joint_abs(key, name):
            lo, hi = A[name]["range"]
            return jax.random.uniform(key, (12,), minval=lo, maxval=hi)

        return {
            "k_u": per_joint(keys[0], "k_u", self._p0.k_u),
            "k_e": per_joint(keys[1], "k_e", self._p0.k_e),
            "R": per_joint(keys[2], "R", self._p0.R),
            "J_m": per_joint(keys[3], "J_m", self._p0.J_m),
            "tau_c": per_joint(keys[4], "tau_c", self._p0.tau_c),
            "b_v": per_joint(keys[5], "b_v", self._p0.b_v),
            "kp": per_joint(keys[6], "kp", self._p0.kp),
            "deadband": per_joint_abs(keys[7], "deadband_abs"),
            "punch": per_joint_abs(keys[8], "punch_abs"),
            "u_bat": scalar(keys[9], S, "u_bat_abs"),
            "sag": scalar(keys[10], S, "sag_ohm_abs"),
            "delay": (scalar(keys[11], B, "delay_s_abs") * CTRL_HZ).astype(jnp.int32),
        }

    # ----------------------------------------------------------------- step
    def step(self, state: State, action: jax.Array) -> State:
        info = dict(state.info)
        rng, k_obs, k_push, k_dir = jax.random.split(info["rng"], 4)
        info["rng"] = rng

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
            tau = self._torque(p, target, q, w, u_bat, sag)
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
            gap_lo, gap_hi = self._ranges["push"]["interval_s_abs"]["range"]
            info["next_push"] = jnp.where(
                due, t + jax.random.uniform(k_push, (), minval=gap_lo, maxval=gap_hi),
                info["next_push"])

        # ---- what happened
        quat = self._sensor(ps, self._s_quat)
        gyro = self._sensor(ps, self._s_gyro)
        gravity_b = _rotate_inv(quat, jnp.array([0.0, 0.0, -1.0]))
        lin_vel_b = _rotate_inv(quat, ps.qvel[0:3])
        touch = jnp.array([self._sensor(ps, a)[0] for a in self._s_touch])
        in_contact = touch > 1.0                       # N; a foot carrying weight

        foot_xy = ps.site_xpos[self._foot_site][:, :2]
        foot_vel_xy = (foot_xy - info["foot_xy"]) / self.dt
        info["foot_xy"] = foot_xy

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
            foot_vel_xy=foot_vel_xy, in_contact=in_contact.astype(jnp.float32),
            done=done, dt=self.dt)
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
