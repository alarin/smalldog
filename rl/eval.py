#!/usr/bin/env python
"""
eval.py — the honest number.

    python eval.py runs/20260831-120000              # the standard battery
    python eval.py runs/... --course                 # the obstacle course
    python eval.py runs/... --shot out.png           # ... and a frame to look at

`train_ppo.py`'s progress line is the training environment reporting on itself:
stochastic policy, randomised servos, randomised ground, reward shaped by the
same weights the policy was optimising. It is the right thing to watch while a
run is going and the wrong thing to quote afterwards.

This file quotes. Three differences, each deliberate:

  DETERMINISTIC   the policy's mean action, not a sample. The robot runs the
                  mean; a stochastic rollout flatters a policy that is only
                  upright because a lucky sample caught it.
  NOMINAL SERVO   params/st3215.json as it stands, no per-episode draw. The
                  randomisation is there to make the policy robust, not to make
                  the score look better by averaging over twelve lucky motors.
  VANILLA MUJOCO  the sim-to-sim pass steps the SAME policy through the CPU
                  engine instead of MJX. MJX and MuJoCo are not the same
                  integrator over the same contacts, and a policy that only
                  works in one of them has learned the solver, not the robot.
                  This is also the only place the obstacle course exists: its
                  two cylinder logs are exactly what MJX cannot collide.

What the numbers mean, and what they do not
-------------------------------------------
Distances are not comparable to `3d/CLAUDE.md` step 6. Those were measured on
the mac, and `WSL.md` says plainly that MuJoCo is deterministic for a given
model, seed and version but not across platforms — and that the flat trot sits
near a bifurcation where 11 g moves it by 180 mm. The comparison that means
something is this box against this box: the analytic trot's baseline here is
781.6 mm in a 5 s trot on flat and 617.9 mm on the committed heightfield,
measured with `ros2/tools/standalone_sim.py --headless [--terrain]`.

Both numbers moved with a594d57, which gave the ROS 2 model the 2.94 N*m the CAD
specifies instead of a rounded 3.0, and the two axes did not move alike. Flat
went 781.4 -> 781.6 mm, which is the deterministic same answer. The heightfield
went 595.5 -> 617.9 mm on this one seed — measured here on both trees, so it is
the servo and not the platform. That is +22 mm from -2 % of torque, and it is
NOT evidence that rough ground is sensitive to the servo: a594d57 swept seeds
7-12 and got 622 +-27 -> 618 +-37 mm, so a single terrain seed cannot tell a
22 mm shift from its own spread. Quote the flat number against a policy; quote
a terrain number only against the same seed, and never as a measurement of the
model.

Both moved again with 7f66997/1046e06, which reshaped the shins and the base and
put total mass at 2.4994 kg, +3.1 g. Re-measured here on both scenes, same
machine, same seed, `ros2/tools/standalone_sim.py --headless [--terrain]`:

    flat          781.6 -> 781.7 mm     the deterministic same answer
    heightfield   617.9 -> 552.3 mm     on seed 7 alone

The second line is a seed-7 observation and NOT a measurement of the model.
a594d57's sweep put the seed-to-seed spread at ~32 mm sd, so -66 mm is about two
of those, and a single seed cannot tell a real shift from its own noise -- which
is exactly the trap a594d57 recorded the pair to keep anyone out of. Settling it
needs the s7..s12 sweep, which needs `generate_model.py --terrain-seed`, which
needs cadquery; this box has none. Until it runs somewhere that does, the
heightfield baseline is something to score a policy against on seed 7, not
evidence about what the reshaped shins did to rough ground.

The battery could not see that change and the sim-to-sim pass could, which is
worth knowing before reaching for either. Same checkpoint, same policy, 10 s:
the MJX rows moved by at most 14 mm of x, against 13 mm between two runs of the
SAME model, so nothing there is distinguishable. The vanilla-MuJoCo rollout went
369.5 -> 324.1 mm forward and +233.1 -> +318.3 mm sideways, and that path
reproduces bit for bit across processes -- it returned the same 369.5/+233.1
twice before the model changed. On this box the CPU pass is the sharp instrument
for a change in the MODEL, and the noisy GPU battery is the sharp one for a
change in the POLICY, because only the battery averages 64 of them.

Worth keeping next to it: a594d57 measured 11 g of mass moving the flat trot
778 -> 597 mm while 2 % of servo torque moved it 0.2 mm, so mass is the axis this
gait is sensitive on. This change was 3.1 g along that axis and flat did not move
at all. The bifurcation is a threshold, not a slope, and neither 11 g nor 3 g is
a rate.

One run of this file is not a measurement either, and the spread is worth
carrying. The same checkpoint, the same seed, the same flags, run twice as
separate processes on this box (--seconds 2, 64 rollouts):

    up, v>lim   identical
    vx, vy      +-0.002 m/s
    |err|       +-0.003
    x           +-0.004 m
    yaw         +-0.027 rad/s

XLA does not promise a fixed reduction order across processes and MJX contacts
amplify whatever order it picks. Two runs that differ by less than the spread
above have not disagreed about anything. The row that matters is the last one:
achieved yaw is printed to three decimals and the third is noise, so on a policy
that turns at 0.07 rad/s the turn column is noise-dominated outright.

Measured 2026-08-31 in both directions, before and after the battery's loop
became a lax.scan: the same-code pairs and the across-rewrite pairs have the
SAME spread, column for column. That is what says the rewrite changed nothing —
not the fact that the numbers looked close.

And the standing caveat has changed shape rather than gone away.
`params/st3215.json` IS a fit now — `fit_bam.py`, 44 runs at three voltages — so
the numbers below are this policy's score against a measured servo rather than a
datasheet one, and `actuator.load()` prints nothing because there is nothing to
warn about. What it does not mean is that the servo is solved: the fit's own
`source` says UNDER-DETERMINED (its analytic passes read freeswing and hold
only), and its derived free speed of 5.90 rad/s overshoots the 3.86 the hardware
does, because the real servo stops at a firmware plateau the law does not carry
(PLAN.md 3c). Read `p.source` before quoting a number as "against the servo".
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a runs/<name> directory written by train_ppo.py, "
                                "or one of its runs/<name>/ckpt/<step> checkpoints")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--episodes", type=int, default=64,
                    help="MJX rollouts per command in the battery")
    ap.add_argument("--course", action="store_true",
                    help="also run the obstacle course, in vanilla MuJoCo")
    ap.add_argument("--terrain", action="store_true",
                    help="sim-to-sim on the heightfield instead of the plane")
    ap.add_argument("--terrain-seeds", type=int, default=0, metavar="N",
                    help="N more heightfield rollouts from scattered start points "
                         "(+-1.2 m, joint scatter +-1.1 deg), reported as mean +- sd "
                         "of distance and a fall count. One rollout is a coin: "
                         "the same policy read upright and FELL on consecutive "
                         "days from the one start pose. Implies --terrain.")
    ap.add_argument("--host-loop", action="store_true",
                    help="score against the HOST-loop servo (MODE 2: no goal profile, "
                         "165 Hz held duty, current fold — actuator.Params.host_loop) "
                         "whatever the run trained on. A run that trained --host-loop "
                         "is scored that way without this flag; this is for the "
                         "acceptance test in ideas/TRAIN_HOST_LOOP.md, where s4 "
                         "(firmware-trained) has to walk at 0.2 in the host-loop model.")
    ap.add_argument("--cmd", type=float, nargs=3, default=None, metavar=("VX", "VY", "YAW"),
                    help="the vanilla-MuJoCo pass's command instead of 0.4 0 0")
    ap.add_argument("--com-offset", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="move the base link's centre of mass by this xyz, metres, on every "
                         "pass. The nominal is the CAD's +10 mm and the robot on the scale "
                         "is not there: params/domain_rand.json's com_offset_m_abs band "
                         "is the measurement, so its midpoint is the honest score.")
    ap.add_argument("--shot", default=None, help="write one frame here")
    ap.add_argument("--mem-fraction", type=float, default=0.60)
    ap.add_argument("--json", default=None)
    return ap.parse_args()


# The analytic trot's baseline on THIS box, from
# `ros2/tools/standalone_sim.py --headless [--terrain]`.  The duration is part of
# the number and travels with it: the trot has a start-up transient, so 781.6 mm
# in 5 s is not 1563 mm in 10 s, and a rollout at any other --seconds cannot be
# divided into it or held against it.
BASELINE_SECONDS = 5.0
BASELINE_MM = {"flat": 781.7, "heightfield": 552.3}   # 1046e06's model, seed 7

# The battery. Each is (label, vx, vy, yaw): what we ask, in the body frame.
BATTERY = [
    ("stand",          0.00,  0.00,  0.00),
    ("walk 0.2 m/s",   0.20,  0.00,  0.00),
    ("walk 0.4 m/s",   0.40,  0.00,  0.00),
    ("walk 0.6 m/s",   0.60,  0.00,  0.00),
    ("back 0.3 m/s",  -0.30,  0.00,  0.00),
    ("strafe 0.25",    0.00,  0.25,  0.00),
    ("turn 0.8 rad/s", 0.00,  0.00,  0.80),
    ("arc",            0.35,  0.00,  0.60),
]


def main():
    a = parse()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import jaxenv
    jaxenv.configure(a.mem_fraction)

    import jax
    import jax.numpy as jnp
    from brax.io import model as brax_io_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    import actuator
    import model as model_mod
    from env import Walk, rotate_inv, check_obs_width
    from env.walk import CTRL_HZ

    run_dir = a.run.rstrip("/")
    with open(os.path.join(run_dir, "run.json")) as f:
        meta = json.load(f)
    targs = meta["args"]

    host = bool(a.host_loop or targs.get("host_loop", False))
    p = dataclasses.replace(actuator.load(), host_loop=host)
    print(f"\nrun         {run_dir}")
    print(f"servo       {'HOST loop (MODE 2, kp %.0f on the Pi at %.0f Hz, no profile)' % (p.kp * 1000, p.host_hz) if host else 'firmware loop (goal profile %.0f rad/s^2)' % p.goal_acc}"
          + ("  <- --host-loop, the run trained on the firmware loop" if host and not targs.get("host_loop") else ""))
    # a runs/<name>/ckpt/<step> directory carries the same params + run.json,
    # minus the wall clock: the run is still going when one of these is scored
    trained = (f"{meta['step']/1e6:.1f} of {targs['num_timesteps']/1e6:.1f} M steps"
               if "wall_clock_min" not in meta else
               f"{targs['num_timesteps']/1e6:.1f} M steps")
    print(f"trained     {trained}, "
          f"{targs['num_envs']} envs, boxes {targs['boxes']}, "
          f"terrain {targs['terrain']}"
          + (f", {meta['wall_clock_min']:.1f} min" if "wall_clock_min" in meta else ""))

    env = Walk(terrain=targs["terrain"], n_boxes=targs["boxes"], host_loop=host,
               com_offset=a.com_offset)
    if a.com_offset is not None:
        print(f"com         base_link centre of mass moved by "
              f"{tuple(round(v * 1000, 1) for v in a.com_offset)} mm on every pass")

    # ---- the policy, deterministic
    networks = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(128, 128, 128),
        value_hidden_layer_sizes=(256, 256, 256))
    params = brax_io_model.load_params(os.path.join(run_dir, "params"))
    check_obs_width(params, env.observation_size, run_dir)
    if len(params) > 2:
        params = params[:2]
    policy = ppo_networks.make_inference_fn(networks)(params, deterministic=True)
    policy_jit = jax.jit(policy)

    # ================================================== MJX, the battery
    print(f"\n== MJX, deterministic, {a.episodes} rollouts x {a.seconds:g} s "
          f"=================")
    print(f"{'command':<16}{'up':>6}{'vx':>9}{'vy':>8}{'yaw':>8}"
          f"{'|err|':>8}{'x':>9}{'v>lim':>9}")

    n_steps = int(a.seconds * CTRL_HZ)

    @jax.jit
    def battery_rollout(cmd):
        """One command, `episodes` rollouts, the whole step loop on the device.

        The loop is a lax.scan and not a Python `for`.  Both compile the same MJX
        step; the difference is that the Python version hands control back to the
        host once per step and pays a dispatch on each, and at 50 Hz that is 500
        round trips per command with nothing to overlap them.

        That dispatch is NOT what the battery costs, which is worth writing down
        because it looks like it should be.  Measured here on the RTX 3070, 64
        rollouts x 10 s, mean over the 7 intervals between printed rows:

            Python for      197 s per command
            lax.scan        182 s per command      about 8 %

        The MJX step is the bottleneck, not the host.  The cross-check is that
        this box trains the same model at ~1.9 batch-steps/s and evaluates it at
        ~2.8, which is the same number twice, and the GPU reads 71 % busy during
        both — that was work, not a starved queue.  The scan stays for the two
        reasons that survive the measurement: it is one call instead of 500, and
        per-step data (frames, traces) becomes a scan output rather than 500 host
        round trips.  It did not make the battery fast.

        The command is re-stamped inside the loop rather than once before it, and
        that is no longer a precaution: `Walk.step` DOES resample the command
        now, at every episode boundary, gated on the `info["episode_done"]` key
        that brax's EpisodeWrapper writes. This loop steps the raw env, so there
        is no such key and no resampling — but the battery's premise is that the
        command is HELD, and the re-stamp is what makes that true of the state
        rather than of an argument about which wrappers are in play.
        """
        st = jax.vmap(env.reset)(
            jax.random.split(jax.random.PRNGKey(0), a.episodes))

        def one_step(carry, _):
            st, alive, vsum, over, n_live = carry
            st = st.replace(info={**st.info, "command": cmd})
            act, _ = policy_jit(st.obs, jax.random.PRNGKey(0))
            st = jax.vmap(env.step)(st, act)
            live = 1.0 - st.done
            alive = alive * live
            q = st.pipeline_state
            # The command is given in the BODY frame, so the achieved velocity has
            # to be measured there too — comparing a body-frame command against a
            # world-frame velocity looks fine until the robot turns.
            quat = jax.lax.dynamic_slice(
                q.sensordata, (0, env._s_quat[0]), (a.episodes, 4))
            vb = jax.vmap(lambda qq, vv: rotate_inv(qq, vv, jnp))(quat, q.qvel[:, 0:3])
            vsum = vsum + jnp.concatenate([vb[:, :2], q.qvel[:, 5:6]], 1) * alive[:, None]
            # ANY joint past the limit on this control step, not how many. The
            # legend under the table says "control steps with any joint past
            # ... rad/s" and the sum counted joint-steps, so a policy with three
            # joints over read three times worse than one with one.
            over = over + jnp.any(
                jnp.abs(q.qvel[:, env._vadr]) > env._vel_limit, axis=1) * alive
            # Per-environment alive-step counter. `vsum` accumulates only while
            # an environment is alive, so dividing it by the mean fraction alive
            # at the END is a bias: a row where half the envs fall at step 10 and
            # half survive divides a nearly-full sum by 0.5. Each environment
            # divides by its own step count instead.
            n_live = n_live + alive
            return (st, alive, vsum, over, n_live), None

        init = (st, jnp.ones(a.episodes), jnp.zeros((a.episodes, 3)),
                jnp.zeros(a.episodes), jnp.zeros(a.episodes))
        (st, alive, vsum, over, n_live), _ = jax.lax.scan(
            one_step, init, None, length=n_steps)
        return alive, vsum, over, n_live, st.pipeline_state.qpos[:, 0]

    results = {}

    for label, vx, vy, yaw in BATTERY:
        cmd = jnp.tile(jnp.array([vx, vy, yaw]), (a.episodes, 1))
        alive, vsum, over, n_live, x_end = battery_rollout(cmd)

        n_alive = float(jnp.mean(alive))
        v = np.asarray(vsum) / np.maximum(np.asarray(n_live), 1.0)[:, None]
        err = float(np.mean(np.linalg.norm(v[:, :2] - np.array([vx, vy]), axis=1)))
        x = float(jnp.mean(x_end))
        results[label] = dict(upright_fraction=n_alive, vx=float(v[:, 0].mean()),
                              vy=float(v[:, 1].mean()), yaw=float(v[:, 2].mean()),
                              track_err=err, x_m=x,
                              vel_limit_violations=float(jnp.mean(over)))
        print(f"{label:<16}{n_alive:6.2f}{v[:,0].mean():9.3f}{v[:,1].mean():8.3f}"
              f"{v[:,2].mean():8.3f}{err:8.3f}{x:9.3f}"
              f"{float(jnp.mean(over)):9.0f}")

    print("  up = fraction still upright at the end. vx/vy/yaw are the achieved "
          "means against\n  the commanded value to their left; |err| is the "
          "linear tracking error. v>lim counts\n  control steps with any joint "
          f"past {env._vel_limit:.2f} rad/s — the servo's no-load speed.")

    # ========================================== vanilla MuJoCo, sim-to-sim
    print("\n== sim-to-sim: the SAME policy, vanilla MuJoCo ====================")
    surfaces = [("flat", False, False)]
    if a.terrain_seeds:
        a.terrain = True
    if a.terrain or targs["terrain"]:
        surfaces.append(("heightfield", True, False))
    if a.course:
        surfaces.append(("obstacle course", True, True))

    sim = {}
    sim_cmd = tuple(a.cmd) if a.cmd else (0.4, 0.0, 0.0)
    for name, terrain, logs in surfaces:
        mj, notes = model_mod.build(terrain=terrain, n_boxes=0, mjx_safe=not logs,
                                    com_offset=a.com_offset)
        out = rollout_mujoco(mj, policy_jit, env, p, cmd=sim_cmd,
                             seconds=a.seconds, shot=a.shot if name == "flat" else None)
        sim[name] = out
        print(f"  {name:<16} travelled {out['x_m']*1000:7.1f} mm in {a.seconds:g} s, "
              f"y {out['y_m']*1000:+7.1f} mm, upright {out['upright']:+.3f}, "
              f"{'FELL at %.1f s' % out['fell_at'] if out['fell'] else 'stayed up'}")
    print(f"  commanded {sim_cmd}. The analytic trot's baseline on this box is "
          f"{BASELINE_MM['flat']:.1f} mm\n  on flat and {BASELINE_MM['heightfield']:.1f} mm "
          f"on the committed heightfield (ros2/tools/standalone_sim.py),\n"
          f"  both over {BASELINE_SECONDS:g} s.")
    if abs(a.seconds - BASELINE_SECONDS) > 1e-9:
        print(f"  NOT comparable as printed: the rollout above ran {a.seconds:g} s against a "
              f"{BASELINE_SECONDS:g} s baseline,\n  and the trot does not hold one velocity, "
              f"so the two do not scale into each other.\n"
              f"  Re-run with --seconds {BASELINE_SECONDS:g} to put them side by side.")

    if a.terrain_seeds:
        mj, _ = model_mod.build(terrain=True, n_boxes=0, mjx_safe=True,
                                com_offset=a.com_offset)
        runs = [rollout_mujoco(mj, policy_jit, env, p, cmd=sim_cmd,
                               seconds=a.seconds, seed=s) for s in range(a.terrain_seeds)]
        xs = np.array([r["x_m"] for r in runs]) * 1000
        falls = sum(r["fell"] for r in runs)
        sim["heightfield_seeds"] = dict(n=a.terrain_seeds, x_mm_mean=float(xs.mean()),
                                        x_mm_sd=float(xs.std(ddof=1)) if len(xs) > 1 else 0.0,
                                        falls=int(falls), runs=runs)
        print(f"\n== heightfield, {a.terrain_seeds} scattered starts, {a.seconds:g} s ==========")
        print(f"  travelled {xs.mean():7.1f} +- {xs.std(ddof=1) if len(xs) > 1 else 0:5.1f} mm, "
              f"{falls} of {a.terrain_seeds} down"
              + ("  (at " + ", ".join(f"{r['fell_at']:.1f}" for r in runs if r["fell"]) + " s)" if falls else ""))

    if not p.fitted:
        print("\n!! Every number above is against the DATASHEET servo, not a fit.")
        print(f"!! {p.source}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(run=run_dir, battery=results, sim_to_sim=sim,
                           actuator_fitted=p.fitted), f, indent=2)
        print(f"\nwrote {a.json}")
    return 0


def rollout_mujoco(mj, policy_jit, env, p, cmd, seconds, shot=None, seed=None):
    """Step the policy through the CPU engine.

    The observation is built by env.assemble_obs and env.stack_obs with xp=np,
    and the torque by actuator.py with xp=np — the same three functions MJX
    calls, on the other backend. That is the point: if this disagrees with the MJX rollout, the
    disagreement is the physics, not two different policies.

    `env` is here for the same reason: the action scale and the control period
    come off it rather than being written out again. They were `* 0.35` and
    `1/50.0` literals, which is two more places an action-space change has to
    reach before this stops silently scoring the policy on a different robot.
    """
    import jax
    import mujoco
    import numpy as np

    from env import assemble_obs, stack_obs, init_hist
    import actuator
    import model as model_mod

    P = model_mod.robot_params()
    qadr, vadr, act = model_mod.joint_order(mj, P)
    lo, hi = model_mod.limits(P, soft=True)
    q0 = model_mod.stance_qpos(mj, P)
    stance_j = q0[qadr]

    d = mujoco.MjData(mj)
    d.qpos[:] = q0
    if seed is not None:
        # A different patch of the heightfield and a different first frame:
        # the scatter gait.py uses (1.1 deg, 3 mm), plus a start point anywhere
        # in the middle of the 4 x 4 m field. Distance is measured from it.
        g = np.random.default_rng(seed)
        d.qpos[0:2] = g.uniform(-1.2, 1.2, 2)
        d.qpos[qadr] += g.uniform(-0.02, 0.02, 12)
        # Stand on the ground that is actually there: a ray straight down from
        # above the start point, so a start on a bump is a start ON it and not
        # inside it. mj_ray needs geoms placed, hence the forward first.
        mujoco.mj_forward(mj, d)
        geomid = np.array([-1], dtype=np.int32)
        # geomgroup: only group 0 — the ground and the course furniture. The
        # robot's collision geoms are group 3 and its meshes group 2, and a ray
        # from above would otherwise report the robot's own back as the ground.
        ground_only = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        # Under each FOOT, not under the base: the field varies by 5 cm across
        # the stance, and a foot spawned inside a bump is a fall at 0.1 s.
        # The highest of the four is the one the robot has to stand on.
        ground = -1.0
        for fx, fy in [(0.09, 0.076), (0.09, -0.076), (-0.09, 0.076), (-0.09, -0.076)]:
            dist = mujoco.mj_ray(mj, d, np.array([d.qpos[0] + fx, d.qpos[1] + fy, 1.0]),
                                 np.array([0.0, 0.0, -1.0]), ground_only, 1, -1, geomid)
            if dist >= 0:
                ground = max(ground, 1.0 - dist)
        d.qpos[2] = max(ground, 0.0) + q0[2] + g.uniform(-0.003, 0.003)
    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    mujoco.mj_forward(mj, d)

    def sadr(name):
        i = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(mj.sensor_adr[i]), int(mj.sensor_dim[i])

    aq, _ = sadr("imu_quat")
    ag, _ = sadr("imu_gyro")
    aa, _ = sadr("imu_accel")

    dt_ctrl = float(env.dt)
    n_sub = int(round(dt_ctrl / mj.opt.timestep))
    last_action = np.zeros(12)
    loop = actuator.loop_state(stance_j, xp=np)     # the servo loop's state, either mode
    hist = None                       # filled from the first frame, not from zeros
    command = np.array(cmd, float)
    u_bat = 12.0                      # nominal pack; the battery test is elsewhere
    fell, fell_at = False, float("nan")

    for k in range(int(seconds / dt_ctrl)):
        frame, gravity_b = assemble_obs(
            quat=d.sensordata[aq:aq + 4], gyro=d.sensordata[ag:ag + 3],
            accel=d.sensordata[aa:aa + 3],
            qpos_j=d.qpos[qadr], qvel_j=d.qvel[vadr], stance_j=stance_j,
            last_action=last_action, command=command, xp=np)
        # Same stacking function the env uses, for the same reason assemble_obs
        # is one function: a history assembled differently here would make this
        # a test of two policies rather than of two engines.
        if hist is None:
            hist = init_hist(frame, xp=np)
            obs = hist.reshape(-1)
        else:
            obs, hist = stack_obs(hist, frame, xp=np)
        action, _ = policy_jit(obs, jax.random.PRNGKey(0))
        action = np.asarray(action)
        last_action = action
        target = np.clip(stance_j + action * env._action_scale, lo, hi)

        for _ in range(n_sub):
            # sag = 0: a nominal pack, deliberately. The battery test is the
            # place the supply is swept; this pass is about the two engines
            # disagreeing, and it can only be that if everything else is held.
            #
            # tau_c_external: the Coulomb floor is MuJoCo's frictionloss on
            # every MuJoCo path (actuator.friction, model.build_spec).
            # actuator.servo_step is the same function walk.py's scan calls:
            # the firmware's goal profile, or the Pi's held 165 Hz duty.
            loop, d.ctrl[act] = actuator.servo_step(
                p, loop, target, d.qpos[qadr], d.qvel[vadr], float(mj.opt.timestep),
                u_bat, 0.0, xp=np, tau_c_external=True)
            mujoco.mj_step(mj, d)

        if -gravity_b[2] < 0.4 and not fell:
            fell, fell_at = True, k * dt_ctrl

    if shot:
        _write_shot(mj, d, shot)

    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, d.qpos[3:7])
    return dict(x_m=float(d.qpos[0]) - x0, y_m=float(d.qpos[1]) - y0,
                z_m=float(d.qpos[2]), upright=float(R.reshape(3, 3)[2, 2]),
                fell=bool(fell), fell_at=float(fell_at))


def _write_shot(mj, d, path, w=1280, h=960):
    """One offscreen frame. In a non-interactive session a viewer window verifies
    nothing (WSL.md); a png does."""
    import mujoco
    import jaxenv
    jaxenv.configure_gl()
    try:
        r = mujoco.Renderer(mj, h, w)
        r.update_scene(d)
        px = r.render()
        print(f"  GL_RENDERER {jaxenv.gl_renderer()}")
    except Exception as e:
        print(f"  (no frame: {type(e).__name__}: {e}; try MUJOCO_GL=osmesa)")
        return
    try:
        import PIL.Image
        PIL.Image.fromarray(px).save(path)
        print(f"  wrote {path}")
    except ImportError:
        print("  (no PIL; frame not written)")


if __name__ == "__main__":
    raise SystemExit(main())
