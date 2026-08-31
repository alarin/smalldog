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

And the standing caveat, until the bench runs: `params/st3215.json` is the vendor
datasheet, not a fit. Every number below is this policy's score against the
datasheet servo. `actuator.load()` says so at the top of every run and it is not
to be silenced.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys

import numpy as np


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a runs/<name> directory written by train_ppo.py")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--episodes", type=int, default=64,
                    help="MJX rollouts per command in the battery")
    ap.add_argument("--course", action="store_true",
                    help="also run the obstacle course, in vanilla MuJoCo")
    ap.add_argument("--terrain", action="store_true",
                    help="sim-to-sim on the heightfield instead of the plane")
    ap.add_argument("--shot", default=None, help="write one frame here")
    ap.add_argument("--mem-fraction", type=float, default=0.60)
    ap.add_argument("--json", default=None)
    return ap.parse_args()


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
    import mujoco
    from brax.io import model as brax_io_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    import actuator
    import model as model_mod
    from env import Walk, assemble_obs, rotate_inv

    run_dir = a.run.rstrip("/")
    with open(os.path.join(run_dir, "run.json")) as f:
        meta = json.load(f)
    targs = meta["args"]

    p = actuator.load()
    print(f"\nrun         {run_dir}")
    print(f"trained     {targs['num_timesteps']/1e6:.1f} M steps, "
          f"{targs['num_envs']} envs, boxes {targs['boxes']}, "
          f"terrain {targs['terrain']}, {meta['wall_clock_min']:.1f} min")

    env = Walk(terrain=targs["terrain"], n_boxes=targs["boxes"])

    # ---- the policy, deterministic
    networks = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(128, 128, 128),
        value_hidden_layer_sizes=(256, 256, 256))
    params = brax_io_model.load_params(os.path.join(run_dir, "params"))
    if len(params) > 2:
        params = params[:2]
    policy = ppo_networks.make_inference_fn(networks)(params, deterministic=True)
    policy_jit = jax.jit(policy)

    # ================================================== MJX, the battery
    print(f"\n== MJX, deterministic, {a.episodes} rollouts x {a.seconds:g} s "
          f"=================")
    print(f"{'command':<16}{'up':>6}{'vx':>9}{'vy':>8}{'yaw':>8}"
          f"{'|err|':>8}{'x':>9}{'v>lim':>9}")

    n_steps = int(a.seconds * 50)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    results = {}

    for label, vx, vy, yaw in BATTERY:
        keys = jax.random.split(jax.random.PRNGKey(0), a.episodes)
        st = reset(keys)
        cmd = jnp.tile(jnp.array([vx, vy, yaw]), (a.episodes, 1))
        st = st.replace(info={**st.info, "command": cmd})

        alive = jnp.ones(a.episodes)
        vsum = jnp.zeros((a.episodes, 3))
        tau_sq = jnp.zeros(a.episodes)
        over = jnp.zeros(a.episodes)
        for _ in range(n_steps):
            act, _ = policy_jit(st.obs, jax.random.PRNGKey(0))
            st = step(st, act)
            st = st.replace(info={**st.info, "command": cmd})
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
            over = over + jnp.sum(
                jnp.clip(jnp.abs(q.qvel[:, env._vadr]) - env._vel_limit, 0.0, None) > 0,
                axis=1) * alive

        n_alive = float(jnp.mean(alive))
        v = np.asarray(vsum) / max(n_steps, 1)
        v = v / max(n_alive, 1e-6)
        err = float(np.mean(np.linalg.norm(v[:, :2] - np.array([vx, vy]), axis=1)))
        x = float(jnp.mean(st.pipeline_state.qpos[:, 0]))
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
    print(f"\n== sim-to-sim: the SAME policy, vanilla MuJoCo ====================")
    surfaces = [("flat", False, False)]
    if a.terrain or targs["terrain"]:
        surfaces.append(("heightfield", True, False))
    if a.course:
        surfaces.append(("obstacle course", True, True))

    sim = {}
    for name, terrain, logs in surfaces:
        mj, notes = model_mod.build(terrain=terrain, n_boxes=0, mjx_safe=not logs)
        out = rollout_mujoco(mj, policy_jit, env, p, cmd=(0.4, 0.0, 0.0),
                             seconds=a.seconds, shot=a.shot if name == "flat" else None)
        sim[name] = out
        print(f"  {name:<16} travelled {out['x_m']*1000:7.1f} mm, "
              f"y {out['y_m']*1000:+7.1f} mm, upright {out['upright']:+.3f}, "
              f"{'FELL at %.1f s' % out['fell_at'] if out['fell'] else 'stayed up'}")
    print("  commanded 0.4 m/s forward. The analytic trot's baseline on this box is\n"
          "  781.6 mm in 5 s on flat and 617.9 mm on the committed heightfield "
          "(ros2/tools/standalone_sim.py).")

    if not p.fitted:
        print(f"\n!! Every number above is against the DATASHEET servo, not a fit.")
        print(f"!! {p.source}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(run=run_dir, battery=results, sim_to_sim=sim,
                           actuator_fitted=p.fitted), f, indent=2)
        print(f"\nwrote {a.json}")
    return 0


def rollout_mujoco(mj, policy_jit, env, p, cmd, seconds, shot=None):
    """Step the policy through the CPU engine.

    The observation is built by env.assemble_obs with xp=np and the torque by
    actuator.py with xp=np — the same two functions MJX calls, on the other
    backend. That is the point: if this disagrees with the MJX rollout, the
    disagreement is the physics, not two different policies.
    """
    import jax
    import mujoco
    import numpy as np

    from env import assemble_obs
    import actuator
    import model as model_mod

    P = model_mod.robot_params()
    qadr, vadr, act = model_mod.joint_order(mj, P)
    lo, hi = model_mod.limits(P, soft=True)
    q0 = model_mod.stance_qpos(mj, P)
    stance_j = q0[qadr]

    d = mujoco.MjData(mj)
    d.qpos[:] = q0
    mujoco.mj_forward(mj, d)

    def sadr(name):
        i = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(mj.sensor_adr[i]), int(mj.sensor_dim[i])

    aq, _ = sadr("imu_quat")
    ag, _ = sadr("imu_gyro")

    dt_ctrl = 1.0 / 50.0
    n_sub = int(round(dt_ctrl / mj.opt.timestep))
    last_action = np.zeros(12)
    command = np.array(cmd, float)
    u_bat = 12.0                      # nominal pack; the battery test is elsewhere
    fell, fell_at = False, float("nan")

    for k in range(int(seconds / dt_ctrl)):
        obs, gravity_b = assemble_obs(
            quat=d.sensordata[aq:aq + 4], gyro=d.sensordata[ag:ag + 3],
            qpos_j=d.qpos[qadr], qvel_j=d.qvel[vadr], stance_j=stance_j,
            last_action=last_action, command=command, xp=np)
        action, _ = policy_jit(obs, jax.random.PRNGKey(0))
        action = np.asarray(action)
        last_action = action
        target = np.clip(stance_j + action * 0.35, lo, hi)

        for _ in range(n_sub):
            q = d.qpos[qadr]
            w = d.qvel[vadr]
            duty = actuator.duty(p, target - q, w, xp=np)
            i = (duty * u_bat - p.k_e * w) / p.R
            volt = u_bat - 0.0 * np.sum(np.abs(i))     # nominal: no sag
            d.ctrl[act] = actuator.motor_torque(p, duty * volt, w, xp=np)
            mujoco.mj_step(mj, d)

        if -gravity_b[2] < 0.4 and not fell:
            fell, fell_at = True, k * dt_ctrl

    if shot:
        _write_shot(mj, d, shot)

    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, d.qpos[3:7])
    return dict(x_m=float(d.qpos[0]), y_m=float(d.qpos[1]), z_m=float(d.qpos[2]),
                upright=float(R.reshape(3, 3)[2, 2]), fell=bool(fell),
                fell_at=float(fell_at))


def _write_shot(mj, d, path, w=1280, h=960):
    """One offscreen frame. In a non-interactive session a viewer window verifies
    nothing (WSL.md); a png does."""
    import mujoco
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        r = mujoco.Renderer(mj, h, w)
        r.update_scene(d)
        px = r.render()
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
