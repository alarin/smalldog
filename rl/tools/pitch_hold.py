#!/usr/bin/env python
"""
pitch_hold.py — the two acceptance numbers eval.py does not print (ideas/TRAIN_HOST_LOOP.md).

    python tools/pitch_hold.py runs/20260913-s4 runs/20260916-host-s0 [--host-loop]

For each run, vanilla MuJoCo with the same actuator.servo_step eval.py uses:
  1. body pitch std UNDER WAY: cmd 0.2 for 6 s, pitch sampled at 50 Hz over the last 4 s,
     plus the time the body first passes 0.05 m (start-up);
  2. a stand at cmd 0 from a pitch perturbation: the robot is dropped at rest with the base
     pitched +-15 deg (nose down / nose up), 4 s; did it stay up, and where the pitch settled.
--host-loop scores every run in the host-loop model whatever it trained on; a run whose
run.json says host_loop is scored that way regardless.
"""
import argparse, dataclasses, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--host-loop", action="store_true")
    ap.add_argument("--vx", type=float, default=0.2)
    a = ap.parse_args()

    import jaxenv
    jaxenv.configure(0.3)
    import jax, mujoco
    from brax.io import model as brax_io_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    import actuator, model as model_mod
    from env import Walk, assemble_obs, stack_obs, init_hist, check_obs_width

    P = model_mod.robot_params()
    mj, _ = model_mod.build(terrain=False, n_boxes=0)
    qadr, vadr, act = model_mod.joint_order(mj, P)
    lo, hi = model_mod.limits(P, soft=True)
    q0 = model_mod.stance_qpos(mj, P)
    stance_j = q0[qadr]

    def sadr(name):
        i = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(mj.sensor_adr[i])
    aq, ag, aa = sadr("imu_quat"), sadr("imu_gyro"), sadr("imu_accel")

    def pitch_of(quat):
        w, x, y, z = quat
        return float(np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))))

    def rollout(policy, p, env, cmd, seconds, pitch0_deg=0.0):
        d = mujoco.MjData(mj)
        d.qpos[:] = q0
        if pitch0_deg:
            q = np.zeros(4)
            mujoco.mju_axisAngle2Quat(q, np.array([0.0, 1.0, 0.0]), np.radians(pitch0_deg))
            d.qpos[3:7] = q
            d.qpos[2] += 0.03                    # a hand-over: dropped, not placed
        mujoco.mj_forward(mj, d)
        dt = float(env.dt); n_sub = int(round(dt / mj.opt.timestep))
        loop = actuator.loop_state(stance_j, xp=np)
        last, hist = np.zeros(12), None
        cmdv = np.array(cmd, float)
        pitches, xs, fell = [], [], False
        for k in range(int(seconds / dt)):
            frame, g = assemble_obs(quat=d.sensordata[aq:aq+4], gyro=d.sensordata[ag:ag+3],
                                    accel=d.sensordata[aa:aa+3], qpos_j=d.qpos[qadr],
                                    qvel_j=d.qvel[vadr], stance_j=stance_j, last_action=last,
                                    command=cmdv, xp=np)
            if hist is None:
                hist = init_hist(frame, xp=np); obs = hist.reshape(-1)
            else:
                obs, hist = stack_obs(hist, frame, xp=np)
            action, _ = policy(obs, jax.random.PRNGKey(0))
            last = np.asarray(action)
            target = np.clip(stance_j + last * env._action_scale, lo, hi)
            for _ in range(n_sub):
                loop, d.ctrl[act] = actuator.servo_step(p, loop, target, d.qpos[qadr], d.qvel[vadr],
                                                        float(mj.opt.timestep), 12.0, 0.0, xp=np,
                                                        tau_c_external=True)
                mujoco.mj_step(mj, d)
            pitches.append(pitch_of(d.qpos[3:7])); xs.append(float(d.qpos[0]))
            if -g[2] < 0.4:
                fell = True
        return np.array(pitches), np.array(xs), fell

    for run in a.runs:
        run = run.rstrip("/")
        targs = json.load(open(os.path.join(run, "run.json")))["args"]
        host = a.host_loop or targs.get("host_loop", False)
        p = dataclasses.replace(actuator.load(quiet=True), host_loop=host)
        env = Walk(host_loop=host)
        nets = ppo_networks.make_ppo_networks(
            observation_size=env.observation_size, action_size=env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
            policy_hidden_layer_sizes=(128, 128, 128), value_hidden_layer_sizes=(256, 256, 256))
        prm = brax_io_model.load_params(os.path.join(run, "params"))
        check_obs_width(prm, env.observation_size, run)
        policy = jax.jit(ppo_networks.make_inference_fn(nets)(prm[:2], deterministic=True))

        print(f"\n== {run}  ({'host loop' if host else 'firmware loop'})")
        pit, xs, fell = rollout(policy, p, env, (a.vx, 0, 0), 6.0)
        start = next((i * float(env.dt) for i, x in enumerate(xs) if x > 0.05), float("nan"))
        under_way = pit[100:]
        print(f"  cmd {a.vx:g}: {xs[-1]*1000:6.0f} mm in 6 s, start-up (0.05 m) at {start:.2f} s, "
              f"pitch under way mean {under_way.mean():+.1f} deg, std {under_way.std():.2f} deg, "
              f"p-p {under_way.max()-under_way.min():.1f} deg{'  FELL' if fell else ''}")
        for p0 in (-15.0, 15.0):
            pit, xs, fell = rollout(policy, p, env, (0, 0, 0), 4.0, pitch0_deg=p0)
            print(f"  stand from pitch {p0:+.0f} deg: {'FELL' if fell else 'stayed up'}, "
                  f"pitch at 1 s {pit[49]:+.1f}, at 4 s {pit[-1]:+.1f} deg, drifted {abs(xs[-1])*1000:.0f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
