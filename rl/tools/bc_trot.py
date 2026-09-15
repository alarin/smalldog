#!/usr/bin/env python
"""
bc_trot.py — a PPO starting point that already walks under the real servo.

    python tools/bc_trot.py --name 20260915-bc-trot            # -> runs/<name>/params
    python train_ppo.py --init-from runs/20260915-bc-trot ...  # then PPO from it

Why: against the measured goal profile (actuator.Params.goal_acc = 8 rad/s^2)
ten PPO runs converged to standing — from s4, from scratch, on three rewards and
two curricula. The reward now pays an honest trot more than standing (env/
rewards.py), but nothing PPO can sample from a standing policy under a servo
that filters its per-step noise to nothing ever looks like a trot, so the
gradient toward one never appears. This script supplies the trot: it rolls the
analytic TrotGait (ros2/smalldog_walker, period 1.35 s — the one that walks on
the hardware at 0.08 m/s) through the training environment at the environment's
own commands, records (observation, action) pairs, and fits the PPO policy
network to them by regression. The result is saved in train_ppo's checkpoint
format — (normaliser, policy, value) — with a fresh value network and the
policy's noise head set to a small, stated std, so `--init-from` picks it up
and PPO refines a walker instead of searching for one.

The observation is the environment's, history and all; the network has to
infer the gait phase from the joint positions and its own last action, which is
what makes the clone a closed-loop policy and not a clock. A little action
noise is executed during collection (the label stays clean) so the data covers
states a step off the nominal path — the cheap half of DAgger.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time

_RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RL)
sys.path.insert(0, os.path.join(os.path.dirname(_RL), "ros2", "smalldog_walker"))
os.chdir(_RL)

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400, help="control steps per rollout (8 s)")
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--exec-noise", type=float, default=0.08,
                    help="std of the noise added to the EXECUTED action; labels stay clean")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--std", type=float, default=0.15, help="the policy's noise std after cloning")
    ap.add_argument("--period", type=float, default=1.35)
    ap.add_argument("--mem-fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=None, metavar="NPZ",
                    help="skip collection: fit to a data.npz a previous run saved")
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128, 128])
    a = ap.parse_args()

    import jaxenv
    jaxenv.configure(a.mem_fraction)
    import jax
    import jax.numpy as jnp
    import optax
    from brax.io import model as brax_io_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from env import Walk, check_obs_width
    from env.walk import ACTION_SCALE
    import model as model_mod
    from smalldog_walker.gait import TrotGait

    P = model_mod.robot_params()
    env = Walk()
    # env.dt is a jax scalar (brax derives it from sys.opt.timestep); handed to
    # the gait as-is every one of its arithmetic ops became a device dispatch —
    # 140 ms per joint_targets() against 0.01 ms with a Python float.
    dt = float(env.dt)
    stance = np.array(env._stance_j)
    obs_size, act_size = env.observation_size, 12
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))

    def make_gait():
        g = TrotGait(P)
        g.period = a.period
        g.swing_height = 0.022
        g.max_step = 0.06
        g.body_height = 0.158
        return g

    # ---------------------------------------------------------- collect
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    OBS, ACT = [], []
    t0 = time.time()
    out = os.path.join(_RL, "runs", a.name)
    os.makedirs(out, exist_ok=True)
    for r in range(a.rollouts if a.data is None else 0):
        key, k = jax.random.split(key)
        st = reset(jax.random.split(k, a.envs))
        cmd = np.asarray(st.info["command"])
        gaits = [make_gait() for _ in range(a.envs)]
        for i, g in enumerate(gaits):
            g.stride_max = max(g.stride_max, 0.3 * a.period / 2.0)
            # TrotGait starts from a straight leg and ramps into its cycle over
            # ~20 steps; that ramp is not a gait and stays out of the data.
            for _ in range(40):
                g.joint_targets(dt, *[float(c) for c in cmd[i]])
        alive = np.ones(a.envs, bool)
        for t in range(a.steps):
            q = np.stack([g.joint_targets(dt, *[float(c) for c in cmd[i]]) for i, g in enumerate(gaits)])
            clean = ((q - stance) / ACTION_SCALE).astype(np.float32)
            OBS.append(np.asarray(st.obs)[alive]); ACT.append(clean[alive])
            noisy = clean + rng.normal(0.0, a.exec_noise, clean.shape).astype(np.float32)
            st = step(st, jnp.asarray(noisy))
            alive &= np.asarray(st.done) < 0.5
            if t % 100 == 99:
                print(f"  rollout {r + 1} step {t + 1}/{a.steps}, {alive.sum()} upright, {time.time() - t0:.0f} s", flush=True)
        print(f"rollout {r + 1}/{a.rollouts}: {alive.sum()}/{a.envs} upright at {a.steps * dt:.0f} s, "
              f"{sum(len(o) for o in OBS)} samples, {time.time() - t0:.0f} s")
    if a.data:
        d = np.load(a.data); X, Y = d["obs"], d["act"]
        print(f"data      {a.data}")
    else:
        X = np.concatenate(OBS).astype(np.float32); Y = np.concatenate(ACT).astype(np.float32)
        np.savez_compressed(os.path.join(out, "data.npz"), obs=X, act=Y)
    print(f"data      {X.shape[0]} (obs, action) pairs; |action| p50 {np.median(np.abs(Y)):.3f}, "
          f"max {np.abs(Y).max():.2f}")

    # ---------------------------------------------------------- networks
    networks = ppo_networks.make_ppo_networks(
        observation_size=obs_size, action_size=act_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(a.hidden),
        value_hidden_layer_sizes=(256, 256, 256))
    norm = running_statistics.init_state(jax.ShapeDtypeStruct((obs_size,), jnp.float32))
    norm = running_statistics.update(norm, jnp.asarray(X))
    key, k_pi, k_v = jax.random.split(key, 3)
    policy_params = networks.policy_network.init(k_pi)
    value_params = networks.value_network.init(k_v)
    dist = networks.parametric_action_distribution
    # softplus(s) + min_std = std  ->  the raw value the noise head should sit at
    target_raw = float(np.log(np.expm1(max(a.std - 1e-3, 1e-3))))

    def loss_fn(params, obs, act):
        logits = networks.policy_network.apply(norm, params, obs)
        loc, scale = jnp.split(logits, 2, axis=-1)
        mode = jnp.tanh(loc)                     # = dist.mode(logits)
        l_act = jnp.mean((mode - act) ** 2)
        l_std = jnp.mean((scale - target_raw) ** 2)
        return l_act + 0.1 * l_std, (l_act, l_std)

    opt = optax.adam(a.lr)
    opt_state = opt.init(policy_params)

    @jax.jit
    def train_step(params, opt_state, obs, act):
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, obs, act)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, l, aux

    n = X.shape[0]
    Xj, Yj = jnp.asarray(X), jnp.asarray(Y)
    for ep in range(a.epochs):
        perm = rng.permutation(n)
        tot = 0.0; nb = 0
        for i in range(0, n - a.batch + 1, a.batch):
            idx = jnp.asarray(perm[i:i + a.batch])
            policy_params, opt_state, l, (la, ls) = train_step(policy_params, opt_state, Xj[idx], Yj[idx])
            tot += float(la); nb += 1
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep:3d}  action mse {tot / nb:.5f}  (rms {np.sqrt(tot / nb):.4f} of an "
                  f"action in [-1, 1]), std-head mse {float(ls):.4f}")

    # ---------------------------------------------------------- verify: the clone drives the env
    policy = jax.jit(ppo_networks.make_inference_fn(networks)((norm, policy_params), deterministic=True))
    key, k = jax.random.split(key)
    st = reset(jax.random.split(k, 64))
    st.info["command"] = jnp.tile(jnp.array([0.15, 0.0, 0.0]), (64, 1))
    x0 = np.asarray(st.pipeline_state.qpos[:, 0])
    alive = np.ones(64, bool)
    for t in range(250):
        act, _ = jax.vmap(policy)(st.obs, jax.random.split(jax.random.PRNGKey(t), 64))
        st = step(st, act)
        alive &= np.asarray(st.done) < 0.5
    x1 = np.asarray(st.pipeline_state.qpos[:, 0])
    dx = (x1 - x0)[alive]
    print(f"verify    clone at cmd 0.15 for 5 s: {alive.sum()}/64 upright, forward "
          f"{dx.mean() if len(dx) else float('nan'):.2f} m mean ({dx.min() if len(dx) else float('nan'):.2f}"
          f" .. {dx.max() if len(dx) else float('nan'):.2f}); the IK trot itself does ~0.4")

    # ---------------------------------------------------------- save, train_ppo's way
    params = (norm, policy_params, value_params)
    check_obs_width(params, obs_size, a.name)
    brax_io_model.save_params(os.path.join(out, "params"), params)
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump(dict(args=vars(a), history=[], samples=int(n),
                       note="behaviour-cloned from the IK trot by tools/bc_trot.py; "
                            "the value network is a fresh init"), f, indent=2)
    print(f"wrote     {out}/params  — train_ppo.py --init-from runs/{a.name}")


if __name__ == "__main__":
    main()
