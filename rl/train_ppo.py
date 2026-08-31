#!/usr/bin/env python
"""
train_ppo.py — PPO on MJX, on the 8 GB card this box has.

    python train_ppo.py --smoke                    # 2 min, proves the loop closes
    python train_ppo.py                            # the real run
    python train_ppo.py --boxes 24 --num-envs 1024 # rough ground, smaller batch

Everything about the defaults is set by VRAM, not by FLOPs. rl/README.md says it
plainly and this file is where it becomes numbers:

  * JAX preallocates 75 % of the card on the first device call. On this box the
    Windows desktop is already holding 0.5-1.5 GB, so that preallocation FAILS
    and XLA falls back through a retry ladder, printing several screens of
    RESOURCE_EXHAUSTED before continuing. It is not fatal and it is not the
    training running out of memory — but it is unreadable, and it hides the OOM
    you actually care about. So the fraction is set BEFORE jax is imported, and
    it is set low.
  * num_envs is the memory knob. 2048 is the top of the budget for this robot;
    1024 with procedural boxes, because every box multiplies the collision pairs
    MJX allocates for. If you see an XLA allocation error, halve num_envs before
    you touch anything else — an OOM here surfaces as an allocation failure deep
    in a compiled kernel and says nothing about the model.

The run writes to runs/, which is gitignored and stays that way. A result worth
keeping is a number in a commit message or a file in params/ (rl/CLAUDE.md), so
this script prints the numbers and eval.py produces the honest ones.
"""
from __future__ import annotations

import argparse
import datetime
import functools
import json
import os
import sys
import time


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-timesteps", type=int, default=60_000_000)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--episode-length", type=int, default=500,  # 10 s at 50 Hz
                    help="control steps per episode; 500 = 10 s at 50 Hz")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--unroll-length", type=int, default=20)
    ap.add_argument("--num-minibatches", type=int, default=32)
    ap.add_argument("--updates-per-batch", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--entropy-cost", type=float, default=1e-2)
    ap.add_argument("--discounting", type=float, default=0.97)
    ap.add_argument("--num-evals", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--terrain", action="store_true", help="heightfield scene")
    ap.add_argument("--boxes", type=int, default=0, help="procedural terrain boxes")
    ap.add_argument("--no-randomize", action="store_true",
                    help="turn off the per-environment MODEL randomisation "
                         "(the per-episode servo draw stays; it lives in the env)")
    ap.add_argument("--mem-fraction", type=float, default=0.60,
                    help="XLA_PYTHON_CLIENT_MEM_FRACTION. Low on purpose: the "
                         "Windows desktop owns some of this card.")
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="a two-minute run that proves the loop closes and "
                         "nothing NaNs. Not a policy.")
    return ap.parse_args()


def main():
    a = parse()
    if a.smoke:
        a.num_timesteps, a.num_envs, a.episode_length = 400_000, 256, 200
        a.num_evals, a.batch_size, a.num_minibatches = 2, 64, 8

    # BEFORE jax is imported. Nothing below this line may import jax earlier.
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(a.mem_fraction))

    import jax
    from brax.io import model as brax_io_model
    from brax.training.agents.ppo import train as ppo
    from brax.training.agents.ppo import networks as ppo_networks

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import actuator
    from env import Walk, domain_randomize

    name = a.name or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", name)
    os.makedirs(out, exist_ok=True)

    p = actuator.load()          # the un-quiet load: it says what it is, every run
    print(f"\ndevices     {jax.devices()}")
    print(f"run         {out}")

    env = Walk(terrain=a.terrain, n_boxes=a.boxes)
    eval_env = Walk(terrain=a.terrain, n_boxes=a.boxes)
    for n in env.build_notes:
        print(f"model       {n}")
    print(f"env         obs {env.observation_size}, act {env.action_size}, "
          f"dt {env.dt*1000:.1f} ms, {env._n_frames} physics steps per control step")
    print(f"budget      {a.num_envs} envs x {a.episode_length} steps "
          f"({a.episode_length/50:.0f} s), {a.num_timesteps/1e6:.1f} M steps, "
          f"mem_fraction {a.mem_fraction}")

    randomization = None
    if not a.no_randomize:
        randomization = functools.partial(domain_randomize, n_boxes=a.boxes)

    networks = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128),
        value_hidden_layer_sizes=(256, 256, 256))

    history = []
    t0 = time.time()

    def progress(step, metrics):
        el = time.time() - t0
        # Key names matter here. brax reports the SUM over the episode for every
        # metric except those whose name ends in `per_step`, and the episode
        # length lives under `avg_episode_length`, not `episode_length` — asking
        # for the wrong key returns the default and prints nan forever.
        r = float(metrics.get("eval/episode_reward", float("nan")))
        ln = float(metrics.get("eval/avg_episode_length", float("nan")))
        trk = float(metrics.get("eval/episode_track_err_xy_per_step", float("nan")))
        vx = float(metrics.get("eval/episode_vx_body_per_step", float("nan")))
        row = dict(step=int(step), reward=r, ep_len=ln, track_err_xy=trk,
                   vx_body=vx, seconds=el)
        history.append(row)
        print(f"  {step/1e6:7.2f} M  reward {r:9.2f}  ep_len {ln:6.1f}"
              f"  track_err {trk:6.3f} m/s  vx {vx:+6.3f} m/s  {el/60:6.1f} min",
              flush=True)

    train = functools.partial(
        ppo.train,
        num_timesteps=a.num_timesteps,
        num_envs=a.num_envs,
        episode_length=a.episode_length,
        batch_size=a.batch_size,
        unroll_length=a.unroll_length,
        num_minibatches=a.num_minibatches,
        num_updates_per_batch=a.updates_per_batch,
        learning_rate=a.learning_rate,
        entropy_cost=a.entropy_cost,
        discounting=a.discounting,
        num_evals=a.num_evals,
        reward_scaling=1.0,
        normalize_observations=True,
        action_repeat=1,
        gae_lambda=0.95,
        clipping_epsilon=0.2,
        max_grad_norm=1.0,
        network_factory=networks,
        randomization_fn=randomization,
        seed=a.seed)

    print("\n  step        reward      ep_len   tracking       vx       elapsed")
    make_policy, params, _ = train(environment=env, eval_env=eval_env,
                                   progress_fn=progress)

    brax_io_model.save_params(os.path.join(out, "params"), params)
    meta = dict(
        args=vars(a), history=history,
        wall_clock_min=(time.time() - t0) / 60.0,
        actuator=dict(fitted=p.fitted, source=p.source),
        note=("params/st3215.json is NOT a fit — this policy is trained against "
              "the datasheet servo. Run robot/bench/sweep.py and fit_bam.py, then "
              "retrain, before this goes anywhere near hardware."
              if not p.fitted else "trained against a fitted actuator"),
    )
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nwrote {out}/params and run.json  ({meta['wall_clock_min']:.1f} min)")
    print("runs/ is gitignored. eval.py is the honest number — the line above is "
          "the training environment reporting on itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
