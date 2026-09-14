#!/usr/bin/env python
"""
export_onnx.py — step 6: the policy and its observation normaliser as ONE graph.

    python export_onnx.py runs/<name>                 # -> runs/<name>/policy.onnx + policy.json
    python export_onnx.py runs/<name> --out policy    # -> policy.onnx + policy.json anywhere
    python export_onnx.py runs/<name> --bench         # ... and time it on this CPU

Input  `obs`    float32 [N, OBS_SIZE]   the stacked observation, exactly as env.stack_obs
                                        returns it (newest frame first, raw units)
Output `action` float32 [N, 12]         in [-1, 1]; target = stance + action * ACTION_SCALE,
                                        clipped to the soft limits — the robot side of it
                                        is robot/runtime/policy.py

Why one graph
-------------
brax keeps the running-statistics normaliser OUTSIDE the network: the policy is
`tanh(mlp((obs - mean) / std))[:12]`, and `mean`/`std` are a separate object in the
saved params. Exporting the MLP alone would hand the robot a network that expects
already-normalised input and a JSON of 240 means and 240 stds to apply first — two
files that can drift apart, and the drift is silent (the action just gets worse).
So the normaliser is baked in as the first two nodes, and the only numbers the
runtime needs beside the graph are the ones it cannot derive: the action scale,
the stance and the limits. Those go in the JSON sidecar, read out of the same
robot_params.json the training environment used, with the run name and commit.

The graph is built by hand with onnx.helper rather than traced through jax2onnx or
tf2onnx: it is eleven nodes, and hand-built means no converter dependency and no
converter-specific op that onnxruntime on an aarch64 Orange Pi might lack. Every
op is opset-13 core: Sub, Div, MatMul, Add, Sigmoid, Mul, Slice, Tanh.

What is verified before the file is written
-------------------------------------------
The graph is run through onnxruntime on 512 random observations drawn at the
normaliser's own scale and compared with brax's `make_inference_fn(..., deterministic=True)`
on the same input. The tolerance is 1e-5 absolute; float32 through eight nodes
does not accumulate more than that, so a larger difference is a wrong graph, not
precision, and the export refuses. The sidecar records the measured maximum.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="runs/<name> or runs/<name>/ckpt/<step>: a directory "
                                "with brax `params` in it")
    ap.add_argument("--out", default=None,
                    help="output path without extension (default: <run>/policy)")
    ap.add_argument("--bench", action="store_true",
                    help="time single-observation inference on this CPU")
    ap.add_argument("--n-check", type=int, default=512)
    return ap.parse_args()


def load_policy_arrays(run_dir):
    """(mean, std, [(W, b), ...]) as float32 numpy, from brax's saved params."""
    from brax.io import model as brax_io_model
    params = brax_io_model.load_params(os.path.join(run_dir, "params"))
    if len(params) < 2:
        sys.exit(f"{run_dir}/params: expected (normaliser, policy[, value]), got {len(params)}")
    norm, policy = params[0], params[1]
    mean = np.asarray(norm.mean, np.float32)
    std = np.asarray(norm.std, np.float32)
    layers = policy["params"]
    names = sorted(layers, key=lambda n: int(n.split("_")[-1]))     # hidden_0 .. hidden_3
    dense = [(np.asarray(layers[n]["kernel"], np.float32),
              np.asarray(layers[n]["bias"], np.float32)) for n in names]
    return mean, std, dense, params


def build_graph(mean, std, dense, action_size):
    """The ONNX ModelProto: normalise -> (Dense, swish) x (n-1) -> Dense -> Slice -> Tanh."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    obs_size = mean.shape[0]
    inits = [numpy_helper.from_array(mean, "norm_mean"),
             numpy_helper.from_array(std, "norm_std")]
    nodes = [helper.make_node("Sub", ["obs", "norm_mean"], ["x_centred"]),
             helper.make_node("Div", ["x_centred", "norm_std"], ["h0"])]
    h = "h0"
    for i, (W, b) in enumerate(dense):
        inits += [numpy_helper.from_array(W, f"W{i}"), numpy_helper.from_array(b, f"b{i}")]
        nodes += [helper.make_node("MatMul", [h, f"W{i}"], [f"z{i}_mm"]),
                  helper.make_node("Add", [f"z{i}_mm", f"b{i}"], [f"z{i}"])]
        if i < len(dense) - 1:
            # swish(x) = x * sigmoid(x): brax's linen.swish, the PPO default.
            nodes += [helper.make_node("Sigmoid", [f"z{i}"], [f"s{i}"]),
                      helper.make_node("Mul", [f"z{i}", f"s{i}"], [f"h{i+1}"])]
            h = f"h{i+1}"
        else:
            h = f"z{i}"
    # The last layer emits (loc, scale) side by side; the deterministic action is
    # tanh(loc) — brax's NormalTanhDistribution.mode(). scale is dropped here.
    inits += [numpy_helper.from_array(np.array([0], np.int64), "sl_start"),
              numpy_helper.from_array(np.array([action_size], np.int64), "sl_end"),
              numpy_helper.from_array(np.array([1], np.int64), "sl_axes")]
    nodes += [helper.make_node("Slice", [h, "sl_start", "sl_end", "sl_axes"], ["loc"]),
              helper.make_node("Tanh", ["loc"], ["action"])]
    graph = helper.make_graph(
        nodes, "smalldog_policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["N", obs_size])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["N", action_size])],
        initializer=inits)
    model = helper.make_model(graph, producer_name="smalldog rl/export_onnx.py",
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def main():
    a = parse()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import jaxenv
    jaxenv.configure(0.10, platforms="cpu")
    import jax
    import onnxruntime as ort
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    import model as model_mod
    from env import check_obs_width
    from env.walk import (ACTION_SCALE, CTRL_HZ, OBS_FRAME, OBS_HIST, OBS_SCALE_ACCEL,
                          OBS_SCALE_GYRO, OBS_SCALE_QVEL, OBS_SIZE)

    run_dir = a.run.rstrip("/")
    out = a.out or os.path.join(run_dir, "policy")
    mean, std, dense, params = load_policy_arrays(run_dir)
    check_obs_width(params, OBS_SIZE, run_dir)
    action_size = dense[-1][0].shape[1] // 2
    print(f"run       {run_dir}")
    print(f"policy    {mean.shape[0]} -> " + " -> ".join(str(W.shape[1]) for W, _ in dense)
          + f"  (last layer is loc|scale, {action_size} actions)")

    model = build_graph(mean, std, dense, action_size)

    # ---- the reference: brax's own deterministic inference on the same params
    networks = ppo_networks.make_ppo_networks(
        observation_size=OBS_SIZE, action_size=action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(W.shape[1] for W, _ in dense[:-1]),
        value_hidden_layer_sizes=(256, 256, 256))
    policy = jax.jit(ppo_networks.make_inference_fn(networks)(params[:2], deterministic=True))
    rng = np.random.default_rng(0)
    obs = (mean + std * rng.standard_normal((a.n_check, OBS_SIZE))).astype(np.float32)
    ref = np.asarray(jax.vmap(lambda o: policy(o, jax.random.PRNGKey(0))[0])(obs))

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    got = sess.run(["action"], {"obs": obs})[0]
    err = float(np.abs(got - ref).max())
    print(f"verify    onnxruntime vs brax on {a.n_check} observations: max |diff| = {err:.2e}")
    if err > 1e-5:
        sys.exit("!! the graph does not reproduce the policy — not written")

    # ---- write
    import onnx
    onnx.save(model, out + ".onnx")
    P = model_mod.robot_params()
    lo, hi = model_mod.limits(P, soft=True)
    mj, _ = model_mod.build()
    q0 = model_mod.stance_qpos(mj, P)
    qadr, _, _ = model_mod.joint_order(mj, P)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         text=True).strip()
    except Exception:
        commit = None
    run_json = os.path.join(run_dir, "run.json")
    meta = json.load(open(run_json)) if os.path.exists(run_json) else {}
    side = dict(
        run=os.path.basename(run_dir), commit=commit, trained_steps=meta.get("args", {}).get("num_timesteps"),
        onnx=os.path.basename(out) + ".onnx", verify_max_abs_diff=err,
        ctrl_hz=CTRL_HZ, obs_hist=OBS_HIST, obs_frame=OBS_FRAME, obs_size=OBS_SIZE,
        obs_layout=["gravity_b[3]", "gyro[3] * obs_scale_gyro", "accel[3] * obs_scale_accel",
                    "qpos_j[12] - stance", "qvel_j[12] * obs_scale_qvel", "last_action[12]",
                    "command[3] (vx, vy, yaw_rate)"],
        obs_scale_gyro=OBS_SCALE_GYRO, obs_scale_accel=OBS_SCALE_ACCEL, obs_scale_qvel=OBS_SCALE_QVEL,
        history="newest frame first; at reset all OBS_HIST frames are copies of the first",
        action_scale_rad=ACTION_SCALE, joint_names=P["joint_names"],
        stance_rad=[float(v) for v in q0[qadr]],
        soft_lo_rad=[float(v) for v in lo], soft_hi_rad=[float(v) for v in hi],
        target="clip(stance + action * action_scale_rad, soft_lo, soft_hi)",
        joint_velocity_limit_rad_s=P["joint_velocity_limit"],
        source="rl/export_onnx.py",
    )
    with open(out + ".json", "w") as f:
        json.dump(side, f, indent=2)
    size_kb = os.path.getsize(out + ".onnx") / 1024
    print(f"wrote     {out}.onnx ({size_kb:.0f} kB) and {out}.json")

    if a.bench:
        sess = ort.InferenceSession(out + ".onnx", providers=["CPUExecutionProvider"])
        one = obs[:1]
        for _ in range(50):
            sess.run(["action"], {"obs": one})
        t = []
        for _ in range(500):
            t0 = time.perf_counter(); sess.run(["action"], {"obs": one}); t.append(time.perf_counter() - t0)
        t = np.array(t) * 1e6
        print(f"bench     one observation: p50 {np.percentile(t, 50):.0f} us, "
              f"p95 {np.percentile(t, 95):.0f} us, max {t.max():.0f} us "
              f"of a {1e6/CTRL_HZ:.0f} us tick")
    return 0


if __name__ == "__main__":
    sys.exit(main())
