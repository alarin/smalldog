#!/usr/bin/env python
# RAW docstring: the ascii art below contains \_ , and python 3.12+ reports a
# lone backslash-underscore in a normal string literal as a SyntaxWarning.
r"""
drift.py -- the sideways displacement, split into the two things that cause it.

    RUN=runs/<run>             python tools/drift.py
    RUN=runs/<run>/ckpt/<step> SEEDS=8 SECONDS=10 CMDV=0.4 python tools/drift.py

    world lateral velocity  y' = u*sin(theta) + v*cos(theta)
                                 \_________/   \_________/
                                  the robot     the robot
                                  is pointed    is sliding
                                  sideways      sideways

The split is the whole point. A policy that walks perfectly straight along its
own nose while the nose slowly turns, and a policy that points straight ahead
and crabs, both show the same y in a video, and they need opposite fixes: the
first is a heading problem the yaw rate can see, the second is a contact problem
it cannot. Chasing the wrong one costs a run. It cost this project one.

Seeds scatter the initial joint angles by 1.1 deg and the drop height by 3 mm.
Seed 0 is unperturbed and prints the trajectory, which is the diagnostic view;
the pooled mean +- sd underneath is the comparable number. A single rollout is
not a measurement -- see the note in tools/gait.py about the 1.9 mm that wasn't.

Also prints mean action per joint. A roll actuator whose mean sits at +0.34 on
one side and -0.43 on the other is a policy leaning, and that shows up here
before it shows up anywhere else.

Runs on the CPU (JAX_PLATFORMS=cpu) so it can be used while a run holds the GPU.
That is now set rather than claimed: jaxenv.configure(platforms="cpu").
"""
import os, sys, json
_RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RL); os.chdir(_RL)
import jaxenv; jaxenv.configure(0.10, platforms="cpu")

import numpy as np, jax, mujoco
from brax.io import model as brax_io_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import actuator, model as model_mod
from env import Walk, assemble_obs, stack_obs, init_hist
from env.walk import ACTION_SCALE, CTRL_HZ

RUN = os.environ.get("RUN")
if not RUN:
    raise SystemExit("set RUN=runs/<run> or RUN=runs/<run>/ckpt/<step>")
SECONDS = float(os.environ.get("SECONDS", "10.0"))
SEEDS = int(os.environ.get("SEEDS", "8"))
CMD = np.array([float(os.environ.get("CMDV", "0.4")), 0.0, 0.0])

targs = json.load(open(f"{RUN}/run.json"))["args"]
env = Walk(terrain=targs["terrain"], n_boxes=targs["boxes"])
nets = ppo_networks.make_ppo_networks(
    observation_size=env.observation_size, action_size=env.action_size,
    preprocess_observations_fn=running_statistics.normalize,
    policy_hidden_layer_sizes=(128, 128, 128), value_hidden_layer_sizes=(256, 256, 256))
prm = brax_io_model.load_params(f"{RUN}/params")
if len(prm) > 2:
    prm = prm[:2]
policy_jit = jax.jit(ppo_networks.make_inference_fn(nets)(prm, deterministic=True))

p = actuator.load()
mj, _ = model_mod.build(terrain=False, n_boxes=0, mjx_safe=True)
P = model_mod.robot_params()
qadr, vadr, act = model_mod.joint_order(mj, P)
lo, hi = model_mod.limits(P, soft=True)
q0 = model_mod.stance_qpos(mj, P); stance_j = q0[qadr]
sadr = lambda n: int(mj.sensor_adr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SENSOR, n)])
aq, ag, aa = sadr("imu_quat"), sadr("imu_gyro"), sadr("imu_accel")
dt = 1 / CTRL_HZ
n_sub = int(round(dt / mj.opt.timestep))
names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i)) for i in act]


def rollout(seed):
    d = mujoco.MjData(mj); d.qpos[:] = q0
    if seed:                                  # seed 0 is the unperturbed trajectory
        rng = np.random.default_rng(seed)
        d.qpos[qadr] += rng.normal(0, 0.02, 12)
        d.qpos[2] += rng.normal(0, 0.003)
    mujoco.mj_forward(mj, d)
    last_action = np.zeros(12); hist = None; rows = []
    for k in range(int(SECONDS / dt)):
        fr_, _ = assemble_obs(
            quat=d.sensordata[aq:aq + 4], gyro=d.sensordata[ag:ag + 3],
            accel=d.sensordata[aa:aa + 3], qpos_j=d.qpos[qadr], qvel_j=d.qvel[vadr],
            stance_j=stance_j, last_action=last_action, command=CMD, xp=np)
        if hist is None:
            hist = init_hist(fr_, xp=np); obs = hist.reshape(-1)
        else:
            obs, hist = stack_obs(hist, fr_, xp=np)
        a_, _ = policy_jit(obs, jax.random.PRNGKey(0)); a_ = np.asarray(a_)
        last_action = a_
        target = np.clip(stance_j + a_ * ACTION_SCALE, lo, hi)
        for _ in range(n_sub):
            q, w = d.qpos[qadr], d.qvel[vadr]
            d.ctrl[act] = actuator.motor_torque(
                p, actuator.duty(p, target - q, w, xp=np) * 12.0, w, xp=np,
                tau_c_external=True)
            mujoco.mj_step(mj, d)
        R = np.zeros(9); mujoco.mju_quat2Mat(R, d.qpos[3:7]); R = R.reshape(3, 3)
        vb = R.T @ d.qvel[0:3]
        rows.append((k * dt, d.qpos[0], d.qpos[1], np.arctan2(R[1, 0], R[0, 0]),
                     vb[0], vb[1], d.qvel[5], a_.copy()))
        if d.qpos[2] < 0.08:
            break
    cols = [np.array([r[i] for r in rows]) for i in range(7)]
    return (*cols, np.array([r[7] for r in rows]),
            d.qpos[2] >= 0.08)


print(f"{RUN}   {SEEDS} seed(s) x {SECONDS:g} s, commanded vx {CMD[0]:g} m/s")
fin_x, fin_y, fin_h, fin_s, fell = [], [], [], [], 0
A0 = None; ACTS = []
for s in range(SEEDS):
    t, X, Y, TH, U, V, YR, A, ok = rollout(s)
    if not ok or len(t) < 20:
        fell += 1; continue
    head = np.cumsum(U * np.sin(TH)) * dt
    slip = np.cumsum(V * np.cos(TH)) * dt
    fin_x.append(X[-1] * 1000); fin_y.append(Y[-1] * 1000)
    fin_h.append(head[-1] * 1000); fin_s.append(slip[-1] * 1000)
    ACTS.append(A)
    if s == 0:
        A0 = A
        print("\nseed 0, unperturbed:")
        print(f"{'t':>5}{'x mm':>9}{'y mm':>9}{'head deg':>10}{'u m/s':>8}"
              f"{'v m/s':>8}{'yaw r/s':>9}{'y_head':>9}{'y_slip':>9}")
        for i in range(0, len(t), max(1, len(t) // 20)):
            print(f"{t[i]:5.1f}{X[i]*1000:9.1f}{Y[i]*1000:9.1f}{np.degrees(TH[i]):10.2f}"
                  f"{U[i]:8.3f}{V[i]:8.3f}{YR[i]:9.3f}{head[i]*1000:9.1f}{slip[i]*1000:9.1f}")

if not fin_x:
    raise SystemExit("fell on every seed")
sd = lambda a: f" +- {np.std(a):.0f}" if len(a) > 1 else ""
print(f"\nover {len(fin_x)} seed(s)" + (f", {fell} fell" if fell else "") + ":")
print(f"  travelled x   {np.mean(fin_x):+8.0f}{sd(fin_x)} mm")
print(f"  world y       {np.mean(fin_y):+8.0f}{sd(fin_y)} mm")
print(f"    of which heading {np.mean(fin_h):+8.0f}{sd(fin_h)} mm")
print(f"    of which slip    {np.mean(fin_s):+8.0f}{sd(fin_s)} mm")

# Mirror symmetry. A straight-ahead command is symmetric under y -> -y and the
# robot is too (model.py places the four hips as translated copies at +-36 mm
# with identity quaternions, and check below: leg masses and mirrored ipos agree
# to 0.000 g / 0.000 mm, stance CoM y is -0.03 mm). So the time-averaged policy
# should be symmetric, and any residual is the policy leaning of its own accord.
#
# The sign convention is the trap. Reflecting through the x-z plane sends an
# axial vector to (-wx, +wy, -wz), and every joint axis here is +x for roll and
# +y for pitch and knee, in bodies that are NOT rotated. So symmetry demands
#
#     roll        a[left] = -a[right]      residual = a[left] + a[right]
#     pitch/knee  a[left] = +a[right]      residual = a[left] - a[right]
#
# Read the raw means without that and an opposite-signed roll pair looks like a
# dramatic lean when it is the one thing that is behaving. That mistake was made
# here, out loud, before this block was written.
print("\nmirror residual, mean over seeds (0 = symmetric):")
res = {}
for pair in (("fl", "fr"), ("rl", "rr")):
    for kind in ("roll", "pitch", "knee"):
        il = names.index(f"{pair[0]}_{kind}"); ir = names.index(f"{pair[1]}_{kind}")
        vals = [A[:, il].mean() + (A[:, ir].mean() if kind == "roll" else -A[:, ir].mean())
                for A in ACTS]
        res[f"{pair[0]}/{pair[1]} {kind}"] = (float(np.mean(vals)), float(np.std(vals)))
for k, (mu, sg) in sorted(res.items(), key=lambda z: -abs(z[1][0])):
    bar = "#" * min(40, int(abs(mu) * 100))
    print(f"    {k:<16}{mu:+.4f}" + (f" +- {sg:.4f}" if len(ACTS) > 1 else "")
          + f"   {bar}")

# A0 is seed 0's action trace, and seed 0 is the one seed that can be MISSING
# from the pooled set: it is the unperturbed rollout, and if it is the rollout
# that fell, the loop above skipped it. Without this guard the summary below
# ends in AttributeError on None, after every number it was going to print.
if A0 is None:
    print("\nseed 0 fell, so there is no unperturbed action trace to print. The "
          "pooled numbers above stand; re-run with a different SEEDS to see one.")
else:
    print("\nmean action per joint, seed 0 (policy output, before scaling):")
    for n, m in sorted(zip(names, A0.mean(axis=0)), key=lambda z: z[0]):
        print(f"    {n:<16}{m:+.4f}")
