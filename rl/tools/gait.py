#!/usr/bin/env python
"""
gait.py -- what the walk actually looks like, in numbers.

    RUN=runs/<run>             python tools/gait.py
    RUN=runs/<run>/ckpt/<step> SEEDS=8 CMDS=0.2,0.4 python tools/gait.py

Per commanded speed: peak-to-peak foot lift on each leg, the dominant frequency
of each foot's height signal, the stride that implies, and how much of the
action signal sits above 10 Hz. The last one is the buzz -- a policy whose
action power is half above 10 Hz on a 50 Hz controller is chattering, not
walking, and no reward curve will say so.

SEEDS is not optional decoration. Each seed scatters the initial joint angles by
1.1 deg and the drop height by 3 mm, and the spread that comes back is the only
thing that says whether a difference between two policies is a difference. It
was worth 8 seeds on the first comparison this tool was written for: the gait
frequency separated by 1.17 Hz against a 0.10 Hz spread and was real, while the
foot lift moved 1.9 mm against a 1.2 mm standard error and was not.

Point it at two runs' checkpoints at the SAME step. Comparing a 33 M checkpoint
against a 60 M one measures training, not the change you made.

Runs on the CPU (JAX_PLATFORMS=cpu) so it can be used while a run holds the GPU.
That is now set rather than claimed: jaxenv.configure(platforms="cpu").
"""
import os, sys, json, numpy as np
_RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RL); os.chdir(_RL)
import jaxenv; jaxenv.configure(0.10, platforms="cpu")
import jax, mujoco
from brax.io import model as brax_io_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import actuator, model as model_mod
from env import Walk, assemble_obs, stack_obs, init_hist
from env.walk import ACTION_SCALE, CTRL_HZ

RUN = os.environ.get("RUN")
if not RUN:
    raise SystemExit("set RUN=runs/<run> or RUN=runs/<run>/ckpt/<step>")
SECONDS = float(os.environ.get("SECONDS", "6.0"))
SEEDS = int(os.environ.get("SEEDS", "8"))
CMDS = [float(c) for c in os.environ.get("CMDS", "0.2,0.4").split(",")]
BOXES = int(os.environ.get("BOXES", "0"))       # raise this many in the robot's path
BOXH = float(os.environ.get("BOXH", "0.022"))   # their top height, m; 0.022 is the domain_rand cap
SETTLE = 1.0                       # s discarded: the drop, not the gait

targs = json.load(open(f"{RUN}/run.json"))["args"]
env = Walk(terrain=targs["terrain"], n_boxes=targs["boxes"])
nets = ppo_networks.make_ppo_networks(
    observation_size=env.observation_size, action_size=env.action_size,
    preprocess_observations_fn=running_statistics.normalize,
    policy_hidden_layer_sizes=(128, 128, 128),
    value_hidden_layer_sizes=(256, 256, 256))
prm = brax_io_model.load_params(f"{RUN}/params")
if len(prm) > 2:
    prm = prm[:2]
pol = jax.jit(ppo_networks.make_inference_fn(nets)(prm, deterministic=True))

p = actuator.load()
mj, _ = model_mod.build(terrain=False, n_boxes=0, mjx_safe=True)
P = model_mod.robot_params()
qadr, vadr, act = model_mod.joint_order(mj, P)
lo, hi = model_mod.limits(P, soft=True)
q0 = model_mod.stance_qpos(mj, P); st_j = q0[qadr]
sadr = lambda n: int(mj.sensor_adr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SENSOR, n)])
aq, ag, aa = sadr("imu_quat"), sadr("imu_gyro"), sadr("imu_accel")
LEGS = ["fl", "fr", "rl", "rr"]
fsid = [mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, f"{l}_foot_site") for l in LEGS]
dt = 1 / CTRL_HZ
n_sub = int(round(dt / mj.opt.timestep))


def model_for(rng):
    """A freshly compiled model with the boxes in the corridor the robot walks.

    Compiled, not edited: model.compile_with_boxes explains why writing
    geom_pos on a live model gives you a box that draws but does not collide.

    The corridor matters too. model.BOX_PATCH_M scatters over +-4 m square,
    which is the right order for where an episode ends up and the wrong shape
    for where it goes -- the robot leaves the origin forwards and stays inside
    about +-0.4 m laterally. At 24 boxes and density 0.275 that is 0.22 boxes
    met per 4 m episode: one box every five episodes is not a curriculum. Here
    BOXES=n means n boxes actually underfoot.
    """
    tops = rng.uniform(0.0, BOXH, BOXES)
    x = rng.uniform(0.5, 4.0, BOXES)                         # not under the start
    y = rng.uniform(-0.35, 0.35, BOXES)
    return model_cache(tuple(tops), tuple(x), tuple(y))


_CACHE = {}


def model_cache(tops, x, y):
    key = (tops, x, y)
    if key not in _CACHE:
        _CACHE[key] = model_mod.compile_with_boxes(
            list(tops), list(zip(x, y)), mjx_safe=True)[0]
    return _CACHE[key]


def rollout(cmd, seed, perturb=True):
    """-> (foot heights, actions, distance, seconds) after the settle window."""
    rng = np.random.default_rng(seed)
    mjs = model_for(rng) if BOXES else mj
    d = mujoco.MjData(mjs); d.qpos[:] = q0
    if perturb:
        d.qpos[qadr] += rng.normal(0, 0.02, 12)     # 1.1 deg of joint scatter
        d.qpos[2] += rng.normal(0, 0.003)           # 3 mm of drop height
    mujoco.mj_forward(mjs, d)
    la = np.zeros(12); hist = None; FZ = []; ACT = []; x0 = None
    for k in range(int(SECONDS / dt)):
        fr_, _ = assemble_obs(
            quat=d.sensordata[aq:aq + 4], gyro=d.sensordata[ag:ag + 3],
            accel=d.sensordata[aa:aa + 3], qpos_j=d.qpos[qadr], qvel_j=d.qvel[vadr],
            stance_j=st_j, last_action=la, command=cmd, xp=np)
        if hist is None:
            hist = init_hist(fr_, xp=np); obs = hist.reshape(-1)
        else:
            obs, hist = stack_obs(hist, fr_, xp=np)
        a_, _ = pol(obs, jax.random.PRNGKey(0)); a_ = np.asarray(a_); la = a_
        tgt = np.clip(st_j + a_ * ACTION_SCALE, lo, hi)
        for _ in range(n_sub):
            q, w = d.qpos[qadr], d.qvel[vadr]
            d.ctrl[act] = actuator.motor_torque(
                p, actuator.duty(p, tgt - q, w, xp=np) * 12.0, w, xp=np)
            mujoco.mj_step(mjs, d)
        if k == int(SETTLE / dt):
            x0 = d.qpos[0]
        if k >= int(SETTLE / dt):
            FZ.append([d.site_xpos[i][2] for i in fsid]); ACT.append(a_.copy())
        if d.qpos[2] < 0.08:
            break
    n = len(FZ)
    return np.array(FZ), np.array(ACT), (d.qpos[0] - x0 if n else 0.0), n * dt


def spectrum(FZ, ACT):
    """Dominant foot-height frequency and the share of action power above 10 Hz.

    The bin spacing comes from `dt` and the sample count, not from the rollout's
    wall duration: a run that ended early has fewer samples, and rfftfreq already
    knows that. It used to take a `T` argument and ignore it, which is the kind
    of parameter that looks like it is doing the thing it is named for.
    """
    n = len(FZ); ff = np.fft.rfftfreq(n, dt)
    dom = []
    for i in range(4):
        s = FZ[:, i] - FZ[:, i].mean()
        Pw = np.abs(np.fft.rfft(s)) ** 2; Pw[0] = 0
        dom.append(ff[np.argmax(Pw)])
    As = ACT - ACT.mean(0)
    PA = np.abs(np.fft.rfft(As, axis=0)) ** 2; PA[0] = 0
    return np.mean(dom), PA[ff > 10].sum() / PA.sum()


print(f"{RUN}   {SEEDS} seed(s) x {SECONDS:g} s, first {SETTLE:g} s discarded"
      + (f", {BOXES} boxes up to {BOXH*1000:.0f} mm in the corridor" if BOXES else ", flat"))
for cmd_v in CMDS:
    cmd = np.array([cmd_v, 0.0, 0.0])
    L, F, V, H, fell = [], [], [], [], 0
    for s in range(SEEDS):
        # A fixed seed even at SEEDS=1. `None` used to be passed there, which
        # skipped the joint scatter -- fine -- but also handed
        # np.random.default_rng(None) to the box placement, so a single-seed run
        # with BOXES>0 scattered a different course every time and was not
        # comparable with itself. Seed 0 is the unperturbed pose either way;
        # `perturb` is what says whether to scatter the joints.
        FZ, ACT, dx, T = rollout(cmd, s, perturb=SEEDS > 1)
        if len(FZ) < 20:
            fell += 1; continue
        L.append((FZ.max(0) - FZ.min(0)) * 1000)
        f, hf = spectrum(FZ, ACT)
        F.append(f); V.append(dx / T); H.append(hf)
    if not L:
        print(f"\n=== command vx {cmd_v} m/s: fell on {fell}/{SEEDS} seeds"); continue
    L = np.array(L); F = np.array(F); V = np.array(V); H = np.array(H)
    se = lambda x: x.std() / max(1, np.sqrt(len(x)))
    print(f"\n=== command vx {cmd_v} m/s" + (f"   ({fell} seed(s) fell)" if fell else ""))
    print("  peak foot lift   " + "  ".join(
        f"{l} {L[:, i].mean():5.1f}" + (f"+-{L[:, i].std():.1f}" if len(L) > 1 else "")
        for i, l in enumerate(LEGS)) + " mm")
    if len(L) > 1:
        print(f"  lift pooled      {L.mean():5.1f} +- {L.std():.1f} mm  "
              f"(se of the mean {se(L.reshape(-1)):.1f})")
    print(f"  gait             {F.mean():5.2f}" + (f" +- {F.std():.2f}" if len(F) > 1 else "")
          + f" Hz   -> stride {V.mean() / F.mean() * 1000:.0f} mm, "
            f"{50 / F.mean():.1f} control steps per cycle")
    print(f"  speed            {V.mean():5.3f}" + (f" +- {V.std():.3f}" if len(V) > 1 else "")
          + " m/s")
    print(f"  action power >10 Hz {H.mean() * 100:.0f}%" +
          (f" +- {H.std() * 100:.0f}" if len(H) > 1 else "") + "   (Nyquist is 25 Hz)")
