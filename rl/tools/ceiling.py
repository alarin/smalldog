#!/usr/bin/env python
"""
ceiling.py -- is the policy slow, or is the servo?

    RUN=runs/<run>            python tools/ceiling.py
    RUN=runs/<run>/ckpt/<step> python tools/ceiling.py

Commands a trained policy 0.2 / 0.4 / 0.6 / 0.8 m/s through the CPU engine with
the nominal servo, and prints what it answers alongside the joint speeds it used
to answer. The column that matters is p95 against joint_velocity_limit: a policy
whose p95 sits at 96 % of the motor's no-load speed and does not move when you
ask for more is not under-trained, it is geared out of the command.

Written after a run was launched to fix a "0.6 m/s tops out at 0.406" defect
that turned out to be arithmetic. See the Commands docstring in env/walk.py.

Runs on the CPU (JAX_PLATFORMS=cpu) so it can be used while a run holds the GPU.
"""
import os, sys, json, numpy as np
_RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RL); os.chdir(_RL)
import jaxenv; jaxenv.configure(0.10)
import jax, mujoco
from brax.io import model as brax_io_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import actuator, model as model_mod
from env import Walk, assemble_obs, stack_obs, init_hist
RUN=os.environ["RUN"]; SECONDS=6.0
targs=json.load(open(f"{RUN}/run.json"))["args"]
env=Walk(terrain=targs["terrain"],n_boxes=targs["boxes"])
nets=ppo_networks.make_ppo_networks(observation_size=env.observation_size,
 action_size=env.action_size,preprocess_observations_fn=running_statistics.normalize,
 policy_hidden_layer_sizes=(128,128,128),value_hidden_layer_sizes=(256,256,256))
prm=brax_io_model.load_params(f"{RUN}/params")
if len(prm)>2: prm=prm[:2]
pol=jax.jit(ppo_networks.make_inference_fn(nets)(prm,deterministic=True))
p=actuator.load(); mj,_=model_mod.build(terrain=False,n_boxes=0,mjx_safe=True)
P=model_mod.robot_params(); qadr,vadr,act=model_mod.joint_order(mj,P)
lo,hi=model_mod.limits(P,soft=True); q0=model_mod.stance_qpos(mj,P); st_j=q0[qadr]
VLIM=float(P["joint_velocity_limit"])
sadr=lambda n:int(mj.sensor_adr[mujoco.mj_name2id(mj,mujoco.mjtObj.mjOBJ_SENSOR,n)])
aq,ag,aa=sadr("imu_quat"),sadr("imu_gyro"),sadr("imu_accel")
print(f"joint_velocity_limit (= MEASURED no-load, 2026-09-11) {VLIM:.2f} rad/s")
print(f"{'cmd':>5}{'got':>8}{'ratio':>7}{'|w|p50':>8}{'p95':>7}{'max':>7}{'over%':>7}{'stallish%':>10}")
for CMDV in (0.2,0.4,0.6,0.8):
    CMD=np.array([CMDV,0.0,0.0])
    d=mujoco.MjData(mj); d.qpos[:]=q0; mujoco.mj_forward(mj,d)
    dt=1/50.0; n_sub=int(round(dt/mj.opt.timestep)); la=np.zeros(12); hist=None
    W=[]; BX0=None; alive=True
    for k in range(int(SECONDS/dt)):
        fr_,_=assemble_obs(quat=d.sensordata[aq:aq+4],gyro=d.sensordata[ag:ag+3],
            accel=d.sensordata[aa:aa+3],qpos_j=d.qpos[qadr],qvel_j=d.qvel[vadr],
            stance_j=st_j,last_action=la,command=CMD,xp=np)
        if hist is None: hist=init_hist(fr_,xp=np); obs=hist.reshape(-1)
        else: obs,hist=stack_obs(hist,fr_,xp=np)
        a_,_=pol(obs,jax.random.PRNGKey(0)); a_=np.asarray(a_); la=a_
        tgt=np.clip(st_j+a_*0.35,lo,hi)
        for _ in range(n_sub):
            q,w=d.qpos[qadr],d.qvel[vadr]
            d.ctrl[act]=actuator.motor_torque(p,actuator.duty(p,tgt-q,w,xp=np)*12.0,w,xp=np)
            mujoco.mj_step(mj,d)
        if k==int(0.5/dt): BX0=d.qpos[0]
        if k>=int(0.5/dt): W.append(np.abs(d.qvel[vadr]).copy())
        if d.qpos[2]<0.08: alive=False; break
    W=np.array(W); v=(d.qpos[0]-BX0)/(len(W)*dt) if BX0 is not None and len(W) else float('nan')
    over=(W>VLIM).mean()*100; stall=(W>0.85*VLIM).mean()*100
    print(f"{CMDV:>5.1f}{v:>8.3f}{v/CMDV:>7.2f}{np.median(W):>8.2f}"
          f"{np.percentile(W,95):>7.2f}{W.max():>7.2f}{over:>7.1f}{stall:>10.1f}"
          + ("" if alive else "   FELL"))
