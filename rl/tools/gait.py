#!/usr/bin/env python
"""
gait.py -- what the walk actually looks like, in numbers.

    RUN=runs/<run>             python tools/gait.py
    RUN=runs/<run>/ckpt/<step> python tools/gait.py

Per commanded speed: peak-to-peak foot lift on each leg, the dominant frequency
of each foot's height signal, the stride that implies, and how much of the
action signal sits above 10 Hz. The last one is the buzz -- a policy whose
action power is half above 10 Hz on a 50 Hz controller is chattering, not
walking, and no amount of looking at the reward curve will say so.

Point it at two runs' checkpoints at the SAME step to compare gaits; comparing a
33 M checkpoint against a 60 M one measures training, not the change you made.

Runs on the CPU (JAX_PLATFORMS=cpu) so it can be used while a run holds the GPU.
"""
import os, sys, json, numpy as np
_RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RL); os.chdir(_RL)
import jaxenv; jaxenv.configure(0.60)
import jax, mujoco
from brax.io import model as brax_io_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import actuator, model as model_mod
from env import Walk, assemble_obs, stack_obs, init_hist
RUN=os.environ.get("RUN","runs/20260831-162549"); SECONDS=6.0
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
sadr=lambda n:int(mj.sensor_adr[mujoco.mj_name2id(mj,mujoco.mjtObj.mjOBJ_SENSOR,n)])
LEGS=["fl","fr","rl","rr"]
fsid=[mujoco.mj_name2id(mj,mujoco.mjtObj.mjOBJ_SITE,f"{l}_foot_site") for l in LEGS]

for CMDV in (0.2, 0.4):
    CMD=np.array([CMDV,0.0,0.0])
    d=mujoco.MjData(mj); d.qpos[:]=q0; mujoco.mj_forward(mj,d)
    aq,ag,aa=sadr("imu_quat"),sadr("imu_gyro"),sadr("imu_accel")
    dt=1/50.0; n_sub=int(round(dt/mj.opt.timestep)); la=np.zeros(12); hist=None
    FZ=[]; BX=[]; ACT=[]
    for k in range(int(SECONDS/dt)):
        fr_,_=assemble_obs(quat=d.sensordata[aq:aq+4],gyro=d.sensordata[ag:ag+3],
            accel=d.sensordata[aa:aa+3],
            qpos_j=d.qpos[qadr],qvel_j=d.qvel[vadr],stance_j=st_j,last_action=la,
            command=CMD,xp=np)
        if hist is None:
            hist=init_hist(fr_,xp=np); obs=hist.reshape(-1)
        else:
            obs,hist=stack_obs(hist,fr_,xp=np)
        a_,_=pol(obs,jax.random.PRNGKey(0)); a_=np.asarray(a_); la=a_
        tgt=np.clip(st_j+a_*0.35,lo,hi)
        for _ in range(n_sub):
            q,w=d.qpos[qadr],d.qvel[vadr]
            d.ctrl[act]=actuator.motor_torque(p,actuator.duty(p,tgt-q,w,xp=np)*12.0,w,xp=np)
            mujoco.mj_step(mj,d)
        FZ.append([d.site_xpos[s][2] for s in fsid]); BX.append(d.qpos[0]); ACT.append(a_.copy())
    FZ=np.array(FZ); BX=np.array(BX); ACT=np.array(ACT)
    n=len(FZ); freqs=np.fft.rfftfreq(n, dt)
    print(f"\n=== command vx {CMDV} m/s ===")
    print(f"  foot lift peak-to-peak: " + "  ".join(
        f"{l} {(FZ[:,i].max()-FZ[:,i].min())*1000:.0f}mm" for i,l in enumerate(LEGS)))
    dom=[]
    for i,l in enumerate(LEGS):
        s=FZ[:,i]-FZ[:,i].mean(); P_=np.abs(np.fft.rfft(s))**2
        P_[0]=0; f=freqs[np.argmax(P_)]; dom.append(f)
        print(f"  {l}: dominant foot-z frequency {f:5.2f} Hz")
    F=float(np.mean(dom)); v=(BX[-1]-BX[0])/(n*dt)
    print(f"  speed {v:.3f} m/s, gait {F:.2f} Hz -> STRIDE {v/F*1000:.0f} mm, "
          f"{50/F:.1f} control steps per cycle")
    # action chatter: how much of the action signal is at the Nyquist end
    As=ACT-ACT.mean(0); PA=np.abs(np.fft.rfft(As,axis=0))**2; PA[0]=0
    hi_frac=PA[freqs>10].sum()/PA.sum()
    print(f"  action power above 10 Hz: {hi_frac*100:.0f}%   (Nyquist is 25 Hz)")
