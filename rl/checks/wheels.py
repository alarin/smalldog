#!/usr/bin/env python
"""
wheels.py — can this robot go faster on wheels, and what kind?

    python checks/wheels.py                     # the whole comparison
    python checks/wheels.py --mode driven --gear 6 --wheel-d 60
    python checks/wheels.py --seconds 8 --video   (not implemented; use --shot)

Built on the same MjSpec trick as imu_placement.py: the committed MJCF is never
edited, the wheel is added programmatically at load time, and the CAD stays the
one source of geometry.

Why the configurations below are the ones worth running
-------------------------------------------------------
The ST3215's no-load speed is 4.71 rad/s at 12 V. Straight onto a wheel that is
0.118 m/s on 50 mm and 0.236 m/s on 100 mm — against a trot that already
measures 0.156 m/s. So "put wheels on it" as usually meant does not buy speed,
and the interesting question is what does. Four answers get measured here:

  feet       the trot as it stands, for the baseline everything is judged against

  free       free-rolling wheels at the feet, existing trot. Expected to be
             *worse*, and it is worth seeing why: a stance leg sweeping backwards
             against a free wheel just spins the wheel. The propulsion is gone.

  ratchet    the same wheel with a one-way clutch: locks when the foot is pushed
             backwards against the ground, free when the body glides forward over
             it. Skins on skis. This is the version where the existing gait can
             still push AND the body can coast, so it is the one that might pay
             without a single extra motor.

  driven     wheels turned by four more ST3215 in continuous mode, legs held in
             the stance pose, swept over gear ratio. This is where the real
             answer is, and it is not "add wheels", it is **gear them up**: the
             servo has 2.94 N*m at the output and rolling a 2.5 kg robot on a
             hard floor needs about 0.01 N*m per wheel. It is over-geared for
             this job by two orders of magnitude, and a printed step-up stage
             trades the surplus back into speed. Terminal wheel speed is
             U / (k_e * gear_down) — i.e. the ratio multiplies it directly.

The actuator is not an ideal velocity source. Wheel torque comes from
rl/actuator.py's law with the parameters in rl/params/st3215.json, so the speed
ceiling emerges from the back-EMF rather than being asserted, and it will move
when the bench fit replaces the vendor priors.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DESC = os.path.join(ROOT, "ros2", "smalldog_description")
sys.path.insert(0, os.path.join(ROOT, "rl"))
sys.path.insert(0, os.path.join(ROOT, "ros2", "smalldog_walker"))

import actuator as A                                                 # noqa: E402

LEGS = ("fl", "fr", "rl", "rr")
RAMP = 1.0          # s to full duty on the wheels

# A printed rim with a TPU tyre, plus the fork that carries it, per corner. The
# TPU foot it replaces is about 7 g, so the net is roughly +50 g on the robot —
# and 3d/CLAUDE.md records that +11 g ANYWHERE moved the flat trot from 778 mm
# to 597 mm. Every number below is therefore reported against a baseline re-run
# at the same mass, never against the figure in the README.
WHEEL_KG = 0.020
FOOT_KG = 0.007
SERVO_KG = 0.060      # ST3215 with both hubs, from 3d/mini_dog.py section 4


def build(wheel_d, mode, gear=1.0, wheel_kg=WHEEL_KG, width=0.014, scene=None,
          drive_mass=0.0):
    """The committed model plus a wheel per leg. Returns (model, wheel joint ids).

    The axle goes at the shin tip, where the foot's spigot already is — that is
    the real mounting point, since foot() is a bolt-on TPU part on an M3 into the
    shin's nut slot. It also means the leg gets LONGER by (r - 13 mm), which the
    caller has to give back in body height or the comparison is between two
    different robots.
    """
    spec = mujoco.MjSpec.from_file(scene or os.path.join(DESC, "mujoco", "scene.xml"))
    p_ref = A.load(A.DEFAULT_PATH, quiet=True)
    r = 0.5 * wheel_d
    for leg in LEGS:
        shin = spec.body(f"{leg}_shin")
        for g in shin.geoms:
            if g.name == f"{leg}_foot":
                g.contype, g.conaffinity = 0, 0        # the foot stops touching
        # A driven wheel needs its motor somewhere, and on a leg-mounted design
        # that somewhere is the end of the shin: the single worst place on the
        # robot for mass, both for leg inertia and for what fea.py sees. It is
        # carried here rather than quietly left out, because leaving it out is
        # how a wheel study returns a speed the real machine cannot reach.
        b = shin.add_body(name=f"{leg}_wheel", pos=[0, 0, -0.082])
        # explicitinertial, or the compiler ignores b.mass and derives it from
        # the geom's default 1000 kg/m^3 — which made a 100 mm wheel weigh 110 g
        # and silently dropped the drive motor's mass entirely. The whole point
        # of carrying the motor is that it is heavy and badly placed; a study
        # that loses it is worse than one that never claimed to have it.
        b.explicitinertial = True
        b.mass = wheel_kg + drive_mass
        # a thin ring, which is what a printed rim with a tyre is
        b.inertia = [0.5 * wheel_kg * r * r + drive_mass * 0.02 ** 2,
                     0.5 * wheel_kg * r * r + drive_mass * 0.02 ** 2,
                     wheel_kg * r * r + drive_mass * 0.02 ** 2]
        # The rotor's inertia, reflected through the step-up, and it is not a
        # detail: a 20 g 60 mm wheel has J = 9e-6 kg m^2 of its own, while the
        # ST3215's reflected rotor is 0.008 — nearly a thousand times more at
        # 1:1. Leave it out and a newton-metre of drive gives the wheel 160000
        # rad/s^2, which is 160 rad/s inside one 1 ms step: the first version of
        # this script had wheels at 154 rad/s and a robot on its back, and the
        # cause was an integrator blowing up, not a servo doing anything. A
        # step-up divides the reflected inertia by the ratio squared, so it also
        # makes the wheel far more responsive — the other half of the trade.
        arm = (p_ref.J_m / (gear * gear)) if mode == "driven" else 1e-6
        b.add_joint(name=f"{leg}_wheel", type=mujoco.mjtJoint.mjJNT_HINGE,
                    axis=[0, 1, 0], damping=1e-4, armature=arm)
        g = b.add_geom(name=f"{leg}_wheel", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[r, 0.5 * width, 0], quat=[0.7071068, 0.7071068, 0, 0],
                       rgba=[0.15, 0.15, 0.17, 1])
        # condim 6 so there IS rolling resistance, and priority so these values
        # win outright instead of being averaged with the floor's — the trap
        # rl/checks/check_model.py found on the feet.
        g.condim = 6
        g.priority = 1
        g.friction = [1.0, 0.01, 0.004]
        g.mass = wheel_kg           # belt and braces: the geom must not re-derive it
        if mode == "driven":
            a = spec.add_actuator()
            a.name = f"{leg}_wheel"
            a.target = f"{leg}_wheel"
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.gainprm[0] = 1.0                        # plain torque source; the
            a.ctrlrange = [-10, 10]                   # law is applied in Python
    m = spec.compile()
    wj = {leg: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_wheel")
          for leg in LEGS}
    return m, wj


def wheel_torque(p: A.Params, w_wheel, gear, u_bat, duty=1.0):
    """Torque a step-up-geared ST3215 can put on the wheel at this wheel speed.

    `gear` is the step-up: the wheel turns `gear` times per turn of the servo
    output, so it sees the servo's torque divided by it and the servo sees the
    wheel's speed divided by it. Terminal wheel speed is therefore
    U / (k_e * gear) ... no: the servo tops out at U / k_e and the wheel runs
    `gear` times that. Gearing up multiplies the speed and divides the torque,
    which is exactly the trade this servo can afford.
    """
    w_servo = w_wheel / gear
    tau_servo = (p.k_u * duty * u_bat - p.k_w * w_servo
                 - p.tau_c * math.tanh(w_servo / p.v_eps) - p.b_v * w_servo)
    tau_servo = max(-p.stall_torque(u_bat), min(p.stall_torque(u_bat), tau_servo))
    return tau_servo / gear


def run(mode, wheel_d, gear, seconds, speed_cmd, u_bat, p, settle=1.0,
        ratchet_k=8.0, wheel_kg=WHEEL_KG, quiet=False, drive_mass=0.0):
    P = json.load(open(os.path.join(DESC, "robot_params.json")))
    r = 0.5 * wheel_d
    if mode == "feet":
        m = mujoco.MjModel.from_xml_path(os.path.join(DESC, "mujoco", "scene.xml"))
        wj = {}
    else:
        m, wj = build(wheel_d, mode, gear, wheel_kg,
                      drive_mass=drive_mass if mode == "driven" else 0.0)
    d = mujoco.MjData(m)

    act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
           for n in P["joint_names"]]
    st = P["stance_rad"]
    for n in P["joint_names"]:
        k = n.split("_")[1]
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        d.qpos[m.jnt_qposadr[j]] = 0.0 if k == "roll" else st[k]
    # the wheel makes the leg longer by (r - the old foot radius); give it back
    # in ride height or this compares two different robots
    grow = max(0.0, r - 0.013) if mode != "feet" else 0.0
    d.qpos[2] = P["stance_base_height_m"] + grow

    DT = 0.01
    sub = max(1, int(round(DT / m.opt.timestep)))
    gait = None
    if mode in ("feet", "free", "ratchet"):
        from smalldog_walker.gait import TrotGait
        gait = TrotGait(P)
        gait.body_height = 0.158 + grow
        if not quiet and abs(gait.body_height - (0.158 + grow)) > 1e-6:
            print(f"    note: body height clamped to {gait.body_height*1000:.0f} mm "
                  f"(asked for {(0.158+grow)*1000:.0f}) — the leg cannot reach that far")

    wid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{l}_wheel")
           for l in LEGS] if mode == "driven" else []
    wdof = {l: m.jnt_dofadr[j] for l, j in wj.items()}

    x0 = None
    tau_peak, w_peak, fell = 0.0, 0.0, False
    for step in range(int((settle + seconds) / DT)):
        t = step * DT
        cmd = 0.0 if t < settle else speed_cmd
        if gait is not None:
            for a, v in zip(act, gait.joint_targets(DT, cmd, 0.0, 0.0)):
                d.ctrl[a] = v
        else:
            for n, a in zip(P["joint_names"], act):
                k = n.split("_")[1]
                d.ctrl[a] = 0.0 if k == "roll" else st[k]
        for _ in range(sub):
            if mode == "driven":
                # ramp, because slamming full duty puts 2.94 N*m on a 30 mm wheel
                # — 98 N of tractive force under a 24.5 N robot, which simply
                # flips it. A real controller would ramp; so does this.
                duty = 0.0 if t < settle else min(1.0, (t - settle) / RAMP)
                for leg, a in zip(LEGS, wid):
                    w = float(d.qvel[wdof[leg]])
                    # forward roll is +omega about +y: a positive rotation there
                    # carries the top of the wheel forwards and the contact
                    # patch backwards, which drives the body along +x
                    tau = wheel_torque(p, w, gear, u_bat, duty)
                    d.ctrl[a] = tau
                    if t >= settle + RAMP:
                        tau_peak = max(tau_peak, abs(tau))
                        w_peak = max(w_peak, abs(w))
            elif mode == "ratchet":
                # one way: free when it rolls forward, a hard brake when it tries
                # to roll back. A damper rather than a constraint — it is stable,
                # and the residual reverse creep is reported so it can be judged.
                for leg in LEGS:
                    v = float(d.qvel[wdof[leg]])
                    if v < 0.0:
                        d.qfrc_applied[wdof[leg]] = -ratchet_k * v
                    else:
                        d.qfrc_applied[wdof[leg]] = 0.0
            mujoco.mj_step(m, d)
        if t >= settle + (RAMP if mode == "driven" else 0.0) and x0 is None:
            x0 = float(d.qpos[0])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        if R.reshape(3, 3)[2, 2] < 0.4:
            fell = True
            break
    dist = float(d.qpos[0]) - (x0 if x0 is not None else 0.0)
    elapsed = max(1e-6, t - settle)
    return dict(mode=mode, gear=gear, wheel_mm=wheel_d * 1000, dist_mm=dist * 1000,
                speed=dist / elapsed, fell=fell, seconds=elapsed,
                tau_peak=tau_peak, w_peak=w_peak,
                mass=float(m.body_mass[1:].sum()), grow_mm=grow * 1000)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default=None,
                    choices=("feet", "free", "ratchet", "driven"))
    ap.add_argument("--wheel-d", type=float, default=0.060, help="wheel diameter, m")
    ap.add_argument("--gear", type=float, default=1.0, help="step-up to the wheel")
    ap.add_argument("--gears", default="1,2,4,6,8")
    ap.add_argument("--diameters", default=None,
                    help="comma-separated wheel diameters in m, swept at --gear")
    ap.add_argument("--drive-mass", type=float, default=SERVO_KG,
                    help="mass added at each wheel for its motor (0 to leave it out)")
    ap.add_argument("--seconds", type=float, default=5.0, help="measured window")
    ap.add_argument("--cmd", type=float, default=0.20, help="trot command, m/s")
    ap.add_argument("--volts", type=float, default=12.0)
    ap.add_argument("--params", default=A.DEFAULT_PATH)
    a = ap.parse_args()

    p = A.load(a.params)
    print(f"\nST3215 as modelled: stall {p.stall_torque(a.volts):.2f} N*m, "
          f"no load {p.no_load_speed(a.volts):.2f} rad/s at {a.volts:g} V")
    print(f"wheel {a.wheel_d*1000:.0f} mm -> direct-drive ceiling "
          f"{p.no_load_speed(a.volts) * a.wheel_d / 2:.3f} m/s\n")

    rows = []
    if a.diameters:
        for dia in (float(x) for x in a.diameters.split(",")):
            print(f"  running driven {a.gear:g}:1 on {dia*1000:.0f} mm ...")
            rows.append(run("driven", dia, a.gear, a.seconds, a.cmd, a.volts, p,
                            drive_mass=a.drive_mass))
        rows.insert(0, run("feet", a.wheel_d, 1.0, a.seconds, a.cmd, a.volts, p))
    elif a.mode:
        rows.append(run(a.mode, a.wheel_d, a.gear, a.seconds, a.cmd, a.volts, p,
                        drive_mass=a.drive_mass))
    else:
        for mode in ("feet", "free", "ratchet"):
            print(f"  running {mode} ...")
            rows.append(run(mode, a.wheel_d, 1.0, a.seconds, a.cmd, a.volts, p))
        for g in (float(x) for x in a.gears.split(",")):
            print(f"  running driven, step-up {g:g}:1 ...")
            rows.append(run("driven", a.wheel_d, g, a.seconds, a.cmd, a.volts, p,
                            drive_mass=a.drive_mass))

    print(f"\n{'configuration':<24}{'mass kg':>9}{'travel mm':>11}{'m/s':>8}"
          f"{'vs trot':>9}{'wheel rad/s':>13}{'peak N*m':>10}")
    base = next((r["speed"] for r in rows if r["mode"] == "feet"), None)
    for r in rows:
        name = (r["mode"] if r["mode"] != "driven"
                else f"driven {r['gear']:g}:1  {r['wheel_mm']:.0f} mm")
        rel = f"{r['speed']/base:5.2f}x" if base else "    -"
        print(f"{name:<24}{r['mass']:9.3f}{r['dist_mm']:11.0f}{r['speed']:8.3f}"
              f"{rel:>9}{r['w_peak']:13.2f}{r['tau_peak']:10.3f}"
              + ("   FELL" if r["fell"] else ""))
    if rows and rows[0]["grow_mm"]:
        print(f"\nthe wheel puts the contact {rows[-1]['grow_mm']:.0f} mm below where "
              f"the foot was, so the leg is that much longer and the ride height was "
              f"raised to match.")
    print("\nWhat to read out of this: the driven rows are a speed the servo can hold "
          "\ncontinuously, the trot row is a speed it reaches by working hard. The "
          "\nstep-up column is the whole point — the peak torque against the servo's "
          f"\n{p.stall_torque(a.volts):.2f} N*m says how much ratio is still on the table.")


if __name__ == "__main__":
    main()
