#!/usr/bin/env python
"""
policy.py — the RL policy on the real robot: a `Runtime` source that reads an ONNX.

    python runtime/policy.py POLICY_DIR --selftest                 # loopback bus, fake IMU, timing
    python runtime/policy.py POLICY_DIR --port /dev/ttyUSB0 --preflight
    python runtime/policy.py POLICY_DIR --port /dev/ttyUSB0 --stand --seconds 10
    python runtime/policy.py POLICY_DIR --port /dev/ttyUSB0 --vx 0.2 --seconds 8
    python runtime/policy.py POLICY_DIR --port /dev/ttyUSB0 --no-imu ...    # blind; see below

POLICY_DIR holds `policy.onnx` and `policy.json` as `rl/export_onnx.py` writes them.
The graph is the normaliser and the network in one; the JSON is everything the
robot side cannot derive — the observation layout and scalings, the action
scale, the stance and the soft limits, in `joint_names` order, which is also
`calib.joints` order (both come from robot_params.json; `PolicySource` checks).

The observation is `rl/env/walk.py`'s, term for term
------------------------------------------------------
    gravity_b[3]  gyro[3]*0.25  accel[3]*0.1  (q - stance)[12]  w*0.1[12]  last_action[12]  cmd[3]
stacked OBS_HIST deep, newest first, the first frame copied OBS_HIST times at
start — the same `assemble_obs`/`stack_obs` arithmetic, restated here in plain
Python because `robot/` imports neither jax nor `rl/env` (its dependency list is
pyserial and numpy, and onnxruntime is the one addition). The scalings are read
from the JSON rather than typed, so they cannot drift from the training frame.

`q` and `w` come straight off the bus feedback the loop already reads every
tick (`Runtime.read()`), in radians and rad/s after `calib`'s centre and sign.
`gravity_b`, `gyro`, `accel` come from `imu.bmi088.Attitude` (I2C) — the projected
gravity the policy was trained on is a filtered thing here, not a sensor.

The action becomes a target the same way as in training:
    target = clip(stance + action * action_scale, soft_lo, soft_hi)
`Runtime.send()` clamps once more against calib's own soft limits and holds the
last goal on a NaN, so a broken graph cannot become a lunge.

--no-imu is a bench mode, not a walking mode
--------------------------------------------
It feeds the policy a level, still IMU — gravity (0, 0, -1), zero rates, +g on
z — so the loop can be timed against the servos without the chip. The policy
then walks blind: it will not know it is tipping. Keep the robot on the bench
stand or held. `--selftest` uses the same stub.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from feetech.bus import Bus                                          # noqa: E402
from runtime.calib import CALIB, Calibration                          # noqa: E402
from runtime.loop import CTRL_HZ, FollowingLoopback, Runtime         # noqa: E402
from runtime.safety import Limits, Tripped                           # noqa: E402


class StillIMU:
    """A level robot that never moves. For --selftest and --no-imu."""

    def update(self, dt):
        return (0.0, 0.0, -1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 9.80665)


class LiveIMU:
    """imu.bmi088 read every tick through its complementary filter."""

    def __init__(self, chip, bias=(0.0, 0.0, 0.0)):
        from imu.bmi088 import Attitude
        self.chip = chip
        self.att = Attitude(bias=bias)

    def update(self, dt):
        accel, gyro = self.chip.read()
        return self.att.update(accel, gyro, dt)


class PolicySource:
    """source(dt, feedback) -> 12 targets in calib.joints order, from the ONNX."""

    def __init__(self, policy_dir, calib: Calibration, imu, command=(0.0, 0.0, 0.0)):
        import onnxruntime as ort
        with open(os.path.join(policy_dir, "policy.json")) as f:
            self.meta = m = json.load(f)
        if list(m["joint_names"]) != list(calib.joints):
            raise SystemExit(f"policy joint order {m['joint_names']} is not calib's "
                             f"{calib.joints}; both come from robot_params.json — regenerate")
        self.joints = list(calib.joints)
        self.stance = np.array(m["stance_rad"], np.float32)
        self.lo = np.array(m["soft_lo_rad"], np.float32)
        self.hi = np.array(m["soft_hi_rad"], np.float32)
        self.scale = float(m["action_scale_rad"])
        self.k_gyro, self.k_acc, self.k_qvel = (m["obs_scale_gyro"], m["obs_scale_accel"],
                                               m["obs_scale_qvel"])
        self.hist_n, self.frame_n = int(m["obs_hist"]), int(m["obs_frame"])
        self.imu = imu
        self.command = np.array(command, np.float32)
        self.last_action = np.zeros(12, np.float32)
        self.hist = None
        self.sess = ort.InferenceSession(os.path.join(policy_dir, m["onnx"]),
                                         providers=["CPUExecutionProvider"])
        self.infer_us = []
        self.blind_ticks = 0
        self.nonfinite = 0             # ticks whose frame or action had a NaN/inf
        self.log = []                  # (frame, action, target) per tick when --log

    def set_command(self, vx, vy, wz):
        self.command = np.array([vx, vy, wz], np.float32)

    def frame(self, dt, fb):
        g, w, acc = self.imu.update(dt)
        q = np.array([fb[n]["q"] if fb[n] else self.stance[i] for i, n in enumerate(self.joints)],
                     np.float32)
        qd = np.array([fb[n]["w"] if fb[n] else 0.0 for n in self.joints], np.float32)
        self.blind_ticks += sum(1 for n in self.joints if fb[n] is None)
        f = np.concatenate([
            np.asarray(g, np.float32),
            np.asarray(w, np.float32) * self.k_gyro,
            np.asarray(acc, np.float32) * self.k_acc,
            q - self.stance,
            qd * self.k_qvel,
            self.last_action,
            self.command,
        ])
        assert f.shape[0] == self.frame_n, (f.shape, self.frame_n)
        return f

    def __call__(self, dt, fb):
        f = self.frame(dt, fb)
        if self.hist is None:
            self.hist = np.repeat(f[None], self.hist_n, axis=0)
        else:
            self.hist = np.concatenate([f[None], self.hist[:-1]])
        obs = self.hist.reshape(1, -1).astype(np.float32)
        t0 = time.perf_counter()
        a = self.sess.run(["action"], {"obs": obs})[0][0]
        self.infer_us.append((time.perf_counter() - t0) * 1e6)
        if not (np.isfinite(f).all() and np.isfinite(a).all()):
            # send() would hold the last goal on a NaN and nothing would say so
            self.nonfinite += 1
        self.last_action = a.astype(np.float32)
        target = np.clip(self.stance + a * self.scale, self.lo, self.hi)
        if self.log is not None:
            self.log.append((f.copy(), a.astype(np.float32), target.astype(np.float32)))
        return [float(v) for v in target]

    def save_log(self, path):
        if not self.log:
            return
        fr, ac, tg = (np.stack(x) for x in zip(*self.log))
        np.savez(path, frame=fr, action=ac, target=tg, stance=self.stance,
                 joints=np.array(self.joints), command=self.command)
        print(f"log: {len(self.log)} ticks -> {path}")

    def report(self):
        t = np.array(self.infer_us) if self.infer_us else np.zeros(1)
        return (f"inference p50 {np.percentile(t, 50):.0f} us, p95 {np.percentile(t, 95):.0f} us, "
                f"max {t.max():.0f} us of a {1e6/CTRL_HZ:.0f} us tick; "
                f"{self.blind_ticks} joint reads missing, {self.nonfinite} non-finite ticks; "
                f"action |a| p50 {np.median(np.abs(np.stack([l[1] for l in self.log]))) if self.log else float('nan'):.3f} "
                f"max {max((float(np.abs(l[1]).max()) for l in self.log), default=float('nan')):.3f}")


def selftest(policy_dir, seconds=5.0):
    calib = Calibration.load(CALIB)
    bus = Bus(transport=FollowingLoopback(calib.ids), discard_echo=False)
    rt = Runtime(bus, calib, hz=CTRL_HZ)
    src = PolicySource(policy_dir, calib, StillIMU(), command=(0.2, 0.0, 0.0))
    print(f"policy {src.meta['run']} @ {src.meta['commit']}, obs {src.hist_n}x{src.frame_n}")
    with rt:
        rt.engage([float(v) for v in src.stance], ramp_s=0.5)
        rep = rt.run(src, seconds=seconds)
    print("loop  ", {k: v for k, v in rep.items() if k in ("ticks", "overruns", "hz", "p95_ms", "mean_ms")}
          if isinstance(rep, dict) else rep)
    print("policy", src.report())
    # the stub IMU is level and still, so the policy should be walking, not frozen:
    moved = float(np.abs(src.last_action).max())
    print(f"last action max |a| = {moved:.2f}  (a frozen graph reads 0.00)")
    return 0 if moved > 0.01 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy_dir")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--stand", action="store_true", help="hold the stance under the policy at command 0")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--ramp", type=float, default=2.0)
    ap.add_argument("--no-imu", action="store_true", help="bench mode: a still, level IMU stub")
    ap.add_argument("--i2c-bus", type=int, default=1, help="the N of /dev/i2c-N")
    ap.add_argument("--acc-addr", type=lambda x: int(x, 0), default=None)
    ap.add_argument("--gyro-addr", type=lambda x: int(x, 0), default=None)
    ap.add_argument("--bias-seconds", type=float, default=3.0, help="gyro bias, robot still")
    ap.add_argument("--temp-c", type=float, default=60.0)
    ap.add_argument("--current-a", type=float, default=2.5)
    ap.add_argument("--volt-min", type=float, default=9.9)
    ap.add_argument("--track-rad", type=float, default=0.6)
    ap.add_argument("--log", metavar="FILE.npz", help="record every tick's frame, action, target")
    a = ap.parse_args()

    if a.selftest:
        return selftest(a.policy_dir, a.seconds)
    if not a.port:
        sys.exit("--port or --selftest")

    calib = Calibration.load(CALIB)
    bus = Bus(a.port, a.baud)
    limits = Limits(temp_c=a.temp_c, current_a=a.current_a, volt_min=a.volt_min,
                    q_err_rad=a.track_rad)
    rt = Runtime(bus, calib, hz=CTRL_HZ, limits=limits)
    pre = rt.preflight(a.port)
    if not pre["ok"]:
        print("!! not every servo answered; refusing to move")
        return 1
    if a.preflight:
        return 0

    if a.no_imu:
        print("!! no IMU: the policy is BLIND. Bench stand or hands on the robot.")
        imu = StillIMU()
    else:
        from imu.bmi088 import BMI088, measure_bias
        chip = BMI088(a.i2c_bus, a.acc_addr, a.gyro_addr)
        chip.configure()
        print(f"IMU: hold still {a.bias_seconds:g} s for the gyro bias ...")
        bias = measure_bias(chip, a.bias_seconds, CTRL_HZ)
        print("gyro bias rad/s: " + " ".join(f"{b:+.4f}" for b in bias))
        imu = LiveIMU(chip, bias)
        g, _, acc = imu.update(1.0 / CTRL_HZ)
        print(f"gravity now {g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f} (level reads 0 0 -1)")
        if g[2] > -0.9:
            print("!! the IMU does not read level; check imu/bmi088.py AXES before walking")
            return 1

    cmd = (0.0, 0.0, 0.0) if a.stand else (a.vx, a.vy, a.wz)
    src = PolicySource(a.policy_dir, calib, imu, command=cmd)
    print(f"policy {src.meta['run']} @ {src.meta['commit']}   command vx {cmd[0]:+.2f} "
          f"vy {cmd[1]:+.2f} wz {cmd[2]:+.2f}   {a.seconds:g} s")
    print("stance, deg: " + "  ".join(f"{n} {math.degrees(v):+.0f}" for n, v in zip(calib.joints, src.stance)))
    code = 0
    try:
        with rt:
            rt.engage([float(v) for v in src.stance], ramp_s=a.ramp)
            rt.run(src, seconds=a.seconds)
            rt.relax([float(v) for v in src.stance], ramp_s=1.0)
    except KeyboardInterrupt:
        print("\ninterrupted")
    except Tripped as e:
        print(f"\n!! TRIPPED: {e}")
        code = 1
    finally:
        print("policy", src.report())
        if a.log:
            src.save_log(a.log)
    return code


if __name__ == "__main__":
    sys.exit(main())
