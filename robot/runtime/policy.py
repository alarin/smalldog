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

The heading hold (`HeadingHold`, on by default with the IMU)
-------------------------------------------------------------
The policy walks an arc at wz = 0 — it has no heading to hold, only a rate. The runtime
latches the heading at the start and steers `cmd[2]` to keep it, exactly as the IK
trot's `gait.py` does; `--wz` other than 0 releases it. `--no-hold` is the policy as
trained. The log records the command the policy was GIVEN, hold included.

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
from runtime.mode2 import KP, KD, KFF                                 # noqa: E402
from runtime.safety import Limits, Tripped                           # noqa: E402


def pose_line(rt, imu, calib):
    """One line of where the body and the joints are right now (a fresh IMU read)."""
    if hasattr(imu, "reset"):
        imu.reset()
    g, _, _ = imu.update(1.0 / CTRL_HZ)
    fb = rt.read()
    return (f"pitch {math.degrees(math.asin(max(-1.0, min(1.0, g[0])))):+.0f} roll "
            f"{math.degrees(math.atan2(-g[1], -g[2])):+.0f} deg; q/goal deg: " + "  ".join(
            f"{n} {math.degrees(fb[n]['q']):+.0f}/{math.degrees(rt.goal[n]):+.0f}" if fb[n] else f"{n} ?"
            for n in calib.joints))


def sit_pose(calib: Calibration):
    """The fold: knee at its soft limit, hip pitch putting the foot straight under the
    hip (the IK trot's LegKinematics, foot at x = 0 and the shortest leg) — 98 mm from
    the pitch axis against 158 at stance; the knee's hard stop is 110 deg, so a relaxed
    robot cannot sink much further. The pose to sit down into and stand up from.

    Standing up from anywhere else is not safe: torque cut AT STANCE lets the front
    collapse onto its knees (the relaxed IMU read 13-29 deg nose-down), and a straight
    ramp from that kneel — to stance, or through the gait's 15 cm "sit" — pushes off the
    knees and pivots the nose-heavy robot onto its face (2026-09-17, four times, under
    either gain set, joints exactly on target the whole way). Feet under the hips first.
    None if the walker package is not beside this tree."""
    import os as _os
    repo = _os.path.dirname(_os.path.dirname(HERE))
    sys.path.insert(0, _os.path.join(repo, "ros2", "smalldog_walker"))
    try:
        from runtime.calib import load_params
        from smalldog_walker.gait import TrotGait
    except ImportError:
        return None
    gait = TrotGait(load_params())
    if list(gait.joint_names) != list(calib.joints):
        return None
    knee = gait.limits["knee"]
    q = []
    for l in gait.legs:
        kin = gait.kin[l]
        # foot under the pitch axis at the knee limit: fk's u = -l1 sin h - l2 sin(h + k) = 0
        h = min((math.radians(d / 2) for d in range(-180, 1)),
                key=lambda h: abs(kin.l1 * math.sin(h) + kin.l2 * math.sin(h + knee)))
        q += [0.0, h, knee]
    return [float(v) for v in calib.clamp(q)]


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
        self.gravity = (0.0, 0.0, -1.0)     # the last read, for the guard's tilt check

    def update(self, dt):
        accel, gyro = self.chip.read()
        out = self.att.update(accel, gyro, dt)
        self.gravity = out[0]
        return out

    def reset(self):
        """Forget the filtered gravity: the next update snaps to the accelerometer."""
        self.att.g = None

    def roll_pitch(self):
        """(roll, pitch) rad from the last gravity read: x forward, y left, z up."""
        gx, gy, gz = self.gravity        # as servo_node.attitude_from_gravity
        return math.atan2(-gy, -gz), math.asin(max(-1.0, min(1.0, gx)))


class PolicySource:
    """source(dt, feedback) -> 12 targets in calib.joints order, from the ONNX."""

    def __init__(self, policy_dir, calib: Calibration, imu, command=(0.0, 0.0, 0.0),
                 hold: "HeadingHold | None" = None):
        import onnxruntime as ort
        self.hold = hold
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
        cmd = self.command
        if self.hold is not None:
            cmd = np.array([cmd[0], cmd[1], self.hold(dt, w, float(cmd[2]))], np.float32)
        f = np.concatenate([
            np.asarray(g, np.float32),
            np.asarray(w, np.float32) * self.k_gyro,
            np.asarray(acc, np.float32) * self.k_acc,
            q - self.stance,
            qd * self.k_qvel,
            self.last_action,
            cmd,
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


class HeadingHold:
    """Steer the policy's yaw-rate command so the body keeps the heading it started on.

    The policy has no heading in its observation — gyro, gravity, joints, its own last
    action and the command — so with wz = 0 it walks a slow arc: in vanilla MuJoCo the
    shipped 20260915-bc-ft turns −10° in 5 s at cmd 0.1 and −21° at 0.2 (measured
    2026-09-15, `rl/eval.py`'s actuator law). It does answer wz (±0.3 → ±73° in 5 s), so
    the same trick `smalldog_walker/gait.py` plays on the IK trot works here: latch the
    heading when the operator asks for no turn, and command the turn that closes the
    error. kp 3 with the 0.5 rad/s clamp holds the sim to ±1° at 8 s (peak 7° in the
    first stride); kp 1.5 left −4°, an integral term bought nothing.

    Heading is the integrated gyro after `measure_bias` — no magnetometer, so it drifts
    at the residual bias (~0.01 deg/s standing) and is a line-holder, not a compass.
    """

    def __init__(self, kp=3.0, kd=0.05, wz_max=0.5, cmd_eps=0.02):
        self.kp, self.kd, self.wz_max, self.cmd_eps = kp, kd, wz_max, cmd_eps
        self.yaw = 0.0
        self.ref = None
        self.applied = []

    def __call__(self, dt, gyro, wz_cmd):
        """(wz to hand the policy) given this tick's gyro (rad/s) and the operator's wz."""
        self.yaw += gyro[2] * dt
        if abs(wz_cmd) > self.cmd_eps:
            self.ref = None                   # a commanded turn: let go, re-latch after
            return wz_cmd
        if self.ref is None:
            self.ref = self.yaw
        e = (self.ref - self.yaw + math.pi) % (2 * math.pi) - math.pi
        wz = max(-self.wz_max, min(self.wz_max, self.kp * e - self.kd * gyro[2]))
        self.applied.append(wz)
        return wz

    def report(self):
        if not self.applied:
            return "heading hold: never engaged"
        a = np.array(self.applied)
        return (f"heading hold: yaw now {math.degrees(self.yaw):+.1f} deg from start, "
                f"wz applied mean {a.mean():+.3f} max {np.abs(a).max():.2f} rad/s, "
                f"{100 * np.mean(np.abs(a) >= self.wz_max - 1e-6):.0f} % of ticks at the clamp")


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
    ap.add_argument("--mode0", action="store_true",
                    help="the servo's own position loop (MODE 0) instead of the host loop")
    ap.add_argument("--mode2", action="store_true", help=argparse.SUPPRESS)   # the default
    ap.add_argument("--sub-hz", type=float, default=0.0,
                    help="--mode2: pace the host loop, Hz (0: as fast as the bus goes)")
    # The MODEL's gains, not the trot's: actuator.Params.kp 5.22 duty/rad, no kd, and no
    # feed-forward — a policy's target can step 0.1 rad in a tick, which kff 200 turned
    # into a 1000-duty kick and a dive at command 0 (2026-09-16). 5220 is 5.22 in
    # register units; s4 walks with these.
    ap.add_argument("--kp", type=float, default=5220.0, help="host loop, duty per rad")
    ap.add_argument("--kd", type=float, default=0.0, help="host loop, duty per rad/s")
    ap.add_argument("--kff", type=float, default=0.0, help="host loop, duty per rad/s of target")
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
    ap.add_argument("--ramp-model-gains", action="store_true",
                    help="stand up under the policy's gains too (default: the trot's stiffer, "
                         "damped loop for the ramp and the settle, the model's from the hand-over)")
    ap.add_argument("--settle", type=float, default=1.5, metavar="S",
                    help="after the engage ramp, hold the stance (no policy) for S s while the IMU "
                         "filter re-converges on the STANDING robot; 0 to hand over at once")
    ap.add_argument("--stand-before", type=float, default=0.0, metavar="S",
                    help="hold the stance under the policy at command 0 for S s before the walk")
    ap.add_argument("--stand-after", type=float, default=0.0, metavar="S",
                    help="after the walk, hold the stance under the policy at command 0 for S s "
                         "(a LiDAR slice wants the robot standing, not sagging with torque off)")
    ap.add_argument("--no-hold", action="store_true",
                    help="no heading hold: hand the policy --wz as given (it arcs; see HeadingHold)")
    ap.add_argument("--hold-kp", type=float, default=3.0, help="rad/s of turn per rad of heading error")
    ap.add_argument("--hold-max", type=float, default=0.5, help="rad/s, clamp on the hold's turn")
    a = ap.parse_args()

    if a.selftest:
        return selftest(a.policy_dir, a.seconds)
    if not a.port:
        sys.exit("--port or --selftest")

    calib = Calibration.load(CALIB)
    bus = Bus(a.port, a.baud)
    limits = Limits(temp_c=a.temp_c, current_a=a.current_a, volt_min=a.volt_min,
                    q_err_rad=a.track_rad)
    if not a.mode0:
        from runtime.mode2 import Mode2Runtime
        gains = dict(kp=a.kp, kd=a.kd, kff=a.kff)
        # smooth=False: the policy was trained against a PD that saw its target at once
        rt = Mode2Runtime(bus, calib, hz=CTRL_HZ, limits=limits, sub_hz=a.sub_hz,
                          smooth=False, **gains)
        print("MODE 2: host position loop (runtime/mode2.py)")
    else:
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
        # relaxed, the robot sags up to ~30 deg nose-down; only a robot on its side is
        # wrong here. The standing check is after the settle.
        if g[2] > -0.5:
            print("!! the IMU reads more than 60 deg off level; robot over, or imu/bmi088.py AXES")
            return 1

    cmd = (0.0, 0.0, 0.0) if a.stand else (a.vx, a.vy, a.wz)
    hold = None if (a.no_hold or a.no_imu) else HeadingHold(kp=a.hold_kp, wz_max=a.hold_max)
    src = PolicySource(a.policy_dir, calib, imu, command=cmd, hold=hold)
    print("heading hold: " + ("off" if hold is None else
                              f"kp {hold.kp:g}, clamp {hold.wz_max:g} rad/s, lets go when |wz| > {hold.cmd_eps:g}"))
    print(f"policy {src.meta['run']} @ {src.meta['commit']}   command vx {cmd[0]:+.2f} "
          f"vy {cmd[1]:+.2f} wz {cmd[2]:+.2f}   {a.seconds:g} s")
    print("stance, deg: " + "  ".join(f"{n} {math.degrees(v):+.0f}" for n, v in zip(calib.joints, src.stance)))
    # the guard's tilt trip: the source reads the chip every tick, this reads the result
    tilt = (lambda k, dt, fb: rt.guard.attitude(dt, *imu.roll_pitch())) if hasattr(imu, "roll_pitch") else None
    code = 0
    try:
        with rt:
            # Standing up is not what the model's gains are for: kp 5220 with no damping
            # lifting 12 loaded servos from a sagging pose ended nose-down and over
            # twice (2026-09-17: relaxed 13-16 deg nose-down -> fell during the settle;
            # from straight legs the same ramp was fine). Ramp and settle under the
            # IK trot's loop, then hand the policy the PD it was trained against.
            if not a.mode0 and not a.ramp_model_gains:
                rt.kp, rt.kd, rt.kff, rt.smooth = KP, KD, KFF, True
            stance = [float(v) for v in src.stance]
            sit = sit_pose(calib)
            fb0 = rt.read()
            print("measured, deg: " + "  ".join(f"{n} {math.degrees(fb0[n]['q']):+.0f}" if fb0[n] else f"{n} ?"
                                               for n in calib.joints))
            if sit is None:
                print("!! no sit pose (smalldog_walker not found): standing up straight from here")
                rt.engage(stance, ramp_s=a.ramp)
            else:
                print("fold, deg: " + "  ".join(f"{n} {math.degrees(v):+.0f}" for n, v in zip(calib.joints, sit)))
                rt.engage(sit, ramp_s=a.ramp)          # feet under the hips first
                print("after the fold ramp: " + pose_line(rt, imu, calib), flush=True)
                rt.engage_ramp_to(stance, ramp_s=a.ramp)
                print("after the stance ramp: " + pose_line(rt, imu, calib), flush=True)
            if a.settle > 0:
                # The filter was initialised at bias time, in the RELAXED pose (a sagging
                # robot reads 6-16 deg nose-down), and nothing ticked it through the
                # ramp. Handed over stale, the policy's first frames saw a nose-down
                # robot that was standing level, reacted, and the shaking never let
                # the filter settle (2026-09-17: two dives, one fall). Snap to the
                # accelerometer on the standing robot and give it a time constant.
                if hasattr(imu, "reset"):
                    imu.reset()
                rt.run(lambda dt, fb: (imu.update(dt), stance)[1], seconds=a.settle, on_tick=tilt)
                g = imu.gravity if hasattr(imu, "gravity") else (0.0, 0.0, -1.0)
                pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, g[0]))))
                print(f"settled {a.settle:g} s: gravity {g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f} "
                      f"(pitch {pitch_deg:+.1f} deg)", flush=True)
                if g[2] > -0.9:
                    raise Tripped(f"standing at stance the IMU is {math.degrees(math.acos(-g[2])):.0f} deg "
                                  "off level; not handing that to the policy")
            if not a.mode0 and not a.ramp_model_gains:
                rt.kp, rt.kd, rt.kff, rt.smooth = gains["kp"], gains["kd"], gains["kff"], False
                print(f"hand-over: host loop kp {rt.kp:g} kd {rt.kd:g} kff {rt.kff:g}, no smoothing", flush=True)
            if a.stand_before > 0:
                src.set_command(0.0, 0.0, 0.0)
                rt.run(src, seconds=a.stand_before, on_tick=tilt)
                src.set_command(*cmd)
                print(f"walking: vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} wz {cmd[2]:+.2f}", flush=True)
            rt.run(src, seconds=a.seconds, on_tick=tilt)
            if a.stand_after > 0:
                src.set_command(0.0, 0.0, 0.0)
                rt.run(src, seconds=a.stand_after, on_tick=tilt)
            if not a.mode0 and not a.ramp_model_gains:
                rt.kp, rt.kd, rt.kff, rt.smooth = KP, KD, KFF, True
            rt.relax(sit if sit is not None else stance, ramp_s=1.5)
    except KeyboardInterrupt:
        print("\ninterrupted")
    except Tripped as e:
        print(f"\n!! TRIPPED: {e}")
        code = 1
        try:                                   # where the joints actually are, for the read-back
            fb = rt.read()
            print("at the trip, deg q/goal: " + "  ".join(
                f"{n} {math.degrees(fb[n]['q']):+.0f}/{math.degrees(rt.goal[n]):+.0f}" if fb[n] else f"{n} ?"
                for n in calib.joints))
            if hasattr(imu, "gravity"):
                g = imu.gravity
                print(f"gravity {g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f}")
        except Exception as ex:                # noqa: BLE001 — a read-back must not mask the trip
            print(f"(no read-back: {ex})")
    finally:
        print(rt.report_lines())
        print("policy", src.report())
        if hold is not None:
            print(hold.report())
        if a.log:
            src.save_log(a.log)
    return code


if __name__ == "__main__":
    sys.exit(main())
