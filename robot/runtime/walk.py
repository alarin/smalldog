"""
walk.py — the trot, on the real robot. The CLI that wires everything together.

    python runtime/walk.py --dry-run --profile           # no hardware at all
    python runtime/walk.py --port /dev/ttyUSB0 --preflight
    python runtime/walk.py --port /dev/ttyUSB0 --stand --seconds 10
    python runtime/walk.py --port /dev/ttyUSB0 --profile
    python runtime/walk.py --port /dev/ttyUSB0            # keyboard teleop
    python runtime/walk.py --port /dev/ttyUSB0 --mode0    # the firmware's loop, for comparison

The servos run in MODE 2 — the position loop on the host, `runtime/mode2.py` — since
2026-09-16: the firmware's own loop profiles every goal at 7.7 rad/s² and cannot follow
the trot (`ideas/FAST_SERVOS.md`). `--mode0` is the old behaviour, kept for comparison.

        W/S forward/back   A/D strafe   Q/E turn   space stop
        R/F body up/down   ,/. slower/faster   T torque off/on   Ctrl-C quit

The gait is imported, not reimplemented
---------------------------------------
`ros2/smalldog_walker/smalldog_walker/gait.py` is the one copy of the trot, the
way `rl/actuator.py` is the one copy of the servo law, and it is pure Python — it
imports `math` and its own `leg_kinematics`, and nothing else. So this file adds
`ros2/smalldog_walker` to the path and imports `TrotGait` exactly as
`ros2/tools/standalone_sim.py` does. Despite the path, no part of ROS is involved
and `robot/`'s dependency list is unchanged. A second trot living here would drift
from the one the simulator reports on, and then the sim would be reporting on a
robot that does not exist.

The IMU is not here yet, and the gait knows
-------------------------------------------
`TrotGait.feedback()` is optional by construction: supply nothing and it is exactly
the blind open-loop trot. That is what runs today — the BMI088 has not arrived, and
the RL policy cannot run at all without one (nine of the 48 numbers in its
observation come off that chip). Measured cost of blindness, from
`ros2/README.md`'s own sweep: 1.8 deg of heading drift over 1.2 m on flat ground,
against 1.65 m of sideways travel on rough. Flat floor, blind, is fine.

What *is* available without the IMU is foot contact, and it is nearly free here.
`smalldog_walker/contact.py` infers contact from the knee servo's load minus what
the same leg reads at the same phase in free air (AUC 0.86, and as good as a
perfect foot sensor for the gait's purposes). Its stated worry is cost — "reading
four knee loads per tick costs four round trips on top of the position writes" —
and this loop retires it: `Runtime.read()` already SyncReads all fifteen feedback
bytes from all twelve servos every tick, load included, so the reading has already
been paid for. `--baseline` records the free-air curve with the robot hanging;
`--contact` uses it. Both are opt-in, because the threshold is in Feetech's Present
Load units and has to be re-found on hardware.
"""
from __future__ import annotations

import argparse
import math
import os
import select
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "ros2", "smalldog_walker"))

from feetech.bus import Bus                                          # noqa: E402
from runtime.calib import CALIB, Calibration, load_params            # noqa: E402
from runtime.loop import CTRL_HZ, FollowingLoopback, Runtime         # noqa: E402
from runtime.mode2 import DutyLoopback, Mode2Runtime, parse_duty_caps  # noqa: E402
from runtime.ticklog import TickLog                                   # noqa: E402
from runtime.safety import Limits, Tripped                           # noqa: E402
from smalldog_walker.contact import Baseline, ServoContact           # noqa: E402
from smalldog_walker.gait import TrotGait                            # noqa: E402

#: The scripted demo, as (seconds, vx m/s, vy m/s, wz rad/s). Runs without a
#: keyboard, which is what makes it the thing to put on a soak test — and the
#: thing to show friends without also demonstrating your typing.
PROFILE = [(2.0, 0.0, 0.0, 0.0), (6.0, 0.20, 0.0, 0.0), (1.5, 0.0, 0.0, 0.0),
           (4.0, 0.0, 0.0, 0.6), (1.5, 0.0, 0.0, 0.0), (4.0, -0.15, 0.0, 0.0),
           (2.0, 0.0, 0.0, 0.0)]


def build_gait(params, args) -> TrotGait:
    gait = TrotGait(params)
    # order matters: the body-height setter clamps against swing and step
    gait.period = args.period
    gait.swing_height = args.swing
    gait.max_step = args.max_step
    gait.body_height = args.height
    return gait


# ------------------------------------------------------- what the servos can deliver
#
# The trot as tuned asks for more joint speed than these servos have, and the way that
# fails on hardware is not a limp - it is a DRAG.  `gait.py` rate-limits its own output,
# so a demand over that is clipped before a servo ever sees it and the commanded foot
# path is simply not the path that is flown.  The limiter was `joint_velocity_limit *
# 0.85` = 4.0 rad/s when this was written; since 2026-09-11 it is the achievable ceiling
# `joint_rate_ceiling_rad_s` = 3.15 - a fifth lower, which only sharpens what follows.
# Measured 2026-09-08 at 2.55 kg: at 0.20 m/s the demand is 7.55 rad/s, 89 % over, and the
# robot travelled 0.067 m/s - a third of what it was told - with the knee-load residual
# peaking in the half the gait calls SWING on all four legs, which is a foot that never
# leaves the ground.  It also costs current (0.86 A peaks against 0.28 standing) and it
# destroys contact sensing, because there is no free-air half of the cycle left to
# subtract against.
#
# This lives here and not in `gait.py` on purpose.  **The simulation does not have this
# bug.**  The same clipping happens there, but MuJoCo's feet run at mu ~ 1.2 against a
# real 0.3..0.5 on a bare bench, so the sim robot grips and still makes 0.156 m/s.  Fixing
# it inside the gait would move the sim and everything tuned against it - the flat and
# terrain distances in `ros2/README.md`, and what `rl/` trains on - to cure something only
# the hardware suffers from.  So the runtime picks a deliverable operating point and says
# what it changed; the gait is left exactly as the sim and the policy know it.
#
RATE_MARGIN = 0.95          # of the gait's own limiter; 1.0 exactly is not a place to sit
#: In MODE 2 (`--mode2`, `runtime/mode2.py`) the profile is gone and the ceiling is the
#: motor's: 4.7 rad/s at the pack's 11.3 V (`ideas/FAST_SERVOS.md`, limit 1 — the bridge
#: at full duty, the 3.86 no-load figure being the firmware's plateau). Not `verify`: it
#: is the bench's number; what is unmeasured is how much of it a loaded leg keeps.
MODE2_RATE_LIMIT = 4.7


def joint_rate_demand(params, speed, period, swing, max_step, height, wz=0.0):
    """Peak commanded joint rate over one cycle, rad/s — `TrotGait.rate_demand`, the one
    copy of the sweep, on a gait tuned like the one about to run. `wz` because a turn is
    not free: the outer legs stride further, so the same forward speed costs more joint
    rate while turning; every turning command in this file goes through it.
    """
    g = TrotGait(params)
    g.period, g.swing_height, g.max_step, g.body_height = period, swing, max_step, height
    g.stride_max = 1e3                      # so period_for cannot pin the period here
    return g.rate_demand(speed, 0.0, wz)


def feasible_gait(params, args, limit=None):
    """Pick a (speed, period) these servos can actually fly.  Returns (speed, period, note).

    Two constraints, and both have to hold or the gait is lying about something:

      * the joint-rate demand has to fit under the limiter, or the foot path is clipped;
      * the stride, `speed * period / 4`, has to stay under `max_step`, or the gait clamps
        it and the stance foot stops travelling as far as the body does - which is slip
        designed in, and it is how a scan of "feasible" settings produces 0.40 m/s that
        cannot possibly be walked.

    Preferring the SHORTEST period that fits keeps the body unsupported for as little time
    as possible, which is what `TrotGait.period_for` is protecting on rough ground.
    """
    if limit is None:
        limit = TrotGait(params).max_joint_rate
    target = limit * RATE_MARGIN

    def best_period(speed):
        for period in [x / 20.0 for x in range(int(20 * args.period), 20 * 4 + 1)]:
            if speed * period / 4.0 > args.max_step:
                break                        # longer only makes the stride worse
            if joint_rate_demand(params, speed, period, args.swing,
                                 args.max_step, args.height) <= target:
                return period
        return None

    speed = abs(args.speed)
    period = best_period(speed) if speed > 1e-6 else args.period
    if period is not None:
        note = ("" if abs(period - args.period) < 1e-9 else
                f"period {args.period:.2f} -> {period:.2f} s to stay under "
                f"{target:.1f} rad/s")
        return speed, period, note

    # No period works at this speed.  Back the speed off until one does, rather than
    # command a gait that will drag: 0.067 m/s of dragging is slower than 0.15 walked.
    while speed > 0.01:
        speed = round(speed - 0.01, 3)
        period = best_period(speed)
        if period is not None:
            return speed, period, (
                f"!! {abs(args.speed):.2f} m/s needs more joint speed than these servos "
                f"have; capped to {speed:.2f} m/s at period {period:.2f} s")
    return abs(args.speed), args.period, (
        "!! no feasible period found; running as commanded, expect the feet to drag")


def feasible_turn(params, args, period, limit=None, step=0.05):
    """The largest |wz| that fits under the limiter, for a turn ON THE SPOT.

    `feasible_gait` above asks only about forward motion, so until 2026-09-11 every
    turning command in this file went out unchecked: the profile's 0.6 rad/s and the
    teleop's `--turn` 1.2, against a limiter the straight line had been fitted to sit
    0.2 rad/s under. A turn is not free — it lengthens the outer legs' stride — and
    the cost is steep: at this period a spin at 1.2 rad/s demands 4.13 rad/s of joint,
    a third over the 3.15 ceiling, which is the drag that made 0.20 m/s useless.

    ON THE SPOT and not "while walking", because that is the only turn either caller
    ever commands: `Teleop.key` zeroes the whole command vector before setting one
    axis, and every turning step in `PROFILE` has vx = vy = 0. Fitting the combined
    case instead would disable turning outright — at 0.11 m/s the straight line
    already uses 2.98 of the 2.99 available — and would be answering a question
    nothing in this file asks.

    Searches down from what was asked for, so a turn that already fits comes back
    unchanged and nothing is slowed for the sake of it.
    """
    if limit is None:
        limit = TrotGait(params).max_joint_rate
    target = limit * RATE_MARGIN
    wz = abs(args.turn)
    while wz > 1e-9:
        if joint_rate_demand(params, 0.0, period, args.swing, args.max_step,
                             args.height, wz=wz) <= target:
            return wz, ("" if abs(wz - abs(args.turn)) < 1e-9 else
                        f"turn {abs(args.turn):.2f} -> {wz:.2f} rad/s to stay under "
                        f"{target:.1f} rad/s")
        wz = round(wz - step, 3)
    return 0.0, (f"!! no turn rate fits under {target:.1f} rad/s at period "
                 f"{period:.2f} s; turning is disabled. Lengthen --period first.")


def stance_pose(gait, dt, seconds=1.5):
    """Where the gait wants the joints with no command — the pose to stand up into.

    Ticking it rather than computing it is deliberate: the gait rate-limits its own
    output from wherever it last was, so ticking to convergence leaves its internal
    `_q_prev` equal to the pose the servos are about to be ramped to, and the first
    tick of the real loop then continues from there instead of stepping.
    """
    q = None
    for _ in range(max(1, int(seconds / dt))):
        q = gait.joint_targets(dt, 0.0, 0.0, 0.0)
    return q


# ------------------------------------------------------------------- teleop
class Teleop:
    """Raw stdin, same bindings as the ROS 2 keyboard node and the MuJoCo viewer.

    Needs its own focused TTY. Without one (a pipe, a service, `nohup`) it reports
    that and the caller falls back to the scripted profile, rather than running a
    robot that cannot be told to stop.
    """
    MOVE = {"w": ("x", +1), "s": ("x", -1), "a": ("y", +1),
            "d": ("y", -1), "q": ("z", +1), "e": ("z", -1)}

    def __init__(self, gait, speed=0.20, turn=1.2):
        self.gait = gait
        self.speed, self.turn = speed, turn
        #: `,`/`.` may not climb past what `feasible_gait` fitted. They used to run
        #: to a hard-coded 0.45 m/s, four times the current ceiling, which made the
        #: keyboard the one way into this file that walked straight past the whole
        #: feasibility fit — two keypresses and the feet are dragging again, with
        #: nothing printed to say so. The caller passes the FITTED speed in.
        self.speed_max = speed
        self.cmd = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.enabled = True
        self._fd = None
        self._old = None

    @staticmethod
    def available() -> bool:
        return sys.stdin.isatty()

    def __enter__(self):
        import termios, tty
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        import termios
        if self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        sys.stdout.write("\n")
        return False

    def poll(self):
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if not ch:
                return
            if ch == "\x03":                       # Ctrl-C in cbreak mode
                raise KeyboardInterrupt
            self.key(ch.lower())

    def key(self, k):
        g = self.gait
        if k in self.MOVE:
            axis, s = self.MOVE[k]
            self.cmd = {"x": 0.0, "y": 0.0, "z": 0.0}
            self.cmd[axis] = s * (self.turn if axis == "z" else self.speed)
        elif k == " ":
            self.cmd = {"x": 0.0, "y": 0.0, "z": 0.0}
        elif k in "rf":
            g.body_height = g.body_height + (0.004 if k == "r" else -0.004)
        elif k in ",.":
            self.speed = max(0.02, min(self.speed_max,
                                       self.speed + (0.01 if k == "." else -0.01)))
        elif k == "t":
            self.enabled = not self.enabled
            self.cmd = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.status()

    def status(self):
        sys.stdout.write(
            f"\rvx {self.cmd['x']:+.2f}  vy {self.cmd['y']:+.2f}  wz {self.cmd['z']:+.2f}"
            f"   speed {self.speed:.2f}/{self.speed_max:.2f}"
            f"  height {self.gait.body_height*1000:3.0f} mm"
            f"   gait {'on ' if self.enabled else 'OFF'}   ")
        sys.stdout.flush()

    def __call__(self):
        self.poll()
        if not self.enabled:
            return (0.0, 0.0, 0.0)
        return (self.cmd["x"], self.cmd["y"], self.cmd["z"])


def clamp_profile(steps, vmax, wzmax=None):
    """The scripted demo, with every commanded velocity brought inside the fit.

    `PROFILE` carries its own velocities, so the feasibility fit on `--speed` does not
    reach it: without this the demo would still command 0.20 m/s and drag its feet on the
    one run that is meant to be shown to people.  Direction and timing are untouched; only
    the magnitude is capped, so the shape of the demo survives.

    `wzmax` because the profile's fourth step is a 0.6 rad/s turn on the spot and it was
    the half `feasible_gait` never looked at — see `feasible_turn`.
    """
    out = []
    for secs, vx, vy, wz in steps:
        sc = min(1.0, vmax / max(abs(vx), abs(vy), 1e-9))
        if wzmax is not None:
            wz = max(-wzmax, min(wzmax, wz))
        out.append((secs, vx * sc, vy * sc, wz))
    return out


def profile_source(steps):
    """A scripted command sequence; raises StopIteration when it runs out."""
    state = {"t": 0.0, "i": 0}

    def cmd(dt):
        state["t"] += dt
        while state["i"] < len(steps) and state["t"] > steps[state["i"]][0]:
            state["t"] -= steps[state["i"]][0]
            state["i"] += 1
        if state["i"] >= len(steps):
            raise StopIteration
        return steps[state["i"]][1:]
    return cmd


# ------------------------------------------------------------------ contact
def contact_feeder(gait, baseline, threshold, sign):
    """{leg: bool} from the knee servos' load, out of the feedback we already read."""
    est = ServoContact(baseline, threshold=threshold, sign=sign, legs=tuple(gait.legs))

    def feed(dt, fb):
        load, phase = {}, {}
        for l in gait.legs:
            f = fb.get(f"{l}_knee")
            if f is None:
                continue
            load[l] = f["load"]
            phase[l] = gait.leg_phase(l)
        if not load:
            return None
        return est.update(dt, phase, load)
    return feed


def record_baseline(rt, gait, args, out_path):
    """Hang the robot up, trot in free air, average the knee load per gait phase."""
    b = Baseline(gait=dict(period=gait.period, swing_height=gait.swing_height,
                           body_height=gait.body_height, speed=args.speed))
    cmd = (args.speed, 0.0, 0.0)

    def source(dt, fb):
        for l in gait.legs:
            f = fb.get(f"{l}_knee")
            if f is not None:
                b.add(l, gait.leg_phase(l), f["load"])
        return gait.joint_targets(dt, *cmd)

    rt.run(source, seconds=args.seconds or 20.0)
    b.finish()
    cov = b.coverage()
    print("phase coverage: " + ", ".join(f"{l} {100*v:.0f} %" for l, v in cov.items()))
    if min(cov.values()) < 0.9:
        # NOT "run it longer". The phase advances dt/period per tick, so the bin
        # index steps by nbin*dt/period — one bin or more at any period under about
        # 1.2 s at 50 Hz — and a bin the phase never lands in is never filled
        # however long the run is. It is aliasing, and the only two cures are a
        # longer period (a smaller step) or fewer bins. robot/README.md, "Two bins
        # in sixty".
        per_tick = (1.0 / args.hz) / max(gait.period, 1e-9) * b.nbin
        print(f"!! thin coverage: the phase advances {per_tick:.2f} bins per tick at "
              f"period {gait.period:.2f} s, {args.hz:.0f} Hz, {b.nbin} bins.")
        print("!! This is ALIASING, not a short run — a bin the phase never lands in "
              "stays empty however\n!! long you record. Lengthen --period, or record "
              "against fewer bins.")
    print("saved:", b.save(out_path))


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--dry-run", action="store_true",
                    help="loopback bus, no hardware; implies --profile")
    ap.add_argument("--calib", default=CALIB)
    ap.add_argument("--hz", type=float, default=CTRL_HZ)
    ap.add_argument("--mode0", action="store_true",
                    help="the servo's own position loop (MODE 0) instead of the host loop")
    ap.add_argument("--mode2", action="store_true", help=argparse.SUPPRESS)   # the default
    ap.add_argument("--sub-hz", type=float, default=0.0,
                    help="--mode2: pace the host loop, Hz (0: as fast as the bus goes)")
    ap.add_argument("--kp", type=float, default=None, help="--mode2: duty per rad")
    ap.add_argument("--kd", type=float, default=None, help="--mode2: duty per rad/s")
    ap.add_argument("--kff", type=float, default=None, help="--mode2: duty per rad/s of target")
    ap.add_argument("--duty-cap", default="", metavar="SPEC",
                    help="--mode2: per-joint duty ceiling, e.g. pitch=600,fl_roll=800 (a joint "
                         "name or a kind: roll/pitch/knee). 600 at 12 V is the ~2.3 N*m the "
                         "firmware's own protection allowed; mode2.py, 'A torque ceiling'")

    ap.add_argument("--preflight", action="store_true", help="check everything, no motion")
    ap.add_argument("--stand", action="store_true", help="stand up and hold, no gait")
    ap.add_argument("--profile", action="store_true", help="run the scripted demo")
    ap.add_argument("--go", type=float, metavar="S",
                    help="stand 2 s, walk straight at --speed for S seconds, stand; a distance run")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--ramp", type=float, default=2.0, help="s to stand up in")
    ap.add_argument("--no-sit", action="store_true", help="cut torque where it stands")
    ap.add_argument("--log", metavar="FILE.npz",
                    help="record every tick: feedback, goals, gait phase (bench/pack_sag.py)")
    ap.add_argument("--imu", action="store_true",
                    help="read the BMI088 every tick into the --log (heading, attitude, footfalls)")
    ap.add_argument("--i2c-bus", type=int, default=1)
    ap.add_argument("--bias-seconds", type=float, default=3.0)

    ap.add_argument("--speed", type=float, default=0.20, help="m/s for the teleop keys")
    ap.add_argument("--turn", type=float, default=1.2, help="rad/s for the teleop keys")
    ap.add_argument("--height", type=float, default=0.158, help="body height, m")
    ap.add_argument("--period", type=float, default=0.45)
    ap.add_argument("--swing", type=float, default=0.022)
    ap.add_argument("--max-step", type=float, default=0.060)

    ap.add_argument("--baseline", metavar="FILE",
                    help="record the free-air knee-load curve (robot HANGING) and exit")
    ap.add_argument("--as-commanded", action="store_true",
                    help="do NOT fit the gait to the servo's joint speed; run --speed and "
                         "--period exactly as given, feet dragging and all")
    ap.add_argument("--contact", metavar="FILE", help="use a recorded baseline")
    # 50 is measured, on this robot, at 2.55 kg: the residual reads 65..104 units through
    # stance and 1..10 through swing, against an air baseline that repeats to ~2 units.
    # It is NOT the 0.5 the MuJoCo study used - that was in N.m, this is Present Load.
    ap.add_argument("--contact-threshold", type=float, default=50.0)
    # "auto" takes each leg's knee sign from the calibration, which is what the hardware
    # needs: the hubs are mirrored left to right, so a scalar leaves two legs unable to
    # fire whichever way it is set.  An integer still works, for a sensor that is not the
    # servo.  See smalldog_walker/contact.py, ServoContact.
    ap.add_argument("--contact-sign", default="auto")

    ap.add_argument("--temp-c", type=float, default=Limits.temp_c)
    ap.add_argument("--current-a", type=float, default=Limits.current_a)
    ap.add_argument("--volt-min", type=float, default=Limits.volt_min)
    # The tracking limit is the one Limits itself says is unmeasured, and it is the one
    # that stops a loaded run before it can measure anything. Settable so the honest
    # number can be found on the ground, which is the only place it exists.
    ap.add_argument("--track-rad", type=float, default=Limits.q_err_rad,
                    help="tracking-error trip, rad (default %(default)s)")
    ap.add_argument("--tilt-deg", type=float, default=math.degrees(Limits.tilt_rad),
                    help="--imu: body roll or pitch that trips, deg (default %(default).0f)")
    a = ap.parse_args()
    a.mode2 = not a.mode0                  # MODE 2 is the runtime (ideas/FAST_SERVOS.md)

    params = load_params()
    calib = Calibration.load(a.calib, params) if os.path.exists(a.calib) \
        else Calibration.default(params)
    if not getattr(calib, "measured", False) and not a.dry_run:
        print("!! this calibration is defaults, not this robot. Torque stays off.")
        print("!! run: python runtime/calib.py --capture, then --sign all")
        return 2

    limit = MODE2_RATE_LIMIT if a.mode2 else None
    if not a.as_commanded:
        speed, period, note = feasible_gait(params, a, limit=limit)
        a.speed, a.period = speed, period
        if note:
            print(note)
        turn, tnote = feasible_turn(params, a, a.period, limit=limit)
        a.turn = turn
        if tnote:
            print(tnote)
    gait = build_gait(params, a)
    if a.mode2:
        # the gait's own limiter is the position-mode ceiling; lift it to the motor's,
        # or every command over 3.3 rad/s is clipped before the host loop sees it
        gait.max_joint_rate = MODE2_RATE_LIMIT
        print(f"MODE 2: host position loop, joint rate ceiling {MODE2_RATE_LIMIT} rad/s")
    if not a.as_commanded:
        # period_for() only ever SHORTENS the period, pinning it at 2*stride_max/speed, so
        # a period chosen for feasibility has to be admissible there too or it is silently
        # discarded.  This raises the pin exactly far enough for the period just chosen and
        # no further, so the schedule keeps shortening with speed the way it was tuned to.
        gait.stride_max = max(gait.stride_max, a.speed * a.period / 2.0)
    if list(gait.joint_names) != list(calib.joints):
        print("!! the gait and the calibration disagree about joint order — a leg would\n"
              "!! be driven by another leg's servo. Both read robot_params.json, so one\n"
              f"!! of them is stale:\n!!   gait  {list(gait.joint_names)}\n"
              f"!!   calib {list(calib.joints)}")
        return 2
    r = gait.reach_info()
    print(f"trot: period {gait.period:.2f} s, swing {r['swing_height']*1000:.0f} mm, "
          f"body {r['body_height']*1000:.0f} mm of {r['height_min']*1000:.0f}.."
          f"{r['height_max']*1000:.0f}")
    dem = joint_rate_demand(params, a.speed, gait.period, gait.swing_height,
                            gait.max_step, gait.body_height)
    print(f"  {a.speed:.2f} m/s demands {dem:.2f} rad/s of "
          f"{gait.max_joint_rate:.2f} available"
          + ("" if dem <= gait.max_joint_rate else "  !! CLIPPED — the feet will drag"))
    if a.turn:
        # A turn on the spot, which is the only turn either caller commands — see
        # `feasible_turn`. Printed beside the forward demand because it is usually
        # the larger of the two and was never shown at all before 2026-09-11.
        dturn = joint_rate_demand(params, 0.0, gait.period, gait.swing_height,
                                  gait.max_step, gait.body_height, wz=a.turn)
        print(f"  {a.turn:.2f} rad/s of turn on the spot demands {dturn:.2f} rad/s"
              + ("" if dturn <= gait.max_joint_rate
                 else "  !! CLIPPED — the feet will drag in the turn"))

    if a.dry_run:
        loop = DutyLoopback if a.mode2 else FollowingLoopback
        bus = Bus(transport=loop(calib.ids), discard_echo=False)
        a.profile = a.profile or not a.stand
    else:
        bus = Bus(a.port, a.baud)

    limits = Limits(temp_c=a.temp_c, current_a=a.current_a, volt_min=a.volt_min,
                    q_err_rad=a.track_rad, tilt_rad=math.radians(a.tilt_deg))
    if a.mode2:
        gains = {k: v for k, v in dict(kp=a.kp, kd=a.kd, kff=a.kff).items() if v is not None}
        try:
            caps = parse_duty_caps(a.duty_cap, calib.joints)
        except ValueError as e:
            sys.exit(f"--duty-cap: {e}")
        rt = Mode2Runtime(bus, calib, hz=a.hz, limits=limits, sub_hz=a.sub_hz,
                          duty_cap=caps, **gains)
    else:
        rt = Runtime(bus, calib, hz=a.hz, limits=limits)

    pre = rt.preflight(None if a.dry_run else a.port)
    if not pre["ok"]:
        print("!! not every servo answered; refusing to move")
        return 1
    if a.preflight:
        return 0

    contact = None
    if a.contact:
        base = Baseline.load(a.contact)
        drift = base.mismatch(dict(period=gait.period, swing_height=gait.swing_height,
                                   body_height=gait.body_height, speed=a.speed))
        for k, (was, now) in drift.items():
            print(f"!! baseline was recorded at {k}={was}, running at {now}")
        if drift:
            # The curve is the load of ONE trajectory; at another period or speed it
            # is measuring something else and every residual is off by the
            # difference. There is no hardware fix from here — a baseline is a
            # 30 s recording of the robot HANGING — so say the two ways out rather
            # than just the complaint. This fires by default today: the baselines on
            # file were recorded before the rate ceiling became the measured 3.15
            # rad/s, and `feasible_gait` now picks a different operating point.
            recorded = {k: v[0] for k, v in drift.items()}
            flags = " ".join(f"--{k} {v:g}" for k, v in recorded.items()
                             if k in ("speed", "period"))
            print("!! The residual it produces is not this gait's. Either re-record "
                  "the baseline at\n"
                  "!! the operating point you are about to walk at "
                  f"(--baseline FILE), or run at the\n"
                  f"!! point it was recorded at: --as-commanded {flags}".rstrip())
        if str(a.contact_sign) == "auto":
            sign = {l: calib.sign[f"{l}_knee"] for l in gait.legs}
            shown = " ".join(f"{l}{v:+d}" for l, v in sign.items())
        else:
            sign = int(a.contact_sign)
            shown = f"{sign:+d} on every leg"
        contact = contact_feeder(gait, base, a.contact_threshold, sign)
        print(f"contact from knee load: {a.contact}")
        print(f"  threshold {a.contact_threshold:g} Present Load, sign {shown}")
    else:
        print("no IMU, no contact baseline: this is the blind open-loop trot")

    dt = 1.0 / a.hz
    q_stand = stance_pose(gait, dt)
    # Print it before torque, not after. A sign that is wrong in calib.json shows up
    # here as a knee at -60 degrees, and reading one line is cheaper than watching a
    # leg fold the wrong way under 2.5 kg.
    print("standing pose, deg: " + "  ".join(
        f"{l} " + "/".join(f"{math.degrees(q_stand[gait.joint_names.index(f'{l}_{k}')]):+.0f}"
                           for k in ("roll", "pitch", "knee")) for l in gait.legs))

    imu = None
    if a.imu and a.log:
        # The walker never looks at it — the trot stays open-loop. It is logged so the run
        # can be READ: yaw from the gyro says whether it held a line, the accelerometer
        # says when the feet really land against when the gait thinks they do.
        from imu.bmi088 import BMI088, measure_bias
        from runtime.policy import LiveIMU
        chip = BMI088(a.i2c_bus)
        chip.configure()
        print(f"IMU: hold still {a.bias_seconds:g} s for the gyro bias ...")
        bias = measure_bias(chip, a.bias_seconds, a.hz)
        print("gyro bias rad/s: " + " ".join(f"{b:+.4f}" for b in bias))
        imu = LiveIMU(chip, bias)
    elif a.imu:
        print("!! --imu does nothing without --log")

    code = 0
    tick = None
    try:
        with rt:
            rt.engage(q_stand, ramp_s=a.ramp)

            if a.baseline:
                print("the robot must be HANGING — feet off the ground — for this.")
                record_baseline(rt, gait, a, a.baseline)
            elif a.stand:
                print("standing. Ctrl-C to sit down.")
                tick = TickLog(rt, gait, lambda: (0.0, 0.0, 0.0), imu) if a.log else None
                rt.run(lambda dt_, fb: gait.joint_targets(dt_, 0.0, 0.0, 0.0),
                       seconds=a.seconds, on_tick=tick)
            elif a.go or a.profile or not Teleop.available():
                if a.go:
                    steps = [(2.0, 0.0, 0.0, 0.0), (a.go, a.speed, 0.0, 0.0), (1.5, 0.0, 0.0, 0.0)]
                    print(f"straight run: {a.go:.0f} s at {a.speed:.2f} m/s, "
                          f"about {a.go * a.speed:.1f} m")
                else:
                    if not a.profile:
                        print("no TTY for the keyboard; running the scripted profile")
                    steps = PROFILE if a.as_commanded else clamp_profile(PROFILE, a.speed, a.turn)
                cmd = profile_source(steps)

                last = [0.0, 0.0, 0.0]

                def source(dt_, fb):
                    if contact:
                        c = contact(dt_, fb)
                        if c:
                            gait.feedback(contact=c)
                    last[:] = cmd(dt_)
                    return gait.joint_targets(dt_, *last)
                tick = TickLog(rt, gait, lambda: last, imu) if a.log else None
                rt.run(source, seconds=a.seconds, on_tick=tick)
            else:
                with Teleop(gait, a.speed, a.turn) as tele:
                    tele.status()

                    last = [0.0, 0.0, 0.0]

                    def source(dt_, fb):
                        if contact:
                            c = contact(dt_, fb)
                            if c:
                                gait.feedback(contact=c)
                        last[:] = tele()
                        return gait.joint_targets(dt_, *last)
                    tick = TickLog(rt, gait, lambda: last, imu) if a.log else None
                    rt.run(source, seconds=a.seconds, on_tick=tick)

            if not a.no_sit:
                h = gait.body_height
                gait.body_height = 0.0            # the setter clamps to the lowest reachable
                rt.relax(stance_pose(gait, dt), ramp_s=1.5)
                gait.body_height = h
    except KeyboardInterrupt:
        print("\ninterrupted")
    except Tripped as e:
        print(f"\n!! TRIPPED: {e}")
        code = 1
    finally:
        print(rt.report_lines())
        if a.log and tick is not None:
            tick.save(a.log, dict(period=gait.period, speed=a.speed, swing_height=gait.swing_height,
                                  body_height=gait.body_height, hz=a.hz, mode2=a.mode2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
