"""
walk.py — the trot, on the real robot. The CLI that wires everything together.

    python runtime/walk.py --dry-run --profile           # no hardware at all
    python runtime/walk.py --port /dev/ttyUSB0 --preflight
    python runtime/walk.py --port /dev/ttyUSB0 --stand --seconds 10
    python runtime/walk.py --port /dev/ttyUSB0 --profile
    python runtime/walk.py --port /dev/ttyUSB0            # keyboard teleop

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

    def __init__(self, gait, speed=0.20, turn=1.2, log=print):
        self.gait, self.log = gait, log
        self.speed, self.turn = speed, turn
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
            self.speed = max(0.05, min(0.45, self.speed + (0.05 if k == "." else -0.05)))
        elif k == "t":
            self.enabled = not self.enabled
            self.cmd = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.status()

    def status(self):
        sys.stdout.write(
            f"\rvx {self.cmd['x']:+.2f}  vy {self.cmd['y']:+.2f}  wz {self.cmd['z']:+.2f}"
            f"   speed {self.speed:.2f}  height {self.gait.body_height*1000:3.0f} mm"
            f"   gait {'on ' if self.enabled else 'OFF'}   ")
        sys.stdout.flush()

    def __call__(self):
        self.poll()
        if not self.enabled:
            return (0.0, 0.0, 0.0)
        return (self.cmd["x"], self.cmd["y"], self.cmd["z"])


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
        print("!! thin coverage — run it longer, or the curve has bins it invented")
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

    ap.add_argument("--preflight", action="store_true", help="check everything, no motion")
    ap.add_argument("--stand", action="store_true", help="stand up and hold, no gait")
    ap.add_argument("--profile", action="store_true", help="run the scripted demo")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--ramp", type=float, default=2.0, help="s to stand up in")
    ap.add_argument("--no-sit", action="store_true", help="cut torque where it stands")

    ap.add_argument("--speed", type=float, default=0.20, help="m/s for the teleop keys")
    ap.add_argument("--turn", type=float, default=1.2, help="rad/s for the teleop keys")
    ap.add_argument("--height", type=float, default=0.158, help="body height, m")
    ap.add_argument("--period", type=float, default=0.45)
    ap.add_argument("--swing", type=float, default=0.022)
    ap.add_argument("--max-step", type=float, default=0.060)

    ap.add_argument("--baseline", metavar="FILE",
                    help="record the free-air knee-load curve (robot HANGING) and exit")
    ap.add_argument("--contact", metavar="FILE", help="use a recorded baseline")
    ap.add_argument("--contact-threshold", type=float, default=0.5)
    ap.add_argument("--contact-sign", type=int, default=-1)

    ap.add_argument("--temp-c", type=float, default=Limits.temp_c)
    ap.add_argument("--current-a", type=float, default=Limits.current_a)
    ap.add_argument("--volt-min", type=float, default=Limits.volt_min)
    a = ap.parse_args()

    params = load_params()
    calib = Calibration.load(a.calib, params) if os.path.exists(a.calib) \
        else Calibration.default(params)
    if not getattr(calib, "measured", False) and not a.dry_run:
        print("!! this calibration is defaults, not this robot. Torque stays off.")
        print("!! run: python runtime/calib.py --capture, then --sign all")
        return 2

    gait = build_gait(params, a)
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

    if a.dry_run:
        bus = Bus(transport=FollowingLoopback(calib.ids), discard_echo=False)
        a.profile = a.profile or not a.stand
    else:
        bus = Bus(a.port, a.baud)

    limits = Limits(temp_c=a.temp_c, current_a=a.current_a, volt_min=a.volt_min)
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
        contact = contact_feeder(gait, base, a.contact_threshold, a.contact_sign)
        print(f"contact from knee load: {a.contact}")
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

    code = 0
    try:
        with rt:
            rt.engage(q_stand, ramp_s=a.ramp)

            if a.baseline:
                print("the robot must be HANGING — feet off the ground — for this.")
                record_baseline(rt, gait, a, a.baseline)
            elif a.stand:
                print("standing. Ctrl-C to sit down.")
                rt.run(lambda dt_, fb: gait.joint_targets(dt_, 0.0, 0.0, 0.0),
                       seconds=a.seconds)
            elif a.profile or not Teleop.available():
                if not a.profile:
                    print("no TTY for the keyboard; running the scripted profile")
                cmd = profile_source(PROFILE)

                def source(dt_, fb):
                    if contact:
                        c = contact(dt_, fb)
                        if c:
                            gait.feedback(contact=c)
                    return gait.joint_targets(dt_, *cmd(dt_))
                rt.run(source, seconds=a.seconds)
            else:
                with Teleop(gait, a.speed, a.turn) as tele:
                    tele.status()

                    def source(dt_, fb):
                        if contact:
                            c = contact(dt_, fb)
                            if c:
                                gait.feedback(contact=c)
                        return gait.joint_targets(dt_, *tele())
                    rt.run(source, seconds=a.seconds)

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
    return code


if __name__ == "__main__":
    raise SystemExit(main())
