"""
calib.py — which servo is which joint, where its zero is, and which way it turns.

    python runtime/calib.py --selftest                    # no hardware
    python runtime/calib.py --port /dev/ttyUSB0 --check   # ping all 12, nothing moves
    python runtime/calib.py --port /dev/ttyUSB0 --capture
    python runtime/calib.py --port /dev/ttyUSB0 --sign fl_roll
    python runtime/calib.py --port /dev/ttyUSB0 --sign all

Three numbers per joint, and none of them can be derived from the CAD:

  **id**       which servo answers for this joint. Free choice, so it is made once
               here — `fl_roll` = 1 through `rr_knee` = 12, in the order
               `robot_params.json` lists the joints — and the servos get programmed
               to match before assembly (`3d/README.md`, "Assembly order", step 2).
               Print `--ids` and set them with the Feetech tool over the URT-1.

  **centre**   the count the servo reads at the model's mechanical zero — legs
               straight down. The hub can be bolted onto the output in any of four
               positions and the servo's own OFFSET register may be anything, so
               this is a measurement of the assembled robot and nothing else.
               `--capture` takes it: torque off, hold the pose, read.

  **sign**     whether a positive joint angle in the model is a rising or a falling
               count. Also an assembly fact — which face of the fork the driven hub
               ended up on — and the one that cannot be read off the bus at all,
               because the servo happily reports its own counts either way. `--sign`
               moves the joint a little and asks a human which way it went. Getting
               it wrong on one knee is a leg that drives itself into the ground the
               moment torque comes on, so this is asked rather than assumed.

The model's conventions, which the `--sign` prompts are written against
(`3d/README.md`, "Simulation export"; `leg_kinematics.py`'s FK is the arithmetic):
+X forward, +Y left, +Z up, zero pose = legs straight down, and the joint axes are
**identical on all four legs** — the mirrored legs carry mirrored limits, not
mirrored axes. So a positive angle does the same thing on every leg:

    roll   about +X   the foot swings toward the robot's LEFT (+Y)
    pitch  about +Y   the foot swings BACKWARD (-X)
    knee   about +Y   the shin folds BACKWARD and the leg gets SHORTER

The result is one JSON file for one physical robot. It belongs in git — it is a
measurement that has to reach the Orange Pi, and the repository is the only thing
that crosses between machines.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from feetech import registers as R                                   # noqa: E402
from feetech.bus import Bus, BusError, Servo                         # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
PARAMS = os.path.join(REPO, "ros2", "smalldog_description", "robot_params.json")
CALIB = os.path.join(HERE, "calib.json")

#: What a positive angle does, per joint kind. Used verbatim in the --sign prompt.
POSITIVE = {
    "roll":  "the foot swings toward the robot's LEFT (+Y)",
    "pitch": "the foot swings BACKWARD (-X, toward the tail)",
    "knee":  "the shin folds BACKWARD and the leg gets SHORTER",
}


def load_params(path=PARAMS) -> dict:
    with open(path) as f:
        return json.load(f)


def kind(joint: str) -> str:
    return joint.split("_")[1]


def default_ids(joint_names) -> dict:
    """1..12 in `robot_params.json` order. The servos are programmed to match."""
    return {n: i + 1 for i, n in enumerate(joint_names)}


class Calibration:
    """The per-robot numbers, plus the soft limits the runtime clamps against.

    `attach(bus)` builds one `feetech.bus.Servo` per joint, which is where the
    counts<->radians conversion lives. There is deliberately no second copy of that
    arithmetic here: a calibration that converted its own way would disagree with
    the driver in exactly the cases that matter, off-centre and near the limits.
    """

    def __init__(self, joints, ids=None, centre=None, sign=None, soft=None,
                 params=None, note=""):
        self.joints = list(joints)
        self.id = dict(ids or default_ids(self.joints))
        self.centre = dict(centre or {n: R.COUNTS_PER_TURN // 2 for n in self.joints})
        self.sign = dict(sign or {n: +1 for n in self.joints})
        self.soft = dict(soft or {})
        self.params = params or {}
        self.note = note
        missing = [n for n in self.joints if n not in self.id or n not in self.centre]
        if missing:
            raise ValueError(f"calibration is missing id/centre for {missing}")

    # ------------------------------------------------------------------ build
    @classmethod
    def default(cls, params=None, path=PARAMS):
        p = params or load_params(path)
        joints = list(p["joint_names"])
        soft = {n: p["joint_soft_limits_rad"][kind(n)] for n in joints}
        return cls(joints, soft=soft, params=p,
                   note="defaults: ids 1..12 in robot_params order, centre 2048, "
                        "sign +1. NOT a measurement — run --capture and --sign.")

    @classmethod
    def load(cls, path=CALIB, params=None):
        with open(path) as f:
            d = json.load(f)
        p = params or load_params()
        joints = list(d.get("joints", p["joint_names"]))
        soft = d.get("soft") or {n: p["joint_soft_limits_rad"][kind(n)] for n in joints}
        c = cls(joints, ids={k: int(v) for k, v in d["id"].items()},
                centre={k: int(v) for k, v in d["centre"].items()},
                sign={k: int(v) for k, v in d["sign"].items()},
                soft={k: float(v) for k, v in soft.items()}, params=p,
                note=d.get("note", ""))
        c.captured = d.get("captured")
        c.measured = bool(d.get("measured", False))
        return c

    def save(self, path=CALIB):
        with open(path, "w") as f:
            json.dump(dict(
                joints=self.joints, id=self.id, centre=self.centre, sign=self.sign,
                soft=self.soft, note=self.note, measured=getattr(self, "measured", False),
                captured=getattr(self, "captured", None) or time.strftime("%Y-%m-%dT%H:%M:%S"),
            ), f, indent=1)
        return path

    # ------------------------------------------------------------------- use
    @property
    def ids(self):
        """Servo ids in joint order — the order every SyncWrite and SyncRead uses."""
        return [self.id[n] for n in self.joints]

    def attach(self, bus: Bus) -> dict:
        return {n: Servo(bus, self.id[n], centre=self.centre[n], sign=self.sign[n])
                for n in self.joints}

    def clamp(self, q):
        """Clamp a joint vector to the soft limits. The runtime's last line."""
        return [max(-self.soft[n], min(self.soft[n], float(v)))
                for n, v in zip(self.joints, q)]

    def describe(self) -> str:
        w = max(len(n) for n in self.joints)
        head = (f"{'joint':<{w}}  id  centre  sign   soft\n"
                f"{'-' * w}  --  ------  ----   ----")
        rows = [f"{n:<{w}}  {self.id[n]:>2}  {self.centre[n]:>6}  {self.sign[n]:>+4}"
                f"   {self.soft.get(n, float('nan')):.2f}" for n in self.joints]
        tail = "" if getattr(self, "measured", False) else \
            "\n!! not measured: these are defaults, not this robot. --capture, --sign."
        return "\n".join([head] + rows) + tail


# ------------------------------------------------------------------ procedures
def capture(bus, calib, joints=None, settle=0.2):
    """Torque off, read where the joints actually are, call that zero.

    The pose is the model's mechanical zero — legs straight down — because that is
    what every downstream number is referenced to: `stance_rad`, the soft limits,
    the gait's IK and the policy's observation. Assembly order step 6.
    """
    names = list(joints or calib.joints)
    servos = calib.attach(bus)
    for n in names:                       # torque off first: the pose is set by hand
        servos[n].torque(False)
    time.sleep(settle)
    got = {}
    for n in names:
        got[n] = servos[n].feedback()["counts"]
    for n in names:
        calib.centre[n] = got[n]
    calib.measured = True
    calib.captured = time.strftime("%Y-%m-%dT%H:%M:%S")
    return got


def probe_sign(bus, calib, joint, amount=0.15, dwell=0.6, ask=input, out=print):
    """Move one joint a little and ask a human which way it went.

    Only that servo is torqued, and only for as long as the move takes: with eleven
    other joints limp the robot is not going to walk off the bench, and a wrong sign
    here moves the joint 8.6 degrees the other way rather than into a hard stop.
    The amount is small for the same reason — this is a question, not a trajectory.
    """
    s = calib.attach(bus)[joint]
    q0 = s.feedback()["q"]
    s.goal(q0)                            # no jump when torque comes on
    s.torque(True)
    try:
        s.goal(q0 + amount)
        time.sleep(dwell)
        out(f"\n  {joint}: commanded +{math.degrees(amount):.0f} deg.")
        out(f"  A POSITIVE angle should be: {POSITIVE[kind(joint)]}")
        a = (ask("  Is that what happened? [y/n] ") or "").strip().lower()
        s.goal(q0)
        time.sleep(dwell)
    finally:
        s.torque(False)
    calib.sign[joint] = calib.sign[joint] if a.startswith("y") else -calib.sign[joint]
    return calib.sign[joint]


def check(bus, calib, out=print) -> dict:
    """Ping every id, read every joint, and say whether the robot is where the
    calibration thinks it is. Nothing moves."""
    servos = calib.attach(bus)
    rows, bad = {}, []
    for n in calib.joints:
        i = calib.id[n]
        if not bus.ping(i):
            rows[n] = None
            bad.append(f"{n} (id {i}): no answer")
            continue
        try:
            rows[n] = servos[n].feedback()
        except BusError as e:
            rows[n] = None
            bad.append(f"{n} (id {i}): {e}")
    w = max(len(n) for n in calib.joints)
    out(f"{'joint':<{w}}  id   counts     q deg   volt   temp")
    for n in calib.joints:
        fb = rows[n]
        if fb is None:
            out(f"{n:<{w}}  {calib.id[n]:>2}   --- no answer ---")
            continue
        out(f"{n:<{w}}  {calib.id[n]:>2}  {fb['counts']:>7}  {math.degrees(fb['q']):>+7.1f}"
            f"  {fb['volt']:>5.1f}  {fb['temp']:>5.0f}")
    if bad:
        out("\n!! " + "\n!! ".join(bad))
    return rows


def registers(bus, calib) -> dict:
    """Every servo's control registers, which any fit is conditional on.

    `robot/README.md`, "The bench, in order": a fit is only valid for the P/D/I
    coefficients, dead zone, punch and acceleration the servo had at the time, and
    the robot then has to run with the same ones. Recorded on every run so that the
    day `rl/params/st3215.json` stops saying `fitted: false`, the comparison is
    already possible.
    """
    return {n: calib.attach(bus)[n].registers() for n in calib.joints}


# ============================================================== self-test
def _selftest() -> int:
    from feetech.loopback import LoopbackBus
    ok = True

    def chk(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name}: {got!r}" +
              ("" if good else f" != {want!r}"))

    c = Calibration.default()
    chk("12 joints", len(c.joints), 12)
    chk("ids 1..12", c.ids, list(range(1, 13)))
    chk("fl_roll is 1", c.id["fl_roll"], 1)
    chk("rr_knee is 12", c.id["rr_knee"], 12)

    # the soft limits come from robot_params.json, per joint kind
    p = load_params()
    chk("knee soft limit", c.soft["fl_knee"], p["joint_soft_limits_rad"]["knee"])
    chk("clamp bites", c.clamp([9.0] * 12)[2], p["joint_soft_limits_rad"]["knee"])
    chk("clamp is symmetric", c.clamp([-9.0] * 12)[0], -p["joint_soft_limits_rad"]["roll"])

    bus = Bus(transport=LoopbackBus({i: {} for i in c.ids}), discard_echo=False)
    servos = c.attach(bus)
    chk("attach covers every joint", sorted(servos) == sorted(c.joints), True)

    # counts <-> rad goes through feetech.bus.Servo, including the sign
    s = servos["fl_knee"]
    chk("centre is zero rad", s.to_rad(c.centre["fl_knee"]), 0.0)
    quarter = c.centre["fl_knee"] + R.COUNTS_PER_TURN // 4
    chk("+90 deg", round(math.degrees(s.to_rad(quarter)), 6), 90.0)
    c.sign["fl_knee"] = -1
    s = c.attach(bus)["fl_knee"]
    chk("sign flips the reading", round(math.degrees(s.to_rad(quarter)), 6), -90.0)
    c.sign["fl_knee"] = +1

    # capture: the loopback's registers start at zero, so zero is what comes back
    for i in c.ids:
        bus.io.set(i, R.PRESENT_POSITION, 1900)
    got = capture(bus, c, settle=0.0)
    chk("capture reads every joint", len(got), 12)
    chk("capture moves the centre", c.centre["fl_roll"], 1900)
    chk("capture marks it measured", c.measured, True)
    chk("captured centre is the new zero", c.attach(bus)["fl_roll"].to_rad(1900), 0.0)

    # torque must be off after a capture: the pose is set by hand
    chk("capture leaves torque off", bus.io.get(1, R.TORQUE_ENABLE), 0)

    # probe_sign: "n" flips, "y" keeps, and torque is off either way
    was = c.sign["fl_pitch"]
    chk("no flips on yes", probe_sign(bus, c, "fl_pitch", dwell=0.0,
                                      ask=lambda _: "y", out=lambda *_: None), was)
    chk("flips on no", probe_sign(bus, c, "fl_pitch", dwell=0.0,
                                  ask=lambda _: "n", out=lambda *_: None), -was)
    chk("probe leaves torque off", bus.io.get(c.id["fl_pitch"], R.TORQUE_ENABLE), 0)

    # round trip through the file
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "calib.json")
    c.save(path)
    c2 = Calibration.load(path)
    chk("round trip: centre", c2.centre, c.centre)
    chk("round trip: sign", c2.sign, c.sign)
    chk("round trip: ids", c2.id, c.id)
    chk("round trip: measured", c2.measured, True)

    print("calib:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--file", default=CALIB, help="calibration JSON (default runtime/calib.json)")
    ap.add_argument("--dry-run", action="store_true", help="loopback bus, no hardware")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ids", action="store_true", help="print the joint -> servo id map and exit")
    ap.add_argument("--check", action="store_true", help="ping and read everything; nothing moves")
    ap.add_argument("--capture", action="store_true", help="read the mechanical-zero pose")
    ap.add_argument("--sign", metavar="JOINT", help="probe one joint's direction, or 'all'")
    ap.add_argument("--joints", help="comma-separated subset for --capture")
    ap.add_argument("--registers", action="store_true", help="dump the control registers")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()

    calib = Calibration.load(a.file) if os.path.exists(a.file) else Calibration.default()

    if a.ids:
        print(calib.describe())
        print("\nSet these ids over the bus BEFORE assembly — 3d/README.md, "
              "'Assembly order', step 2.")
        return 0

    if a.dry_run:
        from feetech.loopback import LoopbackBus
        bus = Bus(transport=LoopbackBus({i: {} for i in calib.ids}), discard_echo=False)
    else:
        bus = Bus(a.port, a.baud)

    if a.check or not (a.capture or a.sign or a.registers):
        check(bus, calib)
        print()
        print(calib.describe())

    if a.registers:
        print(json.dumps(registers(bus, calib), indent=1))

    if a.capture:
        names = a.joints.split(",") if a.joints else calib.joints
        print("\nTorque is about to come OFF on: " + ", ".join(names))
        print("Hold the robot at the MECHANICAL ZERO pose — legs straight down —")
        print("and support it: nothing is holding the joints up.")
        input("Press Enter when it is there. ")
        got = capture(bus, calib, names)
        for n, v in got.items():
            print(f"  {n:<9} centre = {v}")
        print("saved:", calib.save(a.file))

    if a.sign:
        names = calib.joints if a.sign == "all" else a.sign.split(",")
        for n in names:
            if n not in calib.sign:
                print(f"!! no such joint: {n}")
                continue
            print(f"  {n}: sign is now {probe_sign(bus, calib, n):+d}")
        print("saved:", calib.save(a.file))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
