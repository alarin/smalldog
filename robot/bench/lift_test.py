"""
lift_test.py — is foot contact visible in the servos at all, on this robot, today?

    python bench/lift_test.py --port /dev/ttyUSB0
    python bench/lift_test.py --port /dev/ttyUSB0 --out lift.json

`contact.py` infers contact from the knee's load minus the free-air baseline
(`walk.py --baseline`). That subtraction only means something if a foot taking
the robot's weight actually moves the number the servo reports, and on a
gearbox this stiff it may not: the load has to beat the friction before the
motor does anything at all, and below that the duty is the same loaded or not.

So this asks the question directly. It holds the standing pose and prints cues
while you set the robot down, lift it clear, and set it down again. Same pose
throughout, so the only thing that changes is what is under the feet, and the
difference between the readings IS the contact signal in Present Load units.

**Run it in your own terminal.** The cues are the point, and they are no use
inside a captured pipe.

Measured 2026-09-07, robot at 1.55 kg (no battery, no LiDAR, no Pi — 62 % of
design mass), per leg about 3.8 N:

    fl_knee    56 ->  0 -> 48    delta 52   clear
    fl_roll    40 ->  0 -> 40    delta 40   clear
    rr_pitch   40 ->  0 -> 24    delta 32   clear
    rr_knee   -56 -> -40 -> -48  delta -12  weak
    fr_knee   -32 -> -32 -> -32  delta 0    nothing
    rl_knee    24 ->  24 ->  24  delta 0    nothing

One leg of four (rl) reported nothing on any of its three joints. The two
down readings also disagree with each other — fl_knee 56 then 48 — which is
stiction: the number depends on which way the joint last moved, not on what is
under the foot. Load is quantised in steps of 8 here.

That is the friction band, and it is the same actuator fact `robot/bench` found
from the other side (the ST3215's friction is 6x the model's, and it held the
arm the motor was meant to).

**Re-run at 2.55 kg on 2026-09-08, and the signal is there on all four legs.**
1.55 kg of robot plus a 1 kg plate on the deck — 102 % of the 2.499 kg design
mass, though a plate is one lump where the missing 937 g is spread. Knee only,
which is what `contact.py` reads, |DOWN - LIFTED| in Present Load units:

    leg    plate forward    plate centred
    fl          56               56
    fr          24               24
    rl           8               48
    rr           8               16

Two things came out of running it twice. The front/rear split is **load
distribution, not per-servo scatter**: moving the plate back took rl_knee from 8
to 48 and fr_roll/fr_pitch from 0 to 24/32, while the front legs held. And the
joints whose load barely changed between the two runs reproduced **to the unit**
- fl_knee 56/56, fr_knee 24/24, rl_pitch 32/32, rl_roll 0/0 - which is the
repeatability the 1.55 kg run did not have. Interquartile spread inside every
window of both runs is 0 but for a single 8: the reading is latched, not noisy.
Present Load holds dead flat for seven seconds and steps only when what is under
the foot changes.

The sign is per-leg (fl/rl positive, fr/rr negative), which is `calib.py`'s knee
sign map +1,-1,+1,-1; `contact.py` needs the sign or an absolute value. rr is the
marginal leg at 16, above the baseline's 11-unit repeatability and under the 24
bar - and this is a *static* test with the feet placed by hand, where the gait
wants a touchdown edge inside a swing rather than a calibrated force.

**The first window is not a contact reading.** It is the state the 2 s engage
ramp leaves behind, with the legs driven into the table and the gearbox wound up,
and it runs 2-4x the second window (fl_knee 136 against 56). This file used to
average the two DOWN windows into one `delta`, so the 1.55 kg verdicts above
carry that artifact; `signal` is now DOWN2 - LIFTED alone and `ramped` reports
the other one beside it, for what the wound-up state is worth.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "ros2", "smalldog_walker"))

from feetech.bus import Bus                                          # noqa: E402
from runtime.calib import Calibration, load_params                   # noqa: E402
from runtime.loop import Runtime                                     # noqa: E402
from runtime.safety import Limits, Tripped                           # noqa: E402
from runtime.walk import build_gait, stance_pose                     # noqa: E402

#: (t, what to do). The trace is timestamped, so missing a cue by a second
#: costs nothing — the windows below are cut well inside each phase.
CUES = [(0, "STAND IT ON THE TABLE — all four feet down"),
        (8, "LIFT IT by the body — feet completely clear"),
        (16, "SET IT DOWN again — all four feet"),
        (24, None)]
WINDOWS = [(2, 7.5), (10, 15.5), (18, 23.5)]

#: The same three cues, short enough to be spoken inside the two seconds
#: between the cue and the window that reads it. `--speak` exists because the
#: printed cues are no use when the run is driven from anything but a terminal
#: you are watching, and both hands are on the robot either way.
SPOKEN = ["set it down", "lift it up", "set it down again", "finish"]
WARN_S = 3.0


def _say(text):
    try:
        subprocess.Popen(["say", "-r", "220", text],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--out", help="write the full timestamped trace here")
    ap.add_argument("--height", type=float, default=0.158)
    ap.add_argument("--ramp", type=float, default=2.0)
    ap.add_argument("--speak", action="store_true",
                    help="speak the cues (macOS `say`), for a run you are not watching")
    # Settable for the same reason walk.py's is: this run holds a stance with the
    # legs pushed into a table, which is the loaded case the trip level was measured
    # on, and a limit nobody can set is a limit nobody can measure.
    ap.add_argument("--track-rad", type=float, default=Limits.q_err_rad,
                    help="tracking-error trip, rad (default %(default)s)")
    ap.add_argument("--current-a", type=float, default=Limits.current_a,
                    help="current trip, A of PRESENT_CURRENT (default %(default)s)")
    a = ap.parse_args()
    a.speed, a.turn, a.period, a.swing, a.max_step, a.hz = 0.20, 1.2, 0.45, 0.022, 0.060, 50.0

    params = load_params()
    calib = Calibration.load(params=params)
    if not getattr(calib, "measured", False):
        print("!! this calibration is defaults, not this robot. Torque stays off.")
        return 2
    gait = build_gait(params, a)
    q = stance_pose(gait, 1.0 / a.hz)

    # The temperature limit is lifted because this run is stationary and long, and
    # PRESENT_TEMPERATURE spikes while the motors hold — see Limits.temp_hold_s.
    # Every other limit is left where it is, and that is now true: this loop used
    # to be hand-rolled around rt.read()/rt.send() and never called the guard at
    # all, so for the 24 s a loaded robot stood on the table nothing was watching
    # the current, the voltage, the tracking error or the bus. It runs through
    # `Runtime.run` now, which does the read, the guard and the write in the order
    # the runtime defines, and hands the timing report back at the end.
    rt = Runtime(Bus(a.port, a.baud), calib, hz=a.hz,
                 limits=Limits(temp_c=200.0, temp_warn_c=199.0,
                               q_err_rad=a.track_rad, current_a=a.current_a),
                 log=lambda *_: None)

    trace = []
    print("\ntorque comes on and it ramps to the standing pose over "
          f"{a.ramp:.0f} s. Ctrl-C is safe throughout.\n")
    cue = {"next": 0, "warned": -1, "t0": None}

    def source(dt, fb):
        """One tick: print or speak the cue that is due, record the loads, hold `q`."""
        if cue["t0"] is None:
            cue["t0"] = time.time()
        t = time.time() - cue["t0"]
        nxt = cue["next"]
        if (a.speak and cue["warned"] < nxt and CUES[nxt][0] > WARN_S
                and t >= CUES[nxt][0] - WARN_S):
            cue["warned"] = nxt
            _say("ready to " + SPOKEN[nxt])
        if t >= CUES[nxt][0]:
            if a.speak:
                _say("done, torque off" if CUES[nxt][1] is None else SPOKEN[nxt])
            if CUES[nxt][1] is None:
                raise StopIteration              # the last cue ends the run
            print(f"\n>>> [{CUES[nxt][0]:>2}s] {CUES[nxt][1]}\n")
            cue["next"] = nxt + 1
        else:
            print(f"\r    {CUES[nxt][0] - t:4.1f} s to the next cue ", end="", flush=True)
        trace.append([round(t, 3),
                      {n: (fb[n]["load"] if fb[n] else None) for n in calib.joints}])
        return q

    tripped = None
    with rt:
        try:
            rt.engage(q, ramp_s=a.ramp)
            if a.speak:
                _say("torque on, ramping to the stance")
            # A backstop on top of the cue that ends it: `seconds` is what stops this
            # if a cue is ever removed from CUES without its `None` sentinel.
            rt.run(source, seconds=CUES[-1][0] + 2.0)
        except Tripped as e:
            tripped = e                 # `with rt` has already cut torque

    if tripped is not None:
        # Torque is already off — `with rt` saw to that. The windows are still
        # printed below from whatever was recorded, because a trip at 20 s has two
        # of the three windows in it and the numbers are the point of the run.
        print(f"\n!! TRIPPED: {tripped}")
        print(f"!! after {trace[-1][0] if trace else 0.0:.1f} s of {CUES[-1][0]} — "
              f"the table below is whatever windows that reached.")
    print(rt.report_lines())

    def med(lo, hi, j):
        v = [s[1][j] for s in trace if lo <= s[0] < hi and s[1][j] is not None]
        return statistics.median(v) if v else float("nan")

    print(f"\n{'joint':<9} {'RAMPED':>7} {'LIFTED':>7} {'DOWN':>7} {'signal':>7} {'ramped':>7}  verdict")
    for j in calib.joints:
        d1, up, d2 = (med(*w, j) for w in WINDOWS)
        signal, ramped = d2 - up, d1 - up
        verdict = "clear" if abs(signal) >= 24 else "weak" if abs(signal) >= 8 else "NOTHING"
        print(f"{j:<9} {d1:>7.0f} {up:>7.0f} {d2:>7.0f} "
              f"{signal:>7.0f} {ramped:>7.0f}  {verdict}")
    print("\nsignal = DOWN - LIFTED, and it is the only column that is a contact reading.\n"
          "ramped = RAMPED - LIFTED, the same difference taken against the first window,\n"
          "which is not one: see the note in this file's docstring.\n"
          "A leg whose three joints all read NOTHING cannot be given a contact\n"
          "threshold at this mass: the load never leaves the gearbox's friction band.")
    if a.out:
        json.dump({"cues": CUES, "trace": trace}, open(a.out, "w"))
        print("saved:", a.out)
    return 1 if tripped is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
