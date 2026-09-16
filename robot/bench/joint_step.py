import sys, time, math, json; sys.path.insert(0, ".")
from feetech.bus import Bus
from runtime.calib import Calibration, CALIB
from runtime.mode2 import Mode2Runtime
from runtime.safety import Tripped
name, kp, kd = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
c = Calibration.load(CALIB)
one = Calibration([name], ids={name: c.id[name]}, centre={name: c.centre[name]},
                  sign={name: c.sign[name]}, soft={name: c.soft[name]})
bus = Bus("/dev/ttyACM0", 1_000_000)
rt = Mode2Runtime(bus, one, kp=kp, kd=kd, runaway_rad=0.8)
print(name, "id", c.id[name], "sign", c.sign[name], "centre", c.centre[name], "q0 %.2f" % rt.read()[name]["q"])
rows = []
orig = rt._pd
def pd(fb):
    f = fb[name]
    orig(fb)
    rows.append((time.perf_counter(), f["q"], rt.goal[name], f["w"], rt.duty[name], f["current"], f["load"]))
rt._pd = pd
q0 = rt.read()[name]["q"]
try:
    with rt:
        rt.engage([q0], ramp_s=0.5)
        for tgt in (q0 + 0.2, q0, q0 - 0.2, q0):
            rt.run(lambda dt, fb: [tgt], seconds=0.6)
        rt.relax([q0], ramp_s=0.5)
except Tripped as e:
    print("!! TRIPPED:", e)
print(rt.report_lines())
t0 = rows[0][0]
for r in rows[::4]:
    print("%.3f q %+.3f g %+.3f w %+.2f d %+5d I %.2f L %+5d" % ((r[0]-t0,) + r[1:]))
