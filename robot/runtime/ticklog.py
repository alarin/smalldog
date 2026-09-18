"""
ticklog.py — every tick of a run into one .npz: the black box.

    tick = TickLog(rt, gait, cmd=lambda: last, imu=imu)     # walk.py --log
    rt.run(source, on_tick=tick)
    tick.save("bench/data/go20.npz", dict(period=..., hz=50))

One format for `walk.py --log`, `policy.py --log`'s runtime half and the ROS servo
node's `log:=` — so `bench/pack_sag.py`, `bench/trot_report.py` and
`bench/incident.py` read a recording whichever way the robot was driven. The
explorer run that broke the front hip brackets (2026-09-17) left no recording at
all: `walk.py` and `policy.py` logged on request, the servo node never did, and
the 5 Hz `/diagnostics` is not a record of a 20 ms event. Now the node logs by
default into a ring of the last `minutes`, written on every exit including a trip.

What a row holds, per tick: `t`, `dt`, the twelve servos' `q w load volt temp
current` (`fb`, joints × FIELDS, the same 15-byte read the loop does anyway),
the `goal` handed to the servos, the gait `phase` per leg (nan under a policy),
the `cmd` the source was given (vx vy wz), the IMU triple `imu` (gravity_b,
gyro, accel — what policy.py feeds the network), and in MODE 2 the `duty` the
host loop last wrote per joint, which is the torque the bracket saw. `overruns`
is the runtime's running count, so a late tick is visible beside what it did.
"""
from __future__ import annotations

import collections
import json
import math
import os
import time

FIELDS = ("q", "w", "load", "volt", "temp", "current")


class TickLog:
    def __init__(self, rt, gait=None, cmd=None, imu=None, imu_read=True, minutes=None):
        """`imu_read`: this log is the tick's IMU reader (walk.py: nothing else reads
        the chip, and the read feeds the guard's tilt trip here). False: the source
        or the node read it this tick and `imu.last` is that read (policy.py, the
        servo node) — reading twice would hand two different `dt` to the filter."""
        self.rt, self.gait, self.cmd, self.imu, self.imu_read = rt, gait, cmd, imu, imu_read
        self.joints = list(rt.calib.joints)
        self.legs = list(gait.legs) if gait is not None else \
            list(dict.fromkeys(n.split("_")[0] for n in self.joints))
        self.t = 0.0
        maxlen = int(minutes * 60 * rt.hz) if minutes else None
        self.rows = collections.deque(maxlen=maxlen)
        self.dropped = 0

    def __call__(self, k, dt, fb):
        import numpy as np
        self.t += dt
        fbv = [[fb[n][f] if fb[n] else math.nan for f in FIELDS] for n in self.joints]
        goal = [self.rt.goal[n] for n in self.joints]
        phase = ([self.gait.leg_phase(l) for l in self.gait.legs] if self.gait is not None
                 else [math.nan] * len(self.legs))
        cmd = list(self.cmd()) if self.cmd else [math.nan] * 3
        if self.imu is None:
            imu = [math.nan] * 9
        elif self.imu_read:
            imu = [*sum(self.imu.update(dt), ())]
            self.rt.guard.attitude(dt, *self.imu.roll_pitch())   # Tripped: the robot is over
        else:
            imu = [*sum(self.imu.last, ())]
        duty = [self.rt.duty.get(n, math.nan) for n in self.joints] if hasattr(self.rt, "duty") \
            else [math.nan] * len(self.joints)
        if self.rows.maxlen and len(self.rows) == self.rows.maxlen:
            self.dropped += 1
        self.rows.append((self.t, dt, np.array(fbv, np.float32), np.array(goal, np.float32),
                          np.array(phase, np.float32), np.array(cmd, np.float32),
                          np.array(imu, np.float32), np.array(duty, np.float32),
                          self.rt.overruns))

    def save(self, path, meta: dict | None = None, log=print) -> str | None:
        """Write the ring. `path` may be a directory: then `tick_<stamp>.npz` in it.
        `meta` lands in the `gait` key (pack_sag.py reads period/speed/hz from it)
        and, as JSON, in `meta`. Returns the file written, None if nothing was."""
        import numpy as np
        if not self.rows:
            return None
        if path.endswith(os.sep) or os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, time.strftime("tick_%Y%m%d_%H%M%S.npz"))
        else:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        t, dt, fb, goal, phase, cmd, imu, duty, over = (np.array(x) for x in zip(*self.rows))
        meta = dict(meta or {})
        np.savez(path, t=t, dt=dt, fb=fb, goal=goal, phase=phase, cmd=cmd, imu=imu, duty=duty,
                 overruns=over, fields=np.array(FIELDS), joints=np.array(self.joints),
                 legs=np.array(self.legs), gait=np.array(meta),
                 meta=np.array(json.dumps(meta, default=str)))
        log(f"log: {len(self.rows)} ticks -> {path}"
            + (f" (ring: {self.dropped} older ticks dropped)" if self.dropped else ""))
        return path
