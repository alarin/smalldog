#!/usr/bin/env python
"""
bmi088.py — the BMI088 over I2C, and the gravity vector the policy is trained on.

    python imu/bmi088.py --selftest                      # no hardware: a fake chip
    python imu/bmi088.py --ids                           # find the two dies on the bus
    python imu/bmi088.py                                 # stream, 50 Hz
    python imu/bmi088.py --bias 5                        # 5 s still, gyro bias
    python imu/bmi088.py --level 5                       # 5 s STANDING LEVEL, no levelling
                                                         # (walk.py / robot.launch imu:=false):
                                                         # the mount tilt -> imu/mount.json
    python imu/bmi088.py --i2c-bus 1 --acc-addr 0x19 --gyro-addr 0x69

The BMI088 is two dies with two I2C addresses: accelerometer 0x18 or 0x19 (SDO1
pin), gyroscope 0x68 or 0x69 (SDO2). `--ids` probes all four and says which
answered with the right chip ID (0x1E and 0x0F). The default bus is the Orange
Pi 5 Pro's header I2C; `i2cdetect -y N` shows which N has the pair once the
overlay is on. Both dies return little-endian int16 x, y, z.

What the policy needs, and why not the quaternion
-------------------------------------------------
`rl/env/walk.py`'s frame starts with projected gravity — the world's -z axis
expressed in the body frame — then the gyro in rad/s and the accelerometer in
m/s^2. In simulation gravity comes off a perfect orientation sensor; the robot
has no magnetometer and no such thing, so `Attitude` integrates the gyro
(a fixed world vector seen from a rotating body obeys dg/dt = -w x g) and leans
on the accelerometer's direction slowly enough that a stride's own acceleration
does not tilt it: a complementary filter, 0.02 per tick at 50 Hz, ~1 s time
constant. It is initialised from the accelerometer alone, so the first second
after power-on the robot should be standing still.

Axes
----
`AXES` maps the chip's (x, y, z) onto the model's base_link frame: x forward
(nose), y left, z up. It is a guess until the stream has been read on the
robot: level, it must read gravity ~(0, 0, -1) and accel ~(0, 0, +9.8);
nose DOWN must send gravity's x POSITIVE (the body's x axis now points partly
at the ground, so gravity has a component along it) and the accelerometer's x
negative; rolled onto its LEFT side (y axis toward the ground), gravity's y
positive. Change AXES, not the frame.

Mount
-----
The chip is taped into the Pi's case and AXES says which way round; what is left is the
tape's tilt, a degree or two (measured 2026-09-15: roll -1.4, pitch -0.5). Small, but it
goes straight into the gait's levelling, so `--level` measures the accelerometer on a
level, still, standing robot with levelling off and writes the mount rotation to
`mount.json`; `BMI088` applies it to every read, so the gait, the policy and the ROS
node all see the body's frame, not the tape's. Re-run it whenever the case is opened,
and after any change to AXES (the file is in model axes).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time

# --------------------------------------------------------------- registers
ACC_CHIP_ID, ACC_CHIP_ID_VALUE = 0x00, 0x1E
ACC_DATA = 0x12                    # X LSB .. Z MSB
ACC_CONF, ACC_RANGE = 0x40, 0x41
ACC_PWR_CONF, ACC_PWR_CTRL = 0x7C, 0x7D
ACC_SOFTRESET = 0x7E
ACC_CONF_400HZ_NORMAL = 0xAA       # bwp=normal (0xA), odr=400 Hz (0xA)
ACC_RANGE_6G = 0x01                # full scale = 1.5 * 2^(range+1) g = 6 g

GYR_CHIP_ID, GYR_CHIP_ID_VALUE = 0x00, 0x0F
GYR_DATA = 0x02                    # X LSB .. Z MSB
GYR_RANGE, GYR_BANDWIDTH, GYR_LPM1 = 0x0F, 0x10, 0x11
GYR_SOFTRESET = 0x14
GYR_RANGE_500DPS = 0x02            # 65.536 LSB per deg/s
GYR_BW_400HZ_47HZ = 0x03           # ODR 400 Hz, filter 47 Hz
GYR_LSB_PER_DPS = {0x00: 16.384, 0x01: 32.768, 0x02: 65.536, 0x03: 131.072, 0x04: 262.144}

G = 9.80665

# chip axis -> model axis. Each entry is (chip index, sign) for model x, y, z.
# MEASURED 2026-09-15, the robot tilted by hand with the raw stream running: LiDAR end
# down -> chip y +0.47, LiDAR end up -> chip y -0.67, chip x still; left side down ->
# chip x -0.61. So the chip sits turned 90 deg in the case: body x is chip +y, body y
# is chip -x (right-handed with z up, and the left-side test agrees). The identity it
# shipped with put the pitch on the roll axis and the roll on the pitch axis with the
# sign wrong, and the gait's levelling ran on it as positive feedback: the standing
# robot tilted ITSELF 10 deg nose-down, 7 deg left-down, and the IMU read the opposite.
AXES = ((1, +1), (0, -1), (2, +1))


def remap(v, axes=AXES):
    return tuple(s * v[i] for i, s in axes)


HERE = os.path.dirname(os.path.abspath(__file__))
MOUNT_FILE = os.path.join(HERE, "mount.json")


def _rot_to_up(g_rest):
    """3x3 rotation taking the at-rest accelerometer direction to +z (Rodrigues)."""
    x, y, z = g_rest
    n = math.sqrt(x * x + y * y + z * z)
    a = (x / n, y / n, z / n)
    # axis = a x z_hat, angle = acos(a . z_hat)
    kx, ky, kz = a[1], -a[0], 0.0
    s = math.sqrt(kx * kx + ky * ky)
    c = a[2]
    if s < 1e-9:
        return ((1, 0, 0), (0, 1, 0), (0, 0, 1)) if c > 0 else ((1, 0, 0), (0, -1, 0), (0, 0, -1))
    kx, ky = kx / s, ky / s
    K = ((0, -kz, ky), (kz, 0, -kx), (-ky, kx, 0))
    I = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    KK = tuple(tuple(sum(K[i][m] * K[m][j] for m in range(3)) for j in range(3)) for i in range(3))
    return tuple(tuple(I[i][j] + s * K[i][j] + (1 - c) * KK[i][j] for j in range(3))
                 for i in range(3))


class Mount:
    """The chip's tilt in the body: the rotation that levels it, from `--level`."""

    def __init__(self, accel_at_rest=None, measured=None):
        self.accel_at_rest = tuple(accel_at_rest) if accel_at_rest else None
        self.measured = measured
        self.R = _rot_to_up(self.accel_at_rest) if self.accel_at_rest else None

    def apply(self, v):
        if self.R is None:
            return v
        return tuple(sum(self.R[i][j] * v[j] for j in range(3)) for i in range(3))

    def tilt_deg(self):
        """(roll, pitch) the raw chip reads on a level body — what is being taken out."""
        if self.accel_at_rest is None:
            return 0.0, 0.0
        ax, ay, az = self.accel_at_rest
        n = math.sqrt(ax * ax + ay * ay + az * az)
        gx, gy, gz = -ax / n, -ay / n, -az / n           # gravity = -accel
        return math.degrees(math.atan2(-gy, -gz)), math.degrees(math.asin(max(-1.0, min(1.0, gx))))

    @classmethod
    def load(cls, path=MOUNT_FILE):
        """The saved mount, or an identity one (and `.measured` None) if never levelled."""
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            d = json.load(f)
        return cls(d["accel_at_rest"], d.get("measured"))

    def save(self, path=MOUNT_FILE):
        with open(path, "w") as f:
            json.dump({"accel_at_rest": list(self.accel_at_rest), "measured": self.measured,
                       "note": "imu/bmi088.py --level: the accelerometer, model axes, on a "
                               "level standing robot with levelling off"}, f, indent=2)
            f.write("\n")


# ----------------------------------------------------------------- the chip
ACC_ADDRS, GYR_ADDRS = (0x18, 0x19), (0x68, 0x69)


def probe(bus_no):
    """Which of the four addresses answer with a BMI088 chip ID: {addr: id}."""
    from smbus2 import SMBus
    found = {}
    with SMBus(bus_no) as b:
        for addr, reg, want in [(a, ACC_CHIP_ID, ACC_CHIP_ID_VALUE) for a in ACC_ADDRS] + \
                               [(a, GYR_CHIP_ID, GYR_CHIP_ID_VALUE) for a in GYR_ADDRS]:
            try:
                v = b.read_byte_data(addr, reg)
            except OSError:
                continue
            found[addr] = (v, v == want)
    return found


class BMI088:
    """One smbus2 handle, two addresses, the register dance. Units out: m/s^2, rad/s."""

    def __init__(self, bus=1, acc_addr=None, gyro_addr=None, axes=AXES, mount=None):
        from smbus2 import SMBus
        self.bus = SMBus(bus)
        # the mount tilt out of every read; `mount=Mount()` is the raw chip, for --level
        self.mount = Mount.load() if mount is None else mount
        if acc_addr is None or gyro_addr is None:
            hits = probe(bus)
            accs = [a for a, (v, ok) in hits.items() if ok and a in ACC_ADDRS]
            gyrs = [a for a, (v, ok) in hits.items() if ok and a in GYR_ADDRS]
            if not accs or not gyrs:
                raise RuntimeError(f"BMI088 not found on i2c-{bus}: answered {hits or 'nothing'} "
                                   f"— bus number, overlay, SDO pins or wiring")
            acc_addr, gyro_addr = acc_addr or accs[0], gyro_addr or gyrs[0]
        self.acc_addr, self.gyr_addr = acc_addr, gyro_addr
        self.axes = axes
        self._acc_scale = None
        self._gyr_scale = None

    def _acc_read(self, reg, n=1):
        return bytes(self.bus.read_i2c_block_data(self.acc_addr, reg, n))

    def _acc_write(self, reg, val):
        self.bus.write_byte_data(self.acc_addr, reg, val & 0xFF)

    def _gyr_read(self, reg, n=1):
        return bytes(self.bus.read_i2c_block_data(self.gyr_addr, reg, n))

    def _gyr_write(self, reg, val):
        self.bus.write_byte_data(self.gyr_addr, reg, val & 0xFF)

    def _softreset(self, write, reg):
        # Measured on the board: the gyro die resets on the byte and never ACKs it, so
        # the kernel reports ENXIO on a write that worked (the id reads back after).
        # Swallow that one NACK; every other write still raises.
        try:
            write(reg, 0xB6)
        except OSError as e:
            if e.errno != 6:
                raise
        time.sleep(0.05)

    def ids(self):
        return self._acc_read(ACC_CHIP_ID)[0], self._gyr_read(GYR_CHIP_ID)[0]

    def configure(self):
        acc_id, gyr_id = self.ids()
        if acc_id != ACC_CHIP_ID_VALUE or gyr_id != GYR_CHIP_ID_VALUE:
            raise RuntimeError(f"BMI088 ids wrong: accel 0x{acc_id:02X} (want 0x1E) at "
                               f"0x{self.acc_addr:02X}, gyro 0x{gyr_id:02X} (want 0x0F) at "
                               f"0x{self.gyr_addr:02X}")
        self._softreset(self._acc_write, ACC_SOFTRESET)
        self._acc_write(ACC_PWR_CONF, 0x00); time.sleep(0.005)   # active
        self._acc_write(ACC_PWR_CTRL, 0x04); time.sleep(0.05)    # accel on
        self._acc_write(ACC_CONF, ACC_CONF_400HZ_NORMAL)
        self._acc_write(ACC_RANGE, ACC_RANGE_6G)
        self._acc_scale = 1.5 * (2 ** (ACC_RANGE_6G + 1)) * G / 32768.0

        self._softreset(self._gyr_write, GYR_SOFTRESET)
        self._gyr_write(GYR_RANGE, GYR_RANGE_500DPS)
        self._gyr_write(GYR_BANDWIDTH, GYR_BW_400HZ_47HZ)
        self._gyr_write(GYR_LPM1, 0x00)                          # normal
        self._gyr_scale = math.radians(1.0) / GYR_LSB_PER_DPS[GYR_RANGE_500DPS]
        time.sleep(0.05)
        return acc_id, gyr_id

    def read(self):
        """(accel m/s^2, gyro rad/s), both in the MODEL frame."""
        a = struct.unpack("<hhh", self._acc_read(ACC_DATA, 6))
        w = struct.unpack("<hhh", self._gyr_read(GYR_DATA, 6))
        accel = self.mount.apply(remap(tuple(v * self._acc_scale for v in a), self.axes))
        gyro = self.mount.apply(remap(tuple(v * self._gyr_scale for v in w), self.axes))
        return accel, gyro

    def close(self):
        self.bus.close()


class FakeBMI088:
    """A level, still robot with a little noise. For selftests off the hardware."""

    def __init__(self, tilt_deg=0.0):
        import random
        self.r = random.Random(0)
        self.pitch = math.radians(tilt_deg)

    def configure(self):
        return ACC_CHIP_ID_VALUE, GYR_CHIP_ID_VALUE

    def read(self):
        n = lambda s: self.r.gauss(0.0, s)
        # nose-down pitch: body x tilts toward the ground, so gravity gains a
        # POSITIVE x component and the accelerometer (which reads -gravity) a
        # negative one.
        accel = (-G * math.sin(self.pitch) + n(0.05), n(0.05), G * math.cos(self.pitch) + n(0.05))
        gyro = (n(0.002), n(0.002), n(0.002))
        return accel, gyro

    def close(self):
        pass


# --------------------------------------------------------------- attitude
class Attitude:
    """Projected gravity from gyro + accelerometer: a complementary filter."""

    def __init__(self, alpha=0.02, bias=(0.0, 0.0, 0.0)):
        self.alpha = alpha
        self.bias = tuple(bias)
        self.g = None                      # unit vector, body frame

    @staticmethod
    def _unit(v):
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return tuple(x / n for x in v)

    def update(self, accel, gyro, dt):
        """Returns (gravity_b, gyro_unbiased, accel). gravity_b is a unit vector."""
        w = tuple(gi - bi for gi, bi in zip(gyro, self.bias))
        g_meas = self._unit(tuple(-a for a in accel))      # at rest accel = -g
        if self.g is None:
            self.g = g_meas
        else:
            gx, gy, gz = self.g
            wx, wy, wz = w
            # dg/dt = -w x g, body rates on a world-fixed vector
            dg = (-(wy * gz - wz * gy), -(wz * gx - wx * gz), -(wx * gy - wy * gx))
            g_pred = self._unit((gx + dg[0] * dt, gy + dg[1] * dt, gz + dg[2] * dt))
            a = self.alpha
            self.g = self._unit(tuple((1 - a) * p + a * m for p, m in zip(g_pred, g_meas)))
        return self.g, w, accel


def measure_level(imu, seconds=5.0, hz=50.0):
    """Mean accelerometer over `seconds` of a LEVEL, still, standing robot: the mount."""
    n = int(seconds * hz)
    s = [0.0, 0.0, 0.0]
    for _ in range(n):
        a, _ = imu.read()
        for i in range(3):
            s[i] += a[i]
        time.sleep(1.0 / hz)
    return tuple(v / n for v in s)


def measure_bias(imu, seconds=5.0, hz=50.0):
    """Mean gyro over `seconds` of the robot standing still: the bias to subtract."""
    n = int(seconds * hz)
    s = [0.0, 0.0, 0.0]
    for _ in range(n):
        _, w = imu.read()
        for i in range(3):
            s[i] += w[i]
        time.sleep(1.0 / hz)
    return tuple(v / n for v in s)


# --------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--i2c-bus", type=int, default=1, help="the N of /dev/i2c-N")
    ap.add_argument("--acc-addr", type=lambda x: int(x, 0), default=None, help="0x18 or 0x19")
    ap.add_argument("--gyro-addr", type=lambda x: int(x, 0), default=None, help="0x68 or 0x69")
    ap.add_argument("--ids", action="store_true", help="probe the four addresses and stop")
    ap.add_argument("--bias", type=float, default=0.0, help="seconds still, then print gyro bias")
    ap.add_argument("--level", type=float, default=0.0,
                    help="seconds standing LEVEL with levelling off; writes imu/mount.json")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--selftest", action="store_true", help="fake chip, no hardware")
    a = ap.parse_args()

    if a.selftest:
        imu = FakeBMI088(tilt_deg=10.0)
    else:
        try:
            import smbus2  # noqa: F401
        except ImportError:
            sys.exit("smbus2 is not installed: pip install smbus2 (and the I2C overlay must be on)")
        if a.ids:
            hits = probe(a.i2c_bus)
            for addr, (v, ok) in sorted(hits.items()):
                kind = "accel" if addr in ACC_ADDRS else "gyro"
                print(f"  0x{addr:02X} {kind}: id 0x{v:02X} {'OK' if ok else '?? not a BMI088'}")
            if not hits:
                print(f"  nothing answered on /dev/i2c-{a.i2c_bus}; try `i2cdetect -l` for the bus")
            return 0 if any(ok for _, ok in hits.values()) else 1
        # --level wants the RAW chip; every other run gets the saved mount like the robot
        imu = BMI088(a.i2c_bus, a.acc_addr, a.gyro_addr, mount=Mount() if a.level > 0 else None)
    acc_id, gyr_id = imu.configure()
    print(f"accel id 0x{acc_id:02X}  gyro id 0x{gyr_id:02X}" + ("  (fake)" if a.selftest else
          f"  (i2c-{a.i2c_bus}: 0x{imu.acc_addr:02X} / 0x{imu.gyr_addr:02X})"))
    if not a.selftest:
        m = imu.mount
        if a.level > 0:
            print(f"levelling: the robot must be STANDING on a level floor with levelling off; "
                  f"{a.level:g} s ...")
            rest = measure_level(imu, a.level, a.hz)
            m = Mount(rest, time.strftime("%Y-%m-%d %H:%M"))
            r, p_ = m.tilt_deg()
            m.save()
            imu.mount = m
            print(f"mount: raw chip reads roll {r:+.1f} deg, pitch {p_:+.1f} deg on a level "
                  f"body (accel {rest[0]:+.2f} {rest[1]:+.2f} {rest[2]:+.2f}); wrote {MOUNT_FILE}")
        elif m.measured:
            r, p_ = m.tilt_deg()
            print(f"mount from {MOUNT_FILE} ({m.measured}): taking out roll {r:+.1f}, "
                  f"pitch {p_:+.1f} deg")
        else:
            print(f"no {MOUNT_FILE}: the chip's tilt in the case is NOT taken out; "
                  f"run --level on a standing robot")

    bias = (0.0, 0.0, 0.0)
    if a.bias > 0:
        print(f"hold still {a.bias:g} s ...")
        bias = measure_bias(imu, a.bias, a.hz)
        print("gyro bias rad/s: " + " ".join(f"{b:+.4f}" for b in bias))

    att = Attitude(bias=bias)
    dt = 1.0 / a.hz
    t0 = time.perf_counter()
    k = 0
    try:
        while time.perf_counter() - t0 < a.seconds:
            accel, gyro = imu.read()
            g, w, acc = att.update(accel, gyro, dt)
            if k % int(a.hz / 5) == 0:
                print(f"gravity {g[0]:+.3f} {g[1]:+.3f} {g[2]:+.3f}   "
                      f"gyro {w[0]:+.3f} {w[1]:+.3f} {w[2]:+.3f} rad/s   "
                      f"accel {acc[0]:+.2f} {acc[1]:+.2f} {acc[2]:+.2f} m/s^2")
            k += 1
            time.sleep(max(0.0, t0 + k * dt - time.perf_counter()))
    except KeyboardInterrupt:
        pass
    finally:
        imu.close()
    if a.selftest:
        g = att.g
        ok = g[0] > 0.1 and abs(g[1]) < 0.05 and g[2] < -0.9
        print("selftest: 10 deg nose-down fake reads gravity x positive:", "OK" if ok else "FAIL")
        return 0 if ok else 1
    print("level: expect gravity ~(0, 0, -1), accel ~(0, 0, +9.8). Nose down: gravity x > 0, "
          "accel x < 0. Left side down: gravity y > 0. If not, fix AXES.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
