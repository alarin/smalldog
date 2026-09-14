#!/usr/bin/env python
"""
bmi088.py — the BMI088 over SPI, and the gravity vector the policy is trained on.

    python imu/bmi088.py --selftest                      # no hardware: a fake chip
    python imu/bmi088.py --bus 0 --cs-accel 0 --cs-gyro 1 --ids      # chip IDs only
    python imu/bmi088.py --bus 0 --cs-accel 0 --cs-gyro 1            # stream, 50 Hz
    python imu/bmi088.py --bus 0 --cs-accel 0 --cs-gyro 1 --bias 5   # 5 s still, gyro bias

The BMI088 is two dies with two chip selects (POWER.md: "SPI, 2 x CS"), and the
two speak slightly different SPI: the accelerometer answers a read with one dummy
byte before the data and needs one dummy read after power-up to leave I2C mode;
the gyroscope does neither. Both return little-endian int16 x, y, z.

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
(nose), y left, z up. It is a guess until `--selftest` has been run on the
robot: level, the stream must read gravity ~(0, 0, -1) and accel ~(0, 0, +9.8);
nose DOWN must send gravity's x POSITIVE (the body's x axis now points partly
at the ground, so gravity has a component along it) and the accelerometer's x
negative; rolled onto its LEFT side (y axis toward the ground), gravity's y
positive. Change AXES, not the frame.
"""
from __future__ import annotations

import argparse
import math
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
AXES = ((0, +1), (1, +1), (2, +1))          # UNVERIFIED — see the docstring


def remap(v, axes=AXES):
    return tuple(s * v[i] for i, s in axes)


# ----------------------------------------------------------------- the chip
class BMI088:
    """Two spidev handles and the register dance. Units out: m/s^2 and rad/s."""

    def __init__(self, bus=0, cs_accel=0, cs_gyro=1, hz=10_000_000, axes=AXES):
        import spidev
        self.acc = spidev.SpiDev(); self.acc.open(bus, cs_accel)
        self.gyr = spidev.SpiDev(); self.gyr.open(bus, cs_gyro)
        for dev in (self.acc, self.gyr):
            dev.max_speed_hz = hz
            dev.mode = 0b00
        self.axes = axes
        self._acc_scale = None
        self._gyr_scale = None

    # accel: address | 0x80, one dummy byte, then data
    def _acc_read(self, reg, n=1):
        r = self.acc.xfer2([reg | 0x80, 0x00] + [0x00] * n)
        return bytes(r[2:])

    def _acc_write(self, reg, val):
        self.acc.xfer2([reg & 0x7F, val & 0xFF])

    # gyro: address | 0x80, then data
    def _gyr_read(self, reg, n=1):
        r = self.gyr.xfer2([reg | 0x80] + [0x00] * n)
        return bytes(r[1:])

    def _gyr_write(self, reg, val):
        self.gyr.xfer2([reg & 0x7F, val & 0xFF])

    def ids(self):
        self._acc_read(ACC_CHIP_ID)                 # the dummy read that selects SPI
        time.sleep(0.001)
        return self._acc_read(ACC_CHIP_ID)[0], self._gyr_read(GYR_CHIP_ID)[0]

    def configure(self):
        acc_id, gyr_id = self.ids()
        if acc_id != ACC_CHIP_ID_VALUE or gyr_id != GYR_CHIP_ID_VALUE:
            raise RuntimeError(f"BMI088 not found: accel id 0x{acc_id:02X} (want 0x1E), "
                               f"gyro id 0x{gyr_id:02X} (want 0x0F) — bus, CS or wiring")
        self._acc_write(ACC_SOFTRESET, 0xB6); time.sleep(0.05)
        self._acc_read(ACC_CHIP_ID); time.sleep(0.001)      # SPI mode again after reset
        self._acc_write(ACC_PWR_CONF, 0x00); time.sleep(0.005)   # active
        self._acc_write(ACC_PWR_CTRL, 0x04); time.sleep(0.05)    # accel on
        self._acc_write(ACC_CONF, ACC_CONF_400HZ_NORMAL)
        self._acc_write(ACC_RANGE, ACC_RANGE_6G)
        self._acc_scale = 1.5 * (2 ** (ACC_RANGE_6G + 1)) * G / 32768.0

        self._gyr_write(GYR_SOFTRESET, 0xB6); time.sleep(0.05)
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
        accel = remap(tuple(v * self._acc_scale for v in a), self.axes)
        gyro = remap(tuple(v * self._gyr_scale for v in w), self.axes)
        return accel, gyro

    def close(self):
        self.acc.close(); self.gyr.close()


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
    ap.add_argument("--bus", type=int, default=0)
    ap.add_argument("--cs-accel", type=int, default=0)
    ap.add_argument("--cs-gyro", type=int, default=1)
    ap.add_argument("--ids", action="store_true", help="read the two chip IDs and stop")
    ap.add_argument("--bias", type=float, default=0.0, help="seconds still, then print gyro bias")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--selftest", action="store_true", help="fake chip, no hardware")
    a = ap.parse_args()

    if a.selftest:
        imu = FakeBMI088(tilt_deg=10.0)
    else:
        try:
            imu = BMI088(a.bus, a.cs_accel, a.cs_gyro)
        except ImportError:
            sys.exit("spidev is not installed: pip install spidev (and the SPI overlay must be on)")
    acc_id, gyr_id = imu.configure()
    print(f"accel id 0x{acc_id:02X}  gyro id 0x{gyr_id:02X}" + ("  (fake)" if a.selftest else ""))
    if a.ids:
        return 0

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
