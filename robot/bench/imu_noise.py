"""
imu_noise.py — what the policy's IMU sees from a still robot, as numbers.

    python3 bench/imu_noise.py                 # 10 s at 50 Hz: sd of gyro and accel per axis
    python3 bench/imu_noise.py --seconds 20

Run it twice with the robot standing still on the table: once with the L2 parked and
once spinning (`stream_pcd` running, or `/lidar/spin true` under ROS). The policy reads
the BMI088 at 50 Hz, so this samples at 50 Hz on purpose: a vibration above 25 Hz aliases
into the band the policy is looking at, and that is what matters, not the true spectrum.
Reads the chip directly — no servos, no ROS, no legs needed.
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from imu.bmi088 import BMI088  # noqa: E402


def sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--i2c-bus", type=int, default=1)
    a = ap.parse_args()
    imu = BMI088(a.i2c_bus)
    acc = [[], [], []]
    gyr = [[], [], []]
    dt = 1.0 / a.hz
    t0 = time.perf_counter()
    k = 0
    try:
        while time.perf_counter() - t0 < a.seconds:
            av, wv = imu.read()
            for i in range(3):
                acc[i].append(av[i])
                gyr[i].append(wv[i])
            k += 1
            time.sleep(max(0.0, t0 + k * dt - time.perf_counter()))
    finally:
        imu.close()
    print(f"{k} samples at {a.hz:.0f} Hz over {a.seconds:g} s")
    print("gyro  sd rad/s  x %.4f  y %.4f  z %.4f   peak |w| %.3f" % (
        sd(gyr[0]), sd(gyr[1]), sd(gyr[2]), max(abs(v) for ax in gyr for v in ax)))
    print("accel sd m/s^2  x %.3f  y %.3f  z %.3f   mean z %.2f" % (
        sd(acc[0]), sd(acc[1]), sd(acc[2]), sum(acc[2]) / len(acc[2])))


if __name__ == "__main__":
    main()
