#!/usr/bin/env python3
"""stance_probe.py — where is the CoM? Step the feet forward under a standing robot and
watch the front and rear knee loads meet.

    ros2 launch smalldog_hardware robot.launch.py mode2:=false imu:=true      # standing
    python3 tools/stance_probe.py                                             # 0..50 mm
    python3 tools/stance_probe.py --steps 0 0.02 0.04 --settle 4

Publishes each offset on /smalldog/stance_x (the walker moves all four feet that far
ahead of the hips), waits `settle` s, then averages `window` s of /joint_states effort
(= the servos' Present Load) and /imu. Printed per step: the knee loads front and rear,
their ratio, and the body pitch. The right offset is where front and rear carry the same
and the pitch is 0; the sign of the load flips between the left and right legs, so the
magnitudes are what is compared. Leaves the walker at the last step.

Why: the gait stands the feet under the hips, which puts the support rectangle under the
CAD's CoM (10 mm ahead of centre). On the robot the front knees held 2-4x the rear load
and the IMU read 3-6 deg nose-down in every recording of 2026-09-19.
"""
import argparse
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64

KNEES = ('fl_knee', 'fr_knee', 'rl_knee', 'rr_knee')


class Probe(Node):
    def __init__(self):
        super().__init__('stance_probe')
        self.pub = self.create_publisher(Float64, '/smalldog/stance_x', 10)
        self.create_subscription(JointState, '/joint_states', self.on_js, 10)
        self.create_subscription(Imu, '/imu', self.on_imu, 10)
        self.loads, self.pitch = [], []
        self.collect = False

    def on_js(self, msg):
        if self.collect and msg.effort:
            i = {n: k for k, n in enumerate(msg.name)}
            self.loads.append([msg.effort[i[j]] for j in KNEES])

    def on_imu(self, msg):
        if self.collect:
            q = msg.orientation
            # pitch from servo_node's ZYX quaternion, + nose-down as it reports it
            sinp = 2.0 * (q.w * q.y - q.z * q.x)
            self.pitch.append(math.degrees(math.asin(max(-1.0, min(1.0, sinp)))))

    def sample(self, seconds):
        self.loads, self.pitch, self.collect = [], [], True
        t0 = time.time()
        while time.time() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=0.05)
        self.collect = False
        return np.array(self.loads), np.array(self.pitch)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--steps', type=float, nargs='+', default=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
                    help='stance_x values, m (+ forward)')
    ap.add_argument('--settle', type=float, default=3.0, help='s to wait after each step')
    ap.add_argument('--window', type=float, default=3.0, help='s averaged per step')
    a = ap.parse_args()
    rclpy.init()
    n = Probe()
    print(f'{"feet mm":>8} | {"fl":>6} {"fr":>6} {"rl":>6} {"rr":>6} | front/rear | pitch deg')
    for x in a.steps:
        n.pub.publish(Float64(data=float(x)))
        t0 = time.time()
        while time.time() - t0 < a.settle:
            rclpy.spin_once(n, timeout_sec=0.05)
        loads, pitch = n.sample(a.window)
        if not len(loads):
            print(f'{x*1000:8.0f} | no /joint_states with effort - is the servo node up?')
            continue
        m = np.abs(loads).mean(0)
        front, rear = (m[0] + m[1]) / 2, (m[2] + m[3]) / 2
        p = pitch.mean() if len(pitch) else float('nan')
        print(f'{x*1000:8.0f} | {m[0]:6.0f} {m[1]:6.0f} {m[2]:6.0f} {m[3]:6.0f} | {front/max(rear,1):10.2f} | {p:+.1f}')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
