#!/usr/bin/env python3
"""SmallDog gait node: /cmd_vel -> streamed joint trajectory.

Publishes one-point JointTrajectory messages at `rate` Hz to the
JointTrajectoryController, which is the same pattern the hexapod uses but
streamed continuously instead of action-per-step, because a trot has no
natural step boundary to wait on.
"""
import json, os
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from ament_index_python.packages import get_package_share_directory

from smalldog_walker.gait import TrotGait


class SmallDogWalker(Node):
    def __init__(self):
        super().__init__('smalldog_walker')

        self.declare_parameter('rate', 100.0)
        self.declare_parameter('controller', 'smalldog_controller')
        self.declare_parameter('period', 0.45)
        self.declare_parameter('swing_height', 0.022)
        self.declare_parameter('body_height', 0.158)
        self.declare_parameter('max_step', 0.060)
        self.declare_parameter('cmd_timeout', 0.5)

        share = get_package_share_directory('smalldog_description')
        with open(os.path.join(share, 'robot_params.json')) as f:
            params = json.load(f)

        self.gait = TrotGait(params)
        # order matters: the body-height setter clamps against swing and step
        self.gait.period = self.get_parameter('period').value
        self.gait.swing_height = self.get_parameter('swing_height').value
        self.gait.max_step = self.get_parameter('max_step').value
        self.gait.body_height = self.get_parameter('body_height').value

        ctrl = self.get_parameter('controller').value
        self.pub = self.create_publisher(JointTrajectory, f'/{ctrl}/joint_trajectory', 10)

        self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)
        self.create_subscription(Bool, '/smalldog/enable', self.on_enable, 10)
        self.create_subscription(Float64, '/smalldog/body_height', self.on_height, 10)

        self.cmd = (0.0, 0.0, 0.0)
        self.enabled = True
        # gait phase integrates in SIM time (physics), but "is the operator still
        # sending?" is a wall-clock question: the sim can run many times real time,
        # and then a steady 20 Hz teleop looks stale on the sim clock.
        self._wall = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.last_cmd = self._wall.now()
        self.timeout = self.get_parameter('cmd_timeout').value

        self.rate = float(self.get_parameter('rate').value)
        self.dt = 1.0 / self.rate
        self.timer = self.create_timer(self.dt, self.tick)

        r = self.gait.reach_info()
        self.get_logger().info(
            f'walker up: {len(self.gait.joint_names)} joints -> /{ctrl}/joint_trajectory '
            f'@ {self.rate:.0f} Hz')
        self.get_logger().info(
            f'leg reach {r["d_min"]*1000:.0f}..{r["d_max"]*1000:.0f} mm -> body height '
            f'{r["height_min"]*1000:.0f}..{r["height_max"]*1000:.0f} mm, using '
            f'{r["body_height"]*1000:.0f} mm, swing {r["swing_height"]*1000:.0f} mm')

    def on_cmd_vel(self, msg):
        self.cmd = (msg.linear.x, msg.linear.y, msg.angular.z)
        self.last_cmd = self._wall.now()

    def on_enable(self, msg):
        self.enabled = msg.data
        self.get_logger().info(f'gait {"enabled" if msg.data else "disabled"}')

    def on_height(self, msg):
        self.gait.body_height = msg.data      # the gait clamps to its reachable band

    def tick(self):
        stale = (self._wall.now() - self.last_cmd).nanoseconds * 1e-9 > self.timeout
        cmd = (0.0, 0.0, 0.0) if (stale or not self.enabled) else self.cmd

        q = self.gait.joint_targets(self.dt, *cmd)

        msg = JointTrajectory()
        msg.joint_names = self.gait.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        step = 2.0 * self.dt
        pt.time_from_start = Duration(sec=int(step), nanosec=int((step % 1.0) * 1e9))
        msg.points = [pt]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SmallDogWalker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
