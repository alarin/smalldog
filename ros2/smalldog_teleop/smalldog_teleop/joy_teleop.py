#!/usr/bin/env python3
"""Gamepad teleop for SmallDog: sensor_msgs/Joy -> the same three topics the keyboard sends.

    /cmd_vel               geometry_msgs/Twist
    /smalldog/body_height  std_msgs/Float64
    /smalldog/enable       std_msgs/Bool

Sits behind the `joy` package's `joy_node`, which owns the device; this node only maps.
The layout is the Xbox 360 one the kernel's xpad driver presents for most 2.4 GHz pads
(the robot's "Barrot gamepad" included):

    left stick      forward / back, strafe          axes 1, 0   (up and left are +1)
    right stick X   turn                            axis 3
    D-pad up/down   body up / down                  axis 7, one step per press
    B               stop (zero the command)         button 1
    Start           gait enable / disable           button 7
    LB (hold)       turbo: full speed; without it the sticks reach `slow` of it   button 4

A stick returns to centre, so letting go stops the robot, and so does the pad going
away: `/cmd_vel` is republished at `repeat_rate` because the walker drops a command
older than 0.5 s, but it is republished as ZERO once `/joy` itself is older than
`joy_timeout` — joy_node stops publishing when the receiver drops out, and repeating
the last stick position into that is a robot walking away from a dead controller. The speeds are caps,
the same numbers robot.launch.py fits to the servo (0.08 m/s, 0.65 rad/s), and they are
caps on each axis: forward plus a turn at once is over the joint speed budget and the feet
drag (`robot/README.md`, "The gait is fitted to the servo") — one stick at a time is the
walking that works.
"""
import rclpy
import rclpy.executors
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Float64

# the same band the keyboard node and standalone_sim.py walk over
H_MIN, H_MAX, H_STEP = 0.09, 0.20, 0.005


def _dead(v, band):
    """deadband, then rescale so the edge of the band is 0 and full deflection is 1."""
    if abs(v) < band:
        return 0.0
    return (abs(v) - band) / (1.0 - band) * (1.0 if v > 0 else -1.0)


class JoyTeleop(Node):
    def __init__(self):
        super().__init__('smalldog_joy_teleop')
        p = self.declare_parameter
        p('speed', 0.08)          # m/s at full stick
        p('strafe', 0.05)         # m/s at full stick, sideways
        p('turn', 0.65)           # rad/s at full stick
        p('slow', 0.6)            # of the caps without the turbo button
        p('body_height', 0.158)
        p('deadband', 0.15)
        p('repeat_rate', 20.0)
        p('joy_timeout', 0.5)     # s without /joy -> stop
        p('axis_vx', 1)
        p('axis_vy', 0)
        p('axis_wz', 3)
        p('axis_height', 7)
        p('button_stop', 1)
        p('button_enable', 7)
        p('button_turbo', 4)
        g = lambda k: self.get_parameter(k).value    # noqa: E731

        self.speed, self.strafe, self.turn, self.slow = g('speed'), g('strafe'), g('turn'), g('slow')
        self.height = g('body_height')
        self.band = g('deadband')
        self.ax = {k: int(g(f'axis_{k}')) for k in ('vx', 'vy', 'wz', 'height')}
        self.bt = {k: int(g(f'button_{k}')) for k in ('stop', 'enable', 'turbo')}

        self.cmd = Twist()
        self.enabled = True
        self._prev_buttons = []
        self._prev_height_axis = 0.0
        self._msgs = 0
        self._last_joy = None
        self._joy_timeout = float(g('joy_timeout'))
        self._stale_said = False

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_h = self.create_publisher(Float64, '/smalldog/body_height', 10)
        self.pub_e = self.create_publisher(Bool, '/smalldog/enable', 10)
        self.create_subscription(Joy, '/joy', self.on_joy, 10)
        self.create_timer(1.0 / float(g('repeat_rate')), self.publish_cmd)
        self.get_logger().info(
            f'joy teleop up: {self.speed:.2f} m/s, {self.turn:.2f} rad/s at full stick '
            f'(x{self.slow:.1f} without LB); waiting for /joy')

    @staticmethod
    def _get(seq, i, default=0.0):
        return seq[i] if 0 <= i < len(seq) else default

    def _pressed(self, buttons, i):
        """rising edge of button i"""
        now = self._get(buttons, i, 0)
        was = self._get(self._prev_buttons, i, 0)
        return now and not was

    def on_joy(self, msg):
        self._msgs += 1
        if self._msgs == 1:
            self.get_logger().info(f'/joy is live: {len(msg.axes)} axes, {len(msg.buttons)} buttons')
        self._last_joy = self.get_clock().now()
        a, b = msg.axes, msg.buttons
        scale = 1.0 if self._get(b, self.bt['turbo'], 0) else self.slow

        t = Twist()
        t.linear.x = _dead(self._get(a, self.ax['vx']), self.band) * self.speed * scale
        t.linear.y = _dead(self._get(a, self.ax['vy']), self.band) * self.strafe * scale
        t.angular.z = _dead(self._get(a, self.ax['wz']), self.band) * self.turn * scale
        if self._get(b, self.bt['stop'], 0):
            t = Twist()
        self.cmd = t

        # the D-pad is an axis that reads +-1 while held: one step per press, not per message
        h = self._get(a, self.ax['height'])
        if h and not self._prev_height_axis:
            self.height = max(H_MIN, min(H_MAX, self.height + (H_STEP if h > 0 else -H_STEP)))
            self.pub_h.publish(Float64(data=self.height))
            self.get_logger().info(f'body height {self.height*1000:.0f} mm')
        self._prev_height_axis = h

        if self._pressed(b, self.bt['enable']):
            self.enabled = not self.enabled
            self.pub_e.publish(Bool(data=self.enabled))
            self.get_logger().info(f'gait {"enabled" if self.enabled else "disabled"}')
        self._prev_buttons = list(b)

    def publish_cmd(self):
        stale = (self._last_joy is None or
                 (self.get_clock().now() - self._last_joy).nanoseconds * 1e-9 > self._joy_timeout)
        if stale:
            if self._last_joy is not None and not self._stale_said:
                self._stale_said = True
                self.get_logger().warn(f'no /joy for {self._joy_timeout:g} s — stopping')
            self.cmd = Twist()
        else:
            self._stale_said = False
        self.pub.publish(self.cmd)


def main(args=None):
    rclpy.init(args=args)
    node = JoyTeleop()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        try:
            node.pub.publish(Twist())       # the context may already be down
        except Exception:
            pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
