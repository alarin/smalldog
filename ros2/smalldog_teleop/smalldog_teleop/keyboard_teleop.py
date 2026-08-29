#!/usr/bin/env python3
"""Keyboard teleop for SmallDog.

Turns key presses into:
    /cmd_vel               geometry_msgs/Twist
    /smalldog/body_height  std_msgs/Float64
    /smalldog/enable       std_msgs/Bool

Keys arrive from either source, and both may be live at once:

* the **MuJoCo render window** — `mujoco_ros2_control` republishes every printable key
  pressed over its viewer on `~/key` (`key_topic` here). This is the path the launch file
  uses: no second terminal, just click the sim window and type.
* **this terminal**, read raw, when stdin is a TTY and `read_stdin` is true. That needs
  the focused TTY, so it only works when the node is started by hand
  (`ros2 run smalldog_teleop keyboard`), not from a launch file.
"""
import sys, os, select, termios, tty, threading
import rclpy
import rclpy.executors
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64, String

BANNER = """
SmallDog keyboard teleop
------------------------
  w / s      walk forward / back
  a / d      strafe left / right
  q / e      turn left / right
  space      stop
  r / f      body up / down
  , / .      speed  -  / +
  t          gait enable / disable
  Ctrl-C     quit

held keys are not needed: a press sets the command, space clears it
"""

MOVE = {
    'w': ('x', +1), 's': ('x', -1),
    'a': ('y', +1), 'd': ('y', -1),
    'q': ('z', +1), 'e': ('z', -1),
}


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('smalldog_keyboard_teleop')
        self.declare_parameter('speed', 0.20)
        self.declare_parameter('turn', 1.2)
        self.declare_parameter('body_height', 0.158)
        self.declare_parameter('repeat_rate', 50.0)
        # `~/key` of the mujoco_ros2_control node, which publishes what the viewer window
        # sees. Absolute, because that node is not a child of this one.
        self.declare_parameter('key_topic', '/mujoco_ros2_control_node/key')
        self.declare_parameter('read_stdin', True)

        self.speed = self.get_parameter('speed').value
        self.turn = self.get_parameter('turn').value
        self.height = self.get_parameter('body_height').value
        self.enabled = True
        self.cmd = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_h = self.create_publisher(Float64, '/smalldog/body_height', 10)
        self.pub_e = self.create_publisher(Bool, '/smalldog/enable', 10)
        self.create_subscription(String, self.get_parameter('key_topic').value,
                                 self.on_key_msg, 10)

        rate = float(self.get_parameter('repeat_rate').value)
        self.create_timer(1.0 / rate, self.publish_cmd)

    def on_key_msg(self, msg):
        if msg.data:
            self.on_key(msg.data[0].lower())

    def publish_cmd(self):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = self.cmd['x'], self.cmd['y'], self.cmd['z']
        self.pub.publish(t)

    def status(self):
        return (f"\rvx {self.cmd['x']:+.2f}  vy {self.cmd['y']:+.2f}  wz {self.cmd['z']:+.2f}"
                f"   speed {self.speed:.2f}  height {self.height*1000:3.0f} mm"
                f"   gait {'on ' if self.enabled else 'off'}   ")

    def on_key(self, key):
        if key in MOVE:
            axis, sign = MOVE[key]
            scale = self.turn if axis == 'z' else self.speed
            self.cmd = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            self.cmd[axis] = sign * scale
        elif key == ' ':
            self.cmd = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        elif key in 'rf':
            self.height = max(0.11, min(0.19, self.height + (0.004 if key == 'r' else -0.004)))
            self.pub_h.publish(Float64(data=self.height))
        elif key in ',.':
            self.speed = max(0.05, min(0.45, self.speed + (0.05 if key == '.' else -0.05)))
        elif key == 't':
            self.enabled = not self.enabled
            self.pub_e.publish(Bool(data=self.enabled))
        self.show_status()

    def show_status(self):
        # without a TTY the \r trick is useless — launch buffers by line and prefixes each
        # one — so say it through the logger instead.
        if sys.stdin.isatty():
            sys.stdout.write(self.status())
            sys.stdout.flush()
        else:
            self.get_logger().info(self.status().strip())


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
    # spin on an executor we own, so shutdown is ordered: stop spinning, join, destroy.
    # rclpy.spin() in a daemon thread races the main thread's destroy_node() and segfaults.
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    raw = node.get_parameter('read_stdin').value and sys.stdin.isatty()
    print(BANNER + ('' if raw else '\nkeys come from the MuJoCo window — click it and type\n'))
    settings = termios.tcgetattr(sys.stdin) if raw else None
    if raw:
        node.show_status()
    try:
        if raw:
            tty.setraw(sys.stdin.fileno())
        while rclpy.ok():
            if not raw:
                spin.join(timeout=0.5)
                if not spin.is_alive():
                    break
                continue
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if key == '\x03':
                    break
                node.on_key(key.lower())
    except KeyboardInterrupt:
        pass
    finally:
        if raw:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.cmd = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        node.publish_cmd()
        print()
        executor.shutdown()
        spin.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
