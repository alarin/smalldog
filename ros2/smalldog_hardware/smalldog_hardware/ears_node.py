#!/usr/bin/env python3
"""The robot's ears as a ROS 2 node: "псина, стоп" / "псина, гуляй" (robot/sound/ears.py).

    ros2 run smalldog_hardware ears
    ros2 launch smalldog_hardware robot.launch.py voice:=true ...

    /smalldog/voice      std_msgs/String, every utterance the recognizer finished, as heard
    /smalldog/explore    std_msgs/Bool: false on "стоп" (the explorer cancels its Nav2 goal
                         and the walker stops), true on "гуляй" (it resumes)
    /cmd_vel             one zero Twist on "стоп", so the robot also stops when nobody is
                         exploring — under teleop, or a goal sent by hand

The recognizer runs on its own thread (it blocks on the mic); the node publishes from a
queue on a 20 Hz timer. The reply ("стою" / "гуляю") is `say` on the speaker, from the
recognizer's thread, blocking — so the mic's next 0.5 s is the robot's own voice and
that is fine: it never says a command word.
"""
import os
import queue
import sys
import threading

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, String

# `robot/` by path from this file, as servo_node.py finds the runtime
_HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.environ.get('SMALLDOG_REPO') or os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(REPO, 'robot'))
from sound.ears import Ears, REPLY, say   # noqa: E402


class EarsNode(Node):
    def __init__(self):
        super().__init__('smalldog_ears')
        self.declare_parameter('mic', os.environ.get('MIC', 'plughw:3,0'))
        self.declare_parameter('reply', True)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.voice_pub = self.create_publisher(String, '/smalldog/voice', 10)
        self.explore_pub = self.create_publisher(Bool, '/smalldog/explore', latched)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.events = queue.Queue()
        self.ears = Ears(mic=self.get_parameter('mic').value)
        threading.Thread(target=self.run, daemon=True).start()
        self.create_timer(0.05, self.drain)
        self.get_logger().info(f'listening on {self.ears.mic}: «псина, стоп» / «псина, гуляй»')

    def run(self):
        try:
            for kind, text in self.ears:
                if kind == 'partial':
                    continue
                self.events.put((kind, text))
                if kind == 'cmd' and self.get_parameter('reply').value:
                    say(REPLY[text])
        except Exception as e:   # the mic gone, the model missing: say why, then the node is deaf
            self.events.put(('error', repr(e)))

    def drain(self):
        while not self.events.empty():
            kind, text = self.events.get()
            if kind == 'heard':
                self.voice_pub.publish(String(data=text))
                self.get_logger().info(f'heard: {text!r}')
            elif kind == 'cmd':
                self.get_logger().info(f'>>> {text}')
                self.explore_pub.publish(Bool(data=(text == 'go')))
                if text == 'stop':
                    self.cmd_pub.publish(Twist())
            else:
                self.get_logger().error(f'ears stopped: {text}')


def main():
    rclpy.init()
    node = EarsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.ears.stop = True
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
