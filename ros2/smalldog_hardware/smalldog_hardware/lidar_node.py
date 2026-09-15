#!/usr/bin/env python3
"""The real L2 as a ROS 2 topic: stream_pcd's ULF3 feed -> sensor_msgs/PointCloud2.

    ros2 run smalldog_hardware lidar                       # stream_pcd on 127.0.0.1:9910
    ros2 run smalldog_hardware lidar --ros-args -p source:=10.0.1.47:9910

    /lidar/points        PointCloud2 (x y z intensity, float32), frame lidar_link, 12 Hz,
                         SensorDataQoS — the same shape the sim's MuJoCo node publishes,
                         so smalldog_nav is launched the same way over either

The L2 SDK is C++ and single-client; `3d/tools/stream_pcd.cpp` is the one process that
talks to it, and `robot/slam/slam.py` already reads its feed. This node is that reader
(`slam.Source`, imported, not copied) with a publisher on the end. Points are left in the
sensor's own frame: +Z is the optical axis, which is what the CAD calls `lidar_link`
(3d/lidar.py) — so the cloud lands where the URDF says the sensor is and TF does the rest.

`yaw_offset` (verify): the SDK's X about the axis against the CAD's. lidar_link's +X is
the CAD's, the sensor's is wherever Unitree put it; the two differ by a rotation about +Z
that nobody has measured. Until someone stands the robot facing a wall and reads the
scan, the offset is 0 and a room may come out turned — SLAM does not care, the robot's
heading in the map does.

Stamps: the node's clock at receipt, not the sensor's own `stamp`. The L2's clock is not
the Pi's, and slam_toolbox wants scans on the same clock as TF. At 12 Hz over localhost
the difference is under a frame.
"""
import os
import sys
import threading

import numpy as np
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

_HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.environ.get('SMALLDOG_REPO') or os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(REPO, 'robot'))
from slam.slam import Source, Replay                                  # noqa: E402

FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
          for i, n in enumerate(('x', 'y', 'z', 'intensity'))]


class LidarNode(Node):
    def __init__(self):
        super().__init__('smalldog_lidar')
        self.declare_parameter('source', '127.0.0.1:9910')   # stream_pcd, or a --record .npz
        self.declare_parameter('frame_id', 'lidar_link')
        self.declare_parameter('topic', '/lidar/points')
        self.declare_parameter('yaw_offset', 0.0)           # rad about +Z, SDK -> CAD (verify)
        self.declare_parameter('min_range', 0.0)            # m, 0 = everything the SDK sends
        src = self.get_parameter('source').value
        self.frame = self.get_parameter('frame_id').value
        self.min_r2 = float(self.get_parameter('min_range').value) ** 2
        yaw = float(self.get_parameter('yaw_offset').value)
        c, s = np.cos(yaw), np.sin(yaw)
        self.rot = np.array([[c, -s], [s, c]], dtype=np.float32) if yaw else None

        self.pub = self.create_publisher(PointCloud2, self.get_parameter('topic').value,
                                         qos_profile_sensor_data)
        if src.endswith('.npz'):
            self.source = Replay(src)
        else:
            host, _, port = src.partition(':')
            self.source = Source(host, port or 9910)
        self.n = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        self.create_timer(5.0, self.report)
        self.get_logger().info(f'L2 from {self.source.name} -> {self.pub.topic_name} '
                               f'in {self.frame}')

    def run(self):
        for stamp, acc, xyzi in self.source.frames():
            if not rclpy.ok():
                break
            pts = np.ascontiguousarray(xyzi, dtype=np.float32)
            if self.min_r2 > 0:
                pts = pts[(pts[:, :3] ** 2).sum(axis=1) >= self.min_r2]
            if self.rot is not None:
                pts = pts.copy()
                pts[:, :2] = pts[:, :2] @ self.rot.T
            msg = PointCloud2()
            msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame)
            msg.height, msg.width = 1, len(pts)
            msg.fields = FIELDS
            msg.is_bigendian = False
            msg.point_step, msg.row_step = 16, 16 * len(pts)
            msg.is_dense = True
            msg.data = pts.tobytes()
            self.pub.publish(msg)
            self.n += 1
        err = getattr(self.source, 'err', None)
        self.get_logger().error(f'L2 feed ended after {self.n} frames: {err or "sender closed"}')

    def report(self):
        self.get_logger().info(f'{self.n} frames, lag {self.source.lag}',
                               throttle_duration_sec=30.0)


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.source.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
