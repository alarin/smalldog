#!/usr/bin/env python3
"""The L2's cloud as a 2-D LaserScan, sliced in the level frame — SLAM's and Nav2's input.

    ros2 run smalldog_nav cloud_to_scan --ros-args -r cloud:=/lidar/points

    cloud   sensor_msgs/PointCloud2 in lidar_link (sim: /mujoco_ros2_control_node/lidar/points)
    /scan   sensor_msgs/LaserScan in `target_frame`, one per cloud

Each cloud is moved into `target_frame` (base_footprint: the walker's level frame, floor
at z = 0) with the TF at the cloud's own stamp, cut to the band `min_height..max_height`
and binned by bearing, nearest return per bin; a bin with no return is `inf`, which
slam_toolbox and the costmaps both read as "nothing known", not "clear".

This is pointcloud_to_laserscan's job, and that node is the first thing that was tried.
Under this workspace's Kilted it took every cloud, held it in its TF message filter and
dropped it ~10 s later as "earlier than all the data in the transform cache" while
`can_transform` said yes for the same stamp the whole time; no parameter reaches the
filter. Sixty lines of numpy at 3000 points x 12 Hz is not a cost worth debugging a
message filter for, and the Pi does not have to install the package.

Height band: the L2's axis is 0.25 m up, the walls are what the map is for, and the trot
rocks the body +-3 deg. With the IMU's roll and pitch in the level frame the floor
stays under `min_height` out to `range_max` at the IMU's residual; without the IMU
(base_footprint is then base_link lifted, not levelled) the floor at 3 m rises to
0.16 m at 3 deg, so a scan taken blind wants `min_height` raised or `range_max` cut.
"""
import math

import numpy as np
import rclpy
import rclpy.executors
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException, \
    ConnectivityException


def _rot(q):
    """3x3 from a geometry_msgs Quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _xyz(msg):
    """(n, 3) float64 out of any PointCloud2 with float32 x y z fields."""
    off = {f.name: f.offset for f in msg.fields}
    dt = np.dtype({'names': ['x', 'y', 'z'],
                   'formats': ['<f4'] * 3,
                   'offsets': [off['x'], off['y'], off['z']],
                   'itemsize': msg.point_step})
    a = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
    return np.stack([a['x'], a['y'], a['z']], axis=1).astype(np.float64)


class CloudToScan(Node):
    def __init__(self):
        super().__init__('cloud_to_scan')
        self.declare_parameter('target_frame', 'base_footprint')
        self.declare_parameter('min_height', 0.10)
        self.declare_parameter('max_height', 0.35)
        self.declare_parameter('angle_min', -math.radians(96.0))   # the L2's cone
        self.declare_parameter('angle_max', math.radians(96.0))
        self.declare_parameter('angle_increment', math.radians(0.5))
        self.declare_parameter('range_min', 0.30)
        self.declare_parameter('range_max', 6.0)
        self.declare_parameter('scan_time', 1.0 / 12.0)
        g = lambda n: self.get_parameter(n).value
        self.frame = g('target_frame')
        self.zlo, self.zhi = float(g('min_height')), float(g('max_height'))
        self.a0, self.a1 = float(g('angle_min')), float(g('angle_max'))
        self.da = float(g('angle_increment'))
        self.nbins = int(round((self.a1 - self.a0) / self.da)) + 1
        self.rmin, self.rmax = float(g('range_min')), float(g('range_max'))
        self.scan_time = float(g('scan_time'))

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.pub = self.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
        self.create_subscription(PointCloud2, 'cloud', self.on_cloud, qos_profile_sensor_data)
        self.n = self.dropped = 0
        self.create_timer(10.0, lambda: self.get_logger().info(
            f'{self.n} scans, {self.dropped} clouds without TF', throttle_duration_sec=60.0))

    def on_cloud(self, msg):
        stamp = Time.from_msg(msg.header.stamp)
        try:
            t = self.tf.lookup_transform(self.frame, msg.header.frame_id, stamp)
        except ExtrapolationException:
            # the cloud is a few ms ahead of the last TF: take the latest pose instead of
            # waiting for one - at 100 Hz TF and 0.2 m/s that is under 2 mm
            try:
                t = self.tf.lookup_transform(self.frame, msg.header.frame_id, Time())
            except (LookupException, ExtrapolationException, ConnectivityException):
                self.dropped += 1
                return
        except (LookupException, ConnectivityException):
            self.dropped += 1
            return

        p = _xyz(msg)
        if len(p) == 0:
            return
        tr = t.transform.translation
        p = p @ _rot(t.transform.rotation).T + np.array([tr.x, tr.y, tr.z])
        keep = (p[:, 2] >= self.zlo) & (p[:, 2] <= self.zhi)
        p = p[keep]
        r = np.hypot(p[:, 0], p[:, 1])
        a = np.arctan2(p[:, 1], p[:, 0])
        ok = (r >= self.rmin) & (r <= self.rmax) & (a >= self.a0) & (a <= self.a1)
        r, a = r[ok], a[ok]
        ranges = np.full(self.nbins, np.inf)
        if len(r):
            b = np.clip(np.round((a - self.a0) / self.da).astype(int), 0, self.nbins - 1)
            np.minimum.at(ranges, b, r)

        s = LaserScan()
        s.header.stamp = msg.header.stamp
        s.header.frame_id = self.frame
        s.angle_min, s.angle_max, s.angle_increment = self.a0, self.a1, self.da
        s.time_increment = 0.0
        s.scan_time = self.scan_time
        s.range_min, s.range_max = self.rmin, self.rmax
        s.ranges = ranges.astype(np.float32).tolist()
        self.pub.publish(s)
        self.n += 1


def main(args=None):
    rclpy.init(args=args)
    node = CloudToScan()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
