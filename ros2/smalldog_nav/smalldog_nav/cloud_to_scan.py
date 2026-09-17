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

Drops: a stairwell is empty air in the band, i.e. free space, and the explorer walked the
robot down one (2026-09-16). The L2 looks down as well as out, so the floor's absence is
in the cloud: returns *below* the floor plane, `drop_depth` under z = 0 within
`drop_range`. Each such return is put in the scan as an obstacle at the range where its
ray crossed the floor plane, `r * h / (h - z)` with `h` the sensor's height - that is the
edge of the drop, not the tread it landed on - so a drop is a wall to SLAM and the
costmap alike, and Nav2 keeps `robot_radius` + inflation from it. `drop_depth` 0 turns
it off (a heightfield downhill is a drop too: 0.12 m at 2 m is a 3.4 deg slope).
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
        self.declare_parameter('drop_depth', 0.12)    # m below the floor plane = a drop; a tread is 0.17
        self.declare_parameter('drop_range', 2.0)     # m, how far out to look for one: 3.4 deg of tilt there
        # The floor as the sensor sees it, not as TF says. On the real robot the L2's floor
        # returns at 0.3-1.5 m sat 8 cm BELOW z = 0 at every bearing while the ceiling read
        # its true 2.50 m (2026-09-17): a bias on grazing floor returns, not geometry, and
        # 8 cm of noise under a 12 cm drop threshold flagged a drop in every scan. Each
        # frame a plane z = a + b x + c y is fitted to the near, low returns (least squares
        # after a median cut; the floor also sloped ~2.5 deg with range - residual pitch
        # the IMU's tilt did not carry), and the band and the drop depth are measured from
        # it. 0 turns the fit off (z = 0 is the floor).
        self.declare_parameter('floor_fit_range', 2.0)   # m, the near returns the floor is read from
        self.declare_parameter('floor_fit_min_pts', 50)
        g = lambda n: self.get_parameter(n).value
        self.frame = g('target_frame')
        self.zlo, self.zhi = float(g('min_height')), float(g('max_height'))
        self.a0, self.a1 = float(g('angle_min')), float(g('angle_max'))
        self.da = float(g('angle_increment'))
        self.nbins = int(round((self.a1 - self.a0) / self.da)) + 1
        self.rmin, self.rmax = float(g('range_min')), float(g('range_max'))
        self.scan_time = float(g('scan_time'))
        self.drop_depth, self.drop_range = float(g('drop_depth')), float(g('drop_range'))
        self.drops = 0
        self.floor_range, self.floor_min_pts = float(g('floor_fit_range')), int(g('floor_fit_min_pts'))
        self.floor = np.zeros(3)                         # (a, b, c) of the last fit, carried when a frame has too few
        self.floor_hist = []

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.pub = self.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
        self.create_subscription(PointCloud2, 'cloud', self.on_cloud, qos_profile_sensor_data)
        self.n = self.dropped = 0
        self.create_timer(10.0, self.report)

    def report(self):
        fz = np.median(self.floor_hist) if self.floor_hist else float('nan')
        b_, c_ = self.floor[1], self.floor[2]
        self.get_logger().info(f'{self.n} scans, {self.dropped} clouds without TF, {self.drops} with a '
                               f'drop in view, floor at z {fz:+.3f} m, tilt {math.degrees(math.atan(b_)):+.1f} '
                               f'(pitch) {math.degrees(math.atan(c_)):+.1f} (roll) deg, fit on {len(self.floor_hist)} frames')
        self.floor_hist.clear()

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
        if self.floor_range > 0:
            rr0 = np.hypot(p[:, 0], p[:, 1])
            near = (rr0 >= self.rmin) & (rr0 <= self.floor_range) & (p[:, 2] > -0.25) & (p[:, 2] < 0.05)
            if near.sum() >= self.floor_min_pts:
                q = p[near]
                q = q[np.abs(q[:, 2] - np.median(q[:, 2])) < 0.08]      # walls' feet, legs out
                if len(q) >= self.floor_min_pts:
                    A = np.c_[np.ones(len(q)), q[:, 0], q[:, 1]]
                    self.floor = np.linalg.lstsq(A, q[:, 2], rcond=None)[0]
                    self.floor_hist.append(self.floor[0])
            a_, b_, c_ = self.floor
            p = p - np.c_[np.zeros((len(p), 2)), a_ + b_ * p[:, 0] + c_ * p[:, 1]]   # heights from the fitted floor
        keep = (p[:, 2] >= self.zlo) & (p[:, 2] <= self.zhi)
        r = np.hypot(p[keep, 0], p[keep, 1])
        a = np.arctan2(p[keep, 1], p[keep, 0])
        if self.drop_depth > 0:
            h = float(tr.z) - float(self.floor[0])   # the sensor's height over the floor (at x = y = 0)
            z = p[:, 2]
            rr = np.hypot(p[:, 0], p[:, 1])
            hole = (z < -self.drop_depth) & (rr <= self.drop_range) & (rr > 1e-3)
            if hole.any():
                self.drops += 1
                # where each ray crossed z = 0 on its way down: the edge
                # (an edge under `range_min` - the robot standing at it - is reported at
                # `range_min`: still a wall in front, not a return the consumers discard)
                edge = np.maximum(rr[hole] * h / (h - z[hole]), self.rmin)
                r = np.concatenate([r, edge])
                a = np.concatenate([a, np.arctan2(p[hole, 1], p[hole, 0])])
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
