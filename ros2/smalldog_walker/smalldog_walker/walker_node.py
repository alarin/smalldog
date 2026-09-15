#!/usr/bin/env python3
"""SmallDog gait node: /cmd_vel (+ IMU, foot load) -> streamed joint trajectory.

Publishes one-point JointTrajectory messages at `rate` Hz to the
JointTrajectoryController, which is the same pattern the hexapod uses but
streamed continuously instead of action-per-step, because a trot has no
natural step boundary to wait on.
"""
import json, math, os
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Bool, Float64, Float64MultiArray
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
        # 0 = the gait's own. `period_for` pins the period at 2*stride_max/speed, so a
        # period chosen for the servo's rate ceiling (robot.launch.py: 1.35 s at 0.11
        # m/s) is silently shortened back unless the pin is raised to admit it — the
        # same line robot/runtime/walk.py has after its feasibility fit.
        self.declare_parameter('stride_max', 0.0)
        # 0 = the gait's own. The heading hold adds up to `yaw_max` rad/s of turn ON TOP
        # of the forward command, and on hardware there is no joint speed left for it:
        # at 0.11 m/s the straight line uses the whole ceiling, and the gait's 0.5 rad/s
        # of correction demands 4.65 rad/s of a 3.28 rad/s servo. The feet drag, a
        # planted knee falls 0.7 rad behind its goal and the guard cuts torque (measured,
        # 2026-09-15). robot.launch.py sets the pair (speed, yaw_max) that fits.
        self.declare_parameter('yaw_max', 0.0)
        self.declare_parameter('cmd_timeout', 0.5)
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('foot_load_topic', '/smalldog/foot_load')
        self.declare_parameter('contact_threshold', 1.0)
        # Odometry (README, "Odometry"): the gait's own body velocity integrated, heading
        # from the IMU when it is live. Published as /odom and as two TF frames, the
        # level `base_frame` under the tilting `body_frame`, so a scan sliced in the
        # level frame keeps the floor out. `odom_scale` is the foot slip: 1.0 says the
        # feet do not slide; the sim trot delivers ~0.75 of its command on the flat
        # (README, "Forward speed"), hardware is unmeasured (verify).
        self.declare_parameter('odom', True)
        self.declare_parameter('odom_scale', 1.0)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('body_frame', 'base_link')

        share = get_package_share_directory('smalldog_description')
        with open(os.path.join(share, 'robot_params.json')) as f:
            params = json.load(f)

        self.gait = TrotGait(params)
        # order matters: the body-height setter clamps against swing and step
        self.gait.period = self.get_parameter('period').value
        self.gait.swing_height = self.get_parameter('swing_height').value
        self.gait.max_step = self.get_parameter('max_step').value
        self.gait.body_height = self.get_parameter('body_height').value
        if self.get_parameter('stride_max').value > 0:
            self.gait.stride_max = self.get_parameter('stride_max').value
        if self.get_parameter('yaw_max').value > 0:
            self.gait.yaw_max = self.get_parameter('yaw_max').value

        ctrl = self.get_parameter('controller').value
        self.pub = self.create_publisher(JointTrajectory, f'/{ctrl}/joint_trajectory', 10)

        self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)
        self.create_subscription(Bool, '/smalldog/enable', self.on_enable, 10)
        self.create_subscription(Float64, '/smalldog/body_height', self.on_height, 10)

        # Terrain feedback. Both are optional: with neither topic publishing, the gait is
        # the open-loop trot it has always been, and it falls back to that on its own if
        # either stream stops. Nothing in this workspace publishes them yet — see the
        # "Known gaps" section of the README.
        self.create_subscription(Imu, self.get_parameter('imu_topic').value,
                                 self.on_imu, 10)
        self.create_subscription(Float64MultiArray,
                                 self.get_parameter('foot_load_topic').value,
                                 self.on_foot_load, 10)
        self.contact_threshold = self.get_parameter('contact_threshold').value
        self.contact = None
        self.imu_seen = 0
        self._load_no_imu = False     # said once, in on_foot_load

        self.cmd = (0.0, 0.0, 0.0)
        self.enabled = True
        # gait phase integrates in SIM time (physics), but "is the operator still
        # sending?" is a wall-clock question: the sim can run many times real time,
        # and then a steady 20 Hz teleop looks stale on the sim clock.
        self._wall = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.last_cmd = self._wall.now()
        self.timeout = self.get_parameter('cmd_timeout').value

        self.odom = self.get_parameter('odom').value
        if self.odom:
            self.odom_scale = float(self.get_parameter('odom_scale').value)
            self.odom_frame = self.get_parameter('odom_frame').value
            self.base_frame = self.get_parameter('base_frame').value
            self.body_frame = self.get_parameter('body_frame').value
            self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
            self.tf = TransformBroadcaster(self)
            self._ox = self._oy = self._oyaw = 0.0
            self._yaw0 = None       # IMU yaw at the moment odometry first heard it

        self.rate = float(self.get_parameter('rate').value)
        self.dt = 1.0 / self.rate
        self._last_tick = None
        self.timer = self.create_timer(self.dt, self.tick)

        r = self.gait.reach_info()
        self.get_logger().info(
            f'walker up: {len(self.gait.joint_names)} joints -> /{ctrl}/joint_trajectory '
            f'@ {self.rate:.0f} Hz')
        self.get_logger().info(
            f'leg reach {r["d_min"]*1000:.0f}..{r["d_max"]*1000:.0f} mm -> body height '
            f'{r["height_min"]*1000:.0f}..{r["height_max"]*1000:.0f} mm, using '
            f'{r["body_height"]*1000:.0f} mm, swing {r["swing_height"]*1000:.0f} mm')
        self.get_logger().info(
            f'terrain feedback listening on {self.get_parameter("imu_topic").value} and '
            f'{self.get_parameter("foot_load_topic").value}; open loop until they publish')

    def on_cmd_vel(self, msg):
        self.cmd = (msg.linear.x, msg.linear.y, msg.angular.z)
        self.last_cmd = self._wall.now()

    def on_enable(self, msg):
        self.enabled = msg.data
        self.get_logger().info(f'gait {"enabled" if msg.data else "disabled"}')

    def on_height(self, msg):
        self.gait.body_height = msg.data      # the gait clamps to its reachable band

    def on_imu(self, msg):
        # Say it once, out loud.  Whether this topic is arriving is the single difference
        # between the terrain-aware gait and the blind one, and it is invisible otherwise:
        # the gait degrades silently by design, so a missing IMU looks exactly like a
        # robot that simply walks badly.
        self.imu_seen += 1
        if self.imu_seen == 1:
            self.get_logger().info('IMU is live - terrain feedback active')
        o, w = msg.orientation, msg.angular_velocity
        self.gait.feedback(quat=(o.w, o.x, o.y, o.z), gyro=(w.x, w.y, w.z),
                           contact=self.contact)

    def on_foot_load(self, msg):
        """per-foot load in gait.legs order — a force, a current, anything monotonic.

        The gait only wants "is this foot carrying weight", so the threshold lives here
        and the units stay whatever the source publishes. Held until the next IMU message
        rather than pushed on its own: one feedback() per control-rate stream keeps the
        gait's staleness timer meaningful.
        """
        if len(msg.data) >= len(self.gait.legs):
            self.contact = {l: msg.data[i] > self.contact_threshold
                            for i, l in enumerate(self.gait.legs)}
            # ... and it is HELD, so with no IMU it never reaches gait.feedback() at all.
            # That is a silently open loop, which is exactly the failure this node should
            # not keep to itself.  Once is enough; the IMU handler announces its own
            # arrival the same way.
            if self.imu_seen == 0 and not self._load_no_imu:
                self._load_no_imu = True
                self.get_logger().warn(
                    'foot load is arriving but no IMU yet - contact feedback is held and '
                    'never applied until one does, i.e. the gait is open loop')

    def tick(self):
        stale = (self._wall.now() - self.last_cmd).nanoseconds * 1e-9 > self.timeout
        cmd = (0.0, 0.0, 0.0) if (stale or not self.enabled) else self.cmd

        # The gait's clock is the time that actually passed (on the node's clock: sim time
        # under the sim), not the timer's nominal period. A loaded machine fires a 100 Hz
        # Python timer at 60 and the trot ran at 60 % speed for it — the spin behavior
        # timed out on a robot turning at 2 deg/s. Capped at three periods so a stall
        # does not become one giant stride.
        now = self.get_clock().now()
        if self._last_tick is None:
            dt = self.dt
        else:
            dt = min(max((now - self._last_tick).nanoseconds * 1e-9, 1e-4), 3.0 * self.dt)
        self._last_tick = now
        self.dt = dt

        q = self.gait.joint_targets(dt, *cmd)

        msg = JointTrajectory()
        msg.joint_names = self.gait.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        step = 2.0 * self.dt
        pt.time_from_start = Duration(sec=int(step), nanosec=int((step % 1.0) * 1e9))
        msg.points = [pt]
        self.pub.publish(msg)
        if self.odom:
            self.publish_odom()

    def publish_odom(self):
        """Dead reckoning off the gait plus the IMU's heading.

        Translation: the velocity the stance feet sweep (`gait.body_velocity()`, the
        command after the heading hold and the stride clamps), times `odom_scale`,
        integrated in the odom frame. Heading: the IMU's yaw while it is live (the sim's
        is truth, the robot's an integrated gyro), zeroed where odometry first saw it;
        the commanded turn rate otherwise. It drifts; SLAM's scan matcher is what closes
        it, and `map -> odom` is that node's to publish, not this one's.

        The twist is the gait's own velocity (scaled), all three axes, not a derivative
        of the pose: Nav2's controller reads it as "current speed" and ramps its command
        from there, and the trot's rocking differentiated at 100 Hz read as a robot that
        was barely turning, so the command never rose above the ramp step.
        """
        vx, vy, wz = (v * self.odom_scale for v in self.gait.body_velocity())
        roll, pitch, yaw, live = self.gait.attitude()
        if live:
            if self._yaw0 is None:
                self._yaw0 = yaw - self._oyaw
            yaw_new = _wrap(yaw - self._yaw0)
        else:
            yaw_new = _wrap(self._oyaw + wz * self.dt)
        mid = 0.5 * (self._oyaw + yaw_new)
        c, s = math.cos(mid), math.sin(mid)
        self._ox += (vx * c - vy * s) * self.dt
        self._oy += (vx * s + vy * c) * self.dt
        self._oyaw = yaw_new

        now = self.get_clock().now().to_msg()
        z = self.gait.body_height + self.gait.foot_r

        od = Odometry()
        od.header.stamp = now
        od.header.frame_id = self.odom_frame
        od.child_frame_id = self.base_frame
        od.pose.pose.position.x, od.pose.pose.position.y = self._ox, self._oy
        qw, qx, qy, qz = _quat(0.0, 0.0, self._oyaw)
        od.pose.pose.orientation.w, od.pose.pose.orientation.x = qw, qx
        od.pose.pose.orientation.y, od.pose.pose.orientation.z = qy, qz
        od.twist.twist.linear.x, od.twist.twist.linear.y = vx, vy
        od.twist.twist.angular.z = wz
        # a legged dead reckoning: honest about slip in the plane, tighter on heading
        # when the IMU is speaking
        od.pose.covariance[0] = od.pose.covariance[7] = 0.05
        od.pose.covariance[35] = 0.02 if live else 0.2
        od.twist.covariance[0] = od.twist.covariance[7] = 0.02
        od.twist.covariance[35] = 0.01 if live else 0.1
        self.odom_pub.publish(od)

        t_ob = TransformStamped()
        t_ob.header.stamp = now
        t_ob.header.frame_id = self.odom_frame
        t_ob.child_frame_id = self.base_frame
        t_ob.transform.translation.x, t_ob.transform.translation.y = self._ox, self._oy
        t_ob.transform.rotation.w, t_ob.transform.rotation.x = qw, qx
        t_ob.transform.rotation.y, t_ob.transform.rotation.z = qy, qz

        # the body over its footprint: up by the standing height, tilted as the IMU says
        t_bb = TransformStamped()
        t_bb.header.stamp = now
        t_bb.header.frame_id = self.base_frame
        t_bb.child_frame_id = self.body_frame
        t_bb.transform.translation.z = z
        qw, qx, qy, qz = _quat(roll, pitch, 0.0)
        t_bb.transform.rotation.w, t_bb.transform.rotation.x = qw, qx
        t_bb.transform.rotation.y, t_bb.transform.rotation.z = qy, qz
        self.tf.sendTransform([t_ob, t_bb])


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _quat(roll, pitch, yaw):
    """(w, x, y, z) for a z-y-x Euler triple, the same convention servo_node uses."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def main(args=None):
    rclpy.init(args=args)
    node = SmallDogWalker()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass                    # Ctrl-C, or `ros2 launch` shutting down (Jazzy raises it)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
