#!/usr/bin/env python3
"""The real robot as a ROS 2 node: the walker's joint trajectory in, the servos' state out.

    ros2 run smalldog_hardware servos --ros-args -p port:=/dev/ttyACM0
    ros2 run smalldog_hardware servos --ros-args -p dry_run:=true      # loopback, no hardware

Subscribes  /<controller>/joint_trajectory   trajectory_msgs/JointTrajectory
            the exact topic `smalldog_walker` streams to — the same message it sends the
            JointTrajectoryController in the MuJoCo launch, so the walker does not know
            whether it is driving the simulator or the robot.
Publishes   /joint_states                    sensor_msgs/JointState (position, velocity,
            effort = Present Load), every tick
            /imu                             sensor_msgs/Imu, from the BMI088, every tick
            (`imu:=true`); the walker's default `imu_topic`
            /diagnostics                     diagnostic_msgs/DiagnosticArray, `diag_hz`
            (5 Hz): one status per servo (position, goal, tracking error, load, current,
            volts, the guard's filtered temperature and the raw byte), one for the loop
            (tick rate, overruns, bus errors, torque, how old the walker's goal is) and
            one for the IMU. The level is the guard's own limits: WARN at the warn
            temperature, at half the current trip, at half the tracking trip, or when
            the pack is within 0.5 V of the undervoltage trip; ERROR is a servo that
            did not answer, a stale goal, or the trip itself (published once, on the
            way out). Foxglove's Diagnostics panels read it as is — README, "Watching
            the robot".

No ros2_control, no hardware_interface plugin. `robot/runtime` already is the hardware
interface — the bus driver, the calibration, the safety guard and the 50 Hz tick that
`walk.py` and `policy.py` run on — and it is pure Python. Rewriting it as a C++ plugin so
that a JointTrajectoryController could interpolate a one-point trajectory would put a
second copy of the servo law and the safety limits where the bench cannot see them. So
this node wraps `loop.Runtime` the way `walk.py` does and takes the trajectory itself:
the latest point is the goal, held until the next one.

The tick, and who owns it
-------------------------
`Runtime.run` owns the main thread — it is the real-time loop, and `rclpy` is not one.
An executor spins on a second thread and only ever writes `self._goal` under a lock;
the source the runtime ticks reads it. Publishing happens from the runtime's `on_tick`,
on the loop's thread, which rclpy allows.

Standing up and sitting down
----------------------------
Torque comes on only after the walker's first message: `Runtime.engage` writes the
measured pose as the goal, enables torque, and ramps to that first point over `ramp`
seconds — nothing moves the instant torque arrives, by construction (`loop.py`). On
shutdown the node ramps to the gait's lowest stance and cuts torque, exactly as
`walk.py` sits down; a trip, a bus error or a walker that dies all leave by the same
door, `Runtime.__exit__`, with torque off.

What the walker must be told
----------------------------
The trot the sim runs (period 0.45 s, 0.20 m/s) demands 7.55 rad/s of a servo that has
3.28: on hardware it drags (`robot/README.md`, "The gait is fitted to the servo").
`walk.py` refits the operating point itself; under ROS the walker is a separate node, so
the fit goes in the launch file's parameters — `robot.launch.py` carries the numbers
`walk.py --dry-run` prints (period 1.35 s, 0.11 m/s, turn 0.65 rad/s, stride_max raised
to match). Launch the walker with its sim defaults against this node and it will drag.
"""
import math
import os
import signal
import sys
import threading
import time

import rclpy
import rclpy.executors
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from trajectory_msgs.msg import JointTrajectory

# `robot/` is found the way `robot/runtime/walk.py` finds `ros2/smalldog_walker`: by
# path from this file, through the symlink `--symlink-install` leaves in install/.
# The arrow points the other way here — this node imports the runtime, the runtime never
# imports ROS — and `robot/`'s dependency list stays pyserial + numpy.
_HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.environ.get('SMALLDOG_REPO') or os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(REPO, 'robot'))

from feetech.bus import Bus, BusError                                # noqa: E402
from runtime.calib import CALIB, Calibration, load_params            # noqa: E402
from runtime.loop import CTRL_HZ, FollowingLoopback, Runtime         # noqa: E402
from runtime.safety import Limits, Tripped                           # noqa: E402
from smalldog_walker.gait import TrotGait                            # noqa: E402


def quat_zyx(roll, pitch, yaw):
    """(w, x, y, z) of the body attitude, the ZYX Euler triple `TrotGait.feedback` inverts."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def attitude_from_gravity(g):
    """(roll, pitch) from gravity's unit vector in the body frame: x forward, y left, z up.

    R^T (0, 0, -1) = (sin p, -sin r cos p, -cos r cos p), so a nose-down body reads
    gravity's x positive — `imu/bmi088.py`'s own check — and the inversion is exact.
    """
    gx, gy, gz = g
    return math.atan2(-gy, -gz), math.asin(max(-1.0, min(1.0, gx)))


class ServoNode(Node):
    def __init__(self):
        super().__init__('smalldog_servos')
        p = self.declare_parameter
        p('port', '/dev/ttyACM0')
        p('baud', 1_000_000)
        p('dry_run', False)
        p('calib', CALIB)
        p('hz', CTRL_HZ)
        p('ramp', 2.0)
        p('controller', 'smalldog_controller')
        p('first_goal_timeout', 15.0)      # s to wait for the walker before giving up
        p('settle', 1.0)                   # s after the first goal before standing into it
        p('goal_timeout', 1.0)             # s without a trajectory before it is reported
        p('imu', False)
        p('i2c_bus', 1)
        p('bias_seconds', 3.0)
        p('imu_frame', 'base_link')
        p('temp_c', Limits.temp_c)
        p('current_a', Limits.current_a)
        p('volt_min', Limits.volt_min)
        p('track_rad', Limits.q_err_rad)
        p('diag_hz', 5.0)                  # /diagnostics rate; 0 turns it off
        g = lambda k: self.get_parameter(k).value    # noqa: E731

        self.params = load_params()
        calib_path = g('calib')
        self.calib = (Calibration.load(calib_path, self.params) if os.path.exists(calib_path)
                      else Calibration.default(self.params))
        self.dry = bool(g('dry_run'))
        if not getattr(self.calib, 'measured', False) and not self.dry:
            raise SystemExit('this calibration is defaults, not this robot; torque stays '
                             'off. Run robot/runtime/calib.py --capture, then --sign all')

        self.joints = list(self.calib.joints)
        self.hz = float(g('hz'))
        self.ramp = float(g('ramp'))
        self.first_goal_timeout = float(g('first_goal_timeout'))
        self.settle = float(g('settle'))
        self.goal_timeout = float(g('goal_timeout'))

        self._lock = threading.Lock()
        self._goal = None                  # latest trajectory point, calib.joints order
        self._goal_t = None                # perf_counter when it arrived
        self._stale_said = False
        self._msgs = self._reordered = 0
        self.stop = False                  # set by the signal handler in main()

        ctrl = g('controller')
        self.traj_topic = f'/{ctrl}/joint_trajectory'
        self.create_subscription(JointTrajectory, self.traj_topic, self.on_trajectory, 10)
        self.pub_js = self.create_publisher(JointState, '/joint_states', 10)
        self.pub_imu = self.create_publisher(Imu, '/imu', 10) if g('imu') else None
        self.imu_frame = g('imu_frame')
        self.imu = None                    # LiveIMU, made in start()
        self._yaw = 0.0                    # integrated gyro; drifts, and the gait knows
        self._att = (0.0, 0.0, 0.0)        # roll, pitch, yaw of the last tick, for /diagnostics
        self.diag_every = (int(round(self.hz / float(g('diag_hz')))) if float(g('diag_hz')) > 0
                           else 0)
        self.pub_diag = self.create_publisher(DiagnosticArray, '/diagnostics', 10)

        if self.dry:
            self.bus = Bus(transport=FollowingLoopback(self.calib.ids), discard_echo=False)
        else:
            self.bus = Bus(g('port'), int(g('baud')))
        limits = Limits(temp_c=g('temp_c'), current_a=g('current_a'),
                        volt_min=g('volt_min'), q_err_rad=g('track_rad'))
        self.rt = Runtime(self.bus, self.calib, hz=self.hz, limits=limits,
                          log=lambda s: self.get_logger().info(str(s)))

        # the gait's lowest stance, to sit down into — walk.py's `--no-sit` inverse
        gait = TrotGait(self.params)
        if list(gait.joint_names) != self.joints:
            raise SystemExit(f'the gait and the calibration disagree about joint order:\n'
                             f'  gait  {gait.joint_names}\n  calib {self.joints}')
        gait.body_height = 0.0             # the setter clamps to the lowest reachable
        q = None
        for _ in range(int(1.5 * self.hz)):
            q = gait.joint_targets(1.0 / self.hz, 0.0, 0.0, 0.0)
        self.q_sit = q

        self.get_logger().info(
            f'servos up: {len(self.joints)} joints on '
            f'{"a loopback bus" if self.dry else g("port")} at {self.hz:.0f} Hz, '
            f'listening on /{ctrl}/joint_trajectory')

    # ------------------------------------------------------------- subscriptions
    def on_trajectory(self, msg):
        if not msg.points:
            return
        pt = msg.points[-1]
        names = list(msg.joint_names)
        if names == self.joints:
            q = [float(v) for v in pt.positions]
        else:
            # the walker and the calibration both read robot_params.json, so a
            # different order is a stale file somewhere — reorder, count it, say so once
            try:
                idx = {n: i for i, n in enumerate(names)}
                q = [float(pt.positions[idx[n]]) for n in self.joints]
            except KeyError as e:
                if self._msgs == 0:
                    self.get_logger().error(f'trajectory names {names} lack {e}; ignored')
                self._msgs += 1
                return
            if self._reordered == 0:
                self.get_logger().warn(f'trajectory joint order {names} is not the '
                                       f'calibration\'s; reordering by name')
            self._reordered += 1
        with self._lock:
            self._goal = q
            self._goal_t = time.perf_counter()
        self._msgs += 1

    def latest_goal(self):
        with self._lock:
            return self._goal, self._goal_t

    # ------------------------------------------------------------------ the loop
    def source(self, dt, fb):
        """`Runtime.run`'s source: the latest trajectory point, held when none arrives."""
        if self.stop or not rclpy.ok():
            raise StopIteration
        q, t = self.latest_goal()
        if q is None:
            return [self.rt.goal[n] for n in self.joints]
        age = time.perf_counter() - t
        if age > self.goal_timeout and not self._stale_said:
            self._stale_said = True
            self.get_logger().warn(f'no trajectory for {age:.1f} s; holding the last goal')
        elif age <= self.goal_timeout:
            self._stale_said = False
        return q

    def on_tick(self, k, dt, fb):
        now = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = now
        js.name = self.joints
        js.position = [fb[n]['q'] if fb[n] else math.nan for n in self.joints]
        js.velocity = [fb[n]['w'] if fb[n] else math.nan for n in self.joints]
        js.effort = [float(fb[n]['load']) if fb[n] else math.nan for n in self.joints]
        self.pub_js.publish(js)

        if self.imu is not None:
            g, w, acc = self.imu.update(dt)
            roll, pitch = attitude_from_gravity(g)
            self.rt.guard.attitude(dt, roll, pitch)     # raises Tripped: the robot is over
            self._yaw += w[2] * dt
            self._att = (roll, pitch, self._yaw)
            m = Imu()
            m.header.stamp = now
            m.header.frame_id = self.imu_frame
            qw, qx, qy, qz = quat_zyx(roll, pitch, self._yaw)
            m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z = qw, qx, qy, qz
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = w
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = acc
            # roll/pitch from a complementary filter, yaw integrated: the covariances say so
            m.orientation_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0]
            self.pub_imu.publish(m)

        if self.diag_every and k % self.diag_every == 0:
            self.pub_diag.publish(self.diagnostics(fb, now))

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self, fb, stamp, trip=None) -> DiagnosticArray:
        """One DiagnosticArray: a status per servo, one for the loop, one for the IMU."""
        lim, rt = self.rt.guard.lim, self.rt
        kv = lambda **d: [KeyValue(key=k, value=v) for k, v in d.items()]    # noqa: E731
        msg = DiagnosticArray()
        msg.header.stamp = stamp
        volts = []
        for n in self.joints:
            f = fb.get(n)
            st = DiagnosticStatus(name=f'servo {n}', hardware_id=f'id {self.calib.id[n]}')
            if f is None:
                st.level, st.message = DiagnosticStatus.ERROR, 'no answer on the bus'
                msg.status.append(st)
                continue
            err = f['q'] - rt.goal[n]
            temp = self.rt.guard.temp(n)
            volts.append(f['volt'])
            why = []
            if temp is not None and temp >= lim.temp_warn_c:
                why.append(f'{temp:.0f} C')
            if f['current'] >= 0.5 * lim.current_a:
                why.append(f'{f["current"]:.2f} A')
            if abs(err) >= 0.5 * lim.q_err_rad and rt.torque_on:
                why.append(f'{math.degrees(err):+.0f} deg behind')
            st.level = DiagnosticStatus.WARN if why else DiagnosticStatus.OK
            st.message = ', '.join(why) or 'ok'
            st.values = kv(position_deg=f'{math.degrees(f["q"]):+.1f}',
                           goal_deg=f'{math.degrees(rt.goal[n]):+.1f}',
                           error_deg=f'{math.degrees(err):+.1f}',
                           velocity_rad_s=f'{f["w"]:+.2f}',
                           load=f'{f["load"]:+d}',
                           current_A=f'{f["current"]:.2f}',
                           volt_V=f'{f["volt"]:.1f}',
                           temp_C='' if temp is None else f'{temp:.0f}',
                           temp_raw_C=f'{f["temp"]:.0f}')
            msg.status.append(st)

        st = DiagnosticStatus(name='loop', hardware_id='runtime')
        goal, t = self.latest_goal()
        age = math.nan if t is None else time.perf_counter() - t
        pct = 100.0 * rt.overruns / max(1, rt.ticks)
        if trip is not None:
            st.level, st.message = DiagnosticStatus.ERROR, f'TRIPPED: {trip}'
        elif goal is not None and age > self.goal_timeout:
            st.level, st.message = DiagnosticStatus.ERROR, f'no trajectory for {age:.1f} s'
        elif volts and min(volts) <= lim.volt_min + 0.5:
            st.level, st.message = DiagnosticStatus.WARN, f'pack at {min(volts):.1f} V'
        elif rt.bus_errors:
            st.level, st.message = DiagnosticStatus.WARN, f'{rt.bus_errors} short bus reads'
        else:
            st.level = DiagnosticStatus.OK
            st.message = ('walking' if rt.torque_on else 'torque off') + f', {pct:.1f} % late'
        st.values = kv(torque='on' if rt.torque_on else 'off',
                       hz=f'{rt.hz:.0f}', ticks=str(rt.ticks),
                       overruns=f'{rt.overruns} ({pct:.1f} %)',
                       bus_errors=str(rt.bus_errors),
                       pack_V='' if not volts else f'{min(volts):.1f}..{max(volts):.1f}',
                       goal_age_s='' if t is None else f'{age:.2f}',
                       trajectory_msgs=str(self._msgs),
                       trajectory_publishers=str(self.count_publishers(self.traj_topic)),
                       peak_temp_C=f'{rt.guard.peak["temp"]:.0f}',
                       peak_current_A=f'{rt.guard.peak["current"]:.2f}',
                       peak_error_deg=f'{math.degrees(rt.guard.peak["q_err"]):.1f}')
        msg.status.append(st)

        if self.imu is not None:
            r, p_, y = self._att
            st = DiagnosticStatus(name='imu', hardware_id='bmi088', level=DiagnosticStatus.OK,
                                  message=f'roll {math.degrees(r):+.1f} pitch {math.degrees(p_):+.1f}')
            st.values = kv(roll_deg=f'{math.degrees(r):+.1f}', pitch_deg=f'{math.degrees(p_):+.1f}',
                           yaw_deg=f'{math.degrees(y):+.1f}')
            msg.status.append(st)
        return msg

    def start_imu(self):
        if self.pub_imu is None:
            return
        from runtime.policy import LiveIMU
        if self.dry:
            from imu.bmi088 import FakeBMI088
            chip, bias = FakeBMI088(), (0.0, 0.0, 0.0)
        else:
            from imu.bmi088 import BMI088, measure_bias
            chip = BMI088(int(self.get_parameter('i2c_bus').value))
            chip.configure()
            if chip.mount.measured:
                r, p_ = chip.mount.tilt_deg()
                self.get_logger().info(f'IMU mount ({chip.mount.measured}): taking out roll '
                                       f'{r:+.1f}, pitch {p_:+.1f} deg')
            else:
                self.get_logger().warn('IMU mount not levelled (no imu/mount.json): the '
                                       'chip\'s tilt in the case goes straight into the '
                                       'gait\'s levelling - run imu/bmi088.py --level')
            secs = float(self.get_parameter('bias_seconds').value)
            self.get_logger().info(f'IMU: hold still {secs:g} s for the gyro bias ...')
            bias = measure_bias(chip, secs, self.hz)
            self.get_logger().info('gyro bias rad/s: ' + ' '.join(f'{b:+.4f}' for b in bias))
        self.imu = LiveIMU(chip, bias)

    def run(self) -> int:
        """Preflight, wait for the walker, stand, tick until shutdown, sit. Returns the exit code."""
        log = self.get_logger()
        pre = self.rt.preflight(None if self.dry else self.get_parameter('port').value)
        if not pre['ok']:
            log.error('not every servo answered; refusing to move')
            return 1
        self.start_imu()

        t0 = time.perf_counter()
        while not self.stop and rclpy.ok() and self.latest_goal()[0] is None:
            if time.perf_counter() - t0 > self.first_goal_timeout:
                log.error(f'no trajectory in {self.first_goal_timeout:.0f} s — is the walker '
                          f'up? torque stays off')
                return 1
            time.sleep(0.05)
        if self.stop or not rclpy.ok():
            return 0
        # Two walkers on one topic is a robot flickering between two gaits. The mac
        # grows them: a MuJoCo launch left behind keeps its walker publishing, and this
        # node's first goal was that sim's mid-stride pose.
        n_pub = self.count_publishers(self.traj_topic)
        if n_pub > 1:
            log.error(f'{n_pub} publishers on {self.traj_topic}; a leftover walker? '
                      f'torque stays off')
            return 1
        # The walker's first messages are its own start-up ramp from q = 0 to the
        # stance, rate-limited at the joint ceiling: stand into where it has settled,
        # not into the first frame of that ramp.
        time.sleep(self.settle)
        q0 = self.latest_goal()[0]
        log.info('first goal, deg: ' + '  '.join(
            f'{self.joints[i][:2]} ' + '/'.join(f'{math.degrees(q0[i + j]):+.0f}' for j in range(3))
            for i in range(0, len(self.joints), 3)))

        code = 0
        try:
            with self.rt:
                self.rt.engage(q0, ramp_s=self.ramp)
                log.info('standing; streaming the walker\'s goals')
                self.rt.run(self.source, on_tick=self.on_tick)
                log.info('sitting down')
                self.rt.relax(self.q_sit, ramp_s=1.5)
        except Tripped as e:
            log.error(f'TRIPPED: {e}')
            # one last word on /diagnostics, so the panel says why the robot sat down
            self.pub_diag.publish(self.diagnostics(
                {n: None for n in self.joints}, self.get_clock().now().to_msg(), trip=e))
            code = 1
        except BusError as e:
            log.error(f'bus: {e}')
            code = 1
        finally:
            for line in self.rt.report_lines().split('\n'):
                log.info(line)
            log.info(f'{self._msgs} trajectory messages, {self._reordered} reordered')
        return code


def main(args=None):
    rclpy.init(args=args)
    node = ServoNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    # SIGINT and SIGTERM (Ctrl-C, or `ros2 launch` shutting down) set a flag the loop
    # reads on its next tick, and the loop then leaves through `Runtime.__exit__` with
    # the sit-down ramp in between. Not KeyboardInterrupt: an exception thrown into
    # the middle of a bus transaction is a half-written packet, and a process started
    # in the background inherits SIGINT ignored, which rclpy's handler leaves alone.
    def on_signal(signum, frame):
        node.stop = True
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    code = 1
    try:
        code = node.run()
    finally:
        executor.shutdown()
        spin.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
