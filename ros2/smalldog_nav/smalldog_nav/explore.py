#!/usr/bin/env python3
"""Frontier exploration: walk the room until the map has no edge left.

    ros2 run smalldog_nav explore                 # the launch does this with explore:=true
    ros2 topic pub -1 /smalldog/explore std_msgs/msg/Bool "{data: false}"   # pause / resume

A frontier is a free cell of slam_toolbox's `/map` with an unknown neighbour: the edge of
what has been seen. The node clusters them, walks to the biggest one that is not too far
and not on the blacklist, and repeats until none is left; then it saves the map (if asked)
and walks home. Nav2 does the walking (`navigate_to_pose`); this node only chooses where.

Why it is written here rather than taken from a package: explore_lite (m-explore) is not
in RoboStack or in the Jazzy apt set the Pi runs, and what it does fits in 300 lines
once the map is an OccupancyGrid. The choices that matter for THIS robot:

- **It spins first.** The L2 is bolted axis-forward and its cone is +-96 deg, so the
  first map is a wedge and everything behind the robot is a frontier a hand's breadth
  away. One 2 pi turn (the behavior server's `spin`) fills the ring before any goal is
  chosen; without it the first goals are all "look over your shoulder".
- **Goals are frontier cells, moved back from walls.** A cluster's centroid can sit in
  unknown space or inside the inflation; the goal is the cluster cell nearest the
  centroid among those with no occupied cell within `goal_clearance`, so the planner
  is never handed a pose it must reject.
- **A goal that stops being a frontier is dropped**, not walked out: the map fills in
  as the robot approaches, and standing where the edge used to be buys nothing.
- **Failed goals are blacklisted** by radius. The trot has no reverse and Nav2's recovery
  is a spin and a backup; a frontier behind a chair leg it cannot reach comes up again
  every cycle otherwise.
- **Cost is distance minus a bonus for size**, in metres, so a big edge across the room
  beats a fragment beside the robot, and the robot sweeps rather than dithers.
"""
import math
from collections import deque

import numpy as np
import rclpy
import rclpy.executors
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose, Spin
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException, \
    ConnectivityException

try:
    from slam_toolbox.srv import SaveMap
except ImportError:            # no slam_toolbox on this machine: save_map is just off
    SaveMap = None

FREE, OCC = 25, 65            # /map values: -1 unknown, 0..100 occupancy


def frontiers(grid, clearance_cells):
    """(mask of frontier cells, mask of cells too close to an obstacle) for a 2-D grid."""
    unknown = grid < 0
    free = (grid >= 0) & (grid <= FREE)
    occ = grid >= OCC
    # a free cell with an unknown 4-neighbour
    nb = np.zeros_like(unknown)
    nb[1:, :] |= unknown[:-1, :]
    nb[:-1, :] |= unknown[1:, :]
    nb[:, 1:] |= unknown[:, :-1]
    nb[:, :-1] |= unknown[:, 1:]
    front = free & nb
    # obstacles grown by `clearance_cells`: 4-neighbour dilation that many times, a
    # diamond rather than a disc - close enough, and cheap enough for every map update
    near = occ.copy()
    for _ in range(int(clearance_cells)):
        grown = near.copy()
        grown[1:, :] |= near[:-1, :]
        grown[:-1, :] |= near[1:, :]
        grown[:, 1:] |= near[:, :-1]
        grown[:, :-1] |= near[:, 1:]
        near = grown
    return front, near


def clusters(mask):
    """8-connected components of a boolean grid, as lists of (row, col)."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    out = []
    rows, cols = np.nonzero(mask)
    for r0, c0 in zip(rows, cols):
        if seen[r0, c0]:
            continue
        comp, q = [], deque([(r0, c0)])
        seen[r0, c0] = True
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
        out.append(comp)
    return out


class Explorer(Node):
    def __init__(self):
        super().__init__('smalldog_explore')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('autostart', True)
        self.declare_parameter('spin_first', True)
        self.declare_parameter('min_frontier_m', 0.30)      # cluster length below this is noise
        self.declare_parameter('goal_clearance', 0.28)      # m from any occupied cell
        self.declare_parameter('min_goal_dist', 0.35)       # closer than this is under the robot
        self.declare_parameter('size_gain', 0.5)            # m of distance one m of frontier buys
        self.declare_parameter('blacklist_radius', 0.40)
        self.declare_parameter('goal_timeout', 90.0)        # s, plus dist / 0.05
        self.declare_parameter('replan_period', 1.0)
        self.declare_parameter('done_after', 3)             # empty checks in a row
        self.declare_parameter('save_map', '')              # path stem; '' = do not
        self.declare_parameter('home', True)                # walk back to (0, 0) when done

        g = lambda n: self.get_parameter(n).value
        self.map_frame, self.base_frame = g('map_frame'), g('base_frame')
        self.active = bool(g('autostart'))
        self.grid = None                 # (OccupancyGrid info, np.int8 array)
        self.blacklist = []              # (x, y) in the map frame
        self.goal = None                 # (x, y) being walked to
        self.goal_handle = None
        self.goal_sent_at = None
        self.goal_deadline = 0.0
        self.empty_checks = 0
        self.spun = not bool(g('spin_first'))
        self.spinning = False
        self.spin_tries = 0
        self.done = False
        self.homing = False
        self.goals_sent = self.goals_failed = 0

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(OccupancyGrid, g('map_topic'), self.on_map, latched)
        self.create_subscription(Bool, '/smalldog/explore', self.on_switch, 10)
        self.done_pub = self.create_publisher(Bool, '/smalldog/explored', latched)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin = ActionClient(self, Spin, 'spin')
        self.save = (self.create_client(SaveMap, '/slam_toolbox/save_map')
                     if SaveMap is not None and g('save_map') else None)
        self.create_timer(float(g('replan_period')), self.step)
        self.get_logger().info(
            f'explorer up: {"exploring" if self.active else "idle"} once /map and '
            f'navigate_to_pose are there; /smalldog/explore false|true pauses|resumes')

    # ------------------------------------------------------------------ inputs
    def on_map(self, msg):
        a = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.grid = (msg.info, a)

    def on_switch(self, msg):
        if bool(msg.data) == self.active:
            return
        self.active = bool(msg.data)
        self.get_logger().info('exploring' if self.active else 'paused')
        if not self.active:
            self.cancel()

    def pose(self):
        try:
            t = self.tf.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
        except (LookupException, ExtrapolationException, ConnectivityException):
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    # ------------------------------------------------------------------ the map
    def candidates(self, here):
        """[(cost, x, y, size_m)] over the current map, best first; [] = nothing left."""
        info, grid = self.grid
        res = info.resolution
        g = lambda n: self.get_parameter(n).value
        front, near = frontiers(grid, math.ceil(g('goal_clearance') / res))
        ox, oy = info.origin.position.x, info.origin.position.y
        out = []
        for comp in clusters(front):
            size = len(comp) * res
            if size < g('min_frontier_m'):
                continue
            rc = np.array(comp)
            cen = rc.mean(axis=0)
            ok = [p for p in comp if not near[p]]
            if not ok:
                continue
            r, c = min(ok, key=lambda p: (p[0] - cen[0]) ** 2 + (p[1] - cen[1]) ** 2)
            x, y = ox + (c + 0.5) * res, oy + (r + 0.5) * res
            d = math.hypot(x - here[0], y - here[1])
            if d < g('min_goal_dist'):
                continue
            if any(math.hypot(x - bx, y - by) < g('blacklist_radius')
                   for bx, by in self.blacklist):
                continue
            out.append((d - g('size_gain') * size, x, y, size))
        out.sort()
        return out

    def still_frontier(self, x, y):
        """Is there frontier within a goal tolerance of (x, y) on the latest map?"""
        info, grid = self.grid
        res = info.resolution
        front, _ = frontiers(grid, 0)
        c = int((x - info.origin.position.x) / res)
        r = int((y - info.origin.position.y) / res)
        k = int(0.25 / res)
        sub = front[max(0, r - k):r + k + 1, max(0, c - k):c + k + 1]
        return bool(sub.any())

    # ------------------------------------------------------------------ the loop
    def step(self):
        if not self.active or self.done or self.grid is None:
            return
        here = self.pose()
        if here is None:
            return                      # no map -> base TF yet: SLAM is still coming up
        if not self.spun:
            if not self.spinning:
                self.start_spin()
            return
        if self.homing:
            return
        if self.goal is not None:
            elapsed = (self.get_clock().now() - self.goal_sent_at).nanoseconds * 1e-9
            if elapsed > self.goal_deadline:
                self.get_logger().warn(f'goal ({self.goal[0]:.2f}, {self.goal[1]:.2f}) '
                                       f'timed out after {elapsed:.0f} s, blacklisting')
                self.fail(self.goal)
                self.cancel()
            elif not self.still_frontier(*self.goal):
                self.get_logger().info(f'goal ({self.goal[0]:.2f}, {self.goal[1]:.2f}) '
                                       f'is mapped, moving on')
                self.cancel()
            return
        cands = self.candidates(here)
        if not cands:
            self.empty_checks += 1
            if self.empty_checks >= int(self.get_parameter('done_after').value):
                self.finish(here)
            return
        self.empty_checks = 0
        cost, x, y, size = cands[0]
        self.get_logger().info(f'{len(cands)} frontiers; going to ({x:.2f}, {y:.2f}), '
                               f'{size:.2f} m long, {math.hypot(x-here[0], y-here[1]):.2f} m away')
        self.send_goal(x, y, math.atan2(y - here[1], x - here[0]))

    # ------------------------------------------------------------------ Nav2
    def send_goal(self, x, y, yaw, tag=None):
        if not self.nav.server_is_ready():
            self.get_logger().info('waiting for navigate_to_pose', throttle_duration_sec=5.0)
            return
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)
        self.goal = (x, y)
        self.goal_sent_at = self.get_clock().now()
        here = self.pose() or (x, y, 0.0)
        self.goal_deadline = (float(self.get_parameter('goal_timeout').value)
                              + math.hypot(x - here[0], y - here[1]) / 0.05)
        self.goals_sent += 1
        self.nav.send_goal_async(goal).add_done_callback(self.on_accepted)

    def on_accepted(self, fut):
        h = fut.result()
        if not h.accepted:
            self.get_logger().warn('goal rejected, blacklisting')
            self.fail(self.goal)
            self.goal = None
            return
        self.goal_handle = h
        h.get_result_async().add_done_callback(self.on_result)

    def on_result(self, fut):
        status = fut.result().status
        goal, self.goal, self.goal_handle = self.goal, None, None
        if self.homing:
            self.homing = False
            if status == 4:
                self.get_logger().info('home')
            elif self.home_tries < 3:
                self.get_logger().warn(f'homing ended with status {status}, trying again')
                self.create_timer(3.0, self._retry_home)
            else:
                self.get_logger().warn(f'homing ended with status {status}; staying here')
            return
        if goal is None:
            return
        if status == 4:                                   # SUCCEEDED
            self.get_logger().info(f'reached ({goal[0]:.2f}, {goal[1]:.2f})')
        elif status == 5:                                 # CANCELED, by step()
            pass
        else:                                             # ABORTED, or lost
            self.get_logger().warn(f'goal ({goal[0]:.2f}, {goal[1]:.2f}) failed with '
                                   f'status {status}, blacklisting')
            self.fail(goal)

    def cancel(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self.goal = None

    def fail(self, goal):
        if goal is not None:
            self.blacklist.append(goal)
            self.goals_failed += 1

    def start_spin(self):
        if not self.spin.server_is_ready():
            self.get_logger().info('waiting for the spin behavior', throttle_duration_sec=5.0)
            return
        self.spinning = True
        g = Spin.Goal()
        g.target_yaw = 2.0 * math.pi
        g.time_allowance.sec = 60
        self.get_logger().info('spinning once to see the whole ring first')
        self.spin.send_goal_async(g).add_done_callback(self.on_spin_accepted)

    def on_spin_accepted(self, fut):
        h = fut.result()
        if not h.accepted:
            # the behavior server answers before it is active and says no; try again
            self.spin_tries += 1
            self.spinning = False
            if self.spin_tries >= 10:
                self.get_logger().warn('spin rejected 10 times; exploring from the wedge')
                self.spun = True
            return
        h.get_result_async().add_done_callback(self.on_spin_done)

    def on_spin_done(self, fut):
        self.spun, self.spinning = True, False
        self.get_logger().info(f'spin done (status {fut.result().status})')

    def _retry_home(self):
        # a one-shot: rclpy timers repeat, so the first firing cancels it
        for t in list(self.timers):
            if t.callback is self._retry_home:
                t.cancel()
                self.destroy_timer(t)
        self.go_home()

    # ------------------------------------------------------------------ the end
    def finish(self, here):
        self.done = True
        info, grid = self.grid
        known = int((grid >= 0).sum()) * info.resolution ** 2
        free = int(((grid >= 0) & (grid <= FREE)).sum()) * info.resolution ** 2
        self.get_logger().info(
            f'explored: no frontier left. {known:.1f} m2 known, {free:.1f} m2 free, '
            f'{self.goals_sent} goals, {self.goals_failed} blacklisted')
        self.done_pub.publish(Bool(data=True))
        stem = self.get_parameter('save_map').value
        if self.save is not None and stem:
            if self.save.wait_for_service(timeout_sec=2.0):
                req = SaveMap.Request()
                req.name.data = stem
                # home AFTER the save: slam_toolbox writes the map on its own thread and
                # map -> odom stops meanwhile, and a goal sent then dies on "unable to
                # transform robot pose into global plan's frame"
                def saved(f):
                    self.get_logger().info(f'map saved to {stem}.pgm/.yaml')
                    self.go_home()
                self.save.call_async(req).add_done_callback(saved)
                return
            self.get_logger().warn('/slam_toolbox/save_map is not there; map not saved')
        self.go_home()

    def go_home(self):
        if not bool(self.get_parameter('home').value):
            return
        self.homing = True
        self.home_tries = getattr(self, 'home_tries', 0) + 1
        self.get_logger().info('walking home')
        self.send_goal(0.0, 0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = Explorer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
