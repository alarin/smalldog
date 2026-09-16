#!/usr/bin/env python3
"""
straight_test.py — command the real robot under ROS 2 and MEASURE what it did, from the
LiDAR and the IMU, not by eye.

    python3 tools/straight_test.py --vx 0.08 --seconds 6 --log /tmp/st1.npz    # a straight
    python3 tools/straight_test.py --wz 0.3 --seconds 4                        # a turn
    python3 tools/straight_test.py --vy 0.05 --seconds 4                       # a crab
    python3 tools/straight_test.py --replay /tmp/st1.npz
    python3 tools/straight_test.py --selftest

Needs `robot.launch.py imu:=true lidar:=true` — nothing from `smalldog_nav`. Nothing else
may be publishing /cmd_vel: the pad node repeats zeros at 20 Hz, so run without `joy:=true`.

The LiDAR measurement is a 2-D scan match: `--dwell` seconds of /lidar/points, moved into
base_footprint and cut to the 0.10..0.35 m band (the same slice `smalldog_nav` maps with),
before the command and again after it, and the rigid motion that lays one on the other.
It is independent of the gait's odometry and of slam_toolbox, whose pose leans on that
odometry between scan nodes — the first version of this tool read the map pose and
agreed with the IMU because both were reading the same integrated gyro.

Before it moves it reads the free distance in the direction of travel off the same slice
and refuses a run that would end closer than `--margin` to whatever is there.

What it reports:

  imu     heading change from the IMU's yaw over the command (deg), the gyro's mean rate
  lidar   the robot's turn (deg) and its displacement in the START body frame — forward,
          left — from the scan match, with the match residual as its own error bar
  odom    what the gait's dead reckoning (odom -> base_footprint) says for the same
          stretch, which against the LiDAR's forward distance is `odom_scale` measured

Sign convention everywhere: +x forward, +y left, +yaw counter-clockwise from above. A
+wz command must read a positive turn and a +vy command a positive left; the 2026-09-15
runs read both negative, which is how the mirrored leg map was found.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

Z_LO, Z_HI = 0.10, 0.35          # the slice, as nav.launch.py's cloud_to_scan
R_MIN, R_MAX = 0.30, 6.0


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _yaw(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _xyz(msg):
    step = msg.point_step
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, step)
    return np.stack([buf[:, 4 * i:4 * i + 4].copy().view(np.float32)[:, 0] for i in range(3)], 1)


# ----------------------------------------------------------------- scan match
def thin(p, cell=0.02):
    """one point per 2 cm cell: the fan oversamples near the sensor"""
    if len(p) == 0:
        return p
    k = np.floor(p / cell).astype(np.int64)
    _, idx = np.unique(k[:, 0] * 1_000_003 + k[:, 1], return_index=True)
    return p[np.sort(idx)]


def _icp_t(X, tree, Q, t, inlier, iters):
    """translation only, coarse to fine: the first passes accept matches out to 4x the
    inlier radius so a 30 cm walk is still found from t = 0"""
    for r in (4 * inlier, 2 * inlier, inlier):
        for _ in range(iters):
            d, j = tree.query(X + t)
            m = d < r
            if m.sum() < 20:
                break
            t = (Q[j[m]] - X[m]).mean(0)
    d, _ = tree.query(X + t)
    return t, float(np.mean(np.minimum(d, inlier))), float((d < inlier).mean())


def match(P, Q, yaw_range=60.0, step=0.5, inlier=0.15):
    """Rigid (theta, t) with Q ~ R(theta) P + t: brute force over theta, t by a few ICP
    steps at each, then 0.1 deg around the best. Returns (theta rad, t (2,), residual m,
    inlier fraction). P is the scene before, Q after, both in the robot's frame — so the
    ROBOT turned by -theta and moved by -R(-theta) t in its start frame."""
    from scipy.spatial import cKDTree
    tree = cKDTree(Q)

    def rotated(deg):
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        return th, P @ np.array([[c, s], [-s, c]])

    best = None
    for deg in np.arange(-yaw_range, yaw_range + 1e-6, step):
        th, X = rotated(deg)
        t, score, frac = _icp_t(X, tree, Q, np.zeros(2), inlier, 8)
        if best is None or score < best[0]:
            best = (score, th, t, frac)
    score, th, t, frac = best
    for deg in np.arange(math.degrees(th) - step, math.degrees(th) + step + 1e-6, 0.1):
        th2, X = rotated(deg)
        t2, sc, fr = _icp_t(X, tree, Q, t.copy(), inlier, 4)
        if sc < score:
            score, th, t, frac = sc, th2, t2, fr
    return th, t, score, frac


def robot_motion(P, Q):
    """(turn rad, forward m, left m, residual m, inlier fraction) of the robot between the
    two slices, in the robot's START frame."""
    th, t, res, frac = match(P, Q)
    turn = -th
    c, s = math.cos(turn), math.sin(turn)
    mv = -np.array([c * t[0] - s * t[1], s * t[0] + c * t[1]])
    return turn, float(mv[0]), float(mv[1]), res, frac


# --------------------------------------------------------------------- report
def analyse(r: dict) -> dict:
    out = dict(seconds=float(r["seconds"]), vx=float(r["vx"]), vy=float(r["vy"]), wz=float(r["wz"]))
    yaw, ok = r["imu_yaw"], np.isfinite(r["imu_yaw"])
    if ok.sum() > 2:
        seg = yaw[ok]
        out["imu_yaw_deg"] = math.degrees(_wrap(seg[-1] - seg[0]))
        out["gyro_z_mean_dps"] = math.degrees(float(np.nanmean(r["gyro_z"])))
    P, Q = r["slice0"], r["slice1"]
    if len(P) > 50 and len(Q) > 50:
        turn, fwd, left, res, frac = robot_motion(P, Q)
        out["lidar"] = dict(turn_deg=math.degrees(turn), fwd=fwd, left=left, res=res, frac=frac,
                            n0=len(P), n1=len(Q))
    od = r["odom"]
    ok = np.isfinite(od[:, 0])
    if ok.sum() > 2:
        a, b = od[ok][0], od[ok][-1]
        c, s = math.cos(a[2]), math.sin(a[2])
        dx, dy = b[0] - a[0], b[1] - a[1]
        out["odom"] = dict(fwd=c * dx + s * dy, left=-s * dx + c * dy, turn_deg=math.degrees(_wrap(b[2] - a[2])))
    if "lidar" in out and "odom" in out and abs(out["odom"]["fwd"]) > 0.02:
        out["odom_scale_measured"] = out["lidar"]["fwd"] / out["odom"]["fwd"]
    return out


def report(o: dict) -> str:
    cmd = f"vx {o['vx']:+.2f} vy {o['vy']:+.2f} wz {o['wz']:+.2f} for {o['seconds']:.1f} s"
    s = [f"commanded  {cmd}  ({o['seconds'] * o['vx']:+.2f} m forward, {o['seconds'] * o['vy']:+.2f} m left, "
         f"{math.degrees(o['seconds'] * o['wz']):+.0f} deg)"]
    if "imu_yaw_deg" in o:
        s.append(f"imu        turned {o['imu_yaw_deg']:+.1f} deg; gyro z mean {o['gyro_z_mean_dps']:+.2f} deg/s")
    if "lidar" in o:
        m = o["lidar"]
        s.append(f"lidar      turned {m['turn_deg']:+.1f} deg, moved {m['fwd']:+.3f} m forward, {m['left']:+.3f} m left "
                 f"(match residual {m['res'] * 100:.1f} cm, {m['frac'] * 100:.0f} % inliers, {m['n0']}/{m['n1']} pts)")
    else:
        s.append("lidar      no slice (is the L2 up? robot.launch.py lidar:=true)")
    if "odom" in o:
        m = o["odom"]
        s.append(f"odom       turned {m['turn_deg']:+.1f} deg, moved {m['fwd']:+.3f} m forward, {m['left']:+.3f} m left"
                 + (f"; odom_scale measured {o['odom_scale_measured']:.2f}" if "odom_scale_measured" in o else ""))
    if "lidar" in o:
        L = o["lidar"]
        # the 2-D match searches +-60 deg of yaw: past that, or under 70 % inliers, it
        # lands on the wrong wall (a +103 deg turn read -47 at 51 %, 2026-09-16) - the
        # sign check then goes to the IMU
        weak = L["frac"] < 0.70 or abs(L["turn_deg"]) > 60.0
        if weak:
            s.append(f"lidar      match is weak ({L['frac'] * 100:.0f} % inliers, {L['turn_deg']:+.0f} deg): "
                     "read the IMU line for the turn")
        turn = o.get("imu_yaw_deg", L["turn_deg"]) if weak else L["turn_deg"]
        if abs(o["wz"]) > 0.05 and turn * o["wz"] < 0:
            s.append("!! turned AGAINST the commanded wz")
        if abs(o["vy"]) > 0.01 and L["left"] * o["vy"] < 0:
            s.append("!! crabbed AGAINST the commanded vy")
        if abs(o["vx"]) > 0.01 and L["fwd"] * o["vx"] < 0:
            s.append("!! walked AGAINST the commanded vx")
    return "\n".join(s)


# ------------------------------------------------------------------ live
def run(a) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Imu, PointCloud2
    import tf2_ros

    rclpy.init()
    node = Node("straight_test")
    state = dict(imu=None, cloud=None)
    node.create_subscription(Imu, "/imu", lambda m: state.__setitem__("imu", m), 10)
    node.create_subscription(PointCloud2, a.cloud, lambda m: state.__setitem__("cloud", m),
                             qos_profile_sensor_data)
    pub = node.create_publisher(Twist, "/cmd_vel", 10)
    # the lidar node parks the L2 after `idle_stop` s without a /cmd_vel, and this tool
    # waits for a cloud before it commands anything: hold the head on. /lidar/spin is
    # an override with no "release" (lidar_node.py), so it stays on after the run —
    # which is what SLAM wants anyway
    from std_msgs.msg import Bool
    spin_pub = node.create_publisher(Bool, "/lidar/spin", 10)
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, node)

    def spin(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.01)

    def pose(parent, child):
        try:
            tr = buf.lookup_transform(parent, child, Time())
        except Exception:
            return (math.nan,) * 3
        return tr.transform.translation.x, tr.transform.translation.y, _yaw(tr.transform.rotation)

    def slice_now(seen):
        """the latest cloud, in base_footprint, cut to the band -> (n, 2); None if stale"""
        m = state["cloud"]
        if m is None or id(m) in seen:
            return None
        seen.add(id(m))
        try:
            t = buf.lookup_transform(a.base_frame, m.header.frame_id, Time())
        except Exception:
            return None
        p = _xyz(m)
        tr = t.transform.translation
        p = p @ _rot(t.transform.rotation).T + np.array([tr.x, tr.y, tr.z])
        p = p[(p[:, 2] >= Z_LO) & (p[:, 2] <= Z_HI)][:, :2]
        r = np.hypot(p[:, 0], p[:, 1])
        return p[(r >= R_MIN) & (r <= R_MAX)]

    def dwell(seconds):
        """accumulate `seconds` of slices with the robot standing"""
        pts, seen = [], set()
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.01)
            s = slice_now(seen)
            if s is not None and len(s):
                pts.append(s)
        return thin(np.concatenate(pts)) if pts else np.zeros((0, 2))

    # discovery: with Nav2 up beside the robot launch the first cloud can take > 5 s to arrive
    end = time.monotonic() + 12.0
    while time.monotonic() < end and (state["imu"] is None or state["cloud"] is None):
        spin_pub.publish(Bool(data=True))       # a parked head takes ~3 s to come back
        rclpy.spin_once(node, timeout_sec=0.5)
    spin(1.0)                                   # and a second for /tf_static
    if state["imu"] is None:
        print("!! no /imu — robot.launch.py imu:=true")
    if state["cloud"] is None:
        print(f"!! no {a.cloud} — robot.launch.py lidar:=true")
        if not a.blind:
            rclpy.shutdown()
            return 1

    print(f"standing {a.dwell:.1f} s for the first slice ...")
    P = dwell(a.dwell)
    if len(P) and (a.vx or a.vy) and not a.watch:
        heading = math.atan2(a.vy, a.vx)
        if abs(heading) > math.radians(84.0):
            print("!! the L2 sees +-96 deg forward only: the direction of travel is out of its "
                  "view, the free-space check is BLIND")
        ang = np.arctan2(P[:, 1], P[:, 0]) - heading
        sel = np.abs((ang + np.pi) % (2 * np.pi) - np.pi) < math.radians(12.0)
        free = float(np.hypot(P[sel, 0], P[sel, 1]).min()) if sel.any() else math.inf
        speed = math.hypot(a.vx, a.vy)
        print(f"free {free:.2f} m in the direction of travel (+-12 deg, {len(P)} pts in the slice); "
              f"the run wants {a.seconds * speed:.2f} m + {a.margin:.2f} margin")
        if free < a.seconds * speed + a.margin:
            print(f"!! too close: refusing. --seconds {max(0.0, (free - a.margin) / speed):.0f} would fit")
            rclpy.shutdown()
            return 1

    rows = []
    t0 = time.monotonic()
    tw = Twist()

    def sample():
        m = state["imu"]
        rows.append((time.monotonic() - t0,
                     _yaw(m.orientation) if m else math.nan,
                     m.angular_velocity.z if m else math.nan,
                     *pose(a.odom_frame, a.base_frame)))

    def phase(seconds, vx, vy, wz):
        tw.linear.x, tw.linear.y, tw.angular.z = float(vx), float(vy), float(wz)
        end = time.monotonic() + seconds
        nxt = time.monotonic()
        while time.monotonic() < end:
            pub.publish(tw)
            sample()
            nxt += 1.0 / a.rate
            while time.monotonic() < nxt:
                rclpy.spin_once(node, timeout_sec=0.005)

    if a.watch:
        # somebody else is driving (robot/runtime/policy.py on the bus, this launch in
        # dry_run for the L2 and TF): no command, just the two slices around the walk
        print(f"watching {a.watch:.1f} s, no command published ...")
        end = time.monotonic() + a.watch
        while time.monotonic() < end:
            sample()
            rclpy.spin_once(node, timeout_sec=0.05)
    else:
        print(f"commanding vx {a.vx:+.2f} vy {a.vy:+.2f} wz {a.wz:+.2f} for {a.seconds:.1f} s, then {a.stop:.1f} s stop")
        phase(a.seconds, a.vx, a.vy, a.wz)
        phase(a.stop, 0.0, 0.0, 0.0)
    print(f"standing {a.dwell:.1f} s for the second slice ...")
    Q = dwell(a.dwell)
    rclpy.shutdown()

    arr = np.array(rows, float)
    r = dict(t=arr[:, 0], imu_yaw=arr[:, 1], gyro_z=arr[:, 2], odom=arr[:, 3:6],
             slice0=P, slice1=Q, seconds=a.seconds, vx=a.vx, vy=a.vy, wz=a.wz)
    if a.log:
        np.savez(a.log, **r)
        print(f"log -> {a.log}")
    print(report(analyse(r)))
    return 0


# ------------------------------------------------------------------ selftest
def selftest() -> int:
    """a synthetic room, the robot moved by a known rigid motion, the match must find it"""
    rng = np.random.default_rng(0)
    wall = np.concatenate([np.stack([np.linspace(-1.5, 2.5, 400), np.full(400, 1.5)], 1),
                           np.stack([np.linspace(-1.5, 2.5, 400), np.full(400, -1.5)], 1),
                           np.stack([np.full(300, 2.5), np.linspace(-1.5, 1.5, 300)], 1),
                           np.stack([np.full(300, -1.5), np.linspace(-1.5, 1.5, 300)], 1),
                           np.array([[1.2, 0.4], [1.25, 0.4], [1.2, 0.45], [1.25, 0.45]])])
    fails = 0
    for turn_deg, fwd, left in [(0.0, 0.0, 0.0), (25.0, 0.10, 0.0), (-30.0, 0.30, -0.05), (5.0, 0.0, 0.08)]:
        th = math.radians(turn_deg)
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]])
        Q = (wall - np.array([fwd, left])) @ R          # the room seen from the moved robot
        P = wall + rng.normal(0, 0.01, wall.shape)
        Q = Q + rng.normal(0, 0.01, Q.shape)
        got = robot_motion(thin(P), thin(Q))
        ok = abs(math.degrees(got[0]) - turn_deg) < 0.6 and abs(got[1] - fwd) < 0.02 and abs(got[2] - left) < 0.02
        fails += not ok
        print(f"  {'ok ' if ok else 'FAIL'} truth turn {turn_deg:+.1f} fwd {fwd:+.2f} left {left:+.2f}  ->  "
              f"got {math.degrees(got[0]):+.1f} / {got[1]:+.3f} / {got[2]:+.3f}  (res {got[3] * 100:.1f} cm)")
    print("selftest:", "ok" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vx", type=float, default=0.0, help="m/s forward")
    ap.add_argument("--vy", type=float, default=0.0, help="m/s left")
    ap.add_argument("--wz", type=float, default=0.0, help="rad/s counter-clockwise")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--stop", type=float, default=2.0, help="s of zero command before the second slice")
    ap.add_argument("--dwell", type=float, default=1.5, help="s of cloud in each slice")
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--margin", type=float, default=0.5, help="m to keep between the end and the wall")
    ap.add_argument("--blind", action="store_true", help="run without the LiDAR")
    ap.add_argument("--watch", type=float, default=0.0, metavar="S",
                    help="publish nothing; slice, wait S s while something else walks the "
                         "robot (policy.py with this launch in dry_run), slice again. --vx/--vy/--wz "
                         "then only label the report")
    ap.add_argument("--cloud", default="/lidar/points")
    ap.add_argument("--odom-frame", default="odom")
    ap.add_argument("--base-frame", default="base_footprint")
    ap.add_argument("--log", metavar="FILE.npz")
    ap.add_argument("--replay", metavar="FILE.npz", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.replay:
        for p in a.replay:
            r = dict(np.load(p))
            print(f"== {p}")
            print(report(analyse(r)))
        return 0
    if not (a.vx or a.vy or a.wz or a.watch):
        ap.error("give a command: --vx, --vy and/or --wz (or --watch)")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
