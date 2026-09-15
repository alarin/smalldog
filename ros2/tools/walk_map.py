#!/usr/bin/env python
"""
walk_map.py — does the L2's LiDAR odometry survive a trotting robot?

    python tools/walk_map.py --selftest            # 4 s, checks the harness itself
    python tools/walk_map.py --path still          # the control: stand, register, drift?
    python tools/walk_map.py --path straight       # 2 m in a straight line
    python tools/walk_map.py --path spin           # 360 deg on the spot
    python tools/walk_map.py --path patrol         # both rooms and the doorway (headline)
    python tools/walk_map.py --path patrol --rigid # ... with motion distortion switched off
    python tools/walk_map.py --path patrol --map room.pcd --log run.npz

THE QUESTION
`robot/slam/slam.py` registers the L2's frames with KISS-ICP and accumulates a voxel map.
Its own selftest walks a synthetic room at a constant 0.1 m/s on a perfectly smooth
trajectory, which is not what this robot is: a trot bobs the body at ~2.2 Hz, pitches it
either side of level every stride, and yaws at up to 1.2 rad/s. This measures it against
that. The answer, and what follows from it, is in `robot/README.md`, "SLAM".

**This is not the robot's mapping stack.** `ros2/smalldog_nav` is — 2-D scan into
slam_toolbox with the gait's own odometry, and it explores the room by itself. This bench
measures the *other* frontend, the 3-D one in `robot/slam/`, which nothing is built on.
Keeping the measurement is how that stays a decision rather than an accident.

It trots the REAL gait (`smalldog_walker.gait`, the same one the robot runs) through the
REAL room (`mujoco/scene_room.xml`, the same one `tools/explore.sh` uses) casting the REAL
measured scan pattern (`3d/lidar.py`), hands the frames to the REAL odometry
(`robot/slam/slam.py`), and scores what comes back against MuJoCo's own truth. Nothing
here reimplements any of those four, and — see `Room` below — nothing here keeps a second
copy of the room either.

MOTION DISTORTION IS MODELLED HERE, AND IS NOT MODELLED IN lidar.py
`Scanner.scan()` casts every ray of a frame from the sensor pose at the END of that
frame's window — its docstring says so, and says what to do instead: "cast in chunks per
sim step and accumulate". That is what `capture_frame()` below does. Each 1/12 s frame is
built from `--slice-ms` sub-scans, each cast from the pose the robot actually had at that
instant, and the pieces are concatenated **in the sensor frame** — which is precisely the
distortion a real sweeping lidar hands its consumer, because the consumer treats one frame
as rigid and it is not. `--rigid` restores the end-of-frame behaviour as a control arm,
and the gap between the two arms IS the cost of not deskewing.

WHAT IS HONEST HERE AND WHAT IS NOT
- **The navigation cheats, deliberately.** `Patrol` steers on MuJoCo's ground-truth pose,
  not on the map. This bench measures perception; a robot that wandered into a wall
  because its own estimate was bad would confound the measurement with a story about
  navigation. `smalldog_nav` is the thing that does it for real.
- **The accelerometer is gravity only.** `slam.py` uses it for one thing — levelling the
  world frame off a second of frames with the robot standing still — and at rest gravity
  is all a real one reads. A trotting robot's accelerometer reads specific force and this
  bench does not model that; nothing downstream of levelling looks at it. If anything ever
  fuses that channel, this is a lie that will need fixing first.
- **Self-hits are in the cloud and out of the map score.** The sensor sees the robot's own
  legs (`lidar.py` excludes the chassis it is bolted to and nothing else, on purpose), so
  they are registered exactly as they would be on hardware. They are not room surfaces
  though, so scoring the map against the room has to know which points they were: the
  bench tags them in the intensity channel, which the sim has no other use for
  (`lidar.py`: "NO INTENSITY, deliberately") and which `Odometry` already carries through
  into the map. `--keep-self` scores without the filter, to show what it is worth.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
REPO = os.path.dirname(WS)
CAD = os.path.join(REPO, "3d")
sys.path.insert(0, os.path.join(WS, "smalldog_walker"))
sys.path.insert(0, CAD)
sys.path.insert(0, os.path.join(REPO, "robot"))

import mujoco                                                      # noqa: E402
import lidar as lidar_gen                                          # noqa: E402
from slam.slam import Odometry, level_rotation, write_pcd, yaw_deg  # noqa: E402
from slam import slam as slam_mod                                  # noqa: E402
from smalldog_walker.gait import TrotGait                          # noqa: E402

MJCF = os.path.join(WS, "smalldog_description", "mujoco")
PARAMS = os.path.join(WS, "smalldog_description", "robot_params.json")

SPEED = 0.20        # m/s, the commanded trot — standalone_sim's own regression speed
TURN = 1.2          # rad/s on the spot, likewise
SLICE_MS = 7.0      # sub-scan length; 12 of them to a 83 ms frame.  At 0.2 m/s that is
                    # 1.4 mm of travel inside one slice, well under the 5 cm map voxel,
                    # so the residual un-modelled smear is not what any number here reads.
LEVEL_S = 1.5       # standing still before the run: slam.py wants LEVEL_FRAMES of rest


# ======================================================================================
# the room, read out of the compiled model
# ======================================================================================
class Room:
    """The world's static geometry, straight from MuJoCo, as signed distance functions.

    THIS FILE KEEPS NO COPY OF THE ROOM. `scene_room.xml` is written by
    `smalldog_description/scripts/generate_model.py` and is what `tools/explore.sh` and
    `smalldog_nav` drive the robot around; a second description of the same walls here
    would drift from it the first time anybody moved a crate, and then the map score would
    be measured against a room that does not exist. So the surfaces come out of the model
    that was actually simulated: every geom on the worldbody, by type.

    What that costs is generality of shape — a box, a plane, a sphere, a cylinder and a
    capsule have exact signed distances in a few lines each, and a mesh or a heightfield
    does not. `missing` names any geom this cannot score, so a scene that grows one is a
    printed warning rather than a silently optimistic number.
    """

    def __init__(self, model, data):
        import mujoco
        G = mujoco.mjtGeom
        self.solids, self.missing = [], []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != 0:          # the robot is not part of the room
                continue
            kind = int(model.geom_type[g])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
            pos = np.array(data.geom_xpos[g], dtype=float)
            mat = np.array(data.geom_xmat[g], dtype=float).reshape(3, 3)
            size = np.array(model.geom_size[g], dtype=float)
            if kind in (int(G.mjGEOM_PLANE), int(G.mjGEOM_BOX), int(G.mjGEOM_SPHERE),
                        int(G.mjGEOM_CYLINDER), int(G.mjGEOM_CAPSULE)):
                self.solids.append((kind, name, pos, mat, size))
            else:
                self.missing.append(name)

    # ----------------------------------------------------------------------------------
    @staticmethod
    def _sdf(kind, pos, mat, size, p):
        import mujoco
        G = mujoco.mjtGeom
        q = (p - pos) @ mat                        # into the geom's own frame
        if kind == int(G.mjGEOM_PLANE):
            return q[:, 2]                         # infinite, normal +Z
        if kind == int(G.mjGEOM_SPHERE):
            return np.linalg.norm(q, axis=1) - size[0]
        if kind == int(G.mjGEOM_BOX):
            d = np.abs(q) - size[:3]
            return (np.linalg.norm(np.maximum(d, 0.0), axis=1)
                    + np.minimum(np.max(d, axis=1), 0.0))
        r, h = size[0], size[1]
        radial = np.linalg.norm(q[:, :2], axis=1) - r
        if kind == int(G.mjGEOM_CAPSULE):          # hemispherical caps
            axial = np.abs(q[:, 2]) - h
            body = np.stack([radial, axial], 1)
            outside = np.linalg.norm(np.maximum(body, 0.0), axis=1)
            return np.where(axial <= 0, radial, outside + np.minimum(np.max(body, 1), 0.0))
        axial = np.abs(q[:, 2]) - h                # cylinder, flat caps
        d = np.stack([radial, axial], 1)
        return (np.linalg.norm(np.maximum(d, 0.0), axis=1)
                + np.minimum(np.max(d, axis=1), 0.0))

    def surface_distance(self, pts):
        """Distance from each point to the NEAREST TRUE SURFACE of the room, metres.

        The map's error measure, and deliberately independent of the pose the map was
        built from: a point that landed on a wall scores 0 whether the odometry knew where
        it was or not, so a large value here is smear and mis-registration rather than
        drift. NOT a completeness measure — a map of one wall and nothing else scores
        perfectly."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        if not len(pts) or not self.solids:
            return np.zeros(len(pts))
        best = np.full(len(pts), np.inf)
        for kind, _, pos, mat, size in self.solids:
            best = np.minimum(best, np.abs(self._sdf(kind, pos, mat, size, pts)))
        return best

    def clearance_xy(self, x, y, z=0.12):
        """Horizontal room the robot has at (x, y), negative inside something.

        Evaluated at `z` — knee height — so a doorway's lintel or a table the robot walks
        under is not read as a wall. The floor plane is excluded for the same reason."""
        import mujoco
        p = np.array([[float(x), float(y), float(z)]])
        best = np.inf
        for kind, _, pos, mat, size in self.solids:
            if kind == int(mujoco.mjtGeom.mjGEOM_PLANE):
                continue
            best = min(best, float(self._sdf(kind, pos, mat, size, p)[0]))
        return best

    def clearance_along(self, waypoints, step=0.02):
        """(worst clearance, where) over every segment of a closed-ended path."""
        worst, at = np.inf, None
        for (x0, y0), (x1, y1) in zip(waypoints, waypoints[1:]):
            n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step))
            for i in range(n + 1):
                t = i / n
                x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
                c = self.clearance_xy(x, y)
                if c < worst:
                    worst, at = c, (x, y)
        return worst, at

    def describe(self):
        kinds = {}
        for kind, _, _, _, _ in self.solids:
            kinds[kind] = kinds.get(kind, 0) + 1
        return (f"{len(self.solids)} static geoms"
                + (f", {len(self.missing)} unscorable ({', '.join(self.missing)})"
                   if self.missing else ""))


# ======================================================================================
# the world, and the walker in it
# ======================================================================================
def load(scene="scene_room.xml", seed=0):
    params = json.load(open(PARAMS))
    model = mujoco.MjModel.from_xml_path(os.path.join(MJCF, scene))
    data = mujoco.MjData(model)
    gait = TrotGait(params)
    act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in gait.joint_names]
    if -1 in act:
        raise RuntimeError("actuator/joint name mismatch between MJCF and robot_params.json")
    sc = lidar_gen.Scanner(model, seed=seed)
    if not sc.ok:
        raise SystemExit(f"walk_map: {sc.missing}"
                         "  (run smalldog_description/scripts/generate_model.py)")
    mujoco.mj_forward(model, data)          # geom_xpos/xmat, which Room reads
    return model, data, gait, act, sc, params, Room(model, data)


def sensors(model):
    """The sensors the gait feeds on — same set and same names as standalone_sim.py."""
    def adr(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return model.sensor_adr[i] if i >= 0 else -1
    return dict(quat=adr("imu_quat"), gyro=adr("imu_gyro"),
                contact={l: adr(f"{l}_contact") for l in ("fl", "fr", "rl", "rr")})


def feed(gait, data, sens, blind=False):
    if blind or sens["quat"] < 0:
        return
    q, g = sens["quat"], sens["gyro"]
    gait.feedback(quat=tuple(data.sensordata[q:q + 4]),
                  gyro=tuple(data.sensordata[g:g + 3]),
                  contact={l: data.sensordata[a] > 1e-6
                           for l, a in sens["contact"].items() if a >= 0})


def body_pose(data):
    """(x, y, yaw) of the floating base, from the sim's truth."""
    x, y = float(data.qpos[0]), float(data.qpos[1])
    w, qx, qy, qz = (float(v) for v in data.qpos[3:7])
    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return x, y, yaw


def settle(model, data, gait, act, seconds, sens, blind=False):
    """Hold the stance while physics settles.  dt = 1.0 for the single command, for the
    reason standalone_sim.settle() spells out: a dt of 0 leaves the gait rate-limited to
    3e-4 rad off zero and the robot 'settles' with its legs straight."""
    q = gait.joint_targets(1.0, 0.0, 0.0, 0.0)
    for i, a in enumerate(act):
        data.ctrl[a] = q[i]
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        feed(gait, data, sens, blind)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Patrol:
    """Waypoint follower on the SIM'S OWN TRUTH — see the header, this cheats on purpose.

    Turn on the spot until the bearing is within `tol`, then walk, trimming heading as it
    goes.  Turn-then-walk rather than a curve because that is what the explorer this
    unblocks will do, and because a 90 deg corner walked as an arc puts the yaw rate and
    the forward speed into one number that cannot afterwards be told apart."""

    def __init__(self, waypoints, speed=SPEED, turn=TURN, reach=0.25, tol=math.radians(8)):
        self.wp = list(waypoints)
        self.speed, self.turn, self.reach, self.tol = speed, turn, reach, tol
        self.i = 0
        self.turned = 0.0            # |yaw| commanded on the spot, radians, for the report

    @property
    def done(self):
        return self.i >= len(self.wp)

    def command(self, x, y, yaw, dt):
        if self.done:
            return (0.0, 0.0, 0.0)
        tx, ty = self.wp[self.i]
        if math.hypot(tx - x, ty - y) < self.reach:
            self.i += 1
            return self.command(x, y, yaw, dt)
        err = wrap(math.atan2(ty - y, tx - x) - yaw)
        wz = max(-self.turn, min(self.turn, 2.0 * err))
        if abs(err) > self.tol:
            self.turned += abs(wz) * dt
            return (0.0, 0.0, wz)
        return (self.speed, 0.0, 0.5 * wz)


#: The tour, for `--path patrol`.  `scene_room.xml` is two chambers joined by a doorway at
#: x = 2.5, so this is a circuit of the near room, out through the door, a circuit of the
#: far one and back — which is also roughly what `smalldog_nav`'s explorer ends up doing,
#: and it is the trajectory that matters most here because the doorway is where a frontend
#: has least to register against.  Hand-placed, but never hand-verified: `Room` measures
#: the clearance out of the compiled model and the selftest fails if it is under PATROL_CLEAR.
PATROL = [(1.90, -0.90), (1.90, 1.20), (0.00, 1.20), (-1.05, 0.30), (-1.05, -0.55),
          (1.00, -0.55), (2.60, 0.00), (4.15, -0.90), (4.15, 0.15), (3.20, 0.15),
          (2.40, 0.00), (0.00, 0.00)]
PATROL_CLEAR = 0.25       # m of room either side, at knee height, everywhere on the tour


def path_plan(name, laps):
    """(waypoints, seconds cap, what it is for)."""
    if name == "still":
        return [], 8.0, "standing still — the control arm: any drift here is ICP noise"
    if name == "straight":
        return [(2.0, 0.0)], 30.0, "2 m in a straight line"
    if name == "spin":
        return [], 8.0, "360 deg on the spot"
    if name == "patrol":
        return PATROL * laps, 40.0 + 260.0 * laps, (f"{laps} tour(s) of both rooms"
                                                    " and the doorway between them")
    raise ValueError(name)


# ======================================================================================
# one lidar frame, as the sensor would actually deliver it
# ======================================================================================
def capture_frame(model, data, gait, act, sens, sc, cmd, steps, slice_steps, rigid, blind):
    """Advance the sim one lidar frame and return (xyzi, acc, truth 4x4, n_rays).

    The cloud is the concatenation of sub-scans cast as the robot moves through the frame,
    each in the sensor frame AT THAT INSTANT — so a consumer that treats the frame as
    rigid (every one of them does) sees exactly the smear a real sweeping lidar produces.
    `rigid` casts the whole frame from the end pose instead, which is lidar.Scanner's own
    behaviour and the control arm for the deskew question."""
    t_start = data.time
    pieces, rays = [], 0
    t_slice = t_start
    done = 0
    while done < steps:
        n = min(slice_steps, steps - done)
        for _ in range(n):
            q = gait.joint_targets(model.opt.timestep, *cmd)
            for i, a in enumerate(act):
                data.ctrl[a] = q[i]
            mujoco.mj_step(model, data)
            feed(gait, data, sens, blind)
        done += n
        if not rigid:
            c = sc.scan(model, data, t_slice, data.time)
            pieces.append(c)
            rays += c["n_rays"]
            t_slice = data.time
    if rigid:
        c = sc.scan(model, data, t_start, data.time)
        pieces = [c]
        rays = c["n_rays"]

    local = np.concatenate([p["local"] for p in pieces]) if pieces else np.zeros((0, 3))
    geom = np.concatenate([p["geom"] for p in pieces]) if pieces else np.zeros(0, int)
    # provenance in the intensity channel: 1 = a surface of the room, 0 = the robot itself
    inten = np.where(model.geom_bodyid[geom] != 0, 0.0, 1.0) if len(geom) else np.zeros(0)
    xyzi = np.empty((len(local), 4), np.float32)
    xyzi[:, :3] = local
    xyzi[:, 3] = inten

    R = np.array(data.site_xmat[sc.site], dtype=float).reshape(3, 3)
    truth = np.eye(4)
    truth[:3, :3] = R
    truth[:3, 3] = np.array(data.site_xpos[sc.site], dtype=float)
    acc = R.T @ np.array([0.0, 0.0, 9.81])          # gravity only — see the header
    return xyzi, acc, truth, rays


class VoxelMap:
    """The same map `Odometry` keeps, grown from a pose handed in rather than estimated.

    Only here so the run can build the map a second time from MuJoCo's truth: the gap
    between the two maps' scores is the odometry's contribution, and what is left is the
    sensor, the noise and the smear."""

    def __init__(self, voxel, min_range, max_range):
        self.voxel, self.min_range, self.max_range = voxel, min_range, max_range
        self.keys = set()
        self.pts = []
        self.n = 0

    def add(self, xyzi, pose):
        pts = np.asarray(xyzi[:, :3], dtype=float)
        r = np.linalg.norm(pts, axis=1)
        keep = (r > self.min_range) & (r < self.max_range)
        pts, inten = pts[keep], np.asarray(xyzi[keep, 3], dtype=float)
        if not len(pts):
            return
        world = pts @ pose[:3, :3].T + pose[:3, 3]
        k = np.floor(world / self.voxel).astype(np.int64)
        key = (k[:, 0] * 73856093) ^ (k[:, 1] * 19349669) ^ (k[:, 2] * 83492791)
        key, first = np.unique(key, return_index=True)
        fresh = np.fromiter((kk not in self.keys for kk in key.tolist()), bool, len(key))
        idx = first[fresh]
        self.keys.update(key[fresh].tolist())
        out = np.empty((len(idx), 4), np.float32)
        out[:, :3] = world[idx]
        out[:, 3] = inten[idx]
        self.pts.append(out)
        self.n += len(out)

    def map(self):
        return np.concatenate(self.pts) if self.pts else np.zeros((0, 4), np.float32)


def _B(R):
    b = np.eye(4)
    b[:3, :3] = R
    return b


def score_map(room, pts, to_world, keep_self=False):
    """(n scored, p50, rms, p95) of the distance from each map point to the true room.

    `to_world` is needed and is easy to forget: the map is in the NODE's frame — levelled,
    and with its origin at the sensor's pose on the first registered frame — while the room
    is in MuJoCo's world.  Scoring one against the other without it reads the offset
    between the two origins as map error, which is a quarter of a metre of nothing."""
    if not len(pts):
        return 0, 0.0, 0.0, 0.0
    sel = pts if keep_self else pts[pts[:, 3] > 0.5]
    if not len(sel):
        return 0, 0.0, 0.0, 0.0
    w = sel[:, :3] @ to_world[:3, :3].T + to_world[:3, 3]
    d = room.surface_distance(w)
    return len(d), float(np.median(d)), float(np.sqrt((d ** 2).mean())), float(np.percentile(d, 95))


# ======================================================================================
def run(a):
    model, data, gait, act, sc, params, room = load(a.scene, a.seed)
    sens = sensors(model)
    gait.period, gait.swing_height = a.period, a.swing
    gait.max_step = a.max_step
    gait.body_height = a.height

    steps = int(round((1.0 / sc.frame_hz) / model.opt.timestep))
    slice_steps = max(1, int(round(a.slice_ms / 1000.0 / model.opt.timestep)))
    frame_dt = steps * model.opt.timestep
    waypoints, cap, what = path_plan(a.path, a.laps)
    seconds = a.seconds if a.seconds else cap

    print(f"walk_map: {what}")
    print(f"  room    {os.path.basename(a.scene)}: {room.describe()}")
    print(f"  sensor  cone {math.degrees(sc.cone):.0f} deg, {sc.rate:g} pts/s,"
          f" {sc.frame_hz:g} Hz -> {frame_dt * 1000:.1f} ms frames of"
          f" {'ONE cast at the end pose (rigid)' if a.rigid else f'{steps // slice_steps} sub-scans'}")
    if a.stack > 1:
        print(f"  stack   {a.stack} frames per registration -> a {sc.frame_hz / a.stack:.1f} Hz pose")
    print(f"  gait    {a.speed:g} m/s, {a.turn:g} rad/s, period {gait.period:g} s,"
          f" {'blind' if a.blind else 'IMU + foot contact'}")

    settle(model, data, gait, act, 1.5, sens, a.blind)
    pat = Patrol(waypoints, a.speed, a.turn)

    odo = Odometry(a.max_range, a.min_range, a.voxel, a.threshold, a.map_voxel, a.threads)
    tmap = VoxelMap(odo.map_voxel, odo.cfg.data.min_range, odo.cfg.data.max_range)

    R0 = None
    ups = []
    stack = []                                  # levelled clouds awaiting registration
    # The truth trajectory is accumulated every frame.  Reading it off the REGISTERED
    # frames instead makes --stack 4 report a quarter of the samples and so a shorter
    # walk and less turning, which is a property of the reporting and not of the robot.
    walked, turned, last_xy, last_yaw = 0.0, 0.0, None, None
    est, rel, ms, rays_pf, ret_pf = [], [], [], [], []
    truth0 = None
    n_frames = 0
    spin_left = 2 * math.pi if a.path == "spin" else 0.0
    t_level_end = LEVEL_S
    t_wall = time.perf_counter()

    while data.time < seconds:
        # --- what to command this frame
        if data.time < t_level_end:
            cmd = (0.0, 0.0, 0.0)                  # standing still, for the levelling
        elif a.path == "still":
            cmd = (0.0, 0.0, 0.0)
        elif a.path == "spin":
            cmd = (0.0, 0.0, a.turn) if spin_left > 0 else (0.0, 0.0, 0.0)
            spin_left -= a.turn * frame_dt
        else:
            x, y, yaw = body_pose(data)
            cmd = pat.command(x, y, yaw, frame_dt)

        xyzi, acc, truth, rays = capture_frame(model, data, gait, act, sens, sc, cmd,
                                               steps, slice_steps, a.rigid, a.blind)
        bx, by, byaw = body_pose(data)
        if last_xy is not None:
            walked += math.hypot(bx - last_xy[0], by - last_xy[1])
            turned += wrap(byaw - last_yaw)      # SIGNED, so the trot's per-stride yaw
        last_xy, last_yaw = (bx, by), byaw       # wobble cancels and only net turning counts
        n_frames += 1
        rays_pf.append(rays)
        ret_pf.append(len(xyzi))

        # --- the node's own pipeline, in the order slam.run() does it
        if R0 is None:
            ups.append(acc)
            if len(ups) < slam_mod.LEVEL_FRAMES:
                continue
            up = np.median(np.array(ups), axis=0)
            R0 = level_rotation(up, (0.0, 0.0, 1.0))
            tilt = math.degrees(math.acos(min(1.0, R0[2] @ np.array([0, 0, 1.0]))))
            print(f"  levelled after {len(ups)} frames: tilt {tilt:.1f} deg")
            truth0 = truth.copy()

        lvl = xyzi.copy()
        lvl[:, :3] = xyzi[:, :3] @ R0.T
        stack.append(lvl)
        if len(stack) < a.stack:
            continue
        lvl = np.concatenate(stack) if len(stack) > 1 else stack[0]
        stack = []
        t0 = time.perf_counter()
        pose, _ = odo.step(lvl)
        ms.append((time.perf_counter() - t0) * 1e3)
        est.append(pose.copy())

        # truth in the node's frame: KISS sees L_k = R0 * sensor_k and reports L_k -> L_0
        r = _B(R0) @ np.linalg.inv(truth0) @ truth @ _B(R0.T)
        rel.append(r)
        tmap.add(lvl, r)

        if a.path != "still" and pat.done and a.path == "patrol":
            break
        if a.path == "spin" and spin_left <= -1.0:
            break

    wall = time.perf_counter() - t_wall
    if not est:
        raise SystemExit("walk_map: nothing registered — the run was shorter than levelling")

    est = np.array(est)
    rel = np.array(rel)
    err_t = np.linalg.norm(est[:, :3, 3] - rel[:, :3, 3], axis=1)
    err_y = np.array([abs(wrap(math.radians(yaw_deg(e) - yaw_deg(r))))
                      for e, r in zip(est, rel)])
    path, yaw_t = float(walked), abs(float(turned))
    ms = np.array(ms)

    to_world = truth0 @ _B(R0.T)
    o_n, o_p50, o_rms, o_p95 = score_map(room, odo.map(), to_world, a.keep_self)
    t_n, t_p50, t_rms, t_p95 = score_map(room, tmap.map(), to_world, a.keep_self)

    print(f"\n  {n_frames} frames, {len(est)} registered, {wall:.0f} s wall"
          f" ({data.time:.1f} s simulated)")
    print(f"  cloud   {np.mean(ret_pf):.0f} returns of {np.mean(rays_pf):.0f} rays a frame")
    print(f"  icp     p50 {np.median(ms):.1f} ms  p99 {np.percentile(ms, 99):.1f}"
          f"   (the Pi has {1000 / sc.frame_hz:.0f} between frames)")
    print(f"\n  TRAJECTORY  {path:.2f} m walked, {math.degrees(yaw_t):.0f} deg of net turning"
          f"  (truth, every frame)")
    frac = (f"   = {100 * err_t[-1] / path:.1f} % of path" if path > 0.05
            else "   (standing still: there is no path to be a fraction of)")
    print(f"    position  end {err_t[-1] * 100:6.1f} cm   max {err_t.max() * 100:6.1f} cm{frac}")
    print(f"    yaw       end {math.degrees(err_y[-1]):6.2f} deg  max {math.degrees(err_y.max()):6.2f} deg")
    if yaw_t > 0.1:
        print(f"              {math.degrees(err_y[-1]) / (math.degrees(yaw_t) / 360):.2f} deg per 360 deg turned")
    print(f"\n  MAP vs the room's true surfaces ({'all points' if a.keep_self else 'room points only'})")
    print(f"    from odometry  {o_n:7d} pts   p50 {o_p50 * 1000:5.1f} mm  rms {o_rms * 1000:5.1f}  p95 {o_p95 * 1000:6.1f}")
    print(f"    from truth     {t_n:7d} pts   p50 {t_p50 * 1000:5.1f} mm  rms {t_rms * 1000:5.1f}  p95 {t_p95 * 1000:6.1f}")
    if a.stack > 1:
        print(f"    (the truth map carries the same {a.stack}-frame stacking smear —"
              f" it is the floor for THIS configuration, not for the sensor)")
    print(f"    -> the odometry costs the map {(o_rms - t_rms) * 1000:+.1f} mm rms;"
          f" the rest is the sensor, its {sc.sigma * 1000:.0f} mm noise"
          f"{'' if a.rigid else ' and the smear'}")

    if a.map:
        write_pcd(a.map, odo.map(), f"walk_map.py {a.path}, {len(est)} frames")
        print(f"\n  map -> {a.map}")
    if a.log:
        np.savez(a.log, est=est, rel=rel, ms=ms, err_t=err_t, err_y=err_y,
                 rays=np.array(rays_pf), returns=np.array(ret_pf), R0=R0)
        print(f"  log -> {a.log}")
    return dict(path=path, yaw=yaw_t, end=float(err_t[-1]), max=float(err_t.max()),
                yaw_end=float(math.degrees(err_y[-1])), frames=len(est),
                map_rms=o_rms, truth_rms=t_rms, ms=float(np.median(ms)))


def selftest(a):
    """The harness, not the odometry: 4 s of standing, and everything hooked up right."""
    fails = []

    def chk(name, cond, extra=""):
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}{('  ' + extra) if extra else ''}")
        if not cond:
            fails.append(name)

    model, data, gait, act, sc, _, room = load(a.scene, 0)
    sens = sensors(model)
    steps = int(round((1.0 / sc.frame_hz) / model.opt.timestep))
    settle(model, data, gait, act, 1.0, sens, False)

    # a distorted frame and a rigid one, same instant, same commands
    xyzi, acc, truth, rays = capture_frame(model, data, gait, act, sens, sc,
                                           (0.0, 0.0, 0.0), steps, 7, False, False)
    chk("a frame comes back", len(xyzi) > 500, f"{len(xyzi)} returns of {rays} rays")
    chk("the ray budget is the sensor's", abs(rays - sc.rate / sc.frame_hz) < 30,
        f"{rays} against {sc.rate / sc.frame_hz:.0f}")
    chk("gravity is where it should be", abs(np.linalg.norm(acc) - 9.81) < 1e-6)
    chk("the room is tagged, the robot is not",
        0.0 < (xyzi[:, 3] < 0.5).mean() < 0.2,
        f"{(xyzi[:, 3] < 0.5).mean() * 100:.1f} % of returns are the robot's own legs")
    room_pts = xyzi[xyzi[:, 3] > 0.5]
    w = room_pts[:, :3] @ truth[:3, :3].T + truth[:3, 3]
    d = room.surface_distance(w)
    chk("truth-projected points land on the room",
        float(np.sqrt((d ** 2).mean())) < 4 * sc.sigma,
        f"rms {np.sqrt((d ** 2).mean()) * 1000:.1f} mm against {sc.sigma * 1000:.0f} mm of sensor noise")
    worst, at = room.clearance_along([(0.0, 0.0)] + PATROL)
    chk("the patrol tour is clear of everything in the model", worst >= PATROL_CLEAR,
        f"worst {worst * 1000:.0f} mm at ({at[0]:+.2f}, {at[1]:+.2f}),"
        f" want >= {PATROL_CLEAR * 1000:.0f}")
    chk("every static geom can be scored", not room.missing,
        room.describe())

    print("selftest: OK" if not fails else f"!! selftest FAILED: {', '.join(fails)}")
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--path", default="patrol", choices=("still", "straight", "spin", "patrol"))
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=None, help="cap the run")
    ap.add_argument("--rigid", action="store_true",
                    help="cast each frame from its end pose — lidar.Scanner's own"
                         " behaviour, i.e. no motion distortion.  The control arm")
    ap.add_argument("--slice-ms", type=float, default=SLICE_MS)
    ap.add_argument("--stack", type=int, default=1, metavar="N",
                    help="register N frames at once instead of one.  One L2 frame is a"
                         " few hundred points on ~114 meridian lines, not a surface"
                         " sample, and frame-to-map ICP on it is under-constrained -- see"
                         " robot/README.md, 'SLAM'.  Costs pose rate: N frames is 12/N Hz")
    ap.add_argument("--blind", action="store_true", help="no IMU/contact feedback to the gait")
    ap.add_argument("--scene", default="scene_room.xml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=SPEED)
    ap.add_argument("--turn", type=float, default=TURN)
    ap.add_argument("--height", type=float, default=0.158)
    ap.add_argument("--period", type=float, default=0.45)
    ap.add_argument("--swing", type=float, default=0.022)
    ap.add_argument("--max-step", type=float, default=0.060)
    ap.add_argument("--keep-self", action="store_true",
                    help="score the map without dropping the robot's own legs")
    # the odometry's own knobs, defaulted to exactly what slam.py runs on the robot
    ap.add_argument("--max-range", type=float, default=slam_mod.MAX_RANGE_M)
    ap.add_argument("--min-range", type=float, default=slam_mod.MIN_RANGE_M)
    ap.add_argument("--voxel", type=float, default=slam_mod.VOXEL_M)
    ap.add_argument("--threshold", type=float, default=slam_mod.THRESHOLD_M)
    ap.add_argument("--map-voxel", type=float, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--map", metavar="OUT.pcd")
    ap.add_argument("--log", metavar="OUT.npz")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a)
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
