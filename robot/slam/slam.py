#!/usr/bin/env python3
"""
slam.py — where the robot is and what the room looks like, from the L2, on the Pi.

    python3 slam/slam.py                         # stream_pcd on localhost -> map on :9911
    python3 slam/slam.py --log run.npz --map room.pcd --record raw.npz
    python3 slam/slam.py --replay raw.npz --map room.pcd     # the same run, offline
    python3 slam/slam.py --selftest              # a synthetic room, a known walk, re-found

    # on the mac, the map building live:
    cd 3d && .venv/bin/python tools/pcview.py --stream 10.0.1.47:9911 --accum 100000

What it is: LiDAR odometry (KISS-ICP, pip-installed, C++ underneath) over the L2's frames,
plus a global voxel map accumulated from the registered frames. Every frame the sensor's
pose in a level world frame comes out, and the points that entered the map on that frame
go out on the wire. It is not SLAM in the loop-closing sense: drift is what it is, there
is no graph and nothing is ever re-optimised. For rooms and a dog that walks at 0.1 m/s it
is the map you can look at tonight; loop closure is the next thing, not this thing.

The pieces, and why they are shaped this way
--------------------------------------------
- **The source is `stream_pcd`'s TCP feed, not the SDK.** The L2 SDK is C++ only and the
  streamer already exists (`3d/tools/stream_pcd.cpp`); it accepts ONE client, and this
  node is it. Everyone who wants to look connects to this node instead, on `--serve`,
  which speaks the same ULF3 format — so `pcview.py --stream` needs no changes and, with
  a long `--accum`, draws the map rather than the sensor.
- **Every frame is registered, none dropped.** Odometry is a chain; a dropped frame is a
  doubled velocity step in the constant-velocity prior. If ICP falls behind the sensor the
  backlog is counted and printed (`lag`), which is the number that says whether the Pi
  keeps up, and it did (~15 ms per frame against 83 between them).
- **The world frame is levelled off the L2's own accelerometer**, from the first second
  of frames with the robot standing still: up is +Z, and +X is where the sensor was looking,
  projected onto the floor — on this robot the L2's axis points FORWARD and level
  (`LIDAR_TILT` = 90 in `3d/mini_dog.py`; the onboard capture reads it 2° below level), so
  +X is forward and the levelling rotation is a ~90° one. It is not bolted at 45°; that
  claim was here, and in `ros2/README.md`, and was wrong in both.
  The pose is the SENSOR's, not the body's: the extrinsic is in `3d/mini_dog.py`
  (`lidar_pose()`) and is not in `robot_params.json` yet.
- **No deskew.** The wire carries no per-point time; at 0.1 m/s and 12 Hz that is 8 mm
  across a frame, under the 5 cm map voxel. Turning at 0.65 rad/s it is 3°, which is the
  first thing to fix if turns drift: `stream_pcd` would have to send the SDK's per-point
  `time`.
- **The ICP threshold is fixed** (`--threshold`, 0.3 m), not KISS-ICP's adaptive one. The
  adaptive estimator only learns from frames that moved more than `min_motion_th` (0.1 m,
  a car's number), which this robot never does in one frame, so it would sit on its 2 m
  initial value forever. 0.3 m is a room's scale; the selftest holds at it.
- **The map is a voxel set, not the KISS local map.** KISS forgets what is further than
  `max_range` from the sensor; the map here keeps one point per `--map-voxel` (5 cm) cube
  for good, which is what you want to write out and what a client wants to draw.
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import select
import socket
import struct
import sys
import threading
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

MAGIC = b"ULF3"                  # 3d/tools/stream_pcd.cpp's frame: n u32, stamp f64, acc 3f, xyzi 4f*n
G_UP = (0.0, 0.0, 9.81)          # the "accelerometer" a served world-frame cloud carries: level already

# The L2 on this robot, for the odometry. The sensor's own range is unmeasured (verify,
# ref/lidar/README.md); 10 m is the far wall of every room it has seen.
MAX_RANGE_M = 10.0
MIN_RANGE_M = 0.25               # the nearest thing in the capture was 0.17: the robot itself
VOXEL_M = 0.10                   # KISS's registration voxel; the map's is half of it
THRESHOLD_M = 0.30               # ICP kernel; correspondences to 3x this
LEVEL_FRAMES = 12                # one second of accelerometer at rest


# ======================================================================================
# the wire: ULF3 in, ULF3 out
# ======================================================================================
def _recv_exactly(sock, n, stop=None):
    buf = bytearray()
    while len(buf) < n:
        if stop is not None and stop.is_set():
            raise ConnectionError("stopped")
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("stream closed by the sender")
        buf += chunk
    return bytes(buf)


def pack_frame(stamp, acc, xyzi):
    """One ULF3 message. `xyzi` (n, 4) float32."""
    xyzi = np.ascontiguousarray(xyzi, dtype="<f4")
    return (MAGIC + struct.pack("<Id3f", len(xyzi), float(stamp), *acc) + xyzi.tobytes())


class Source:
    """Frames off `stream_pcd`, every one of them, in order.

    A reader thread fills a queue the main loop drains; the queue's depth is the lag,
    printed every second. Nothing here drops a frame — see the header for why."""

    def __init__(self, host, port):
        self.name = f"{host}:{port}"
        self.sock = socket.create_connection((host, int(port)), timeout=10)
        self.sock.settimeout(10)
        self.q = queue.Queue()
        self.err = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                magic = _recv_exactly(self.sock, 4, self.stop)
                if magic != MAGIC:
                    raise ValueError(f"bad frame magic {magic!r} - wrong port?")
                n, stamp, ax, ay, az = struct.unpack("<Id3f", _recv_exactly(self.sock, 24, self.stop))
                if n == 0:                      # stream_pcd's heartbeat: the L2 is parked
                    continue
                raw = _recv_exactly(self.sock, 16 * n, self.stop)
                xyzi = np.frombuffer(raw, dtype="<f4").reshape(n, 4)
                self.q.put((stamp, (ax, ay, az), xyzi))
        except Exception as e:
            self.err = e
        finally:
            self.q.put(None)
            try:
                self.sock.close()
            except OSError:
                pass

    def frames(self):
        """Yields (stamp, acc, xyzi) until the sender goes away."""
        while True:
            item = self.q.get()
            if item is None:
                return
            yield item

    @property
    def lag(self):
        return self.q.qsize()

    def close(self):
        self.stop.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


class Replay:
    """The same frames out of a `--record` file: what happened, again, without the robot."""

    def __init__(self, path):
        z = np.load(path)
        self.name = path
        self.stamps = z["stamp"]
        self.acc = z["acc"]
        self.off = z["offset"]
        self.xyzi = z["xyzi"]
        self.lag = 0

    def frames(self):
        for i in range(len(self.stamps)):
            yield (float(self.stamps[i]), tuple(self.acc[i]), self.xyzi[self.off[i]:self.off[i + 1]])

    def close(self):
        pass


class Recorder:
    """Every raw frame, verbatim, so a run can be replayed and the odometry argued about."""

    def __init__(self, path):
        self.path = path
        self.stamps, self.acc, self.chunks = [], [], []

    def add(self, stamp, acc, xyzi):
        self.stamps.append(stamp)
        self.acc.append(acc)
        self.chunks.append(np.asarray(xyzi, dtype="<f4"))

    def save(self):
        off = np.zeros(len(self.chunks) + 1, dtype=np.int64)
        off[1:] = np.cumsum([len(c) for c in self.chunks])
        xyzi = np.concatenate(self.chunks) if self.chunks else np.zeros((0, 4), "<f4")
        np.savez(self.path, stamp=np.array(self.stamps), acc=np.array(self.acc, dtype=np.float32),
                 offset=off, xyzi=xyzi)
        return len(self.chunks), len(xyzi)


class Server:
    """Serves ULF3 to any number of viewers. A slow viewer drops frames, never the node:
    each client has a queue two deep and the newest frame wins."""

    def __init__(self, port):
        self.ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ls.bind(("", int(port)))
        self.ls.listen(4)
        self.ls.setblocking(False)
        self.port = int(port)
        self.clients = []                          # (sock, queue)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self.stop.is_set():
            r, _, _ = select.select([self.ls], [], [], 0.5)
            if not r:
                continue
            try:
                s, peer = self.ls.accept()
            except OSError:
                continue
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            q = queue.Queue(maxsize=2)
            with self.lock:
                self.clients.append((s, q))
            print(f"  viewer connected from {peer[0]}")
            threading.Thread(target=self._feed, args=(s, q, peer[0]), daemon=True).start()

    def _feed(self, s, q, who):
        try:
            while not self.stop.is_set():
                msg = q.get()
                if msg is None:
                    break
                s.sendall(msg)
        except OSError:
            pass
        finally:
            with self.lock:
                self.clients = [c for c in self.clients if c[0] is not s]
            try:
                s.close()
            except OSError:
                pass
            print(f"  viewer {who} went away")

    def send(self, msg):
        with self.lock:
            clients = list(self.clients)
        for _, q in clients:
            if q.full():
                try:
                    q.get_nowait()             # the viewer is behind: newest wins
                except queue.Empty:
                    pass
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass

    @property
    def n(self):
        with self.lock:
            return len(self.clients)

    def close(self):
        self.stop.set()
        with self.lock:
            for s, q in self.clients:
                q.put(None)
        self.ls.close()


# ======================================================================================
# levelling: the world frame
# ======================================================================================
def level_rotation(up, forward):
    """R (3x3) taking the sensor frame to a level frame: `up` -> +Z, and `forward`
    (the sensor's optical axis) -> +X once its vertical part is removed.

    Same idea as pcview.level(), plus the yaw so the map's +X means something. `up` is what
    the accelerometer reads at rest — specific force, pointing up."""
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)
    f = np.asarray(forward, float)
    f = f - up * float(f @ up)
    if np.linalg.norm(f) < 1e-6:                   # looking straight up: no heading, pick one
        f = np.cross(up, [0.0, 1.0, 0.0])
    f = f / np.linalg.norm(f)
    l = np.cross(up, f)
    return np.stack([f, l, up])                    # rows are world axes in sensor coordinates


# ======================================================================================
# the odometry
# ======================================================================================
class Odometry:
    def __init__(self, max_range=MAX_RANGE_M, min_range=MIN_RANGE_M, voxel=VOXEL_M,
                 threshold=THRESHOLD_M, map_voxel=None, threads=0):
        from kiss_icp.config import KISSConfig
        from kiss_icp.kiss_icp import KissICP
        from kiss_icp.voxelization import voxel_down_sample
        cfg = KISSConfig()
        cfg.data.max_range = float(max_range)
        cfg.data.min_range = float(min_range)
        cfg.data.deskew = False
        cfg.mapping.voxel_size = float(voxel)
        cfg.adaptive_threshold.fixed_threshold = float(threshold)
        cfg.registration.max_num_threads = int(threads)
        self.cfg = cfg
        self.icp = KissICP(cfg)
        self._down = voxel_down_sample
        self.map_voxel = float(map_voxel or voxel * 0.5)
        self.map_keys = set()
        self.map_pts = []                          # world-frame (n, 4) chunks, one per frame
        self.map_n = 0
        self.pose = np.eye(4)

    def step(self, xyzi):
        """Register one sensor-frame cloud. Returns (pose 4x4, the map's new points (m, 4))."""
        pts = np.ascontiguousarray(xyzi[:, :3], dtype=np.float64)
        self.icp.register_frame(pts, np.zeros(0))
        self.pose = self.icp.last_pose.copy()
        new = self._grow(pts, xyzi[:, 3])
        return self.pose, new

    def _grow(self, pts, inten):
        r = np.linalg.norm(pts, axis=1)
        keep = (r > self.cfg.data.min_range) & (r < self.cfg.data.max_range)
        pts, inten = pts[keep], inten[keep]
        if not len(pts):
            return np.zeros((0, 4), np.float32)
        world = pts @ self.pose[:3, :3].T + self.pose[:3, 3]
        k = np.floor(world / self.map_voxel).astype(np.int64)
        key = (k[:, 0] * 73856093) ^ (k[:, 1] * 19349669) ^ (k[:, 2] * 83492791)
        key, first = np.unique(key, return_index=True)
        fresh = np.fromiter((kk not in self.map_keys for kk in key.tolist()), bool, len(key))
        idx = first[fresh]
        self.map_keys.update(key[fresh].tolist())
        new = np.empty((len(idx), 4), np.float32)
        new[:, :3] = world[idx]
        new[:, 3] = inten[idx]
        self.map_pts.append(new)
        self.map_n += len(new)
        return new

    def map(self):
        return np.concatenate(self.map_pts) if self.map_pts else np.zeros((0, 4), np.float32)


def write_pcd(path, xyzi, comment=None):
    """ASCII PCD, x y z intensity — what pcview.py reads and capture_pcd.cpp writes."""
    xyzi = np.asarray(xyzi, np.float32)
    with open(path, "w") as f:
        if comment:
            f.write(f"# {comment}\n")
        f.write("VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
                f"WIDTH {len(xyzi)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(xyzi)}\nDATA ascii\n")
        np.savetxt(f, xyzi, fmt="%.4f %.4f %.4f %.1f")


def yaw_deg(pose):
    return math.degrees(math.atan2(pose[1, 0], pose[0, 0]))


# ======================================================================================
# the run
# ======================================================================================
def run(source, a):
    odo = Odometry(a.max_range, a.min_range, a.voxel, a.threshold, a.map_voxel, a.threads)
    server = Server(a.serve) if a.serve else None
    rec = Recorder(a.record) if a.record else None
    log = {"stamp": [], "pose": [], "ms": [], "n": []}
    R0 = None
    ups = []
    ms = deque(maxlen=120)
    t_line = time.monotonic()
    frames = 0
    if server:
        print(f"serving the map on :{server.port}  (pcview.py --stream <pi>:{server.port} --accum 100000)")
    try:
        for stamp, acc, xyzi in source.frames():
            frames += 1
            if rec:
                rec.add(stamp, acc, xyzi)
            if R0 is None:                         # levelling: wait for a second of rest
                ups.append(acc)
                if len(ups) < a.level_frames:
                    continue
                up = np.median(np.array(ups), axis=0)
                if np.linalg.norm(up) < 5.0:
                    raise SystemExit(f"!! accelerometer reads {up} — no gravity, cannot level")
                R0 = level_rotation(up, (0.0, 0.0, 1.0))
                tilt = math.degrees(math.acos(min(1.0, R0[2] @ np.array([0, 0, 1.0]))))
                print(f"levelled: tilt {tilt:.1f}°, up {np.round(up, 2)} — world +X is the look direction")
            t0 = time.perf_counter()
            lvl = xyzi.copy()
            lvl[:, :3] = xyzi[:, :3] @ R0.T
            pose, new = odo.step(lvl)
            dt_ms = (time.perf_counter() - t0) * 1e3
            ms.append(dt_ms)
            log["stamp"].append(stamp)
            log["pose"].append(pose)
            log["ms"].append(dt_ms)
            log["n"].append(len(xyzi))
            if server and len(new):
                server.send(pack_frame(stamp, G_UP, new))
            now = time.monotonic()
            if now - t_line >= 1.0:
                t_line = now
                p = pose[:3, 3]
                m = np.array(ms)
                print(f"  f{frames:5d}  x {p[0]:+6.2f} y {p[1]:+6.2f} z {p[2]:+5.2f} m  yaw {yaw_deg(pose):+6.1f}°"
                      f"  icp {np.median(m):4.1f}/{np.percentile(m, 99):4.1f} ms  lag {source.lag:2d}"
                      f"  map {odo.map_n:7d} pts  viewers {server.n if server else 0}")
            if a.seconds and frames >= a.seconds * 12:
                break
    except KeyboardInterrupt:
        print()
    finally:
        source.close()
        if server:
            server.close()
    print(f"{frames} frames; {len(log['pose'])} registered")
    if getattr(source, "err", None):
        print(f"  source ended: {source.err}")
    if log["pose"]:
        m = np.array(log["ms"])
        p = np.array(log["pose"])
        print(f"  icp p50 {np.median(m):.1f} ms  p99 {np.percentile(m, 99):.1f}  max {m.max():.1f}"
              f"  (12 Hz gives 83)")
        d = np.linalg.norm(np.diff(p[:, :3, 3], axis=0), axis=1).sum()
        print(f"  path {d:.2f} m, ends at {np.round(p[-1, :3, 3], 2)} yaw {yaw_deg(p[-1]):+.1f}°,"
              f" map {odo.map_n} pts at {odo.map_voxel * 100:.0f} cm")
    if a.log and log["pose"]:
        np.savez(a.log, stamp=np.array(log["stamp"]), pose=np.array(log["pose"]),
                 ms=np.array(log["ms"]), n=np.array(log["n"]), R0=R0)
        print(f"  poses -> {a.log}")
    if a.map and odo.map_n:
        write_pcd(a.map, odo.map(), f"slam.py map, level world frame, voxel {odo.map_voxel} m,"
                  f" {len(log['pose'])} frames")
        print(f"  map -> {a.map}")
    if rec:
        nf, npts = rec.save()
        print(f"  raw frames -> {a.record} ({nf} frames, {npts} points)")
    return odo, log


# ======================================================================================
# selftest: a room that does not exist, walked through by a sensor that does
# ======================================================================================
def synth_room(rng, pose, n=5000, room=(6.0, 4.0, 2.5), fov_deg=96.0, noise=0.01):
    """The L2's cone from `pose` (4x4, sensor -> world), hitting the walls of a box
    room whose floor is z = 0 and whose corner is the origin. Returns sensor-frame xyzi.

    Rays are drawn uniformly over the cone, not with the real 1/sin pattern — coverage
    is what the odometry needs, and the pattern is `3d/lidar.py`'s business."""
    half = math.radians(fov_deg) / 2
    cos_t = rng.uniform(math.cos(half), 1.0, n)
    sin_t = np.sqrt(1 - cos_t ** 2)
    phi = rng.uniform(0, 2 * math.pi, n)
    d_s = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], 1)    # +Z is the axis
    R, o = pose[:3, :3], pose[:3, 3]
    d = d_s @ R.T
    t_best = np.full(n, np.inf)
    lo, hi = np.zeros(3), np.array(room)
    for ax in range(3):
        for wall in (lo[ax], hi[ax]):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (wall - o[ax]) / d[:, ax]
            ok = (t > 0.05) & (t < t_best)
            hit = o + d * t[:, None]
            for bx in range(3):
                if bx != ax:
                    ok &= (hit[:, bx] >= lo[bx] - 1e-6) & (hit[:, bx] <= hi[bx] + 1e-6)
            t_best = np.where(ok, t, t_best)
    ok = np.isfinite(t_best)
    t_best = t_best[ok] + rng.normal(0, noise, ok.sum())
    p_s = d_s[ok] * t_best[:, None]
    out = np.empty((len(p_s), 4), np.float32)
    out[:, :3] = p_s
    out[:, 3] = 100.0
    return out


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def selftest():
    """A 6 x 4 x 2.5 m room, the sensor 0.2 m up and tilted 45° — NOT the robot's mount,
    which is axis-forward (see the header); this is a generic sensor walk, walking
    2.5 m forward at 0.1 m/s, turning 90° left at 0.5 rad/s, then 1.5 m more — 12 Hz, the
    frames handed to the node exactly as `stream_pcd` would hand them (sensor frame plus
    a gravity vector), and the trajectory it recovers held against the truth."""
    rng = np.random.default_rng(0)
    tilt = math.radians(45.0)
    # sensor -> body, columns = the sensor's axes in the body: optical +Z pitched `tilt`
    # up from body +X, sensor +X along body +Y
    R_sb = np.array([[0.0, -math.sin(tilt), math.cos(tilt)],
                     [1.0, 0.0, 0.0],
                     [0.0, math.cos(tilt), math.sin(tilt)]])
    assert np.allclose(R_sb @ R_sb.T, np.eye(3)) and np.linalg.det(R_sb) > 0
    hz, v, w = 12.0, 0.10, 0.5
    legs = [("go", 2.5 / v), ("turn", math.radians(90) / w), ("go", 1.5 / v)]
    x, y, yaw = 1.0, 1.0, 0.0
    truth, frames = [], []
    for kind, T in legs:
        for _ in range(int(round(T * hz))):
            if kind == "go":
                x += v / hz * math.cos(yaw)
                y += v / hz * math.sin(yaw)
            else:
                yaw += w / hz
            R_bw = _rot("z", math.degrees(yaw))
            pose = np.eye(4)
            pose[:3, :3] = R_bw @ R_sb
            pose[:3, 3] = (x, y, 0.2)
            truth.append(pose)
            frames.append(synth_room(rng, pose))
    up_s = R_sb.T @ np.array([0, 0, 9.81])         # what the accelerometer reads, sensor frame
    # the node's own frame: level, +X where the sensor looked at t = 0 (body +X here, yaw 0)
    R0 = level_rotation(up_s, (0, 0, 1.0))
    odo = Odometry()
    est = []
    t0 = time.perf_counter()
    for f in frames:
        lvl = f.copy()
        lvl[:, :3] = f[:, :3] @ R0.T
        pose, _ = odo.step(lvl)
        est.append(pose)
    ms = (time.perf_counter() - t0) * 1e3 / len(frames)
    # truth in the node's frame. KISS sees the LEVELLED sensor frame L_k = R0 * sensor_k
    # and reports T_k: L_k -> L_0, so the truth is B(R0) T_w0^-1 T_wk B(R0^T).
    def B(R):
        b = np.eye(4)
        b[:3, :3] = R
        return b
    rel = [B(R0) @ np.linalg.inv(truth[0]) @ t @ B(R0.T) for t in truth]
    err_t = np.array([np.linalg.norm(e[:3, 3] - r[:3, 3]) for e, r in zip(est, rel)])
    err_y = np.array([abs((yaw_deg(e) - yaw_deg(r) + 180) % 360 - 180) for e, r in zip(est, rel)])
    path = sum(np.linalg.norm(np.diff(np.array([r[:3, 3] for r in rel]), axis=0), axis=1))
    print(f"selftest: {len(frames)} frames, {path:.2f} m of path with a 90° turn, {ms:.1f} ms/frame")
    print(f"  position error: end {err_t[-1] * 100:.1f} cm, max {err_t.max() * 100:.1f} cm")
    print(f"  yaw error:      end {err_y[-1]:.2f}°, max {err_y.max():.2f}°")
    print(f"  map {odo.map_n} pts at {odo.map_voxel * 100:.0f} cm; room walls span "
          f"{np.round(odo.map().min(0)[:3], 2)} .. {np.round(odo.map().max(0)[:3], 2)}")
    ok = err_t[-1] < 0.05 and err_y[-1] < 2.0 and err_t.max() < 0.10
    print("selftest: OK" if ok else "!! selftest FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default="127.0.0.1:9910", metavar="HOST[:PORT]",
                    help="stream_pcd's address (default 127.0.0.1:9910)")
    ap.add_argument("--replay", metavar="RAW.npz", help="frames from a --record file instead")
    ap.add_argument("--serve", type=int, default=9911, metavar="PORT",
                    help="serve the map's new points as ULF3 here (0 = off; default 9911)")
    ap.add_argument("--log", metavar="FILE.npz", help="every pose, stamp and ICP time")
    ap.add_argument("--map", metavar="FILE.pcd", help="the map on exit, level world frame")
    ap.add_argument("--record", metavar="RAW.npz", help="every raw frame, for --replay")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many (0 = until ctrl-C)")
    ap.add_argument("--max-range", type=float, default=MAX_RANGE_M)
    ap.add_argument("--min-range", type=float, default=MIN_RANGE_M)
    ap.add_argument("--voxel", type=float, default=VOXEL_M, help="registration voxel, m")
    ap.add_argument("--map-voxel", type=float, default=None, help="map voxel, m (default voxel/2)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD_M, help="ICP kernel, m")
    ap.add_argument("--threads", type=int, default=0, help="KISS-ICP threads (0 = all)")
    ap.add_argument("--level-frames", type=int, default=LEVEL_FRAMES,
                    help="frames of accelerometer to level from, robot at rest")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.replay:
        source = Replay(a.replay)
    else:
        host, _, port = a.source.partition(":")
        try:
            source = Source(host, port or 9910)
        except OSError as e:
            raise SystemExit(f"--source {a.source}: {e}\n  is stream_pcd running there?\n"
                             "    ~/unilidar_sdk2/unitree_lidar_sdk/bin/stream_pcd")
    print(f"frames from {source.name}")
    run(source, a)


if __name__ == "__main__":
    main()
