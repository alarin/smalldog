"""
pcview.py — look at a point cloud, from a file or straight out of the sim.

A viewer, nothing more: it does not talk to a sensor and it knows nothing about the
robot's geometry.  Everything it draws is (n,3) points plus an optional per-point scalar,
so the *same* window shows a capture off the real Unitree L2 and the cloud `lidar.py`
predicts for the same scene — which is the only way to argue about the rows still marked
**verify** in README.md (LIDAR_RATE, LIDAR_R_MIN/MAX, LIDAR_SIGMA, and the cone).

    .venv/bin/python tools/pcview.py --sim                    # the modelled L2, standing
    .venv/bin/python tools/pcview.py cap.pcd                  # a real capture
    .venv/bin/python tools/pcview.py cap.pcd --sim -c file    # both, one colour each
    .venv/bin/python tools/pcview.py cap.npy --stats          # no window, numbers only
    .venv/bin/python tools/pcview.py live.pcd --watch         # redraw when the file changes

GETTING REAL POINTS IN
There is no hardware path in this repository — `lidar.py` is a simulation of the L2, not a
driver — so the sensor side is Unitree's own SDK on whatever machine the L2 is plugged
into, and what crosses to here is a *file*.  Any of these work, with no conversion:

    .pcd            ascii or binary (not binary_compressed) — what the SDK's own tools and
                    `pcl_ros` write.  x y z, plus intensity/ring/time if present
    .npy / .npz     an (n,3) or (n,4) array; an .npz may also carry `range`, `geom`,
                    `intensity`, `n_rays` (that is what --save writes)
    .csv .xyz .txt  whitespace- or comma-separated, three or four columns, `#` comments
    .bin            raw float32 x y z i, KITTI-style
    -               the same, on stdin

UNITS ARE METRES here, not the CAD's millimetres: both the L2's SDK and MuJoCo speak
metres, and this file only ever sees their output.  `--mm` divides an input by 1000 for
the one case where somebody exports a cloud in CAD units.

THE FRAME MATTERS FOR HALF THE NUMBERS.  A real capture arrives in the *sensor* frame,
where +Z is the optical axis (that is `lidar_link`, see README.md), so "angle off the
axis" is the cone and "range" is the sensor's own range.  A sim cloud has both: --frame
local matches the sensor, --frame world is where the points sit in the room.  The cone
half-angle and the range histogram this prints are only meaningful in the sensor frame,
and it says so rather than quietly printing a number off a world cloud.
"""
import os, sys, math, time, socket, struct, threading, colorsys, argparse

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

HERE = os.path.dirname(os.path.abspath(__file__))
CAD = os.path.dirname(HERE)

# distinct flat colours for --color file, in order.  Chosen to stay apart on black.
FILE_COLORS = [(1.00, 0.78, 0.20), (0.30, 0.75, 1.00), (0.45, 0.90, 0.45),
               (1.00, 0.45, 0.45), (0.85, 0.55, 1.00), (0.60, 0.60, 0.65)]


# ======================================================================================
# loading.  Every reader returns a dict with at least `xyz`; anything else is extra
# per-point scalars the colour modes can use.
# ======================================================================================
def load_pcd(path):
    """ascii or binary .pcd.  Reads the header for real - field order is not fixed.

    Also picks up the `# imu_accel_xyz` comment tools/capture_pcd.cpp writes: that is the
    L2's own accelerometer, and with the sensor at rest it points straight up, which is
    the only thing that lets the cloud be stood upright.  See level().
    """
    grav = None
    with open(path, "rb") as f:
        hdr, fields, size, typ, count = {}, None, None, None, None
        while True:
            line = f.readline()
            if not line:
                raise SystemExit(f"{path}: no DATA line — truncated .pcd?")
            text = line.decode("ascii", "replace").strip()
            if text.startswith("#"):
                bits = text.lstrip("#").split()
                if bits and bits[0] == "imu_accel_xyz" and len(bits) >= 4:
                    grav = tuple(float(x) for x in bits[1:4])
                continue
            k, _, v = text.partition(" ")
            k = k.upper()
            if k == "FIELDS":   fields = v.split()
            elif k == "SIZE":   size = [int(x) for x in v.split()]
            elif k == "TYPE":   typ = v.split()
            elif k == "COUNT":  count = [int(x) for x in v.split()]
            elif k in ("WIDTH", "HEIGHT", "POINTS"): hdr[k] = int(v)
            elif k == "DATA":
                data = v.strip().lower()
                break
        n = hdr.get("POINTS", hdr.get("WIDTH", 0) * hdr.get("HEIGHT", 1))
        count = count or [1] * len(fields)
        if data == "ascii":
            arr = np.loadtxt(f, dtype=np.float64, ndmin=2)
            cols = {name: arr[:, i] for i, name in enumerate(_expand(fields, count))
                    if i < arr.shape[1]}
        elif data == "binary":
            np_t = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "<u1", ("U", 2): "<u2",
                    ("U", 4): "<u4", ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4"}
            dt = np.dtype([(nm, np_t[(t, s)])
                           for nm, t, s in zip(_expand(fields, count),
                                               _expand(typ, count), _expand(size, count))])
            rec = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
            cols = {nm: rec[nm].astype(np.float64) for nm in dt.names}
        else:
            raise SystemExit(f"{path}: DATA {data} is not supported — re-save as ascii or "
                             "binary (pcl_convert_pcd_ascii_binary, or -1 in PCL's writer)")
    for a in "xyz":
        if a not in cols:
            raise SystemExit(f"{path}: no '{a}' field (has {', '.join(cols)})")
    out = {"xyz": np.column_stack([cols["x"], cols["y"], cols["z"]])}
    for extra in ("intensity", "ring", "time", "t"):
        if extra in cols:
            out[extra] = cols[extra]
    if grav is not None:
        out["gravity"] = grav
    return out


def _expand(vals, count):
    """PCD COUNT > 1 means the field repeats; name the copies f, f_1, f_2 ..."""
    out = []
    for v, c in zip(vals, count):
        out += [v] + [f"{v}_{i}" for i in range(1, c)]
    return out


def load_array(path):
    if path.endswith(".npz"):
        z = np.load(path)
        key = next((k for k in ("local", "xyz", "points", "world") if k in z), None)
        if key is None:
            raise SystemExit(f"{path}: no local/xyz/points/world array (has {', '.join(z)})")
        out = {"xyz": np.asarray(z[key], dtype=float)}
        for extra in ("range", "geom", "intensity", "world", "local"):
            if extra in z and extra != key and np.asarray(z[extra]).ndim == 1:
                out[extra] = np.asarray(z[extra], dtype=float)
        return out
    a = np.load(path) if path.endswith(".npy") else None
    if a is None:                                    # .bin — raw float32 x y z i
        a = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    a = np.atleast_2d(np.asarray(a, dtype=float))
    if a.shape[1] < 3:
        raise SystemExit(f"{path}: {a.shape[1]} columns, need at least 3 (x y z)")
    out = {"xyz": a[:, :3]}
    if a.shape[1] >= 4:
        out["intensity"] = a[:, 3]
    return out


def load_text(path):
    f = sys.stdin if path == "-" else open(path)
    rows = []
    for line in f:
        line = line.split("#")[0].strip().replace(",", " ")
        if not line:
            continue
        try:
            rows.append([float(v) for v in line.split()])
        except ValueError:
            continue                                  # a header line, most likely
    if not rows:
        raise SystemExit(f"{path}: no numeric rows")
    w = min(len(r) for r in rows)
    a = np.array([r[:w] for r in rows], dtype=float)
    if w < 3:
        raise SystemExit(f"{path}: {w} columns, need at least 3 (x y z)")
    out = {"xyz": a[:, :3]}
    if w >= 4:
        out["intensity"] = a[:, 3]
    return out


def load_file(path, mm=False):
    ext = os.path.splitext(path)[1].lower()
    if path != "-" and not os.path.exists(path):
        raise SystemExit(f"{path}: no such file")
    c = (load_pcd(path) if ext == ".pcd" else
         load_array(path) if ext in (".npy", ".npz", ".bin") else
         load_text(path))
    c["xyz"] = np.ascontiguousarray(c["xyz"], dtype=float)
    if mm:
        c["xyz"] /= 1000.0
    finite = np.isfinite(c["xyz"]).all(axis=1)
    if not finite.all():                              # a real sensor emits NaN for no-return
        for k, v in list(c.items()):
            c[k] = v[finite] if np.asarray(v).ndim >= 1 and len(v) == len(finite) else v
    c["name"] = os.path.basename(path) if path != "-" else "stdin"
    c["frame"] = "sensor"                             # what a capture is, unless told otherwise
    return c


# ======================================================================================
# the sim source.  This is lidar.Scanner on the model this tree exports, standing in its
# own stand keyframe — the prediction a real capture gets compared against.
# ======================================================================================
def load_sim(xml=None, terrain=False, frame="local", seconds=1.5, seed=0):
    sys.path.insert(0, CAD)
    try:
        import mujoco, lidar
    except ImportError as e:
        raise SystemExit(f"--sim needs the project venv (mujoco, lidar.py): {e}")
    if xml is None:
        xml = os.path.join(CAD, "out", "sim",
                           "mini_dog_terrain.xml" if terrain else "mini_dog.xml")
    if not os.path.exists(xml):
        raise SystemExit(f"{xml}: not exported yet — run "
                         ".venv/bin/python export_sim.py --check")
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    if m.nkey:                                        # the "stand" keyframe export_sim writes
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.ctrl[:] = m.key_ctrl[0]
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    sc = lidar.Scanner(m, seed=seed)
    if not sc.ok:
        raise SystemExit(f"--sim: {sc.missing}")
    c = sc.scan(m, d)
    out = {"xyz": np.ascontiguousarray(c["local" if frame == "local" else "world"]),
           "range": c["range"], "geom": c["geom"].astype(float),
           "name": f"sim:{os.path.basename(xml)}",
           "frame": "sensor" if frame == "local" else "world",
           "n_rays": c["n_rays"], "geom_names": _geom_names(m, mujoco, c["geom"])}
    return out


def _geom_names(m, mujoco, ids):
    out = {}
    for g in np.unique(ids):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g))
        out[int(g)] = nm or f"geom {int(g)}"
    return out


# ======================================================================================
# the live source.  tools/stream_pcd.cpp on the Pi serves one message per lidar frame;
# this reads them on a thread and keeps the newest, so a slow render drops frames instead
# of falling behind the sensor - which is what you want from a live view and the opposite
# of what a queue would do.
# ======================================================================================
class Stream:
    """Connects to stream_pcd and keeps a sliding window of the last `accum` frames.

    One frame off the L2 is ~5200 points, which looks like nothing.  The sensor's scan is
    non-repetitive, so the honest way to make a live view legible is to show a window of
    recent frames rather than to fake density in one: `accum` frames at 12 Hz is `accum`/12
    seconds of history, and moving the sensor smears it exactly as much as the real dwell
    would.  Default 12 = one second.
    """
    MAGIC = b"ULF3"

    def __init__(self, host, port, accum=12, name=None, do_level=True):
        self.addr = (host, int(port))
        self.level = do_level
        self.accum = max(1, int(accum))
        self.name = name or f"{host}:{port}"
        self.frames = []                       # newest last
        self.gravity = None                    # the L2's accelerometer, newest frame
        self.lock = threading.Lock()
        self.frames_seen = 0
        self.dropped = 0
        self.err = None
        self.stop = threading.Event()
        self.sock = socket.create_connection(self.addr, timeout=10)
        self.sock.settimeout(10)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ---------------------------------------------------------------------------------
    def _recv_exactly(self, n):
        buf = bytearray()
        while len(buf) < n and not self.stop.is_set():
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("stream closed by the sender")
            buf += chunk
        return bytes(buf)

    def _run(self):
        try:
            while not self.stop.is_set():
                magic = self._recv_exactly(4)
                if magic != self.MAGIC:
                    raise ValueError(f"bad frame magic {magic!r} - wrong port?")
                (n,) = struct.unpack("<I", self._recv_exactly(4))
                (stamp,) = struct.unpack("<d", self._recv_exactly(8))
                acc = struct.unpack("<3f", self._recv_exactly(12))
                raw = self._recv_exactly(16 * n)
                a = np.frombuffer(raw, dtype="<f4").reshape(n, 4)
                with self.lock:
                    self.gravity = acc
                    self.frames.append((stamp, a))
                    if len(self.frames) > self.accum:
                        del self.frames[0]
                    self.frames_seen += 1
        except Exception as e:                 # the viewer keeps running on the last cloud
            self.err = e
        finally:
            try:
                self.sock.close()
            except OSError:
                pass

    def close(self):
        self.stop.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def cloud(self):
        """The current window as one cloud dict, in the same shape as load_file()."""
        with self.lock:
            frames = list(self.frames)
            seen = self.frames_seen
        if frames:
            a = np.concatenate([f[1] for f in frames])
        else:
            a = np.zeros((0, 4), dtype=np.float32)
        c = {"xyz": np.ascontiguousarray(a[:, :3], dtype=float),
             "intensity": np.ascontiguousarray(a[:, 3], dtype=float),
             "name": self.name, "frame": "sensor",
             "frames": len(frames), "frames_seen": seen}
        if self.gravity is not None:
            c["gravity"] = self.gravity
        if self.level and len(a):
            c, _ = level(c)                    # stand it up every frame, live
        return c


# ======================================================================================
# standing the cloud upright.  This is the difference between a readable picture and a
# rainbow blob, and it is not cosmetic: a lidar cloud is drawn in the SENSOR's frame, and
# the sensor is almost never level.  On this robot it is bolted at LIDAR_TILT = 45 deg,
# so without this every wall in every capture leans by 45 degrees and "height" means
# nothing.  Sitting on a desk it happens to be a 1.5 deg correction; on the dog it is the
# whole picture.
# ======================================================================================
def level(c, up=None):
    """Rotate a cloud so gravity is -Z, i.e. up is +Z.  Returns a new cloud dict.

    `up` is the measured up direction in the cloud's own frame.  The L2's accelerometer
    gives it directly - at rest an accelerometer reads specific force, which points UP -
    and capture_pcd.cpp writes it into the .pcd header.  Without one there is nothing
    honest to do, so this says so and leaves the cloud alone rather than guessing from the
    points (a room's biggest plane is as likely to be a ceiling or a wall as a floor).
    """
    g = up if up is not None else c.get("gravity")
    if g is None:
        return c, None
    g = np.asarray(g, dtype=float)
    n = np.linalg.norm(g)
    if n < 1e-6:
        return c, None
    g = g / n
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(g, z)
    s_, c_ = np.linalg.norm(v), float(np.dot(g, z))
    if s_ < 1e-9:                                   # already upright (or exactly inverted)
        R = np.eye(3) if c_ > 0 else np.diag([1.0, -1.0, -1.0])
    else:                                           # Rodrigues, g -> +Z
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c_) / (s_ ** 2))
    out = dict(c)
    out["xyz"] = np.ascontiguousarray(c["xyz"] @ R.T)
    out["frame"] = "level"                          # no longer the sensor frame
    out["tilt_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, c_))))
    return out, out["tilt_deg"]


def ground_z(p, q=0.5):
    """Where the floor is, for the grid and for height colouring.

    A robust low percentile, not a plane fit: an upward-looking L2 in a room returns 45 %
    ceiling and almost no floor, so "the biggest plane" is the ceiling and a fit would
    hang the grid there.  The bottom half-percent of a levelled cloud is the floor, or the
    lowest thing the sensor can see, which is what you want the grid to sit on either way.
    """
    return float(np.percentile(p[:, 2], q)) if len(p) else 0.0


# ======================================================================================
# what the cloud says about the sensor.  These are the numbers the **verify** rows want,
# and every one of them is a property of the points, not of anything in mini_dog.py.
# ======================================================================================
def stats(c, verbose=True):
    p = c["xyz"]
    n = len(p)
    frame = c.get("frame")
    sensor = frame == "sensor"
    # Levelling is a pure rotation about the sensor, so RANGE survives it untouched; only
    # the cone does not, because +Z is no longer the optical axis.  Saying "not shown" for
    # both would throw away a number that is still exactly right.
    rangeable = frame in ("sensor", "level")
    r = c["range"] if "range" in c else np.linalg.norm(p, axis=1)
    lo, hi = p.min(axis=0), p.max(axis=0)
    print(f"\n{c['name']}: {n} points, frame = {c.get('frame', '?')}")
    if not n:
        return
    print(f"  bbox      x {lo[0]:+7.3f} … {hi[0]:+7.3f}   y {lo[1]:+7.3f} … {hi[1]:+7.3f}"
          f"   z {lo[2]:+7.3f} … {hi[2]:+7.3f}  m")
    if rangeable:
        q = np.percentile(r, [0, 1, 50, 99, 100])
        print(f"  range     {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f} / {q[3]:.3f} / {q[4]:.3f} m"
              "   (min, 1%, median, 99%, max)")
    if sensor:
        ax = np.degrees(np.arctan2(np.hypot(p[:, 0], p[:, 1]), p[:, 2]))
        az = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
        # arctan2 returns both -180 and +180, which would read as a 37th sector
        cov = len(np.unique((np.floor(az / 10.0).astype(int) % 36)))
        print(f"  cone      off-axis 0 … {ax.max():.1f}°  (99% within {np.percentile(ax, 99):.1f}°)"
              f"   azimuth {cov}/36 ten-degree sectors occupied")
        if verbose:
            # Count per band AND count per steradian.  Equal-width angle bands cover
            # very unequal solid angles, so a flat count histogram is not a uniform
            # sensor - it is one whose density falls off the axis.  Print both, or the
            # first column argues the opposite of what the sensor is doing.
            h, edges = np.histogram(ax, bins=12, range=(0, max(96.0, ax.max())))
            wide = max(1, h.max())
            print(f"      {'band':>13}  {'points':>7}  {'pts/sr':>8}")
            for i, k in enumerate(h):
                t0, t1 = math.radians(edges[i]), math.radians(edges[i + 1])
                sr = 2 * math.pi * (math.cos(t0) - math.cos(t1))
                bar = "#" * int(22 * k / wide)
                print(f"      {edges[i]:5.1f}–{edges[i+1]:5.1f}°  {k:7d}  {k/sr:8.0f}  {bar}")
    elif frame == "level":
        g = ground_z(p)
        below = float((p[:, 2] < 0).mean())
        print(f"  levelled  ground at z = {g:+.2f} m; the cone is not shown because +Z is"
              " no longer the optical axis")
        if below < 0.05:
            # An upward-looking L2 reaches only 6 deg below its own base plane, so in this
            # room it barely sees the floor at all.  Say so: "height above the floor" would
            # be a claim the data does not support, and the zero is really desk height.
            print(f"            only {100*below:.1f}% of points are below the sensor — it"
                  " can see almost no floor, so this zero is the lowest visible surface"
                  " (about desk height), not the room's floor")
        h, edges = np.histogram(p[:, 2] - g, bins=10)
        wide = max(1, h.max())
        for i, k in enumerate(h):
            print(f"      {edges[i]:+5.2f}–{edges[i+1]:+5.2f} m  {k:7d}  "
                  + "#" * int(24 * k / wide))
    else:
        print("  range/cone not shown: this is a world cloud, so distance from the origin"
              " is not the sensor's range.  Use --frame local.")
    if "n_rays" in c and c["n_rays"]:
        print(f"  returns   {n}/{c['n_rays']} rays = {100.0*n/c['n_rays']:.1f}%")
    if "geom_names" in c and "geom" in c:
        ids, cnt = np.unique(c["geom"].astype(int), return_counts=True)
        top = sorted(zip(cnt, ids), reverse=True)[:6]
        print("  hit       " + ", ".join(f"{c['geom_names'].get(int(g), g)} {k}"
                                         for k, g in top))
    if "intensity" in c:
        i = c["intensity"]
        print(f"  intensity {i.min():.1f} … {i.max():.1f}, median {np.median(i):.1f}")


# ======================================================================================
# the window
# ======================================================================================
MODES = ["height", "range", "axis", "intensity", "geom", "file"]


def scalars(c, mode, index):
    """(values, title, categorical) for one cloud under one colour mode, or None."""
    p = c["xyz"]
    if mode == "range":
        return (c["range"] if "range" in c else np.linalg.norm(p, axis=1)), "range, m", False
    if mode == "axis":
        return np.degrees(np.arctan2(np.hypot(p[:, 0], p[:, 1]), p[:, 2])), "off-axis, °", False
    if mode == "height":
        # height above the floor, on a levelled cloud, is the one channel that draws a
        # room the way a person reads one: floor dark, walls a gradient, ceiling bright.
        return p[:, 2] - c.get("ground", 0.0), "height, m", False
    if mode == "intensity":
        return (c["intensity"], "intensity", False) if "intensity" in c else None
    if mode == "geom":
        return (c["geom"], "geom id", True) if "geom" in c else None
    return None                                        # "file" — a flat colour per cloud


# Turbo, at 9 control points.  A hue sweep (VTK's default) is perceptually lumpy - it
# spends most of its length in cyan and green - which on a 60k-point cloud reads as
# texture that is not in the data.  These are Google's turbo values, sampled evenly.
TURBO = [(0.190, 0.072, 0.232), (0.246, 0.545, 0.905), (0.116, 0.855, 0.783),
         (0.263, 0.977, 0.470), (0.633, 0.999, 0.235), (0.904, 0.900, 0.199),
         (0.994, 0.633, 0.164), (0.911, 0.312, 0.062), (0.680, 0.098, 0.016)]


def ramp(lo, hi):
    """A vtkLookupTable over TURBO between lo and hi."""
    ctf = vtk.vtkColorTransferFunction()
    for i, (r, g, b) in enumerate(TURBO):
        ctf.AddRGBPoint(lo + (hi - lo) * i / (len(TURBO) - 1.0), r, g, b)
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    for i in range(256):
        lut.SetTableValue(i, *ctf.GetColor(lo + (hi - lo) * i / 255.0), 1.0)
    lut.SetTableRange(lo, hi)
    lut.Build()
    return lut


class Viewer:
    # (where the camera sits relative to the scene, which way is up ON SCREEN).
    # The up vector is per-view and it is not decoration: "top" looks straight down, so a
    # +Z up vector would be parallel to the view direction, which is degenerate - VTK then
    # picks the roll itself and the picture comes out arbitrarily rotated or mirrored.
    # This is the same defect that made the very first render an empty window with one dot
    # in it; it survived here because a degenerate TOP view still draws something.
    VIEWS = {"iso":   ((-1.0, -1.0, 0.55), (0.0, 0.0, 1.0)),
             "top":   ((0.0, 0.0, 1.0),    (0.0, 1.0, 0.0)),
             "front": ((-1.0, 0.0, 0.10),  (0.0, 0.0, 1.0)),
             "side":  ((0.0, -1.0, 0.10),  (0.0, 0.0, 1.0))}

    def __init__(self, clouds, mode="height", size=2.0, bg=(0.06, 0.07, 0.09), view="iso"):
        self.clouds, self.mode, self.psize, self.view = clouds, mode, size, view
        self.flip = False
        # one ground level for the whole scene, off whichever cloud is levelled: the grid
        # and the height colouring have to agree or the picture lies about where zero is.
        lev = [c for c in clouds if c.get("frame") == "level" and len(c["xyz"])]
        self.ground = ground_z(lev[0]["xyz"]) if lev else 0.0
        for c in clouds:
            c["ground"] = self.ground
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(*bg)
        self.actors, self.polys, self.filters = [], [], []
        for i, c in enumerate(clouds):
            pd = vtk.vtkPolyData()
            pts = vtk.vtkPoints()
            pts.SetData(numpy_to_vtk(np.ascontiguousarray(c["xyz"]), deep=1))
            pd.SetPoints(pts)
            f = vtk.vtkVertexGlyphFilter()
            f.SetInputData(pd); f.Update()
            m = vtk.vtkPolyDataMapper(); m.SetInputConnection(f.GetOutputPort())
            a = vtk.vtkActor(); a.SetMapper(m)
            a.GetProperty().SetPointSize(self.psize)
            a.GetProperty().SetColor(*FILE_COLORS[i % len(FILE_COLORS)])
            self.ren.AddActor(a)
            self.actors.append(a); self.polys.append(pd); self.filters.append(f)
        self.bar = vtk.vtkScalarBarActor()
        self.bar.SetWidth(0.10); self.bar.SetHeight(0.35)
        self.bar.SetPosition(0.885, 0.10)
        self.bar.GetLabelTextProperty().SetColor(0.85, 0.86, 0.88)
        self.bar.GetTitleTextProperty().SetColor(0.85, 0.86, 0.88)
        self.bar.GetLabelTextProperty().SetShadow(0)
        self.bar.GetTitleTextProperty().SetShadow(0)
        self.ren.AddViewProp(self.bar)
        self.grid = self._grid()
        self.ren.AddActor(self.grid)
        self.ren.AddActor(self._origin())
        self.text = vtk.vtkTextActor()
        self.text.GetTextProperty().SetFontSize(15)
        self.text.GetTextProperty().SetColor(0.80, 0.82, 0.86)
        self.text.SetPosition(12, 12)
        self.ren.AddViewProp(self.text)
        self.recolor()

    # ---------------------------------------------------------------- scene furniture
    def _extent(self):
        p = np.vstack([c["xyz"] for c in self.clouds if len(c["xyz"])])
        return float(np.percentile(np.abs(p), 98)) if len(p) else 1.0

    def _grid(self):
        """1 m squares on the FLOOR, out to the cloud — the scale bar, in metres.

        It used to sit on z = 0, which in a sensor-frame cloud is a plane through the
        sensor: on a desk that is a sheet of graph paper hanging in mid-air across the
        middle of the room, crossing every wall.  On a levelled cloud the floor is a real
        place and the grid belongs there.
        """
        e = max(1.0, math.ceil(self._extent()))
        z0 = self.ground
        pts, lines = vtk.vtkPoints(), vtk.vtkCellArray()
        k = 0
        for i in range(int(-e), int(e) + 1):
            for (a, b) in (((i, -e, z0), (i, e, z0)), ((-e, i, z0), (e, i, z0))):
                pts.InsertNextPoint(*a); pts.InsertNextPoint(*b)
                lines.InsertNextCell(2); lines.InsertCellPoint(k); lines.InsertCellPoint(k + 1)
                k += 2
        pd = vtk.vtkPolyData(); pd.SetPoints(pts); pd.SetLines(lines)
        m = vtk.vtkPolyDataMapper(); m.SetInputData(pd)
        a = vtk.vtkActor(); a.SetMapper(m)
        a.GetProperty().SetColor(0.22, 0.24, 0.28); a.GetProperty().SetLineWidth(1)
        return a

    def _origin(self):
        """where the sensor is: a marker at 0,0,0 and its +Z, the optical axis."""
        s = vtk.vtkSphereSource(); s.SetRadius(max(0.02, self._extent() * 0.012))
        s.SetThetaResolution(16); s.SetPhiResolution(16)
        m = vtk.vtkPolyDataMapper(); m.SetInputConnection(s.GetOutputPort())
        a = vtk.vtkActor(); a.SetMapper(m); a.GetProperty().SetColor(1.0, 0.35, 0.35)
        return a

    # ------------------------------------------------------------------------- colour
    def recolor(self):
        vals = [scalars(c, self.mode, i) for i, c in enumerate(self.clouds)]
        got = [v for v in vals if v is not None]
        if self.mode == "file" or not got:
            for i, a in enumerate(self.actors):
                a.GetMapper().ScalarVisibilityOff()
                a.GetProperty().SetColor(*FILE_COLORS[i % len(FILE_COLORS)])
            self.bar.SetVisibility(0)
            self._caption("flat colour per cloud" if self.mode == "file"
                          else f"{self.mode}: not in this cloud")
            return
        cat = got[0][2]
        allv = np.concatenate([v[0] for v in got])
        lo, hi = (float(allv.min()), float(allv.max())) if not cat else (0, 0)
        if not cat:
            lo, hi = float(np.percentile(allv, 1)), float(np.percentile(allv, 99))
            if hi - lo < 1e-9:
                hi = lo + 1e-9
        lut = vtk.vtkLookupTable()
        if cat:
            ids = np.unique(allv)
            lut.SetNumberOfTableValues(max(2, len(ids)))
            lut.Build()
            for j in range(lut.GetNumberOfTableValues()):
                h = (j * 0.61803398875) % 1.0
                lut.SetTableValue(j, *colorsys.hsv_to_rgb(h, 0.65, 1.0), 1.0)
            lut.SetTableRange(float(ids.min()), float(ids.max()) + 1e-9)
        else:
            lut = ramp(lo, hi)
        for i, (a, pd, v) in enumerate(zip(self.actors, self.polys, vals)):
            if v is None:
                a.GetMapper().ScalarVisibilityOff()
                continue
            arr = numpy_to_vtk(np.ascontiguousarray(v[0], dtype=np.float64), deep=1)
            arr.SetName(v[1])
            pd.GetPointData().SetScalars(arr)
            self.filters[i].Update()
            m = a.GetMapper()
            m.SetLookupTable(lut); m.SetScalarRange(lut.GetTableRange())
            m.ScalarVisibilityOn()
        self.bar.SetLookupTable(lut); self.bar.SetTitle(got[0][1]); self.bar.SetVisibility(1)
        self._caption(got[0][1])

    def _aim(self, cam=None):
        cam = cam or self.ren.GetActiveCamera()
        d, up = self.VIEWS.get(self.view, self.VIEWS["iso"])
        if self.flip:                                # up is -Z: for a cloud whose sensor
            d = (d[0], d[1], -d[2])                  # convention turns out to be inverted
            up = (up[0], up[1], -up[2]) if up[2] else up
        # look at the middle of the room, not at the sensor: with an upward-looking L2 the
        # sensor sits at the bottom edge of everything it can see.
        p = np.vstack([c["xyz"] for c in self.clouds if len(c["xyz"])])
        f = np.percentile(p, 50, axis=0) if len(p) else np.zeros(3)
        cam.SetFocalPoint(*f)
        cam.SetPosition(f[0] + d[0], f[1] + d[1], f[2] + d[2])
        cam.SetViewUp(*up)

    def _caption(self, what):
        def label(i, c):
            n = f"{i+1}:{c['name']} ({len(c['xyz'])})"
            if "frames" in c:                    # a live cloud says how deep the window is
                n += f" live {c['frames']}f/{c['frames_seen']}"
            return n
        names = "   ".join(label(i, c) for i, c in enumerate(self.clouds))
        st = getattr(self, "_stream", None)
        if st is not None and st.err is not None:
            names += f"   [stream ended: {st.err}]"

        self.text.SetInput(f"{names}\ncolour {self.mode} — {what}"
                           f"    view {self.view}{' flipped' if self.flip else ''}"
                           "   [c] colour  [v] view  [f] flip  [+/-] size  [g] grid  "
                           "[1-9] show/hide  [p] png  [r] reset  [q] quit")

    # ------------------------------------------------------------------------ the run
    def show(self, png=None, size=(1500, 1000), title="pcview", watch=None, mm=False,
             stream=None, stream_index=0, stream_hz=12.0):
        rw = vtk.vtkRenderWindow()
        rw.AddRenderer(self.ren); rw.SetSize(*size); rw.SetWindowName(title)
        if png:
            rw.SetOffScreenRendering(1)
        # orientation FIRST, then ResetCamera to fit: ResetCamera only moves the camera
        # along the direction it already has, and setting view-up to +Z afterwards makes
        # it parallel to the default -Z view direction, which is degenerate (an empty
        # window with one dot in it, which is exactly what that looked like).
        cam = self.ren.GetActiveCamera()
        self._aim(cam)
        self.ren.ResetCamera()
        self.ren.ResetCameraClippingRange()
        rw.Render()
        if png:
            w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(rw); w2i.Update()
            wr = vtk.vtkPNGWriter(); wr.SetFileName(png)
            wr.SetInputConnection(w2i.GetOutputPort()); wr.Write()
            print("wrote", png)
            return
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(rw)
        iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        axes = vtk.vtkAxesActor()
        omw = vtk.vtkOrientationMarkerWidget()
        omw.SetOrientationMarker(axes); omw.SetInteractor(iren)
        omw.SetViewport(0.0, 0.0, 0.16, 0.24)
        omw.SetEnabled(1); omw.InteractiveOff()
        iren.AddObserver("KeyPressEvent", self._key)
        self._rw, self._iren = rw, iren
        iren.Initialize()
        if watch:
            self._arm_watch(watch, mm)
        if stream is not None:
            self._arm_stream(stream, stream_index, hz=stream_hz)
        iren.Start()

    def _key(self, obj, ev):
        k = obj.GetKeySym()
        if k == "c":
            self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
            self.recolor()
        elif k in ("plus", "equal", "minus"):
            self.psize = max(1.0, self.psize + (0.5 if k != "minus" else -0.5))
            for a in self.actors:
                a.GetProperty().SetPointSize(self.psize)
        elif k == "g":
            self.grid.SetVisibility(not self.grid.GetVisibility())
        elif k and k.isdigit() and 1 <= int(k) <= len(self.actors):
            a = self.actors[int(k) - 1]
            a.SetVisibility(not a.GetVisibility())
        elif k == "p":
            out = os.path.join(CAD, "out", "pcview.png")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(self._rw); w2i.Update()
            wr = vtk.vtkPNGWriter(); wr.SetFileName(out)
            wr.SetInputConnection(w2i.GetOutputPort()); wr.Write()
            print("wrote", out)
        elif k == "f":
            self.flip = not self.flip
            self._aim(); self.ren.ResetCamera(); self.ren.ResetCameraClippingRange()
            self._caption(self.bar.GetTitle() or self.mode)
        elif k == "v":
            names = list(self.VIEWS)
            self.view = names[(names.index(self.view) + 1) % len(names)]
            self._aim(); self.ren.ResetCamera(); self.ren.ResetCameraClippingRange()
            self._caption(self.bar.GetTitle() or self.mode)
        elif k == "r":
            self._aim(); self.ren.ResetCamera(); self.ren.ResetCameraClippingRange()
        self._rw.Render()

    def _arm_stream(self, stream, index, hz=12.0):
        """Swap the live cloud's points on a timer.  Never blocks on the socket: the
        reader thread already has the newest window, and if it has died we keep drawing
        the last good cloud and say so in the caption rather than tearing down a window
        the user is still looking at."""
        self._auto = None

        def tick(obj, ev):
            c = stream.cloud()
            if not len(c["xyz"]) and stream.err is None:
                return
            self.clouds[index] = c
            pd = self.polys[index]
            pts = vtk.vtkPoints()
            pts.SetData(numpy_to_vtk(np.ascontiguousarray(c["xyz"]), deep=1))
            pd.GetPointData().Initialize()       # old scalars are the wrong length
            pd.SetPoints(pts)
            pd.Modified()
            self.filters[index].Update()
            self.recolor()
            if self._auto is None and len(c["xyz"]):
                self.ren.ResetCamera()           # frame the room once, on the first cloud
                self.ren.ResetCameraClippingRange()
                self._auto = True
            self._rw.Render()

        self._iren.AddObserver("TimerEvent", tick)
        self._iren.CreateRepeatingTimer(max(20, int(1000 / hz)))
        self._stream = stream

    # --watch: the file is being rewritten by whatever is capturing; follow it.
    def _arm_watch(self, paths, mm, hz=4.0):
        stamp = {p: _mtime(p) for p in paths}

        def tick(obj, ev):
            changed = [p for p in paths if _mtime(p) != stamp.get(p)]
            if not changed:
                return
            time.sleep(0.05)                           # let the writer finish the file
            for p in changed:
                stamp[p] = _mtime(p)
                i = [j for j, c in enumerate(self.clouds)
                     if c["name"] == os.path.basename(p)]
                if not i:
                    continue
                try:
                    c = load_file(p, mm)
                except SystemExit as e:
                    print(e); continue
                self.clouds[i[0]] = c
                pts = vtk.vtkPoints()
                pts.SetData(numpy_to_vtk(np.ascontiguousarray(c["xyz"]), deep=1))
                pd = self.polys[i[0]]
                pd.GetPointData().Initialize()      # the old scalars are the wrong length
                pd.SetPoints(pts)
                pd.Modified()
                self.filters[i[0]].Update()         # re-emit one vertex cell per point
            self.recolor()
            self._rw.Render()

        self._iren.AddObserver("TimerEvent", tick)
        self._iren.CreateRepeatingTimer(int(1000 / hz))


def _mtime(p):
    try:
        return os.stat(p).st_mtime_ns
    except OSError:
        return None


# ======================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="view a point cloud — a file, or the modelled L2 out of the sim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="units are metres (--mm if your file is in CAD millimetres)")
    ap.add_argument("files", nargs="*",
                    help=".pcd / .npy / .npz / .bin / .csv / .xyz / .txt, or - for stdin")
    ap.add_argument("--sim", action="store_true",
                    help="also scan the modelled L2 (lidar.py) off out/sim/mini_dog.xml")
    ap.add_argument("--sim-xml", metavar="XML", help="scan this MJCF instead")
    ap.add_argument("--terrain", action="store_true",
                    help="--sim on mini_dog_terrain.xml, so the cloud has ground in it")
    ap.add_argument("--frame", choices=("local", "world"), default="local",
                    help="sim cloud in the sensor frame (default) or in world coordinates")
    ap.add_argument("--seed", type=int, default=0, help="scan pattern seed for --sim")
    ap.add_argument("--mm", action="store_true", help="input files are in millimetres")
    ap.add_argument("-c", "--color", choices=MODES, default=None,
                    help="colour by (default: height if the cloud could be levelled,"
                         " else range; [c] cycles in the window)")
    ap.add_argument("--view", choices=("iso", "top", "front", "side"), default="iso",
                    help="camera preset ([v] cycles in the window). top reads like a"
                         " floor plan, which is usually the one that makes a room legible")
    ap.add_argument("--level", dest="level", action="store_true", default=True,
                    help="stand the cloud upright using the sensor's accelerometer"
                         " (the default whenever the capture carries one)")
    ap.add_argument("--no-level", dest="level", action="store_false",
                    help="leave the cloud in the sensor frame")
    ap.add_argument("--up", metavar="X,Y,Z",
                    help="up direction in the cloud's own frame, if the capture has no"
                         " accelerometer in it (e.g. --up 0,0,1)")
    ap.add_argument("--clip", metavar="LO,HI",
                    help="keep only points between these heights above the floor, in"
                         " metres — indoors this is the difference between a room and a"
                         " lid, because the ceiling is often half the cloud."
                         " e.g. --clip -0.3,1.8 for a floor plan")
    ap.add_argument("--max-range", type=float, metavar="M",
                    help="drop points beyond M metres — a handful of far outliers"
                         " stretch the colour scale over everything else")
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--stats", action="store_true", help="print the numbers, no window")
    ap.add_argument("--png", metavar="OUT", help="render offscreen to a PNG and exit")
    ap.add_argument("--save", metavar="OUT.npz",
                    help="write the first cloud back out as .npz (xyz + any scalars)")
    ap.add_argument("--watch", action="store_true",
                    help="reload the files when they change on disk, and redraw")
    ap.add_argument("--stream", metavar="HOST[:PORT]",
                    help="live cloud from tools/stream_pcd.cpp running on the lidar's host"
                         " (default port 9910) — e.g. --stream 10.0.1.47")
    ap.add_argument("--accum", type=int, default=12, metavar="N",
                    help="how many live frames to keep on screen at once (default 12,"
                         " i.e. one second at the L2's 12 Hz)")
    # argparse reads "--clip -0.3,1.8" as a missing value followed by an unknown option,
    # because the value starts with a minus.  A height clip whose lower bound is below the
    # floor is the normal case, so join it up rather than making everyone type "=".
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, tok in enumerate(argv[:-1]):
        if tok in ("--clip", "--up") and argv[i + 1].startswith("-") \
                and not argv[i + 1][1:2].isalpha():
            argv[i] = f"{tok}={argv[i + 1]}"
            del argv[i + 1]
            break
    a = ap.parse_args(argv)

    clouds = [load_file(p, a.mm) for p in a.files]

    stream = stream_index = None
    if a.stream:
        host, _, port = a.stream.partition(":")
        try:
            stream = Stream(host, port or 9910, accum=a.accum, do_level=a.level)
        except OSError as e:
            raise SystemExit(f"--stream {a.stream}: {e}\n"
                             "  is stream_pcd running on that host?  On the Pi:\n"
                             "    ~/unilidar_sdk2/unitree_lidar_sdk/bin/stream_pcd")
        print(f"connected to {stream.name}, waiting for the first frame ...")
        for _ in range(100):                     # ~5 s: 12 Hz, so this is generous
            if stream.frames or stream.err:
                break
            time.sleep(0.05)
        if stream.err and not stream.frames:
            raise SystemExit(f"--stream: {stream.err}")
        stream_index = len(clouds)
        clouds.append(stream.cloud())

    if a.sim or a.sim_xml:
        clouds.append(load_sim(a.sim_xml, a.terrain, a.frame, seed=a.seed))
    if not clouds:
        ap.error("nothing to show — give a file, --sim, or --stream")

    if a.max_range:
        for i, c in enumerate(clouds):
            keep = np.linalg.norm(c["xyz"], axis=1) <= a.max_range
            if not keep.all():
                print(f"  {c['name']}: dropped {int((~keep).sum())} points beyond "
                      f"{a.max_range} m")
                clouds[i] = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep)
                                 else v) for k, v in c.items()}

    up = tuple(float(v) for v in a.up.split(",")) if a.up else None
    if a.level:
        for i, c in enumerate(clouds):
            lc, tilt = level(c, up)
            if tilt is not None:
                clouds[i] = lc
                print(f"  {c['name']}: levelled, {tilt:.1f}° off upright")
            elif c.get("frame") == "sensor":
                print(f"  {c['name']}: no accelerometer in this capture — not levelled"
                      " (pass --up X,Y,Z if you know which way is up)")

    if a.clip:
        lo, _, hi = a.clip.partition(",")
        lo, hi = float(lo), float(hi or "inf")
        lev = [c for c in clouds if c.get("frame") == "level" and len(c["xyz"])]
        if not lev:
            print("  --clip needs a levelled cloud: heights are measured from the floor,"
                  " and an unlevelled cloud has no floor")
        else:
            g = ground_z(lev[0]["xyz"])
            for i, c in enumerate(clouds):
                h = c["xyz"][:, 2] - g
                keep = (h >= lo) & (h <= hi)
                if not keep.all():
                    print(f"  {c['name']}: clipped to {lo:+.2f}..{hi:+.2f} m above the"
                          f" floor, {int(keep.sum())} of {len(keep)} points kept")
                    clouds[i] = {k: (v[keep] if isinstance(v, np.ndarray)
                                     and len(v) == len(keep) else v)
                                 for k, v in c.items()}

    if a.color is None:
        a.color = "height" if any(c.get("frame") == "level" for c in clouds) else "range"

    for c in clouds:
        stats(c, verbose=a.stats)
    if a.save:
        c = clouds[0]
        np.savez(a.save, **{k: v for k, v in c.items()
                            if isinstance(v, np.ndarray)})
        print("wrote", a.save)
    if a.stats:
        if stream is not None:
            stream.close()
        return 0

    v = Viewer(clouds, a.color, a.point_size, view=a.view)
    if a.png:
        v.show(png=a.png)
        if stream is not None:
            stream.close()
        return 0
    # --watch follows real files only; stdin is read once and cannot be re-read.
    watch = [p for p in a.files if p != "-"] if a.watch else None
    if a.watch and not watch:
        print("--watch: nothing to watch (stdin is not a file)")
    try:
        v.show(watch=watch, mm=a.mm, stream=stream, stream_index=stream_index or 0)
    finally:
        if stream is not None:
            stream.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
