"""Print-orientation screen: for every build direction on the sphere, how much of the part
would sit unsupported and how much of it lands on the bed.

Reads `out/stl/*.stl`, which are already in mini_dog's PRINT_ORIENT frame, so a row is an
EXTRA rotation to compose onto the current entry; +Z is the current orientation.  This is
geometry only and it OVER-counts - it cannot see whether a down-facing face has the part's
own body under it, so treat it as a screen and settle the choice by slicing.  It says
nothing about strength: for hip_bracket / thigh / shin the build direction is a strength
decision, and `fea.py --all --orient` is what scores it.

    .venv/bin/python tools/orient_scan.py [part ...]
"""
import numpy as np, struct, sys, math, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESH = 30.0          # Orca support_threshold_angle, degrees from vertical

def read_stl(path):
    with open(path, "rb") as fh:
        head = fh.read(84)
        if head[:5] == b"solid" and b"facet" in fh.read(512):
            raise SystemExit("ascii stl not handled: " + path)
        n = struct.unpack("<I", head[80:84])[0]
        buf = np.frombuffer(fh.read(50 * n), dtype=np.uint8).reshape(n, 50)
    f = buf[:, :48].copy().view("<f4").reshape(n, 4, 3)
    return f[:, 1:, :].astype(float)          # (n, 3, 3) triangle vertices

def metrics(tri, n):
    """(height mm, overhang area mm2, bed area mm2) for build direction n"""
    n = np.asarray(n, float); n /= np.linalg.norm(n)
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ar = 0.5 * np.linalg.norm(cr, axis=1)
    ok = ar > 1e-9
    ar, cr, tri = ar[ok], cr[ok], tri[ok]
    down = -(cr @ n) / (2 * ar)               # 1 = facing straight down
    h = tri.reshape(-1, 3) @ n
    z0, z1 = h.min(), h.max()
    cz = (tri.mean(axis=1)) @ n
    on_bed = cz < z0 + 0.4
    over = float(ar[(down > math.cos(math.radians(90 - THRESH))) & ~on_bed].sum())
    bed  = float(ar[(down > 0.99) & on_bed].sum())
    return float(z1 - z0), over, bed

def dirs(n=1000):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def axis_angle(n):
    """rotation taking direction n to +Z, as (axis, degrees) - the extra PRINT_ORIENT"""
    n = np.asarray(n, float); n /= np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.clip(n @ z, -1, 1))
    if c > 1 - 1e-9:  return (1.0, 0.0, 0.0), 0.0
    if c < -1 + 1e-9: return (1.0, 0.0, 0.0), 180.0
    ax = np.cross(n, z); ax /= np.linalg.norm(ax)
    return tuple(round(float(v), 6) for v in ax), math.degrees(math.acos(c))

if __name__ == "__main__":
    parts = sys.argv[1:] or ["chassis_bottom", "chassis_top", "lidar_mount",
                             "hip_bracket_A", "thigh_A", "shin_A", "foot", "servo_gauge"]
    out = {}
    for p in parts:
        tri = read_stl(os.path.join(ROOT, "out", "stl", p + ".stl"))
        D = dirs(1000)
        m = np.array([metrics(tri, d) for d in D])
        cur = metrics(tri, (0, 0, 1))
        order = np.lexsort((m[:, 0], m[:, 1]))          # overhang, then height
        print(f"\n  {p}   current: overhang {cur[1]:7.0f} mm2  height {cur[0]:5.1f} mm"
              f"  bed {cur[2]:5.0f} mm2")
        # the six axis-aligned directions first - a flip keeps the part flat on the bed,
        # which the free-direction ranking below does not weigh
        for lbl, d in (("+Z (current)", (0, 0, 1)), ("-Z", (0, 0, -1)), ("+X", (1, 0, 0)),
                       ("-X", (-1, 0, 0)), ("+Y", (0, 1, 0)), ("-Y", (0, -1, 0))):
            hh, ov, bd = metrics(tri, d)
            print(f"     {lbl:<13} overhang {ov:7.0f} mm2  height {hh:5.1f} mm  bed {bd:5.0f} mm2")
        seen, rows = [], []
        for i in order:
            d = D[i]
            if any(d @ s > 0.985 for s in seen):        # keep the list diverse
                continue
            seen.append(d)
            ax, ang = axis_angle(d)
            rows.append(dict(dir=[round(float(v), 4) for v in d], axis=ax, angle=round(ang, 1),
                             overhang=round(m[i, 1]), height=round(m[i, 0], 1),
                             bed=round(m[i, 2])))
            print(f"     overhang {m[i,1]:7.0f} mm2  height {m[i,0]:5.1f} mm  bed {m[i,2]:5.0f} mm2"
                  f"   extra rot {ax} {ang:5.1f} deg")
            if len(rows) >= 6:
                break
        out[p] = dict(current=dict(overhang=round(cur[1]), height=round(cur[0], 1),
                                   bed=round(cur[2])), best=rows)
    json.dump(out, open(os.path.join(ROOT, "out", "orient_scan.json"), "w"), indent=1)
