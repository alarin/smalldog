"""
terrain.py — procedural MuJoCo heightfield ground, shared by both sim exporters.

`export_sim.py --terrain` and `../ros2/.../generate_model.py --terrain` both call
`write(dir)` here; it drops a `terrain.png` next to the model and hands back the
`<hfield>`/`<geom>` attributes that put the robot on it instead of on a flat plane.

The field is deterministic (seeded value noise), so two exporters — and two runs — get
the same ground, and a self-test on terrain stays comparable between runs.  It is always
flat inside FLAT_R of the origin, with a smooth fade out to the noise, so the robot
spawns level no matter what the seed does.  FLAT_R is the stance footprint and nothing
more: a wider pad puts the robot in the middle of a parking lot, and since the viewer
opens looking straight at it, the ground reads as flat and the terrain looks broken.

AMP_MM is a gait number, not a taste one — but the gait turned out to have far more room
than expected, so it is set for what reads as terrain in the viewer.  The walker swings a
foot 22 mm above the stance plane; since 2026-08-27 it also levels the body on the IMU and
stands where a foot lands rather than where the profile said it would, which is what makes
that affordable.  Measured on the 5 s headless trot (`tools/standalone_sim.py --headless
--terrain`, 0.20 m/s commanded, 793 mm on flat ground), over twelve terrain seeds:

    relief      blind gait                 with terrain feedback
    +-27 mm     574 +-67 mm, fell 2/12     628 +-79 mm, fell 0/12   <- the default
    +-53 mm     504 +-86 mm, fell 5/12     447 +-70 mm, fell 0/12

Two things that look wrong but are not: the default relief is *taller* than the foot swing
(the slopes are smooth, so the foot lands on a hillside rather than stubbing into a step),
and the blind gait's distance barely responds to the amplitude at all.  A dead-flat
heightfield already costs it ~15 % against `type="plane"` — that penalty is the hfield
contact, not the relief — and +-15 mm costs the same as +-30 mm.  So AMP_MM is not the
knob for "the dog is struggling"; the controller is.

Since 2026-08-28 the field also carries an obstacle course - ramps, walls and logs, laid
along +x, described at COURSE below.  The relief alone is smooth by construction (160 mm
is its longest feature), so a foot always lands on a hillside and never meets an edge; the
course is the part the heightfield cannot express.  It starts at x = 0.95 m, past the
0.66 m the 5 s regression trot reaches, so that measurement is unchanged - verified, both
arms come out at 663 +-62 mm over the same six seeds.  `standalone_sim.py --course` is the
one that walks it.  Six seeds, 25 s at 0.20 m/s:

    relief only          2917 +-187 mm, fell 0/6
    relief + course      2713 +-291 mm, fell 0/6
    ... blind            1353 +-755 mm, fell 3/6

The course cost ~1300 mm and two falls in six until 2026-08-28, and almost none of that
was the obstacles: the gait held no heading, so the first thing an obstacle did was knock
it off course, and it then walked out of the 0.80 m corridor sideways - 1.65 m off the
centreline by 25 s on the default seed.  Closing that loop (`gait.yaw_*`) took the default
seed from 1 obstacle cleared to 5, and the course now costs ~200 mm.  Before reshaping
anything here because "the robot cannot do it", check that it is not simply leaving.

Two traps for anyone re-measuring this:

  * one seed is not a measurement.  At fixed settings the blind trot spreads +-67 mm over
    seeds, and two of twelve seeds put it on its back.  An earlier revision of this
    docstring read a +-18/+-27/+-40 mm table off single runs and concluded the amplitude
    mattered and that "attitude stayed inside 3 deg in every run"; neither survived a
    twelve-seed sweep, where the median tilt is 13 deg blind and 7 deg closed-loop.
  * an obstacle geom's rotation goes in as a *quaternion*.  Both consumers compile with
    <compiler angle="radian">, so an euler written in degrees is read as radians and says
    nothing about it: the first version of this course had a "6 degree" ramp that came out
    tilted 16 degrees the other way and 75 mm tall, and logs turned by 90 radians.  Walls
    carry no rotation, so walls were the only thing that behaved - which is what the
    measurements were saying long before anyone believed them.  Ray-cast the compiled
    scene, not a hand-built test scene: a test scene without the <compiler> line uses
    degrees and will happily confirm geometry the real model does not have.
  * MuJoCo caches a heightfield by *file name* within a process.  Rewriting terrain.png
    per seed and reloading the scene silently reuses whichever field was compiled first,
    so a sweep that looks like twelve terrains is one terrain twelve times.  Give each
    seed its own png.
"""
import math
import os
import numpy as np

AMP_MM     = 30.0     # +- this, before the flat pad is blended in
WAVELEN_MM = 160.0    # longest feature; octaves go down from here
OCTAVES    = 4
HALF_M     = 4.0      # heightfield half-extent, so 8 x 8 m of ground
FLAT_R_M   = 0.16     # dead-flat spawn pad: the stance footprint (+-90 x +-70 mm), no more
FADE_M     = 0.25     # ... blended out to the noise over this much more
BASE_M     = 0.05     # solid slab carried below the lowest point
CELL_MM    = 12.0     # heightfield resolution; keep >=10 cells per wavelength
SEED       = 7


def _value_noise(n, cells, rng):
    """n x n smooth noise in [-1, 1] from a (cells+1)^2 random lattice."""
    lat = rng.uniform(-1.0, 1.0, (cells + 1, cells + 1))
    t = np.linspace(0.0, cells, n)
    i = np.clip(t.astype(int), 0, cells - 1)
    f = t - i
    f = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)          # smootherstep
    fx, fy = np.meshgrid(f, f, indexing="ij")
    ix, iy = np.meshgrid(i, i, indexing="ij")
    a, b = lat[ix, iy],         lat[ix + 1, iy]
    c, d = lat[ix, iy + 1],     lat[ix + 1, iy + 1]
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
            + c * (1 - fx) * fy + d * fx * fy)


def height_mm(amp_mm=AMP_MM, wavelen_mm=WAVELEN_MM, half_m=HALF_M,
              flat_r_m=FLAT_R_M, fade_m=FADE_M, cell_mm=CELL_MM, seed=SEED):
    """Square height map in mm, mean ~0, exactly 0 on the spawn pad."""
    span_mm = 2.0 * half_m * 1000.0
    n = int(round(span_mm / cell_mm)) + 1
    rng = np.random.default_rng(seed)

    h, gain, cells = np.zeros((n, n)), 1.0, max(2, int(round(span_mm / wavelen_mm)))
    norm = 0.0
    for _ in range(OCTAVES):
        h += gain * _value_noise(n, cells, rng)
        norm += gain
        gain *= 0.5
        cells *= 2
    h *= amp_mm / norm

    xs = np.linspace(-half_m, half_m, n)
    r = np.hypot(*np.meshgrid(xs, xs, indexing="ij"))
    t = np.clip((r - flat_r_m) / max(1e-6, fade_m), 0.0, 1.0)
    return h * (t * t * (3.0 - 2.0 * t))                   # smoothstep pad -> noise


def write(dirpath, name="terrain.png", obstacles=True, **kw):
    """Write the heightfield image and return the MJCF numbers that go with it.

    MuJoCo maps the image's 0..1 range onto 0..size[2] *above* the geom's own z, so the
    geom sits at the field's minimum and the spawn pad lands exactly on z = 0.

    `obstacles` adds the course from COURSE below, bedded into *this* field.  It comes
    back as `["obstacles"]`, ready for `obstacle_xml()`; the sampling and the png are
    done from one array so the two can never be built from different noise.
    """
    from PIL import Image
    h = height_mm(**kw)
    lo, hi = float(h.min()), float(h.max())
    span = max(hi - lo, 1e-6)
    img = np.rint((h - lo) / span * 255.0).astype(np.uint8)
    path = os.path.join(dirpath, name)
    Image.fromarray(img, mode="L").save(path)
    half = kw.get("half_m", HALF_M)
    return dict(file=name, nrow=h.shape[0], size=(half, half, span / 1000.0, BASE_M),
                pos_z=lo / 1000.0, amp_mm=max(hi, -lo),
                obstacles=(course_geoms(h, half_m=half,
                                        clear_r_m=kw.get("flat_r_m", FLAT_R_M))
                           if obstacles else []))


def size_attr(t):
    return " ".join(f"{v:.6g}" for v in t["size"])


# =====================================================================================
# The obstacle course.
#
# The heightfield alone is *smooth*: 160 mm is its longest feature and the octaves below
# that are gentler still, so a foot always lands on a hillside and never stubs into
# anything.  That is why the blind trot barely notices the amplitude (see above).  Ramps,
# walls and logs are the part the relief cannot express - a slope the body has to be
# levelled onto, an edge the swing has to clear, and a round crest the foot rolls off.
#
# Everything sits on the centreline y = 0 and is laid out along +x, because that is the
# direction the headless trot walks.  The 5 s run reaches ~0.75 m, so it meets the log and
# the low wall and nothing else; the ramp and the taller pair are there for the viewer and
# for longer runs.  Widths are ~0.8 m against a 92 mm body: walking around is not an
# option, and the gait has no perception to do it with anyway.
#
# Heights are quoted against the ground *at the centreline*, which is where the robot
# crosses.  Away from it the relief runs out from under a wall or a log by up to the full
# +-27 mm, so the ends bridge dips and sink into rises.  That is what a log lying on rough
# ground looks like, and it is why the walls are buried BURY_M deep rather than sized to
# the nominal height.
#
# What the trot can actually do, measured on flat ground so the relief does not muddy it
# (8 s at 0.20 m/s, one obstacle, closed loop; clear ground reaches 1252 mm):
#
#     ramp   4 deg 1240   6 deg 1102   8 deg 1155   10 deg 1122   14 deg  950 mm
#     wall   6 mm  1249   14 mm 1183   18 mm  889   22 mm  474    26 mm  461 mm
#     log    6 mm  1251   14 mm 1141   22 mm  875   30 mm  432 mm
#
# So: ramps to at least 14 deg, walls to ~18 mm, logs to ~22 mm.  The wall cliff between
# 18 and 22 mm is the 22 mm foot swing, exactly; a log gets a few mm more because the foot
# rolls over the crest instead of catching a square edge.  The course below is graded to
# that — the first two are comfortable, the last two are at the limit — and the relief
# under each one moves it by up to the full +-27 mm, so on a rise the far pair is past it.
#
# It starts at x = 0.95 and not sooner on purpose.  The 5 s trot of
# `standalone_sim.py --headless --terrain` reaches ~0.66 m, and CLAUDE.md reads that
# distance as the regression signal for a mass or joint-limit change.  Put an obstacle
# inside it and the number stops meaning that: measured with the course at x = 0.55 it
# fell from 663 +-62 to 486 +-93 mm, so a real regression would have to beat the noise the
# course adds.  `--course` below runs long enough to cross it and is judged separately.
COURSE = (
    dict(kind="log",  x=0.95, r=0.050, h=0.014, w=0.80),
    dict(kind="ramp", x=1.40, run=0.35, rise=0.049, top=0.25, w=0.70),
    dict(kind="wall", x=2.60, t=0.030, h=0.015, w=0.80),
    dict(kind="log",  x=3.00, r=0.070, h=0.022, w=0.80),
    dict(kind="wall", x=3.45, t=0.030, h=0.020, w=0.80),
)
BURY_M  = 0.12    # how far a wall/ramp solid carries below its nominal ground
RAMP_T  = 0.12    # ramp slab thickness; > the relief span, so only the top face shows
RGBA    = dict(wall="0.46 0.44 0.40 1", ramp="0.52 0.48 0.42 1", log="0.35 0.26 0.17 1")


def ground_mm(h, x_m, y_m, half_m=HALF_M):
    """Height of the field at a world (x, y), in mm — as MuJoCo will have built it.

    The array is *not* indexed [x, y].  `height_mm` builds it with indexing="ij" and PIL
    then writes axis 0 as image rows, so once MuJoCo has loaded the png the world x is the
    column and the world y is the row, counted the other way.  Established by ray-casting
    straight down onto the compiled hfield and comparing against both candidates: this one
    lands within a fraction of a mm, the transpose is out by tens of mm.  If an obstacle
    ever floats or drowns, re-run that check before touching the arithmetic below.
    """
    n = h.shape[0]
    c = (x_m + half_m) / (2.0 * half_m) * (n - 1)
    r = (n - 1) - (y_m + half_m) / (2.0 * half_m) * (n - 1)
    c = min(max(c, 0.0), n - 1.0)
    r = min(max(r, 0.0), n - 1.0)
    c0, r0 = int(c), int(r)
    c1, r1 = min(c0 + 1, n - 1), min(r0 + 1, n - 1)
    tc, tr = c - c0, r - r0
    return float((h[r0, c0] * (1 - tc) + h[r0, c1] * tc) * (1 - tr)
                 + (h[r1, c0] * (1 - tc) + h[r1, c1] * tc) * tr)


def _box(name, kind, pos, size, quat=None):
    return dict(name=name, type="box", pos=pos, size=size, quat=quat, rgba=RGBA[kind])


def _quat_y(th):
    """rotation about +y by th radians, as a quaternion.

    Quaternions, not euler, and not by taste: both consumers compile their MJCF with
    <compiler angle="radian">, so an euler written in degrees is silently read as radians.
    A "6 degree" ramp came out tilted 16 the other way and 75 mm tall, and a log written
    euler="90 0 0" was turned by 90 radians.  A quat has no unit to get wrong.
    """
    return (math.cos(th / 2.0), 0.0, math.sin(th / 2.0), 0.0)


def _ramp(tag, kind, x_lo, z_lo, x_hi, z_hi, w):
    """One slab whose *top face* runs from (x_lo, z_lo) to (x_hi, z_hi).

    A rotated box, not a wedge — MuJoCo has no wedge.  RAMP_T is thick enough that the
    underside stays below the ground over the whole run, so what shows is the top face and
    two buried edges.  The centre is the top face's midpoint pushed back along the slab's
    own -z: for a rotation of alpha about y, the local (0,0,T/2) lands at
    (T/2 sin(alpha), 0, T/2 cos(alpha)) with alpha = -th, so the centre is the midpoint
    *plus* c sin(th) in x.  Get that sign backwards and the slab slides ~2 c sin(th) along
    x, which opens a notch at every seam where two pieces meet — invisible in the geom
    numbers, obvious the moment you ray-cast the deck joints.
    """
    run, dz = x_hi - x_lo, z_hi - z_lo
    th = math.atan2(dz, run)                       # >0 climbing, <0 descending
    c = RAMP_T / 2.0
    return _box(tag, kind,
                (x_lo + run / 2.0 + c * math.sin(th), 0.0,
                 (z_lo + z_hi) / 2.0 - c * math.cos(th)),
                (math.hypot(run, dz) / 2.0, w / 2.0, c),
                _quat_y(-th))


def course_geoms(h, course=COURSE, half_m=HALF_M, clear_r_m=FLAT_R_M):
    """Bed `course` into the field `h` and return MJCF geom dicts (SI, metres)."""
    out = []
    for k, ob in enumerate(course, 1):
        kind, x, w = ob["kind"], ob["x"], ob["w"]
        g = ground_mm(h, x, 0.0, half_m) / 1000.0
        if kind == "wall":
            t, hgt = ob["t"], ob["h"]
            half_z = (hgt + BURY_M) / 2.0
            out.append(_box(f"obs{k}_wall", kind, (x, 0.0, g + hgt - half_z),
                            (t / 2.0, w / 2.0, half_z)))
            lo, hi = x - t / 2.0, x + t / 2.0
        elif kind == "log":
            # bedded, not resting: a cylinder sunk until its crest is h above the ground.
            # r > h, so the buried part is what keeps a foot from getting under the edge.
            r, hgt = ob["r"], ob["h"]
            q = math.sqrt(0.5)          # about +x by 90 deg: the axis ends up along y
            out.append(dict(name=f"obs{k}_log", type="cylinder", pos=(x, 0.0, g + hgt - r),
                            size=(r, w / 2.0), quat=(q, q, 0.0, 0.0), rgba=RGBA[kind]))
            lo, hi = x - r, x + r
        elif kind == "ramp":
            run, rise, top = ob["run"], ob["rise"], ob["top"]
            x1, x2, x3 = x + run, x + run + top, x + 2 * run + top
            g_hi = ground_mm(h, x3, 0.0, half_m) / 1000.0
            # both ramps climb to the same deck; referencing the higher of the two ends
            # keeps the far ramp descending even where the relief rises under the deck
            deck = max(g, g_hi) + rise
            gmin = min(ground_mm(h, xx / 100.0, 0.0, half_m) / 1000.0
                       for xx in range(int(x1 * 100), int(x2 * 100) + 1))
            half_z = (deck - gmin + BURY_M) / 2.0
            out += [_ramp(f"obs{k}_ramp_up", kind, x, g, x1, deck, w),
                    _box(f"obs{k}_deck", kind, ((x1 + x2) / 2.0, 0.0, deck - half_z),
                         (top / 2.0, w / 2.0, half_z)),
                    _ramp(f"obs{k}_ramp_dn", kind, x2, deck, x3, g_hi, w)]
            lo, hi = x, x3
        else:
            raise ValueError(f"unknown obstacle {kind!r}")
        if lo < clear_r_m:
            raise ValueError(f"{kind} at x={x} reaches x={lo:.3f}, inside the spawn pad"
                             f" (r={clear_r_m}) — the robot would start on top of it")
    return out


def obstacle_xml(hf, indent="    ", attrs=""):
    """The `["obstacles"]` of a `write()` result as MJCF worldbody lines.

    `attrs` is appended verbatim: pass the *scene's own* floor friction and contact flags,
    so an obstacle is never slipperier or in a different contact group than the ground it
    stands on.  No material — the two exporters name their ground material differently and
    this file is not going to know which.
    """
    x = []
    for g in hf.get("obstacles", ()):
        e = f' quat="{" ".join(f"{v:.8g}" for v in g["quat"])}"' if g["quat"] else ""
        x.append(f'{indent}<geom name="{g["name"]}" type="{g["type"]}"'
                 f' pos="{" ".join(f"{v:.6g}" for v in g["pos"])}"'
                 f' size="{" ".join(f"{v:.6g}" for v in g["size"])}"{e}'
                 f' rgba="{g["rgba"]}"{attrs}/>')
    return "\n".join(x)
