"""
lidar.py — the Unitree L2 as a *sensor*, shared by both sim exporters and by the sim itself.

`mini_dog.py` already knows where the L2 is bolted (`lidar_pose()`, `LIDAR_OPT`) and what
it can see (`LIDAR_FOV*`, `LIDAR_RATE`, `LIDAR_R_*`, `LIDAR_SIGMA`).  This module is the
part that turns those into simulation: the MJCF fragments that put the sensor in the model,
and a `Scanner` that ray-casts it into a point cloud.  Like terrain.py it carries no CAD
and no second copy of a robot dimension.

    export_sim.py                       -> site + <custom> numerics in out/sim/*.xml
    ../ros2/.../generate_model.py       -> the same, in smalldog_description/mujoco/robot.xml
    ../ros2/tools/standalone_sim.py     -> Scanner, for --lidar
    ../ros2/src/mujoco_ros2_control     -> C++ MujocoLidar, the ROS 2 /points publisher

WHERE THE PARAMETERS LIVE, AND WHY THEY ARE IN THE MODEL FILE
The ROS 2 side of this is C++ and cannot import anything here, so the scan parameters are
written into the MJCF as `<custom><numeric name="lidar_*">` and read back out of the
*compiled* model by both consumers.  That is deliberate: the numbers exist once, in
mini_dog.py, and reach the C++ node through the file it already loads instead of through a
launch file that would immediately start drifting.  Nothing reads a lidar constant from
anywhere else, and a model without those numerics simply has no lidar - both consumers
check and say so rather than falling back to a built-in guess.

The scan *pattern* is the one thing that does exist twice, here and in the C++ node, for
the plain reason that one of them cannot call the other.  Both derive every number from
the numerics above; if you change the pattern, change it in both, and the check that they
still agree is `standalone_sim.py --lidar` against the ROS 2 topic on the same scene.

THE PATTERN, AND WHAT IS HONEST ABOUT IT
The L2's internal optics are not published.  What is published is the envelope: a
hemisphere about the sensor's own axis (LIDAR_FOV, plus 6 deg below the base plane in
NEGA), ~LIDAR_RATE points per second, and - the reason it is worth simulating at all -
that the scan is *non-repetitive*: unlike a spinning multi-beam lidar it does not retrace
the same rings, so standing still keeps filling the field in.

What is modelled here is a Risley pair: two counter-rotating wedge prisms, each deflecting
the beam by half the cone angle, which is the classic way to build exactly that kind of
scan.  It reproduces the coverage (0 .. LIDAR_FOV_NEGA from the axis, all azimuths), the
point rate, and the non-repetition - the two spin rates are in the golden ratio, so the
pattern never closes and a longer dwell always adds new directions.

What it does NOT reproduce is the density profile.  A Risley rosette piles its points up
where the sweep turns around, i.e. on the axis and on the rim, while the L2's manual says
its own density is highest at the middle of the vertical FOV (which is what LIDAR_TILT is
set to exploit - see mini_dog.py).  So: use this for geometry, coverage and occlusion; do
not use it to argue about how many returns a particular object gets.  Fixing that needs
the real pattern, not a better guess.

MOTION DISTORTION is not modelled either.  Every point of a frame is cast from the sensor
pose at the end of that frame's window, while a real sweeping lidar moves through it.  At
the trot's 0.2 m/s and FRAME_HZ = 10 that is 20 mm across a frame; if you ever care about
it, cast in chunks per sim step and accumulate in world coordinates - the pose is right
there in mjData either way.
"""
import math

import numpy as np

# --------------------------------------------------------------------------------------
# The pattern model.  These three are NOT sensor specifications - they are this file's
# guess at how the L2 fills its cone, in the same sense that MJ_DAMPING in export_sim.py
# is a guess at the ST3215's gearbox.  Everything the catalogue actually states lives in
# mini_dog.py.
#
# SPIN_A is the first prism's rate.  The second is SPIN_A / -PHI: counter-rotating, and in
# the golden ratio so the rosette never closes on itself.  Any rational ratio does close -
# at 121.6 / -77.7 Hz the whole pattern repeats every 10 s, which is 100 frames and looks
# fine right up until someone parks the robot and wonders why the cloud stops improving.
PHI      = (1.0 + 5.0 ** 0.5) / 2.0
SPIN_A   = 121.6                      # Hz, first prism (7300 rpm, Livox-class hardware)
SPIN_B   = -SPIN_A / PHI              # ... second, counter-rotating: -75.2 Hz
FRAME_HZ = 10.0                       # how often the sim emits an accumulated cloud

# Which geoms a ray may hit, by MuJoCo geom group.  Both exporters draw the printed solids
# as visual-only meshes in group 2 and put the physics in primitives (group 3, and the
# floor in group 0), so the mask below is "see what the physics collides with".  Casting
# against the visual meshes instead would be more faithful and roughly two orders of
# magnitude slower - a lightened bracket is thousands of triangles, and there are 2160
# rays in a frame.  It travels with the other parameters, in the model.
RAY_GROUPS = (1, 1, 0, 1, 1, 1)

# NO INTENSITY, deliberately.  mj_multiRay returns a distance and a geom id and nothing
# about the surface; the version of it in the ROS 2 workspace's MuJoCo (3.3) does not even
# return the normal.  A cloud with a plausible-looking made-up reflectance in it is worse
# than one with none, because somebody downstream will eventually threshold on it.


def spec(nega=True):
    """Everything the sim needs to know about the sensor, in SI, straight from the CAD.

    Imported lazily: this module is also used inside the ROS 2 workspace, where mini_dog
    (and CadQuery under it) is not installed and only the Scanner half is wanted.
    """
    import mini_dog as md
    return dict(
        cone=math.radians(md.LIDAR_FOV_NEGA if nega else md.LIDAR_FOV),
        rate=md.LIDAR_RATE,
        r_min=md.LIDAR_R_MIN / 1000.0,
        r_max=md.LIDAR_R_MAX / 1000.0,
        sigma=md.LIDAR_SIGMA / 1000.0,
        spin=(SPIN_A, SPIN_B),
        frame_hz=FRAME_HZ,
    )


def pose_m():
    """The optical centre and the sensor frame, in metres, in robot coordinates.

    Returns `(pos, quat_wxyz, rpy)`.  The frame is the sensor's own: +Z is the axis of its
    up-hemisphere, so the whole thing is the robot frame turned nose-down about +Y by
    LIDAR_TILT and moved out to the scan core.  That is *not* the usual ROS lidar
    convention of Z-up-X-forward, and it is on purpose: every FOV number in mini_dog.py is
    an angle from this axis, and a frame that hides the axis inside a quaternion would make
    every one of them unreadable.
    """
    import mini_dog as md
    (sx, sy, sz), n = md.lidar_pose()
    p = tuple((c + d * md.LIDAR_OPT) / 1000.0 for c, d in zip((sx, sy, sz), n))
    t = math.radians(md.LIDAR_TILT)
    return p, (math.cos(t / 2), 0.0, math.sin(t / 2), 0.0), (0.0, t, 0.0)


def _f(vals):
    return " ".join(f"{v:.6g}" for v in vals)


def site_xml(indent="      ", name="lidar", body=True):
    """The `<site>` that *is* the sensor, plus the sensor's own body, drawn.

    The site sits at the optical centre and carries the sensor frame; the body is the
    LIDAR_L2_BOX envelope at the pose mini_dog holds for it (`lidar_com()`), which is
    12 mm lower - the scan core is not at the middle of the case.  Until now the L2 was in
    the model only as a lump of mass, and a robot with a tilted empty pedestal is a robot
    whose FOV nobody can see.  It is visual-only and in the visual group, which is not in
    RAY_GROUPS, so the sensor cannot see itself.
    """
    import mini_dog as md
    p, q, _ = pose_m()
    rows = [f'{indent}<site name="{name}" pos="{_f(p)}" quat="{_f(q)}" size="0.004"'
            f' rgba="0.9 0.4 0.1 1"/>']
    if body:
        c = tuple(v / 1000.0 for v in md.lidar_com())
        r, h = md.LIDAR_L2_BOX[0] / 2000.0, md.LIDAR_L2_BOX[2] / 2000.0
        rows.append(f'{indent}<geom name="{name}_body" type="cylinder" group="2"'
                    f' contype="0" conaffinity="0" size="{r:.6g} {h:.6g}"'
                    f' pos="{_f(c)}" quat="{_f(q)}" rgba="0.15 0.15 0.17 1"/>')
    return rows


def custom_xml(indent="  ", name="lidar", nega=True):
    """`<custom>` block carrying the scan parameters into the compiled model.

    This is the whole interface to the C++ node and to Scanner below - see the header.
    """
    s = spec(nega)
    rows = [f'{indent}<custom>']
    for k, v in (("cone", (s["cone"],)), ("rate", (s["rate"],)),
                 ("range", (s["r_min"], s["r_max"])), ("sigma", (s["sigma"],)),
                 ("spin", s["spin"]), ("frame_hz", (s["frame_hz"],)),
                 ("groups", RAY_GROUPS)):
        rows.append(f'{indent}  <numeric name="{name}_{k}" data="{_f(v)}"/>')
    rows.append(f'{indent}</custom>')
    return rows


def urdf_link(parent="base_link", name="lidar_link"):
    """The matching fixed link, so TF has a frame to hang the cloud off in RViz."""
    p, _, rpy = pose_m()
    return [f'  <link name="{name}"/>',
            f'  <joint name="{name}_joint" type="fixed">',
            f'    <parent link="{parent}"/><child link="{name}"/>',
            f'    <origin xyz="{_f(p)}" rpy="{_f(rpy)}"/>',
            f'  </joint>']


# =======================================================================================
# The scanner
# =======================================================================================
def directions(t0, t1, n, cone, spin):
    """`n` unit vectors in the sensor frame, for the window [t0, t1).

    The Risley pair in closed form: two wedges of half-angle `cone/2`, spinning at
    `spin[0]` and `spin[1]` Hz.  Angle from the axis comes out as 0 .. cone and the
    azimuth wraps with the first prism.
    """
    if n <= 0:
        return np.zeros((0, 3))
    t = t0 + (np.arange(n) + 0.5) * (t1 - t0) / n
    a = 0.5 * cone
    sa, ca = math.sin(a), math.cos(a)
    p1 = 2.0 * math.pi * spin[0] * t
    p2 = 2.0 * math.pi * spin[1] * t
    x1 = sa * ca * (1.0 + np.cos(p2))
    y1 = sa * np.sin(p2)
    z1 = ca * ca - sa * sa * np.cos(p2)
    c1, s1 = np.cos(p1), np.sin(p1)
    return np.stack([x1 * c1 - y1 * s1, x1 * s1 + y1 * c1, z1], axis=1)


class Scanner:
    """Casts the L2 out of a compiled MuJoCo model.  Reads its parameters from the model.

    Construct it once per model; call `frame(model, data)` and it hands back a cloud every
    1/frame_hz of simulated time and None in between, which is what a consumer of a real
    lidar sees too.  `scan()` is the unconditional version for one-shot use.
    """

    def __init__(self, model, name="lidar", seed=0):
        import mujoco
        self.mj = mujoco
        self.name = name
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        self.ok = self.site >= 0
        if not self.ok:
            self.missing = f'no site "{name}" in this model'
            return

        def num(key, default=None):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, f"{name}_{key}")
            if i < 0:
                return default
            a, n = model.numeric_adr[i], model.numeric_size[i]
            return np.array(model.numeric_data[a:a + n], dtype=float)

        self.cone = num("cone")
        if self.cone is None:
            self.ok = False
            self.missing = (f'site "{name}" is there but the <custom> numerics are not -'
                            " the model predates lidar.py, regenerate it")
            return
        self.cone = float(self.cone[0])
        self.rate = float(num("rate")[0])
        self.r_min, self.r_max = (float(v) for v in num("range"))
        self.sigma = float(num("sigma")[0])
        self.spin = tuple(float(v) for v in num("spin"))
        self.frame_hz = float(num("frame_hz")[0])
        self.missing = None

        # the body the sensor is bolted to is excluded from its own view: the chassis box
        # would otherwise fill a third of the cloud.  The legs are NOT excluded - they do
        # swing through the forward-down cone at every stride, that is what the real
        # sensor sees, and masking them is the perception side's job.
        self.body = int(model.site_bodyid[self.site])
        self.groups = np.array(num("groups", RAY_GROUPS), dtype=np.uint8)
        self.rng = np.random.default_rng(seed)
        self._t0 = None

    # ---------------------------------------------------------------------------------
    def scan(self, model, data, t0=None, t1=None):
        """One cloud for the window [t0, t1); defaults to the last 1/frame_hz of sim time.

        Returns a dict: `local` (n,3) points in the sensor frame, `world` (n,3), `range`,
        `geom` (hit geom ids) and `n_rays` - how many were cast, so the return fraction is
        visible.  No intensity: see the note by RAY_GROUPS.
        """
        if not self.ok:
            raise RuntimeError(f"lidar.Scanner: {self.missing}")
        t1 = data.time if t1 is None else t1
        t0 = t1 - 1.0 / self.frame_hz if t0 is None else t0
        n = max(0, int(round(self.rate * (t1 - t0))))
        d_local = directions(t0, t1, n, self.cone, self.spin)

        pnt = np.array(data.site_xpos[self.site], dtype=np.float64)
        R = np.array(data.site_xmat[self.site], dtype=np.float64).reshape(3, 3)
        vec = (d_local @ R.T)                      # sensor frame -> world

        dist = np.zeros(n)
        geom = np.full(n, -1, dtype=np.int32)
        # `normal` is not asked for: MuJoCo 3.3 in the ROS 2 workspace has no such
        # argument, and the C++ side has to be able to do exactly what this does.
        self.mj.mj_multiRay(model, data, pnt, vec.reshape(-1), self.groups,
                            1, self.body, geom, dist, None, n, self.r_max)

        hit = (geom >= 0) & (dist >= self.r_min) & (dist <= self.r_max)
        if self.sigma > 0.0:
            dist = dist + self.rng.normal(0.0, self.sigma, n)
            hit &= dist > 0.0
        d_local, vec, dist, geom = d_local[hit], vec[hit], dist[hit], geom[hit]
        return dict(local=d_local * dist[:, None],
                    world=pnt + vec * dist[:, None],
                    range=dist, geom=geom, n_rays=n,
                    time=t1, pos=pnt, mat=R)

    def frame(self, model, data):
        """`scan()` on a 1/frame_hz cadence, None in between."""
        if not self.ok:
            return None
        if self._t0 is None:
            self._t0 = data.time
            return None
        if data.time - self._t0 < 1.0 / self.frame_hz:
            return None
        out = self.scan(model, data, self._t0, data.time)
        self._t0 = data.time
        return out
