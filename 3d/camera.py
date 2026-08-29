"""
camera.py — the IMX415 module as a *sensor*, shared by both sim exporters.

`mini_dog.py` already knows where the module is bolted (`camera_pose()`, `CAM_OPT`) and
what it can see (`camera_fov()`, `CAM_PIX`, `CAM_RATE`).  This module is the part that
turns those into simulation: the MJCF `<camera>` that puts it in the model and the URDF
links that give a ROS 2 image a frame to hang off.  Like lidar.py and terrain.py it
carries no CAD and no second copy of a robot dimension.

    export_sim.py                   -> <camera> in out/sim/*.xml, links in the URDF
    ../ros2/.../generate_model.py   -> the same, in smalldog_description/

WHY THERE ARE TWO FRAMES, AND WHY THAT IS NOT PEDANTRY
`camera_link` is the frame a *robot* wants: +X out of the lens, +Z up, which is REP-103
and which makes CAM_TILT readable as a pitch about +Y.  `camera_optical_frame` is the
frame an *image* wants: +Z out of the lens, +X right, +Y down, which is REP-145 and which
is what every camera_info, every rectification matrix and every cv::projectPoints assumes.
They differ by a fixed (-90, 0, -90) and getting that wrong shows up as a picture that is
right way up and pointing 90 deg off, which is a bug people lose afternoons to.  MuJoCo
uses a third convention again - it looks down its own -Z with +Y up - so the quaternion
below is built from the axes rather than written out, and there is nothing to mistype.

WHAT IS HONEST ABOUT THE SIMULATED CAMERA
MuJoCo renders a pinhole with no distortion, no rolling shutter and no noise, at whatever
resolution the offscreen buffer is set to.  The real module is a rolling-shutter sensor
behind an M12 lens with real barrel distortion, running MJPEG over USB 2.0 at CAM_RATE.
So: use this for framing, occlusion and "can the robot see that from here", and calibrate
the real one before using an image for measurement.  In particular CAM_FOV_D is the
catalogue's diagonal, marked **verify** in mini_dog.py - a lens that is really 78 or 96
diagonal changes every number this file emits.
"""
import math


def _quat(x, y, z):
    """wxyz quaternion of the frame whose axes are the columns (x, y, z)."""
    m = ((x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2]))
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = (0.25*s, (m[2][1]-m[1][2])/s, (m[0][2]-m[2][0])/s, (m[1][0]-m[0][1])/s)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        q = ((m[2][1]-m[1][2])/s, 0.25*s, (m[0][1]+m[1][0])/s, (m[0][2]+m[2][0])/s)
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        q = ((m[0][2]-m[2][0])/s, (m[0][1]+m[1][0])/s, 0.25*s, (m[1][2]+m[2][1])/s)
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        q = ((m[1][0]-m[0][1])/s, (m[0][2]+m[2][0])/s, (m[1][2]+m[2][1])/s, 0.25*s)
    return q


def spec():
    """Everything the sim needs about the camera, in SI, straight from the CAD."""
    import mini_dog as md
    hf, vf = md.camera_fov()
    p, _ = md.camera_pose()
    return dict(pos=tuple(v/1000.0 for v in p), hfov=hf, vfov=vf,
                pix=md.CAM_PIX, rate=md.CAM_RATE, tilt=md.CAM_TILT)


def axes():
    """(forward, up, right) of the module in robot coordinates - mini_dog's own frame."""
    import mini_dog as md
    _, n, u = md.camera_frame()
    return n, u, (n[1]*u[2]-n[2]*u[1], n[2]*u[0]-n[0]*u[2], n[0]*u[1]-n[1]*u[0])


def _f(vals):
    return " ".join(f"{v:.6g}" for v in vals)


def camera_xml(indent="      ", name="camera", body=True):
    """The MJCF `<camera>`, plus the module drawn where it actually is.

    MuJoCo looks down its camera's own -Z with +Y up, so the frame handed to it is
    (right, up, -forward) - built from axes() rather than written out.  fovy is the
    VERTICAL field, which is why camera_fov() returns both."""
    import mini_dog as md
    s = spec()
    n, u, r = axes()
    q = _quat(r, u, tuple(-v for v in n))
    rows = [f'{indent}<camera name="{name}" pos="{_f(s["pos"])}" quat="{_f(q)}"'
            f' fovy="{s["vfov"]:.4g}" resolution="{s["pix"][0]} {s["pix"][1]}"/>']
    if body:
        c = tuple(v/1000.0 for v in md.camera_com())
        rows.append(f'{indent}<geom name="{name}_body" type="cylinder" group="2"'
                    f' contype="0" conaffinity="0"'
                    f' size="{md.CAM_LENS_D/2000.0:.6g} {md.CAM_LENS_H/2000.0:.6g}"'
                    f' pos="{_f(c)}" quat="{_f(_quat(tuple(-v for v in r), u, n))}"'
                    f' rgba="0.1 0.1 0.12 1"/>')
    return rows


def urdf_links(parent="base_link", name="camera"):
    """`camera_link` (REP-103, +X out of the lens) and `camera_optical_frame` (REP-145).

    See the header for why both exist.  The optical frame is a child of the link, so a
    calibration that moves the lens moves exactly one origin here."""
    import mini_dog as md
    p, _ = md.camera_pose()
    p = tuple(v/1000.0 for v in p)
    t = math.radians(md.CAM_TILT)
    return [f'  <link name="{name}_link"/>',
            f'  <joint name="{name}_joint" type="fixed">',
            f'    <parent link="{parent}"/><child link="{name}_link"/>',
            f'    <origin xyz="{_f(p)}" rpy="0 {-t:.6g} 0"/>',
            f'  </joint>',
            f'  <link name="{name}_optical_frame"/>',
            f'  <joint name="{name}_optical_joint" type="fixed">',
            f'    <parent link="{name}_link"/><child link="{name}_optical_frame"/>',
            f'    <origin xyz="0 0 0" rpy="{-math.pi/2:.6g} 0 {-math.pi/2:.6g}"/>',
            f'  </joint>']
