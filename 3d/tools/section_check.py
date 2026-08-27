#!/usr/bin/env python
"""Bending capacity of the shin, straight off the solid - the cheap half of fea.py.

fea.py is the real check and nothing here replaces it.  But a 3D solve costs minutes per
load case, and most of what a shin re-shape does is move section modulus around, which is
a property of the cross-sections alone.  This slices the actual part and reports, per
station, the exact area second moments and the resulting bending stress for a unit force
at the foot - so a profile change can be judged in seconds instead of an hour.

    .venv/bin/python tools/section_check.py
    .venv/bin/python tools/section_check.py --module /path/to/other/mini_dog.py

Fore-aft (about the knee axis) is the direction every fea.py load case bends the shin in:
'stall' is a torque about that axis and the ground cases are a vertical force brought into
the leg plane.  Lateral is reported too because it is what the blade section trades away
and what no load case in fea.py covers.
"""
import argparse, importlib.util, os, sys
import numpy as np

SLAB = 0.5   # mm; thin enough that the out-of-plane term is noise

def load(path):
    if path is None:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import mini_dog
        return mini_dog
    spec = importlib.util.spec_from_file_location("mini_dog_ref", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["mini_dog_ref"] = m
    spec.loader.exec_module(m)
    return m

def section(md, solid, u):
    """area, centroid, area second moments about the section's own centroid, half-widths"""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    z = md.KNEE_Z - u
    slab = md.bxc(md.PITCH_X-60, md.PITCH_X+60, md.LEG_Y-60, md.LEG_Y+60,
                  z-SLAB/2, z+SLAB/2).val()
    try:
        sec = solid.intersect(slab)
        v = sec.Volume()
    except Exception:
        return None
    if v < 1e-6:
        return None
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(sec.wrapped, p)
    com = p.CentreOfMass()
    # MatrixOfInertia() is already referred to the centre of mass, so no parallel axis.
    # (1,1) is the integral of y^2+z^2, (2,2) of x^2+z^2; drop the out-of-plane term and
    # divide by the slab thickness to get the AREA second moments.
    m = p.MatrixOfInertia()
    a = v / SLAB
    cx, cy = com.X(), com.Y()
    ixx = (m.Value(1, 1) - a*SLAB**3/12.0) / SLAB
    iyy = (m.Value(2, 2) - a*SLAB**3/12.0) / SLAB
    bb = sec.BoundingBox()
    return dict(a=a, cx=cx, cy=cy,
                i_fore=iyy, i_lat=ixx,                # about Y (fore-aft) and X (lateral)
                c_fore=max(bb.xmax-cx, cx-bb.xmin), c_lat=max(bb.ymax-cy, cy-bb.ymin))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default=None, help="a different mini_dog.py to profile")
    ap.add_argument("--step", type=float, default=3.0)
    a = ap.parse_args()
    md = load(a.module)
    solid = md.shin().val()
    print(f"\n  shin volume {solid.Volume()/1000:.2f} cm3   L_SHIN {md.L_SHIN:.0f} mm"
          f"   (module: {a.module or 'mini_dog.py'})")
    print("  bending stress is per 1 N of foot force perpendicular to the shin\n")
    print("     u    area    fore-aft            lateral")
    print("    mm     mm2     I      S    MPa/N    I      S    MPa/N")
    worst = (0.0, 0.0)
    u = 24.0
    while u <= md.L_SHIN - 12.0 + 1e-9:
        s = section(md, solid, u)
        if s:
            m = md.L_SHIN - u                      # unit force at the foot
            sf, sl = s["i_fore"]/s["c_fore"], s["i_lat"]/s["c_lat"]
            if m/sf > worst[1]:
                worst = (u, m/sf)
            print(f"  {u:5.1f} {s['a']:7.0f} {s['i_fore']:7.0f} {sf:6.0f} {m/sf:7.4f}"
                  f" {s['i_lat']:7.0f} {sl:6.0f} {m/sl:7.4f}")
        else:
            print(f"  {u:5.1f}       -   (slice failed - OCC boolean, not a hole;"
                  f" cross-check with a neighbouring station)")
        u += a.step
    print(f"\n  worst fore-aft section: u = {worst[0]:.0f} mm, {worst[1]:.4f} MPa per N at the foot")
    print("  This is beam bending only.  It says nothing about the fork, the bolt holes,")
    print("  the interlayer plane or any stress concentration - run fea.py for those.")

if __name__ == "__main__":
    main()
