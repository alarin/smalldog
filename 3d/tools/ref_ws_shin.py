#!/usr/bin/env python
"""Measure the Waveshare DOG PRO lower leg out of ref/ROBOTIC_DOG_-STEP.

Provenance for the SHIN_PROFILE table in mini_dog.py.  That STEP is an aluminium-plate
design and none of its joint spacing transfers to a printed servo-fork robot - but the
SHAPE of its lower leg does, and it is the one real quadruped link this repo owns.

    .venv/bin/python tools/ref_ws_shin.py            # numbers
    .venv/bin/python tools/ref_ws_shin.py --plot     # + out/ref_ws_shin.png

The part is posed in the assembly, so it is first aligned by PCA: the largest principal
direction is the link axis, the smallest is the plate normal.  What comes out:

    joint centres     106.5 mm apart
    plate thickness    12.0 mm, CONSTANT over the whole link
    in-plane depth     26.3 mm at the knee boss -> 8.9 mm just before the ankle
    centreline         bowed ~10 % of the length off the chord between the joints

i.e. a blade - deep in the plane it bends in, thin along the joint axis, tapering the
whole way, and curved.  Not a wedge and not a straight taper.
"""
import os, sys
import numpy as np
import cadquery as cq

STEP = "ref/ROBOTIC_DOG_-STEP/ROBOTIC DOG.step"
LEG_VOL = 17420.0          # the four lower legs are the only solids at this volume

def leg_solid(path=STEP):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - run from the repo root")
    legs = [s for s in cq.importers.importStep(path).solids().vals()
            if abs(s.Volume() - LEG_VOL) < 60]
    if len(legs) != 4:
        raise SystemExit(f"expected 4 lower legs, found {len(legs)}")
    return min(legs, key=lambda s: (s.BoundingBox().xmin, s.BoundingBox().ymin))

def aligned(s):
    """PCA-align: +X along the link, +Y in-plane across it, +Z the plate normal"""
    v, _ = s.tessellate(0.05)
    P = np.array([(p.x, p.y, p.z) for p in v])
    c = P.mean(0)
    Vt = np.linalg.svd(P - c, full_matrices=False)[2]
    pl = cq.Plane(origin=cq.Vector(*c), xDir=cq.Vector(*Vt[0]), normal=cq.Vector(*Vt[2]))
    return s.moved(cq.Location(pl).inverse)

def outline(s, n=44):
    """[(station, y_lo, y_hi, thickness)] by slicing, in the aligned frame"""
    b = s.BoundingBox()
    xs = np.linspace(b.xmin, b.xmax, n+1)
    rows = []
    for i in range(n):
        slab = cq.Solid.makeBox(xs[i+1]-xs[i], 400, 120, cq.Vector(xs[i], -200, -60))
        try:
            sec = s.intersect(slab)
            if sec.Volume() < 1e-6:
                continue
        except Exception:
            continue
        bb = sec.BoundingBox()
        rows.append((0.5*(xs[i]+xs[i+1]), bb.ymin, bb.ymax, bb.zlen))
    return np.array(rows)

def joint_centres(s):
    """(knee hub, ankle eye) in the aligned frame, from the circles at either end"""
    b = s.BoundingBox()
    ends = {"knee": [], "ankle": []}
    for e in cq.Workplane(obj=s).edges().vals():
        if e.geomType() != "CIRCLE":
            continue
        c = e.Center()
        k = "knee" if c.x < 0.5*(b.xmin+b.xmax) else "ankle"
        ends[k].append((e.radius(), np.array([c.x, c.y])))
    out = {}
    for k, v in ends.items():
        bolts = [p for r, p in v if 1.0 < r < 2.5] or [p for r, p in v if r < 7.0]
        out[k] = np.mean(bolts, axis=0)
    return out["knee"], out["ankle"]

def main():
    s = aligned(leg_solid())
    r = outline(s)
    knee, ankle = joint_centres(s)
    L = np.linalg.norm(ankle - knee)
    w, th = r[:, 2]-r[:, 1], r[:, 3]
    mid = 0.5*(r[:, 1]+r[:, 2])
    chord = knee[1] + (r[:, 0]-knee[0])*(ankle[1]-knee[1])/(ankle[0]-knee[0])
    k = (r[:, 0] > knee[0]) & (r[:, 0] < ankle[0])
    print(f"\n  Waveshare DOG PRO lower leg  ({STEP})")
    print(f"    joint centres        {L:8.1f} mm apart")
    print(f"    plate thickness      {np.median(th):8.1f} mm   (min {th.min():.1f}, max {th.max():.1f})")
    print(f"    in-plane depth       {w.max():8.1f} mm at the knee boss")
    print(f"                         {w[k].min():8.1f} mm at the waist  "
          f"({w[k].min()/w.max():.2f} of the boss)")
    print(f"    centreline bow       {(mid-chord)[k].max():+8.1f} mm off the chord"
          f"   ({(mid-chord)[k].max()/L*100:.1f} % of length)")
    print("\n      s/L   depth   thickness")
    for a, ww, tt in zip((r[:, 0]-r[0, 0])/(r[-1, 0]-r[0, 0]), w, th):
        print(f"     {a:5.2f} {ww:7.1f} {tt:9.1f}")
    if "--plot" in sys.argv:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(1, 2, figsize=(9, 7))
        ax[0].fill_betweenx(r[:, 0], r[:, 1], r[:, 2], color="#4b6")
        ax[0].plot(chord, r[:, 0], "k--", lw=1, label="chord")
        ax[0].set_title("in-plane outline"); ax[0].legend(fontsize=8)
        ax[1].fill_betweenx(r[:, 0], -th/2, th/2, color="#46b")
        ax[1].set_title("thickness")
        for a in ax:
            a.set_aspect("equal"); a.grid(alpha=.3); a.invert_yaxis()
        out = os.path.join("out", "ref_ws_shin.png")
        plt.tight_layout(); plt.savefig(out, dpi=110); print("\n  wrote", out)

if __name__ == "__main__":
    main()
