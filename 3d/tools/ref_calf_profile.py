#!/usr/bin/env python
"""Longitudinal profile of the lower-leg link of production quadrupeds.

Provenance for the SHIN_* profile constants in mini_dog.py.  Downloads the collision /
visual meshes shipped with each robot's own URDF and reports, for each station along the
link, the outline width in the two transverse directions.  The point is the *shape of the
profile*, not the absolute sizes - those robots are aluminium and 2.5-3x our scale.

    .venv/bin/python tools/ref_calf_profile.py            # table
    .venv/bin/python tools/ref_calf_profile.py --plot     # out/ref_calf_profile.png

What it shows, consistently across all of them: the lower leg is NOT a straight taper.
It is a big joint boss (15-20 % of the length, sized by the bearing/hub hardware), a
short concave neck out of it, then a long near-constant slender blade, then the ankle.
Section is narrow in the sagittal (bending) plane and wider laterally.
"""
import os, re, sys, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_refcache")
RAW = "https://raw.githubusercontent.com/"
SRC = {
    "unitree_a1":   RAW + "unitreerobotics/unitree_ros/master/robots/a1_description/meshes/calf.dae",
    "unitree_go1":  RAW + "unitreerobotics/unitree_ros/master/robots/go1_description/meshes/calf.dae",
    "unitree_go2":  RAW + "unitreerobotics/unitree_ros/master/robots/go2_description/meshes/calf.dae",
    "unitree_b2":   RAW + "unitreerobotics/unitree_ros/master/robots/b2_description/meshes/calf.dae",
    "mit_minichee": RAW + "mit-biomimetics/Cheetah-Software/master/resources/mini_lower_link.obj",
    "bd_spot":      RAW + "google-deepmind/mujoco_menagerie/main/boston_dynamics_spot/assets/front_left_lower_leg.obj",
}

def fetch(name, url):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name + os.path.splitext(url)[1])
    if not os.path.exists(p):
        urllib.request.urlretrieve(url, p)
    return p

def vertices(path):
    if path.endswith(".obj"):
        v = [[float(x) for x in l.split()[1:4]] for l in open(path, errors="ignore")
             if l.startswith("v ")]
        return np.array(v)
    s = open(path, encoding="utf-8", errors="ignore").read()
    out = []
    for m in re.finditer(r'<float_array[^>]*id="([^"]*)"[^>]*count="(\d+)"[^>]*>(.*?)</float_array>',
                         s, re.S):
        if "position" not in m.group(1).lower():
            continue
        a = np.fromstring(m.group(3), sep=" ")
        out.append(a[:int(m.group(2))].reshape(-1, 3))
    return np.vstack(out)

def profile(path, n=24):
    """(length, [(s/L, w_a, w_b)]) with s measured from the joint (fat) end, in mm"""
    v = vertices(path)
    if np.ptp(v, axis=0).max() < 5.0:            # metres -> mm
        v = v * 1000.0
    ext = np.ptp(v, axis=0)
    ax = int(np.argmax(ext))
    o = [i for i in range(3) if i != ax]
    t = v[:, ax]
    head = np.ptp(v[t < t.min() + 0.15*ext[ax]][:, o[0]])
    tail = np.ptp(v[t > t.max() - 0.15*ext[ax]][:, o[0]])
    if head < tail:
        t = -t
    t = t - t.min()
    rows, e = [], np.linspace(0, t.max(), n+1)
    for i in range(n):
        m = (t >= e[i]) & (t <= e[i+1])
        if m.sum() < 4:
            continue
        rows.append(((i+0.5)/n, np.ptp(v[m][:, o[0]]), np.ptp(v[m][:, o[1]])))
    return ext[ax], rows

def main():
    plot = "--plot" in sys.argv
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(SRC), figsize=(2.6*len(SRC), 7), squeeze=False)
    for j, (name, url) in enumerate(SRC.items()):
        L, rows = profile(fetch(name, url))
        print(f"\n  {name}   link length {L:.0f} mm")
        print("    s/L    sagittal   lateral   (mm)")
        for s, a, b in rows:
            print(f"   {s:5.2f}   {a:7.1f}   {b:7.1f}")
        if plot:
            s = [r[0]*L for r in rows]
            A = axes[0][j]
            A.plot([r[1] for r in rows], s, label="sagittal")
            A.plot([r[2] for r in rows], s, label="lateral")
            A.set_title(name, fontsize=9); A.invert_yaxis(); A.grid(alpha=.3)
            A.set_xlim(0, None)
            if j == 0:
                A.legend(fontsize=8); A.set_ylabel("mm from the knee")
    if plot:
        out = os.path.join(os.path.dirname(HERE), "out", "ref_calf_profile.png")
        plt.tight_layout(); plt.savefig(out, dpi=110)
        print("\n  wrote", out)

if __name__ == "__main__":
    main()
