#!/usr/bin/env python
"""
fea.py - linear static FEA of the printed mini_dog parts.  gmsh (mesh) + sfepy (solve).

Geometry, joint frames and load-bearing dimensions are imported from mini_dog.py, so
editing the model is enough - nothing here has to be kept in sync by hand.  Parts are
rebuilt in memory and meshes are cached on a hash of the geometry itself.

Units: mm, N, MPa.  Displacements come out in mm, stresses in MPa.

    python fea.py --selftest                 # cantilever vs. analytic - run this first
    python fea.py shin_A                     # every load case for one part
    python fea.py --all --mat PETG
    python fea.py shin_A --case stall --size 1.5

Linear static, solid geometry, isotropic material.  Read the caveats printed after a run.
"""
import os, sys, math, argparse, hashlib
import numpy as np

import mini_dog as md

OUT = os.path.join(md.OUT, "fea")

# ---------------------------------------------------------------------------------
# loads
# ---------------------------------------------------------------------------------
G_ACC          = 9.81
# every number below is mini_dog's - fea.py must not carry its own copy
SERVO_STALL_NM = md.SERVO_STALL_NM
N_SERVO        = md.N_SERVO
SERVO_KG       = md.SERVO_KG
BATTERY_KG     = md.BATTERY_KG
ELECTRONICS_KG = md.ELECTRONICS_KG
LIDAR_KG       = md.LIDAR_KG

def robot_mass():
    """mini_dog's own printed-mass estimate plus everything the robot carries.  The
    ground load cases scale with this, so anything bolted on belongs here - the LiDAR is
    230 g on a 2 kg robot and leaving it out made every ground case optimistic."""
    if not md.PARTS:
        md.build()
    printed = sum(wp.val().Volume() / 1000.0 * md.part_rho(n) * qty
                  for n, (wp, qty, _) in md.PARTS.items()) / 1000.0
    return printed + N_SERVO * SERVO_KG + BATTERY_KG + ELECTRONICS_KG + LIDAR_KG

# E [MPa], nu, sigma in-plane [MPa], sigma inter-layer [MPa].  Print-realistic values for
# ~5 walls / 40 % infill, i.e. already below the datasheet numbers for moulded material.
# s_z  = interlayer tensile strength (pulling layers apart)
# s_zs = interlayer shear strength (sliding layers over each other), typically a bit
#        higher than s_z; kept equal here, which is the conservative reading.
MATERIALS = {
    "PETG":   dict(E=1700.0, nu=0.40, s_xy=40.0, s_z=20.0, s_zs=20.0),
    "ASA":    dict(E=1900.0, nu=0.36, s_xy=36.0, s_z=17.0, s_zs=17.0),
    "PLA":    dict(E=2800.0, nu=0.36, s_xy=45.0, s_z=22.0, s_zs=22.0),
    "PAHTCF": dict(E=5500.0, nu=0.38, s_xy=80.0, s_z=35.0, s_zs=35.0),
}

# ---------------------------------------------------------------------------------
# print orientation
#
# An FDM part is transversely isotropic: the layer interfaces are the weak plane.  What
# matters is therefore not von Mises against a single de-rated number, but the traction
# ON the layer plane - and that depends only on the build direction n, never on how the
# part is rotated about n.  Choosing an orientation is choosing one vector.
#
#   sigma_nn = n.S.n          tension across the layers (compression does not open them)
#   tau      = |S.n - sigma_nn n|                  shear along the interface
#   index    = sqrt((sigma_nn+/s_z)^2 + (tau/s_zs)^2)      failure index, 1 = at strength
# ---------------------------------------------------------------------------------
def layer_index(s_el, n, mat):
    """failure index per element for build direction n (sfepy Voigt: xx yy zz xy xz yz)"""
    n = np.asarray(n, float) / np.linalg.norm(n)
    S = np.empty((len(s_el), 3, 3))
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = s_el[:, 0], s_el[:, 1], s_el[:, 2]
    S[:, 0, 1] = S[:, 1, 0] = s_el[:, 3]
    S[:, 0, 2] = S[:, 2, 0] = s_el[:, 4]
    S[:, 1, 2] = S[:, 2, 1] = s_el[:, 5]
    t = S @ n
    snn = t @ n
    tau = np.linalg.norm(t - snn[:, None] * n, axis=1)
    return np.sqrt((np.maximum(snn, 0.0) / mat["s_z"]) ** 2 + (tau / mat["s_zs"]) ** 2)

def printability(coors, tets, n, ang=45.0):
    """(print height, unsupported overhang area, bed contact area) for build direction n"""
    n = np.asarray(n, float) / np.linalg.norm(n)
    t = tets.copy()
    v = np.linalg.det(coors[t[:, 1:]] - coors[t[:, :1]])
    t[v < 0] = t[v < 0][:, [0, 2, 1, 3]]                      # make every tet right-handed
    f = np.vstack([t[:, [0, 2, 1]], t[:, [0, 1, 3]], t[:, [1, 2, 3]], t[:, [0, 3, 2]]])
    _, idx, cnt = np.unique(np.sort(f, axis=1), axis=0, return_index=True, return_counts=True)
    bnd = f[idx[cnt == 1]]                                    # outward-oriented boundary
    a, b = coors[bnd[:, 1]] - coors[bnd[:, 0]], coors[bnd[:, 2]] - coors[bnd[:, 0]]
    cr = np.cross(a, b)
    ar = 0.5 * np.linalg.norm(cr, axis=1)
    nrm = cr / (2 * ar[:, None])
    down = -(nrm @ n)                                         # 1 = facing straight down
    h = coors @ n
    over = float(ar[down > math.cos(math.radians(ang))].sum())
    bed = float(ar[(down > 0.99) & (coors[bnd].mean(axis=1) @ n < h.min() + 0.5)].sum())
    return float(h.max() - h.min()), over, bed

def sphere_dirs(n=400):
    """near-uniform directions on a half sphere (n and -n give the same index)"""
    i = np.arange(n) + 0.5
    z = 1.0 - i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def build_dir(orient):
    """robot-frame layer normal for a mini_dog PRINT_ORIENT entry ((axis), degrees)"""
    ax, ang = orient
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    a = math.radians(-ang)                      # inverse rotation, print frame -> robot
    c, s_, K = math.cos(a), math.sin(a), np.cross(np.eye(3), ax)
    R = np.eye(3) + s_ * K + (1 - c) * (K @ K)
    return R @ np.array([0.0, 0.0, 1.0])

# ---------------------------------------------------------------------------------
# joint axes, straight out of mini_dog's frames
# ---------------------------------------------------------------------------------
def axis_of(joint):
    """(point on the axis, unit direction) in robot coordinates"""
    return {
        "roll":  (np.array([md.ROLL_X, md.ROLL_Y, md.ROLL_Z]), np.array([1.0, 0.0, 0.0])),
        "pitch": (np.array([md.PITCH_X, md.LEG_Y, md.PITCH_Z]), np.array([0.0, 1.0, 0.0])),
        "knee":  (np.array([md.PITCH_X, md.LEG_Y, md.KNEE_Z]), np.array([0.0, 1.0, 0.0])),
    }[joint]

def hub_clamp(joint, slack=0.5):
    """the aluminium hub footprint on the two fork arms - the actually clamped area"""
    p, a = axis_of(joint)
    return lambda c: np.linalg.norm(np.cross(c - p, a), axis=1) < md.HUB_D / 2 + slack

def zone(axis, lo=-1e9, hi=1e9):
    i = "xyz".index(axis)
    return lambda c: (c[:, i] > lo) & (c[:, i] < hi)

def all_of(*fns):
    return lambda c: np.logical_and.reduce([f(c) for f in fns])

def rot_y(vec, deg):
    a = math.radians(deg)
    x, y, z = vec
    return np.array([x * math.cos(a) + z * math.sin(a), y, -x * math.sin(a) + z * math.cos(a)])

# ---------------------------------------------------------------------------------
# what is clamped and where the reaction comes in, per part
#
#   fix   - hub footprint of the joint this link hangs from (the fork arms)
#   load  - the patch the next link pushes on (foot spigot / servo sleeve)
#   pose  - stance angle about +Y, used to bring a world-vertical ground reaction into
#           the part's own frame (parts are modelled legs-straight-down)
# ---------------------------------------------------------------------------------
def part_specs():
    return {
        "shin_A": dict(
            joint="knee",
            pose=md.STAND_PITCH + md.STAND_KNEE,
            fix=hub_clamp("knee"),
            load=zone("z", hi=md.FOOT_Z + 5.0),
            note="knee fork bolted to the servo hubs, ground reaction at the foot spigot",
        ),
        "thigh_A": dict(
            joint="pitch",
            pose=md.STAND_PITCH,
            fix=hub_clamp("pitch"),
            load=zone("z", hi=md.KNEE_Z + 15.0),
            note="hip-pitch fork bolted to the hubs, knee-servo reaction into the sleeve",
        ),
        "hip_bracket_A": dict(
            joint="roll",
            pose=0.0,
            fix=hub_clamp("roll"),
            load=all_of(zone("z", hi=md.PITCH_Z + 15.0),
                        zone("y", lo=md.LEG_Y - md.SLEEVE_LEN / 2 - 4.0)),
            note="roll fork bolted to the hubs, hip-pitch servo reaction into the sleeve",
        ),
    }

def cases(mass_kg):
    w = mass_kg * G_ACC
    return {
        "stand4": dict(kind="ground", n=w / 4.0),
        "stand2": dict(kind="ground", n=w / 2.0),
        "land3g": dict(kind="ground", n=3.0 * w / 2.0),
        "stall":  dict(kind="torque", m=SERVO_STALL_NM * 1000.0),
    }

# ---------------------------------------------------------------------------------
# geometry -> mesh
# ---------------------------------------------------------------------------------
def part_geometry(name):
    """the part's workplane plus a content hash of its *geometry*

    Deliberately not a hash of its STEP file.  OCC stamps a timestamp into the header
    and orders the entities non-deterministically, so the file hash changed on every
    single export - the mesh cache below never hit once, every run re-meshed from
    scratch, and out/fea accumulated a fresh mesh per invocation.  Volume, area, bbox,
    centroid, the topology counts and the sorted face areas are reproducible across
    processes and still move for a 0.05 mm change in a wall thickness.
    """
    if not md.PARTS:
        md.build()
    if name not in md.PARTS:
        raise SystemExit(f"{name} is not a mini_dog part ({', '.join(md.PARTS)})")
    wp = md.PARTS[name][0]
    s = wp.val()
    bb = s.BoundingBox()
    sig = [f"{v:.9g}" for v in (s.Volume(), s.Area(),
                                bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax,
                                *s.Center().toTuple())]
    sig += [str(len(s.Solids())), str(len(s.Faces())),
            str(len(s.Edges())), str(len(s.Vertices()))]
    sig += [f"{a:.9g}" for a in sorted(f.Area() for f in s.Faces())]
    return wp, hashlib.md5("|".join(sig).encode()).hexdigest()[:8]

def mesh_step(step_path, size, mesh_path):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", size * 0.35)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 10)
        gmsh.model.mesh.generate(3)
        gmsh.write(mesh_path)
    finally:
        gmsh.finalize()

def part_mesh(name, size):
    """(mesh path, nodes, tets) - the STEP is written and meshed only when the geometry
    is one no cached mesh was built from, so an unchanged part keeps the same mesh and
    two runs are comparable."""
    import meshio
    wp, tag = part_geometry(name)
    os.makedirs(OUT, exist_ok=True)
    mesh_path = os.path.join(OUT, f"{name}_{tag}_{size}.mesh")
    if not os.path.exists(mesh_path):
        import cadquery as cq
        step_path = os.path.join(OUT, f"_{name}_{tag}.step")
        cq.exporters.export(wp, step_path)
        raw = mesh_path.replace(".mesh", ".msh")
        mesh_step(step_path, size, raw)
        m = meshio.read(raw)
        tets = np.vstack([b.data for b in m.cells if b.type == "tetra"])
        meshio.write(mesh_path, meshio.Mesh(m.points, [("tetra", tets)]),
                     file_format="medit")
        os.remove(raw)
    m = meshio.read(mesh_path)
    return mesh_path, m.points, np.vstack([b.data for b in m.cells if b.type == "tetra"])

def boundary_patch(coors, tets, pred):
    """area, area-weighted centroid and facet count of the outer surface picked by pred"""
    f = np.vstack([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [1, 2, 3]], tets[:, [0, 3, 2]]])
    _, idx, cnt = np.unique(np.sort(f, axis=1), axis=0, return_index=True, return_counts=True)
    bnd = f[idx[cnt == 1]]
    m = pred(coors)
    sel = bnd[m[bnd].all(axis=1)]
    if not len(sel):
        return 0.0, None, 0
    a, b = coors[sel[:, 1]] - coors[sel[:, 0]], coors[sel[:, 2]] - coors[sel[:, 0]]
    ar = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    cen = coors[sel].mean(axis=1)
    return float(ar.sum()), (cen * ar[:, None]).sum(axis=0) / ar.sum(), len(sel)

# ---------------------------------------------------------------------------------
# linear solver
#
# K is symmetric positive definite, so SuperLU's default COLAMD ordering is the wrong
# one for it.  MMD_AT_PLUS_A with pivoting off is ~3x faster on the very same matrix and,
# being an exact factorisation, gives the same answer - checked against the stock solver
# on a pinned mesh, every reported quantity agreed to 1e-11.  The relative residual is
# verified on each solve, so a matrix that is not actually SPD cannot pass silently.
#
# The four load cases of a part differ only in the traction: same mesh, same material,
# same clamp, hence the same K.  Factorising it costs ~31 s and a back-solve ~0.2 s, so
# the factorisation is kept between cases instead of being thrown away and rebuilt four
# times.  It is keyed on the bytes of K itself, so it can never be handed a matrix it
# was not built from - a model change simply misses and re-factorises.
# ---------------------------------------------------------------------------------
_LU = {"digest": None, "lu": None}

def _mtx_digest(mtx):
    h = hashlib.sha1()
    for a in (np.asarray(mtx.shape, np.int64), mtx.indptr, mtx.indices, mtx.data):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def spd_solver():
    """sfepy linear solver: SPD-ordered SuperLU, factorisation reused across load cases"""
    import scipy.sparse.linalg as spl
    from sfepy.solvers.ls import ScipyDirect, standard_call

    class SPDDirect(ScipyDirect):
        name = "ls.spd_direct"
        _parameters = ScipyDirect._parameters

        @standard_call
        def __call__(self, rhs, x0=None, conf=None, eps_a=None, eps_r=None,
                     i_max=None, mtx=None, status=None, **kwargs):
            K = mtx.tocsc()
            digest = _mtx_digest(K)
            if _LU["lu"] is None or digest != _LU["digest"]:
                _LU["lu"], _LU["digest"] = None, None   # drop the old L+U before the new
                _LU["lu"] = spl.splu(K, permc_spec="MMD_AT_PLUS_A",
                                     diag_pivot_thresh=0.0,
                                     options=dict(SymmetricMode=True))
                _LU["digest"] = digest
            x = _LU["lu"].solve(rhs)
            res = np.linalg.norm(K @ x - rhs) / max(np.linalg.norm(rhs), 1e-300)
            if not np.isfinite(res) or res > 1e-6:
                raise RuntimeError(f"the linear solve did not converge (relative residual"
                                   f" {res:.2e}) - is the clamp holding the part?")
            return x

    return SPDDirect({}, method="superlu")

# ---------------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------------
def solve(mesh_path, coors, tets, mat, fix_pred, load_pred, force, order=2, save=None):
    """force: total force vector [N], applied as a uniform traction over the load patch"""
    from sfepy.base.base import IndexedStruct, Struct, output
    from sfepy.discrete.fem import Mesh, FEDomain, Field
    from sfepy.discrete import (FieldVariable, Material, Integral, Function, Functions,
                                Equation, Equations, Problem)
    from sfepy.terms import Term
    from sfepy.discrete.conditions import Conditions, EssentialBC
    from sfepy.solvers.nls import Newton
    from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
    from sfepy.mechanics.tensors import get_von_mises_stress
    from scipy.spatial import cKDTree
    output.set_output(quiet=True)

    area, _, nfac = boundary_patch(coors, tets, load_pred)
    nfix = int(fix_pred(coors).sum())
    if area <= 0:
        raise RuntimeError("the load predicate selects no outer surface")
    if nfix < 20:
        raise RuntimeError(f"the fix predicate selects only {nfix} nodes")

    mesh = Mesh.from_file(mesh_path)
    domain = FEDomain("d", mesh)
    omega = domain.create_region("Omega", "all")
    fns = Functions([
        Function("fsel", lambda c, domain=None: np.where(fix_pred(c))[0]),
        Function("lsel", lambda c, domain=None: np.where(load_pred(c))[0]),
    ])
    fix = domain.create_region("Fix", "vertices by fsel", "facet", functions=fns)
    # a facet region built from a volumetric vertex selection also holds interior facets;
    # without intersecting it with the outer surface the applied force comes out several
    # times too large (7x on the validation beam)
    domain.create_region("Sel", "vertices by lsel", "facet", functions=fns)
    domain.create_region("Surf", "vertices of surface", "facet")
    load = domain.create_region("Load", "r.Sel *f r.Surf", "facet", functions=fns)
    if abs(len(load.facets) - nfac) > 0.02 * nfac + 2:
        raise RuntimeError(f"load region has {len(load.facets)} facets, {nfac} expected")

    field = Field.from_args("fu", np.float64, "vector", omega, approx_order=order)
    u = FieldVariable("u", "unknown", field)
    v = FieldVariable("v", "test", field, primary_var_name="u")
    solid = Material("solid", D=stiffness_from_youngpoisson(3, mat["E"], mat["nu"]))
    trac = Material("trac", val=(np.asarray(force, float) / area).reshape(3, 1))

    integ = Integral("i", order=2 * order)
    t1 = Term.new("dw_lin_elastic(solid.D, v, u)", integ, omega, solid=solid, v=v, u=u)
    t2 = Term.new("dw_surface_ltr(trac.val, v)", integ, load, trac=trac, v=v)
    pb = Problem("elast", equations=Equations([Equation("balance", t1 + t2)]))
    pb.set_bcs(ebcs=Conditions([EssentialBC("fixed", fix, {"u.all": 0.0})]))
    pb.set_solver(Newton({}, lin_solver=spd_solver(), status=IndexedStruct()))
    st = pb.solve()

    uu = st.get_state_parts()["u"].reshape(-1, 3)
    sqp = pb.evaluate(f"ev_cauchy_stress.{2*order}.Omega(solid.D, u)", mode="qp",
                      copy_materials=False)
    vel = get_von_mises_stress(sqp.reshape(-1, 6, 1)).reshape(sqp.shape[0], -1).max(axis=1)
    s_el = pb.evaluate(f"ev_cauchy_stress.{2*order}.Omega(solid.D, u)", mode="el_avg",
                       copy_materials=False).reshape(-1, 6)

    # the peak sitting exactly on the clamped nodes is a numerical singularity - it keeps
    # growing as the mesh is refined.  Report the peak one element-length away from it.
    h = float(np.cbrt(np.abs(np.linalg.det(coors[tets[:, 1:]] - coors[tets[:, :1]])) / 6).mean())
    d = cKDTree(coors[fix_pred(coors)]).query(coors[tets].mean(axis=1))[0]
    free = d > 4.0 * h
    vfree = float(vel[free].max()) if free.any() else float(vel.max())

    if save:
        pb.save_state(save, st, out={"vm": Struct(name="out", mode="cell",
                                                  data=vel.reshape(-1, 1, 1, 1), dofs=None)})
    return dict(s_el=s_el, free=free,
                umax=float(np.linalg.norm(uu, axis=1).max()),
                vfree=vfree, vmax=float(vel.max()), v99=float(np.percentile(vel, 99.0)),
                area=area, n_nodes=len(coors), n_el=len(tets), n_fix=nfix)

# ---------------------------------------------------------------------------------
# self-test: cantilever beam against Euler-Bernoulli
# ---------------------------------------------------------------------------------
def selftest(size=2.0, order=2):
    import gmsh, meshio
    os.makedirs(OUT, exist_ok=True)
    L, W, H, F = 100.0, 10.0, 10.0, 20.0
    mp = os.path.join(OUT, "_beam.mesh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.addBox(0, -W / 2, -H / 2, L, W, H)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.model.mesh.generate(3)
        gmsh.write(mp.replace(".mesh", ".msh"))
    finally:
        gmsh.finalize()
    m = meshio.read(mp.replace(".mesh", ".msh"))
    tets = np.vstack([b.data for b in m.cells if b.type == "tetra"])
    meshio.write(mp, meshio.Mesh(m.points, [("tetra", tets)]), file_format="medit")

    mat = MATERIALS["PLA"]
    r = solve(mp, m.points, tets, mat, zone("x", hi=1e-6), zone("x", lo=L - 1e-6),
              [0, 0, -F], order=order)
    I = W * H ** 3 / 12.0
    d_an, s_an = F * L ** 3 / (3 * mat["E"] * I), F * L * (H / 2) / I
    print(f"  cantilever {L:.0f}x{W:.0f}x{H:.0f} mm, tip load {F:.0f} N, E={mat['E']:.0f} MPa")
    print(f"  tip deflection : FEA {r['umax']:7.4f} mm   analytic {d_an:7.4f} mm"
          f"   -> {100*(r['umax']/d_an-1):+6.1f} %")
    print(f"  root stress    : FEA {r['vfree']:7.2f} MPa  analytic {s_an:7.2f} MPa"
          f"   -> {100*(r['vfree']/s_an-1):+6.1f} %"
          f"   (raw max {r['vmax']:.1f}, singular at the clamp)")
    ok = abs(r["umax"] / d_an - 1) < 0.05 and abs(r["vfree"] / s_an - 1) < 0.15
    print(f"  {'PASS' if ok else 'FAIL'} - deflection within 5 %, stress within 15 %")
    return ok

# ---------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------
def run_part(name, case, cs, matname, size, order, save=True):
    spec = part_specs()[name]
    mat = MATERIALS[matname]
    mesh_path, coors, tets = part_mesh(name, size)
    _, cen, _ = boundary_patch(coors, tets, spec["load"])
    ap, ad = axis_of(spec["joint"])

    c = cs[case]
    if c["kind"] == "ground":
        force, lever = rot_y([0.0, 0.0, c["n"]], -spec["pose"]), None
    else:
        d = (cen - ap) - np.dot(cen - ap, ad) * ad      # axis -> patch, perpendicular part
        lever = float(np.linalg.norm(d))
        force = np.cross(ad, d / lever) * (c["m"] / lever)
    out = os.path.join(OUT, f"{name}_{case}_{matname}.vtk") if save else None
    r = solve(mesh_path, coors, tets, mat, spec["fix"], spec["load"], force,
              order=order, save=out)

    print(f"  {name:14s} {case:7s} F={np.linalg.norm(force):6.1f} N "
          f"{'lever %5.1f mm' % lever if lever else '             '} "
          f"| vM {r['vfree']:6.1f} MPa (p99 {r['v99']:5.1f}) "
          f"| {r['umax']:5.2f} mm | SF {mat['s_xy']/r['vfree']:4.1f} /{mat['s_z']/r['vfree']:5.1f}")
    return r

def orient_report(parts, case_names, cs, matname, size, order):
    """for each part: worst interlayer failure index over the load cases, per build axis"""
    mat = MATERIALS[matname]
    axes = [("robot X", [1, 0, 0]), ("robot Y", [0, 1, 0]), ("robot Z", [0, 0, 1])]
    print(f"\n  interlayer check, {matname}: SF = 1 / max failure index over"
          f" {', '.join(case_names)}\n  (the index folds tension across the layers and shear"
          f" along them; build direction only)")
    for name in parts:
        spec = part_specs()[name]
        mesh_path, coors, tets = part_mesh(name, size)
        free, stresses = None, []
        for case in case_names:
            r = run_part(name, case, cs, matname, size, order, save=False)
            free = r["free"] if free is None else (free & r["free"])
            stresses.append(r["s_el"])
        worst = lambda n: max(layer_index(se, n, mat)[free].max() for se in stresses)
        cur = build_dir(md.PRINT_ORIENT[name])

        def height(n):
            p = coors @ (np.asarray(n, float) / np.linalg.norm(n))
            return float(p.max() - p.min())
        rows = [(nm, np.asarray(v, float), worst(v)) for nm, v in axes]
        best_free = min(sphere_dirs(300), key=worst)
        print(f"\n  {name}   (mini_dog PRINT_ORIENT = {md.PRINT_ORIENT[name]})")
        print(f"     {'build dir':14s} {'SF':>5s}  {'height':>7s} {'overhang':>9s} {'bed':>7s}")
        for nm, v, w in sorted(rows, key=lambda t: t[2], reverse=True):
            for sg in (1, -1):
                hh, ov, bd = printability(coors, tets, sg * v)
                mark = " <- current" if np.dot(sg * v / np.linalg.norm(v), cur) > 0.99 else ""
                print(f"     {nm:>8s} {'+-'[sg < 0]:2s} {1/w:5.2f} {hh:6.1f}mm "
                      f"{ov:7.0f}mm2 {bd:6.0f}mm2{mark}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", nargs="*")
    ap.add_argument("--case", default=None)
    ap.add_argument("--mat", default="PETG", choices=list(MATERIALS))
    ap.add_argument("--size", type=float, default=2.0, help="target element size, mm")
    ap.add_argument("--order", type=int, default=2, help="1 = tet4, 2 = tet10")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--orient", action="store_true",
                    help="rank build directions by the interlayer failure index")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)

    specs = part_specs()
    parts = list(specs) if a.all else a.part
    if not parts:
        ap.error(f"give a part name ({', '.join(specs)}), --all, or --selftest")

    mass = robot_mass()
    cs = cases(mass)
    print(f"\n  m = {mass:.2f} kg -> W = {mass*G_ACC:.1f} N | servo stall {SERVO_STALL_NM} N*m"
          f" | {a.mat} | mesh {a.size} mm, order {a.order}"
          f"\n  {'':22s} {'':21s}   peak von Mises        defl   SF xy / z")
    if a.orient:
        cn = [a.case] if a.case else ["land3g", "stall"]
        orient_report(parts, cn, cs, a.mat, a.size, a.order)
        return

    for name in parts:
        print(f"\n  {name}: {specs[name]['note']}")
        for case in ([a.case] if a.case else list(cs)):
            run_part(name, case, cs, a.mat, a.size, a.order)

    print(f"""
  SF = strength / peak von Mises, in-plane / inter-layer.  Design to the inter-layer one
  unless the part is oriented so the load never crosses a layer boundary.

  Caveats:
   * linear static, solid geometry, isotropic. Real FDM parts are neither solid (infill)
     nor isotropic; the material table is already de-rated for ~5 walls / 40 %.
   * 'stall' bounds what the servo itself can do to the part; an impact is not bounded by
     the servo, and land3g is a 3 g estimate, not a measurement.
   * the clamp is idealised as a rigidly fixed hub footprint - real M2.5 bolts through
     {md.ARM_T:.0f} mm fork arms are softer and put the load into the bolt holes.
   * open {OUT}/*.vtk in ParaView to see where the stress actually sits.""")

if __name__ == "__main__":
    main()
