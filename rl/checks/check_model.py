#!/usr/bin/env python
"""
check_model.py — audit the generated MJCF before anything is trained on it.

    python checks/check_model.py                  # flat scene, numbers only
    python checks/check_model.py --terrain        # ... on the heightfield scene
    python checks/check_model.py --view           # ... plus the passive viewer
    python checks/check_model.py --shot out.png   # ... plus one offscreen frame
    python checks/check_model.py --json out.json  # machine-readable, for CI

Reads the *committed, generated* description in ros2/smalldog_description — no
CadQuery, no ROS, no JAX.  mujoco and numpy are the whole dependency list, so
this runs on the mac, in WSL and on the robot alike.

What it is for.  `3d/CLAUDE.md` already mandates a regeneration check after any
CAD change (`export_sim.py --check`: does it stand, did a mass jump).  This asks
a different question — not "is the model still the robot" but "is the model fit
to train a policy against", which turns on things the standing test cannot see:
whether the actuator's dominant inertia is a measured number or a guess, whether
the contact parameters MuJoCo actually applies are the ones the model asked for,
and whether the joint limits mean the same thing here as they do in the runtime.

Exit code is 0 if nothing FAILed.  WARNs are things that are wrong for RL but
harmless for the trot that lives in ros2/ — they are listed, not fatal, because
this file must stay runnable against the model as it is today.
"""
import argparse, json, math, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
DESC = os.path.abspath(os.path.join(HERE, "..", "..", "ros2", "smalldog_description"))
MJCF = os.path.join(DESC, "mujoco")
PARAMS = os.path.join(DESC, "robot_params.json")

# The ST3215 as the CAD declares it (3d/mini_dog.py section 4).  Repeated here
# only to be checked against, never to be read as a source: if these disagree
# with the model, the model is what ships and this file is what is stale.
SERVO_STALL_NM = 4.50        # MEASURED 2026-09-10, torque rig (vendor said 2.94)
SERVO_NOLOAD_RADS = 3.86     # MEASURED 2026-09-11, free hub (vendor said 4.71)
ENCODER_STEP_RAD = math.radians(360.0 / 4096)   # 0.088 deg, the reported resolution

FAIL, WARN, INFO = "FAIL", "warn", "    "


class Report:
    def __init__(self):
        self.rows, self.n_fail, self.n_warn = [], 0, 0
        self.data = {}

    def say(self, level, text):
        if level == FAIL:
            self.n_fail += 1
        elif level == WARN:
            self.n_warn += 1
        self.rows.append((level, text))
        print(f"  {level}  {text}" if level != INFO else f"        {text}")

    def head(self, title):
        print(f"\n== {title} " + "=" * max(0, 68 - len(title)))


def full_mass_matrix(m, d):
    """mj_fullM's signature moved between MuJoCo 3.x releases; take either."""
    M = np.zeros((m.nv, m.nv))
    try:
        mujoco.mj_fullM(m, d, M)
    except TypeError:
        mujoco.mj_fullM(m, M, d.qM)
    return M


def stance_qpos(m, d, P):
    """Put the robot in the CAD stance, actuators commanding the same pose."""
    mujoco.mj_resetData(m, d)
    for n in P["joint_names"]:
        kind = n.split("_")[1]
        q = 0.0 if kind == "roll" else P["stance_rad"][kind]
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        d.qpos[m.jnt_qposadr[j]] = q
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if a >= 0:
            d.ctrl[a] = q
    d.qpos[2] = P["stance_base_height_m"]
    mujoco.mj_forward(m, d)


# ---------------------------------------------------------------- 1. mass
def check_mass(m, R, P):
    R.head("mass and inertia")
    total = float(m.body_mass[1:].sum())
    print(f"        {'body':<12}{'mass kg':>10}   principal inertia kg m^2")
    for i in range(1, m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        I = m.body_inertia[i]
        print(f"        {n:<12}{m.body_mass[i]:10.5f}   "
              f"[{I[0]:.3e} {I[1]:.3e} {I[2]:.3e}]")
    R.data["total_mass_kg"] = total
    declared = P["total_mass_kg"]
    if abs(total - declared) > 1e-3:
        R.say(FAIL, f"total {total:.5f} kg != robot_params {declared:.5f} kg — "
                    f"the description is out of sync with itself, regenerate it")
    else:
        R.say(INFO, f"total {total:.4f} kg, matches robot_params")

    # A rigid body's principal moments must satisfy the triangle inequality.
    # A violation is not a tuning problem, it is a body that cannot exist, and
    # MuJoCo will integrate it into nonsense rather than refuse it.
    bad = []
    for i in range(1, m.nbody):
        a, b, c = sorted(m.body_inertia[i])
        if a + b < c * (1 - 1e-9):
            bad.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i))
    if bad:
        R.say(FAIL, f"principal inertia violates the triangle inequality: {', '.join(bad)}")
    else:
        R.say(INFO, "all principal inertias are physically realisable")


# ------------------------------------------------------- 2. joint limits
def check_limits(m, R, P):
    R.head("joint limits — three ladders that must not be confused")
    hard, soft = P["joint_limits_rad"], P["joint_soft_limits_rad"]
    print(f"        {'kind':<7}{'CAD ROM':>10}{'MJCF stop':>11}{'soft/gait':>11}"
          f"{'MJ margin':>11}{'soft margin':>13}")
    ok = True
    for kind in ("roll", "pitch", "knee"):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"fl_{kind}")
        mj = float(m.jnt_range[j][1])
        print(f"        {kind:<7}{hard[kind]:10.3f}{mj:11.3f}{soft[kind]:11.3f}"
              f"{hard[kind]-mj:11.3f}{hard[kind]-soft[kind]:13.3f}")
        if not (soft[kind] < mj < hard[kind] + 1e-9):
            ok = False
        if abs(m.jnt_range[j][0] + mj) > 1e-9:
            R.say(FAIL, f"{kind}: MJCF range is not symmetric — {m.jnt_range[j]}")
    if ok:
        R.say(INFO, "ladder is consistent: soft < MuJoCo stop <= CAD ROM")
    else:
        R.say(FAIL, "ladder is inconsistent — a soft limit at or past a hard stop")
    R.say(INFO, "for RL: action range lives inside SOFT, the safety layer clips at SOFT,")
    R.say(INFO, "the MuJoCo stop is a backstop, and the servo's own angle-limit registers")
    R.say(INFO, "get the CAD ROM.  Three numbers, three jobs — do not collapse them.")

    # The two SPEED numbers, which were declared at the top of this file and then
    # never compared against anything.  They are different quantities and the
    # model carries both, so both get checked rather than one being assumed to
    # imply the other.
    vlim = float(P["joint_velocity_limit"])
    ceil = float(P.get("joint_rate_ceiling_rad_s", float("nan")))
    R.data["joint_velocity_limit"] = vlim
    R.data["joint_rate_ceiling_rad_s"] = ceil
    if abs(vlim - SERVO_NOLOAD_RADS) > 1e-6:
        R.say(FAIL, f"joint_velocity_limit {vlim:.3f} != the CAD's measured no-load "
                    f"speed {SERVO_NOLOAD_RADS:.2f} rad/s — one of the two is stale, "
                    f"and it is not the generated model; regenerate, or fix the "
                    f"constant at the top of this file")
    else:
        R.say(INFO, f"joint_velocity_limit {vlim:.2f} rad/s = the MEASURED no-load "
                    f"speed (2026-09-11, free hub).  env/rewards.py's joint_vel "
                    f"penalty is what holds the policy under it, because "
                    f"actuator.py's law has no firmware plateau and runs to 5.90.")
    if not math.isnan(ceil):
        # (forcerange - frictionloss)/damping in the GENERATED model: the fastest
        # the position actuator there can turn a joint.  ros2/smalldog_walker
        # rate-limits its gait against it (3d/CLAUDE.md, 2026-09-11 re-baseline);
        # nothing in rl/ read it at all, and it is the number that says how much
        # of the no-load speed the analytic controller can actually reach.
        R.say(INFO, f"joint_rate_ceiling_rad_s {ceil:.2f} — the generated model's own "
                    f"(forcerange - frictionloss)/damping, {ceil/vlim*100:.0f} % of the "
                    f"no-load speed.  rl/ does NOT train against it: model.py replaces "
                    f"the position actuator with the voltage law, so this ceiling is "
                    f"ros2/smalldog_walker's limit, not this tree's.")
        if ceil > vlim + 1e-6:
            R.say(FAIL, f"the model's rate ceiling {ceil:.2f} is ABOVE the servo's "
                        f"measured no-load speed {vlim:.2f} — the analytic gait would "
                        f"be rate-limited against a speed the hardware does not have")
    else:
        R.say(WARN, "no joint_rate_ceiling_rad_s in robot_params.json — regenerate "
                    "the description; ros2/smalldog_walker's gait limits against it")
    R.data["limits"] = {"hard": hard, "soft": soft,
                        "mjcf": {k: float(m.jnt_range[mujoco.mj_name2id(
                            m, mujoco.mjtObj.mjOBJ_JOINT, f"fl_{k}")][1])
                            for k in ("roll", "pitch", "knee")}}


# ---------------------------------------------------------- 3. actuators
def check_actuators(m, d, R, P):
    R.head("actuators — what the position servo actually is in this model")
    kp = float(m.actuator_gainprm[0][0])
    fr = m.actuator_forcerange[0]
    R.data["kp"] = kp
    R.say(INFO, f"position actuators, kp = {kp:g}, forcerange = "
                f"[{fr[0]:g}, {fr[1]:g}] N*m, dampratio-resolved kv per joint below")

    if not m.actuator_ctrllimited.any():
        R.say(WARN, "no ctrlrange on any actuator: a policy may command a target far "
                    "outside the mechanical range and lean on the joint stop for free. "
                    "The robot cannot — Goal Position is bounded. Set ctrlrange to the "
                    "soft limits before training.")
    if abs(fr[1] - SERVO_STALL_NM) > 1e-6:
        R.say(INFO, f"forcerange {fr[1]:g} is the stall torque {SERVO_STALL_NM:g} rounded up; "
                    f"a constant either way, so it carries no voltage dependence — "
                    f"at 9.9 V (3S empty) the real stall is nearer {SERVO_STALL_NM*9.9/12:.2f} N*m")

    # dampratio resolves kv against the joint's own effective inertia at compile
    # time: kv = 2*sqrt(kp * M_ii).  That is worth saying out loud, because M_ii
    # here is almost entirely `armature`, which is a guess — so the damping is
    # downstream of the same guess, and re-fitting armature moves it too.
    stance_qpos(m, d, P)
    M = full_mass_matrix(m, d)
    R.head("armature — the number the leg dynamics actually hangs on")
    print(f"        {'joint':<10}{'M_ii':>11}{'armature':>11}{'links':>12}"
          f"{'ratio':>9}{'kv':>8}")
    ratios = {}
    for n in ("fl_roll", "fl_pitch", "fl_knee"):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        v = m.jnt_dofadr[j]
        arm, tot = float(m.dof_armature[v]), float(M[v, v])
        link = tot - arm
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        kv = -float(m.actuator_biasprm[a][2])
        ratios[n] = arm / link
        print(f"        {n:<10}{tot:11.6f}{arm:11.6f}{link:12.6f}"
              f"{arm/link:8.1f}x{kv:8.3f}")
    R.data["armature_over_link"] = ratios
    worst = max(ratios.values())
    R.say(WARN if worst > 3 else INFO,
          f"reflected rotor inertia is {worst:.0f}x the link inertia at the knee. "
          f"With a 1:345 gearbox that is expected, and it means the leg's dynamics "
          f"is set by `armature` — which IS measured now (3d/mini_dog.py "
          f"MJ_ARMATURE, the bench free swing at the corrected friction, "
          f"2026-09-09). Kept as a warn because the RATIO is what makes every "
          f"other actuator constant matter, not because the number is unknown.")
    R.say(INFO, "kv = 2*sqrt(kp * M_ii) from dampratio=1, so refitting armature "
                "moves the damping with it. Do not also hand-tune kv.")


# ------------------------------------------------------------ 4. contact
def check_contact(m, d, R, P, settle=3.0):
    R.head("contact — declared vs what the solver applies")
    foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fl_foot")
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for gid, label in ((foot, "fl_foot"), (floor, "floor")):
        if gid < 0:
            continue
        R.say(INFO, f"{label:<9} declared solref={m.geom_solref[gid]} "
                    f"solimp={m.geom_solimp[gid][:3]} friction={m.geom_friction[gid]} "
                    f"priority={m.geom_priority[gid]} solmix={m.geom_solmix[gid]:g}")

    stance_qpos(m, d, P)
    for _ in range(int(settle / m.opt.timestep)):
        mujoco.mj_step(m, d)
    if d.ncon == 0:
        R.say(FAIL, "no contacts after settling — the robot is not standing on anything")
        return
    c = d.contact[0]
    eff_solref, eff_solimp = np.array(c.solref), np.array(c.solimp[:3])
    R.say(INFO, f"applied   solref={eff_solref} solimp={eff_solimp} "
                f"friction={np.array(c.friction[:3])} condim={c.dim}")
    R.data["contact"] = {"declared_foot_solref": m.geom_solref[foot].tolist(),
                         "effective_solref": eff_solref.tolist()}

    # Equal priority means MuJoCo mixes: friction elementwise max, but solref and
    # solimp a solmix-weighted AVERAGE.  So the foot class's deliberately stiff
    # 0.008 s is pulled towards the floor's 0.02 s and neither number is what the
    # model asked for.  It is invisible in the XML and it changes touchdown.
    if abs(eff_solref[0] - m.geom_solref[foot][0]) > 1e-9:
        R.say(WARN, f"the foot's declared solref[0]={m.geom_solref[foot][0]:g} s is not what "
                    f"runs: equal geom priority averages it with the floor's "
                    f"{m.geom_solref[floor][0]:g} s, giving {eff_solref[0]:g} s. Give the foot "
                    f"geoms priority=1 so their own contact parameters win, or set the "
                    f"pair deliberately — but do not leave the touchdown stiffness as "
                    f"the mean of two unrelated choices.")
    if m.opt.impratio <= 1.0:
        R.say(WARN, f"impratio={m.opt.impratio:g}: friction is no stiffer than the normal "
                    f"direction, so feet slip under load more than the friction "
                    f"coefficient suggests. Walking models usually run 10-100.")
    R.say(INFO, "friction 1.2 / solref / solimp are all estimates — 3d/export_sim.py "
                "flags the contact primitives as such. A rubber foot on a lab floor "
                "is measurable with a tilt test; do it before trusting a slip reward.")


# ---------------------------------------------------------- 5. standing
def check_stand(m, d, R, P, settle=3.0):
    R.head(f"standing test — {settle:g} s in the CAD stance, actuators holding it")
    stance_qpos(m, d, P)
    for _ in range(int(settle / m.opt.timestep)):
        mujoco.mj_step(m, d)
    Rm = np.zeros(9)
    mujoco.mju_quat2Mat(Rm, d.qpos[3:7])
    upright = float(Rm.reshape(3, 3)[2, 2])
    z, drift = float(d.qpos[2]), float(np.hypot(d.qpos[0], d.qpos[1]))
    sag = P["stance_base_height_m"] - z
    f = np.zeros(6)
    normal = 0.0
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, f)
        normal += f[0]
    weight = float(m.body_mass[1:].sum()) * 9.81
    R.data.update(stand_z=z, upright=upright, drift_m=drift, sag_m=sag)
    R.say(INFO, f"base z = {z:.4f} m (CAD stance {P['stance_base_height_m']:.4f}, "
                f"sag {sag*1000:+.1f} mm), upright = {upright:+.3f}, "
                f"xy drift = {drift*1000:.1f} mm, contacts = {d.ncon}")
    R.say(INFO, f"normal force {normal:.2f} N vs weight {weight:.2f} N")
    if upright < 0.95:
        R.say(FAIL, f"it fell over (upright {upright:+.3f})")
    if abs(normal - weight) > 0.02 * weight:
        R.say(FAIL, "contact forces do not carry the weight — the solver is not converged")
    # The sag is kp doing its job: holding torque / kp. It is a reading on the
    # actuator, not a defect, and it moves when the actuator model is refit.
    R.say(INFO, f"the {sag*1000:.1f} mm sag is steady-state error of the position "
                f"actuator under load (tau/kp); expect it to move once step 3 "
                f"replaces kp with a fitted voltage law.")


# ----------------------------------------------------------- 6. sensors
def check_sensors(m, R):
    R.head("sensors the RL observation will read")
    want = {"imu_quat": "body orientation (sim only — the robot has no magnetometer)",
            "imu_gyro": "angular velocity, goes straight into the observation",
            # Projected gravity does NOT come from here: env/walk.py rotates
            # imu_quat's world -z into the body frame for that. The
            # accelerometer is its own observation channel, PROPER acceleration,
            # which is gravity plus the body's own motion plus the w x (w x r)
            # artefact imu_placement.py measures — a different signal that
            # happens to contain the same vector at rest.
            "imu_accel": "proper acceleration, its own observation channel",
            "fl_contact": "foot load, for rewards and diagnostics only"}
    for name, why in want.items():
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
        R.say(INFO if i >= 0 else FAIL, f"{name:<12} {'present' if i>=0 else 'MISSING'}  — {why}")
    s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
    if s >= 0:
        pos = m.site_pos[s]
        b = m.site_bodyid[s]
        com = m.body_ipos[b]
        R.say(INFO, f"imu site at {np.array2string(pos, precision=4)} of "
                    f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)}; "
                    f"that body's com is at {np.array2string(com, precision=4)}")
        if np.allclose(pos, 0.0):
            R.say(WARN, "the imu site is at the body origin, which is where no board "
                        "will physically fit. An accelerometer offset from this point "
                        "reads w x (w x r) + alpha x r on top of gravity; measured on "
                        "the existing 0.20 m/s trot that is 3.8 m/s^2 at the 95th "
                        "percentile for a deck mounting, peaking at 9.0 — 42 deg of "
                        "apparent tilt. It is not noise: it correlates with the "
                        "policy's own actions. Place the BMI088 in the CAD and let "
                        "this site follow it BEFORE freezing the observation.")


# ---------------------------------------------- 6b. is the servo randomised
#: Fields of actuator.Params that describe a PHYSICAL servo and therefore vary
#: from one unit to the next, so every one of them has to be randomised or the
#: policy trains against twelve identical motors.  The rest of Params is
#: deliberately NOT here: J_l, theta_bl, k_bl, c_bl are load/backlash structure,
#: duty_max / loop_hz / v_eps / kd are configuration or solver constants, and
#: none of them is a manufacturing spread.
PER_UNIT_FIELDS = ["k_u", "k_e", "R", "J_m", "tau_c", "b_v", "mu_load", "kp",
                   "deadband", "punch"]

#: Per-unit fields that are NOT in the per-episode draw, because the field the
#: PHYSICS reads is a MuJoCo model field and not an actuator.Params attribute.
#: brax's randomization_fn is the only thing that can move one of those, so
#: env/randomize.py draws them, per environment. Field -> (the MjModel array,
#: where the draw lives). A range with nothing reading it is the same defect
#: whichever side of this line it is on, which is why they stay in
#: PER_UNIT_FIELDS and are probed here rather than dropped.
#:
#: `J_m` is inertia: MuJoCo owns the mass matrix and calls it `dof_armature`.
#: `tau_c` is the Coulomb floor: only a stick-slip constraint holds a joint at
#: rest, so it is `dof_frictionloss` and the law is called with
#: `tau_c_external=True` on every MuJoCo path (PLAN.md 2b).
MODEL_SIDE_FIELDS = {
    "J_m":   ("dof_armature", "env/randomize.py, as sys.dof_armature"),
    "tau_c": ("dof_frictionloss", "env/randomize.py, as sys.dof_frictionloss"),
}

#: Operating points the consumption probe below evaluates the law at.  Three,
#: because no single one exercises every register: the first sits just outside
#: the dead zone where `deadband` and `punch` decide the drive, the second is a
#: large error at speed where k_u/k_e/R/b_v dominate, the third is a reversal
#: where the Coulomb terms do.  (err rad, w rad/s)
PROBE_POINTS = ((0.010, 1.0), (0.300, -3.0), (0.002, 0.05))


def _law_moves(base, field, lo, hi, absolute):
    """Does the TRAINING PATH's output move when this field moves across its range?

    `actuator.bus_torque` is that path — inner loop, pack sag, motor torque — and
    it is the same function rl/env/walk.py calls inside its physics scan, which
    is the point: a probe that reimplemented it could drop a term and then report
    the parameter that term reads as unconsumed.  `tau_c_external=True` for the
    same reason: that is what walk.py and eval.py pass, so a field this probe
    says is consumed is one the training path really reads.
    """
    import dataclasses
    import actuator
    nom = getattr(base, field)
    a, b = (lo, hi) if absolute else (lo * nom, hi * nom)
    for err, w in PROBE_POINTS:
        t = [actuator.bus_torque(dataclasses.replace(base, **{field: v}),
                                 err, w, 12.0, 0.03, xp=np, tau_c_external=True)
             for v in (a, b)]
        if abs(float(t[0]) - float(t[1])) > 1e-9:
            return True
    return False


def check_randomisation(R):
    """Every per-unit servo parameter must be drawn per episode AND consumed.

    Two questions, and this check used to ask only half of the first.

    BATCHED exists because `mu_load` shipped unrandomised.  It was in the fit, in
    actuator.Params and in the jax pytree, and every environment still trained
    against one shared value while `tau_c` beside it - the SMALLER of the two
    friction terms - got a per-joint draw.  Nothing caught it: the field exists
    everywhere it is looked for, so the only symptom was a shape, `()` where its
    neighbour was `(n, 12)`, and it took reading `_params()` live on the training
    box to see it.  A missing entry is not a crash and never will be, which is
    exactly why it needs a probe rather than a test of something else.

    CONSUMED exists because the shape test passed two defects that shipped
    anyway.  `J_m` was drawn per joint and fed into an actuator.Params field that
    no function on the training path reads - the inertia the physics uses is
    `dof_armature` - and the bus delay was drawn in seconds and truncated to
    whole ticks, which made all 2000 of 2000 draws zero.  Both came back
    correctly shaped.  So the probe now moves each field across its own range and
    asks whether the law's output moves with it, and looks at the delay in TICKS
    rather than in the seconds nothing downstream uses.

    The draw it probes is the one training uses: `model.EPISODE_DRAW` is walked
    by both `sample_actuator_params` here and `Walk._sample_episode` there.  It
    used to be a second, hand-written copy, which is how a check could be green
    about a draw the environment never called.

    MODEL_SIDE_FIELDS are the two that answer both questions somewhere else.
    `J_m` left the draw on 2026-09-11 because nothing read it; `tau_c` left the
    same day because only MuJoCo's `dof_frictionloss` can hold a joint at rest
    (PLAN.md 2b) and the law is handed `tau_c_external=True` on every MuJoCo
    path.  For those two the probe is inverted - they must be ABSENT from the
    episode draw - and then the compiled training model is asked whether the
    field they moved to is really installed at the fitted nominal.  Whether the
    per-environment draw on top of it is batched is env/randomize.py's own
    `_selftest()`, which this file cannot run: it needs jax, and this file has to
    run on the robot.
    """
    R.head("domain randomisation — is every per-unit servo parameter batched")
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import actuator
        import model as model_mod
    except Exception as e:                                    # pragma: no cover
        R.say(WARN, f"could not import the rl tree: {e}")
        return

    all_ranges = model_mod.domain_ranges()
    ranges = all_ranges["actuator"]
    base = actuator.load(quiet=True)
    n = 256
    draw = model_mod.sample_actuator_params(np.random.default_rng(0), n)
    for f in PER_UNIT_FIELDS:
        key = f if f in ranges else f + "_abs"
        if key not in ranges:
            R.say(FAIL, f"{f:<9} has no range in params/domain_rand.json — every "
                        f"environment will share one value")
            continue
        lo, hi = ranges[key]["range"]
        a = draw.get(f)
        if f in MODEL_SIDE_FIELDS:
            _, where = MODEL_SIDE_FIELDS[f]
            if a is not None:
                R.say(FAIL, f"{f:<9} is drawn per episode, but the training path "
                            f"does not read it there — it belongs in {where}")
            else:
                R.say(INFO, f"{f:<9} {'(model)':<9} x{lo:.2f}..{hi:.2f}  "
                            f"{ranges[key]['evidence']} — randomised in {where}")
            continue
        if a is None:
            R.say(FAIL, f"{f:<9} has a range but the episode draw does not "
                        f"produce it — the range is dead and the value is shared")
        elif np.ndim(a) == 0 or np.size(a) == 1:
            R.say(FAIL, f"{f:<9} came back shape {np.shape(a)} — shared, not sampled")
        elif not _law_moves(base, f, lo, hi, key.endswith("_abs")):
            R.say(FAIL, f"{f:<9} is drawn but the law's output does not move "
                        f"across x{lo:g}..{hi:g} — the draw is dead, like J_m's was")
        else:
            R.say(INFO, f"{f:<9} {str(np.shape(a)):<9} "
                        f"x{lo:.2f}..{hi:.2f}  {ranges[key]['evidence']}")

    # The bus delay, in the unit the env actually indexes its action buffer with.
    # In seconds it always looked fine; it was the conversion that was dead.
    lo_s, hi_s = all_ranges["bus"]["delay_s_abs"]["range"]
    u = np.random.default_rng(1).uniform(0.0, 1.0, n)
    ticks = model_mod.delay_ticks(draw["delay_s"], u, 50.0, xp=np).astype(int)
    share = float(np.mean(ticks > 0))
    R.data["bus_delay"] = {"range_s": [lo_s, hi_s], "share_one_tick_late": share}
    if len(np.unique(ticks)) < 2:
        R.say(FAIL, f"bus delay: {n} draws all came back {ticks[0]} tick(s) — the "
                    f"latency axis is dead. It was, for the whole of step 4: "
                    f"uniform(0, 0.020) s x 50 Hz is [0, 1.0) and int() is 0.")
    else:
        R.say(INFO, f"bus delay  {str(np.shape(ticks)):<9} "
                    f"{lo_s*1000:.2f}..{hi_s*1000:.2f} ms measured "
                    f"(params/bus_timing.json, sync_read + sync_write p50/p95) — "
                    f"{share*100:.0f} % of episodes run one tick late")

    # The fields MuJoCo owns: the draw is in env/randomize.py and needs jax, so
    # what is checkable from here is the NOMINAL those draws multiply. A zero in
    # either is not a narrower spread: model.build_spec() installs
    # `frictionloss <- tau_c` and `armature <- J_m`, and env/randomize.py scales
    # what it finds — so a zero nominal is a field that stays zero however wide
    # the range is, and for tau_c that is no Coulomb friction at all, because the
    # law has already handed the floor away (tau_c_external).
    try:
        m_tr, _ = model_mod.build(terrain=False, n_boxes=0)
        P_tr = model_mod.robot_params()
        _, vadr, _ = model_mod.joint_order(m_tr, P_tr)
        vadr = np.asarray(vadr)
        for f, (attr, where) in MODEL_SIDE_FIELDS.items():
            nom = float(getattr(base, f))
            v = np.asarray(getattr(m_tr, attr))[vadr]
            lo, hi = ranges[f]["range"]
            if np.any(v <= 0.0):
                R.say(FAIL, f"{f:<9} {attr} is {v.min():.4g} on some joint — the "
                            f"draw in {where} is multiplicative, so a zero here "
                            f"is that axis dead whatever the range says")
            elif not np.allclose(v, nom, rtol=1e-6):
                R.say(FAIL, f"{f:<9} {attr} {v.min():.4g}..{v.max():.4g} is not "
                            f"the fitted nominal {nom:.4g} — the draw scales this "
                            f"number, so the whole spread is off by the same factor")
            else:
                R.say(INFO, f"{f:<9} {attr} = {nom:.4g} on 12 joints, "
                            f"x{lo:.2f}..{hi:.2f} per environment in {where}")
    except Exception as e:                                    # pragma: no cover
        R.say(WARN, f"could not build the training model to check the model-side "
                    f"fields ({', '.join(MODEL_SIDE_FIELDS)}): {e}")

    extra = [k for k in ranges if k not in PER_UNIT_FIELDS
             and k.removesuffix("_abs") not in PER_UNIT_FIELDS]
    for k in extra:
        R.say(WARN, f"{k} is in domain_rand.json but not in PER_UNIT_FIELDS — "
                    f"decide whether it is a per-unit spread and list it (adding "
                    f"it to MODEL_SIDE_FIELDS too if MuJoCo owns the field the "
                    f"physics reads), or drop it")


# ------------------------------------------------------------- 7. ledger
def ledger(R):
    R.head("measured vs guessed — what step 3 has to replace")
    for what, where, status in (
        ("link mass, com, inertia", "real solids in 3d/mini_dog.py", "measured"),
        ("joint limits", "swept-boolean ROM scan", "measured"),
        ("link geometry", "CAD", "measured"),
        ("armature", "3d/mini_dog.py MJ_ARMATURE", "MEASURED 2026-09-09 — dominates the leg"),
        ("damping", "3d/mini_dog.py MJ_DAMPING", "MEASURED 2026-09-09, 3 voltages"),
        ("frictionloss", "3d/mini_dog.py MJ_FRICTIONLOSS", "MEASURED 2026-09-09, 2 routes"),
        ("actuator kp", "3d/mini_dog.py MJ_KP", "MEASURED 2026-09-09 — still not a servo model"),
        ("stall torque", "3d/mini_dog.py SERVO_STALL_NM", "MEASURED 2026-09-10, 4.50 N*m"),
        ("no-load speed", "3d/mini_dog.py SERVO_NOLOAD_RADS",
         "MEASURED 2026-09-11, 3.86 rad/s — actuator.py's law gives 5.9"),
        ("foot friction", "3d/export_sim.py", "GUESSED"),
        ("foot solref/solimp", "3d/export_sim.py",
         "GUESSED, but no longer diluted — priority=1 ships"),
        ("backlash", "not in the model at all", "ABSENT"),
        ("encoder quantisation", "actuator.duty() only, not the observation",
         f"HALF ABSENT ({math.degrees(ENCODER_STEP_RAD):.3f} deg/step)"),
        ("bus delay", "rl/params/bus_timing.json", "MEASURED 2026-09-07, 12 servos"),
    ):
        R.say(INFO, f"{what:<24} {status:<34} {where}")


# -------------------------------------------------------------- viewing
def view(m, d, P):
    """Passive viewer in the CAD stance, collision geometry made visible.

    On macOS this must run under mjpython (the viewer needs the main thread);
    ros2/tools/view.sh already solves that for the standalone sim and the same
    incantation works here.
    """
    import mujoco.viewer
    stance_qpos(m, d, P)
    print("\nLook at, in this order:")
    print("  1. the red collision primitives against the grey meshes — a box that")
    print("     does not cover its link is a leg that walks through obstacles;")
    print("  2. the feet: the black spheres are the only things that touch ground;")
    print("  3. the imu site (small sphere at the body origin) against where a")
    print("     BMI088 could physically be bolted — see the sensor section above;")
    print("  4. drive the joints to their stops with the actuator sliders and check")
    print("     nothing interpenetrates at the limit.")
    with mujoco.viewer.launch_passive(m, d) as v:
        v.opt.geomgroup[2] = 1        # visual meshes
        v.opt.geomgroup[3] = 1        # collision primitives
        v.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        v.sync()
        while v.is_running():
            mujoco.mj_step(m, d)
            v.sync()


def _write_png(rgb, path):
    """Minimal PNG writer — zlib and struct, no Pillow.

    The point of this file is that it runs anywhere with mujoco and numpy; adding
    an image library just to save one diagnostic frame would spend that.
    """
    import zlib, struct
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


def shot(m, d, P, path, w=1280, h=960):
    """One offscreen frame of the settled robot, collision primitives shown.

    Needs a GL backend: MUJOCO_GL=egl on a machine with a GPU driver (WSL2 has
    one), osmesa for pure software. Neither exists in every environment — if the
    Renderer cannot be created, say which knob to turn instead of a stack trace.
    """
    stance_qpos(m, d, P)
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    try:
        renderer = mujoco.Renderer(m, h, w)
    except Exception as e:
        print(f"\n--shot: no GL backend ({type(e).__name__}: {e}).\n"
              f"        Try MUJOCO_GL=egl (GPU) or MUJOCO_GL=osmesa (software).")
        return
    with renderer as r:
        opt = mujoco.MjvOption()
        opt.geomgroup[2] = 1
        opt.geomgroup[3] = 1
        r.update_scene(d, camera=-1, scene_option=opt)
        _write_png(r.render(), path)
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None, help="path to a scene xml (default: flat)")
    ap.add_argument("--terrain", action="store_true", help="use scene_terrain.xml")
    ap.add_argument("--settle", type=float, default=3.0, help="standing test length, s")
    ap.add_argument("--view", action="store_true", help="open the passive viewer")
    ap.add_argument("--shot", default=None, help="render one frame to this png")
    ap.add_argument("--json", default=None, help="write the numbers here")
    a = ap.parse_args()

    scene = a.scene or os.path.join(
        MJCF, "scene_terrain.xml" if a.terrain else "scene.xml")
    P = json.load(open(PARAMS))
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)

    print(f"model    {scene}")
    print(f"mujoco   {mujoco.__version__}")
    print(f"         nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} "
          f"dt={m.opt.timestep:g} solver={m.opt.solver} iterations={m.opt.iterations}")

    R = Report()
    check_mass(m, R, P)
    check_limits(m, R, P)
    check_actuators(m, d, R, P)
    check_contact(m, d, R, P, a.settle)
    check_stand(m, d, R, P, a.settle)
    check_sensors(m, R)
    check_randomisation(R)
    ledger(R)

    print("\n== result " + "=" * 68)
    print(f"  {R.n_fail} FAIL, {R.n_warn} warn")
    if a.json:
        json.dump({"scene": scene, "fail": R.n_fail, "warn": R.n_warn,
                   "rows": R.rows, **R.data}, open(a.json, "w"), indent=2)
        print(f"  wrote {a.json}")
    if a.shot:
        shot(m, d, P, a.shot)
    if a.view:
        view(m, d, P)
    return 1 if R.n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
