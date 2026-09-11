"""
randomize.py — the randomisation that lives in the MODEL, for brax's
`randomization_fn`.

There are two randomisations in this tree and the split is not arbitrary:

  HERE, per ENVIRONMENT, fixed for the whole run: things that are fields of the
  MuJoCo model — foot friction, link masses, where the centre of mass actually
  is, the reflected rotor inertia, where the procedural terrain boxes sit. brax
  vmaps the model over these, so every field touched here costs num_envs copies
  of itself in VRAM. That is why the heightfield is not among them: 446k points
  x 2048 environments is 3.6 GB on a card with 8.

  IN model.EPISODE_DRAW, per EPISODE, carried in the env state: the servo, the
  pack, the bus. Those are not model fields — actuator.py computes the torque
  outside MuJoCo — so they can be resampled at every episode boundary, which is
  better, and they cost twelve floats per environment.

The split is not a matter of taste, and getting it wrong is silent. TWO fields
have crossed the line into this file, on the same day and for two different
reasons, and neither move was a preference:

  `J_m` was in the per-episode draw for the whole of step 4 and did nothing: it
  went into an `actuator.Params` field, and not one function on the training path
  reads it - `duty`, `current`, `friction`, `motor_torque` all ignore it, because
  the inertia belongs to the mass matrix and MuJoCo owns that. It is randomised
  HERE now, as `dof_armature`, out of the same range.

  `tau_c` crossed on 2026-09-11 for the opposite reason: the law reads it, but the
  Coulomb floor has to be MuJoCo's `dof_frictionloss` to stick at rest at all
  (model.py docstring 2, PLAN.md 2b), so the MuJoCo callers pass
  `tau_c_external=True` and the law's copy is unused on that path. A model field
  can only be randomised by brax's randomization_fn, so its spread is drawn HERE,
  per environment and fixed for the run, instead of per episode as it was. The
  physical reading is not worse: grease and preload are what this term varies
  with, and they do not change between one episode and the next. What is lost is
  variety per unit of wall-clock, and the cost in VRAM is nv floats.

One rule covers both, and it is the one to apply to the next candidate: randomise
the field the PHYSICS reads. A draw that lands anywhere else lands nowhere, and
nothing crashes when it does.

Both read params/domain_rand.json, and every range in it is labelled `measured`,
`spec` or `guessed`. The masses are measured (real solids in 3d/mini_dog.py) and
are therefore randomised NARROWLY: widening a measured number to be safe throws
away the measurement. The friction is a guess and is randomised wide.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

import model as model_mod


def domain_randomize(sys, rng: jax.Array, ranges: dict | None = None,
                     n_boxes: int = 0, box_geoms=None, joint_dofs=None):
    """brax randomization_fn: (sys, rng) -> (sys_v, in_axes).

    `rng` arrives with one key per environment. Returns the vmapped system and
    the in_axes tree telling brax which fields are batched.

    `box_geoms` is the box geom indices, from model.box_geom_ids(). Pass them.
    They are NOT the last n_boxes geoms — see that function for what the tail
    slice actually points at, and what it did when it was tried.

    `joint_dofs` is the twelve actuated dof indices, `Walk.joint_dofs`. Pass
    them too, for the same reason: `dof_armature` is 18 long here and the first
    six entries are the base's free joint, which has no rotor to reflect.
    """
    ranges = ranges or model_mod.domain_ranges()
    fr_lo, fr_hi = ranges["contact"]["friction"]["range"]
    m_lo, m_hi = ranges["body"]["mass_scale"]["range"]
    c_lo, c_hi = ranges["body"]["com_offset_m_abs"]["range"]
    p_lo, p_hi = ranges["body"]["payload_kg_abs"]["range"]
    h_lo, h_hi = ranges["terrain"]["box_height_m_abs"]["range"]
    d_lo, d_hi = ranges["terrain"]["box_density"]["range"]
    # The two that left model.EPISODE_DRAW, named there rather than spelled out
    # here: which range each of them reads is model.py's to declare, so a rename
    # in params/domain_rand.json breaks in one place instead of silently drawing
    # the wrong band in the other.
    a_block, a_key = model_mod.ARMATURE_RANGE
    j_lo, j_hi = ranges[a_block][a_key]["range"]
    f_block, f_key = model_mod.FRICTIONLOSS_RANGE
    tc_lo, tc_hi = ranges[f_block][f_key]["range"]

    if n_boxes and box_geoms is None:
        raise ValueError(
            "n_boxes without box_geoms. The indices have to come from "
            "model.box_geom_ids(mj_model, n_boxes); this function used to "
            "guess them as the tail of the geom array and the guess was "
            "wrong — it selected the rear-right leg.")
    if joint_dofs is None:
        raise ValueError(
            "joint_dofs is required: the armature draw has to land on the "
            "twelve actuated dofs, and model.joint_order() is the only thing "
            "that knows which those are. Walk carries them as `joint_dofs`.")
    dof_idx = jnp.asarray(joint_dofs, dtype=int)
    box_idx = jnp.asarray(box_geoms, dtype=int) if n_boxes else None
    box_half_z = model_mod.BOX_HALF[2]
    patch = model_mod.BOX_PATCH_M

    def one(key):
        (k_fr, k_m, k_c, k_p, k_j, k_fl,
         k_up, k_h, k_d, k_xy) = jax.random.split(key, 10)

        # -- foot friction. Sliding only; the torsional and rolling components
        #    of MuJoCo's friction triple are not what a printed foot varies in.
        #    One draw scaling EVERY geom's sliding friction, not just the feet:
        #    the shin capsules and the body boxes carry it too, which is right
        #    for a robot that meets an obstacle with a shin as readily as with a
        #    foot, and which is why this says `geom_friction` and not `feet`.
        #    The feet are the ones that matter because they are the ones in
        #    contact, and the floor's own friction is untouched either way.
        fr = sys.geom_friction.at[:, 0].set(
            sys.geom_friction[:, 0] * jax.random.uniform(k_fr, (), minval=fr_lo, maxval=fr_hi))

        # -- the reflected rotor inertia, per servo. This is J_m, and this is the
        #    only place it can be randomised: the physics reads it out of the
        #    mass matrix, so a draw that lands anywhere else lands nowhere. The
        #    range is the same one the per-episode draw used to read.
        arm = sys.dof_armature.at[dof_idx].multiply(
            jax.random.uniform(k_j, (dof_idx.shape[0],), minval=j_lo, maxval=j_hi))

        # -- link masses, narrow, because they are measured.
        mass = sys.body_mass * jax.random.uniform(
            k_m, (sys.nbody,), minval=m_lo, maxval=m_hi)
        # -- plus a payload in the bay, on the base link (body 1; 0 is the world).
        payload = jax.random.uniform(k_p, (), minval=p_lo, maxval=p_hi)
        mass = mass.at[1].add(payload)

        # -- where the mass actually sits. The CAD knows the structure's centre
        #    of mass; it does not know how the harness was dressed.
        ipos = sys.body_ipos.at[1].add(
            jax.random.uniform(k_c, (3,), minval=c_lo, maxval=c_hi))

        # -- the servo's Coulomb friction, which lives in MuJoCo because only a
        #    stick-slip constraint can hold a joint at rest. Multiplicative on
        #    the nominal model.build_spec() installed (the fitted tau_c), and over
        #    the WHOLE dof vector rather than over dof_idx like the armature
        #    above. The asymmetry is deliberate: build_spec() leaves the free
        #    joint's frictionloss at 0, so scaling all nv entries cannot disturb
        #    it and no index has to be hard-coded to protect it. `dof_armature`
        #    takes the index instead, because its free-joint entries are not ours
        #    to scale whatever they hold.
        fl = sys.dof_frictionloss * jax.random.uniform(
            k_fl, (sys.nv,), minval=tc_lo, maxval=tc_hi)

        out = {"geom_friction": fr, "body_mass": mass, "body_ipos": ipos,
               "dof_armature": arm, "dof_frictionloss": fl}

        if n_boxes:
            # Each box is raised to a random top height, or left buried. Density
            # is per environment: some environments are flat ground on purpose,
            # because the robot has to walk on a floor too.
            #
            # `up` and `top` get SEPARATE keys, and the bug that makes that worth
            # a comment is instructive: they were both drawn off k_h, so the two
            # uniforms were the SAME numbers. A box was raised only where that
            # number fell below the density, and its height came from the same
            # number — so the tallest box that could ever be raised was
            # h_lo + dens_max*(h_hi - h_lo) = 12.1 mm, and the 22 mm cap in
            # domain_rand.json was unreachable. The curriculum was half the
            # height it said it was, and every reward-side argument about foot
            # clearance was made against the wrong terrain (rewards.py,
            # FOOT_CLEARANCE_TARGET).
            dens = jax.random.uniform(k_d, (), minval=d_lo, maxval=d_hi)
            up = jax.random.uniform(k_up, (n_boxes,), minval=0.0, maxval=1.0) < dens
            top = jax.random.uniform(k_h, (n_boxes,), minval=h_lo, maxval=h_hi)
            z = jnp.where(up, top - box_half_z, -box_half_z - 1.0)
            xy = jax.random.uniform(k_xy, (n_boxes, 2), minval=-patch, maxval=patch)
            pos = sys.geom_pos.at[box_idx, 0:2].set(xy)
            pos = pos.at[box_idx, 2].set(z)
            out["geom_pos"] = pos

        return out

    fields = jax.vmap(one)(rng)
    sys_v = sys.tree_replace({k: v for k, v in fields.items()})

    in_axes = jax.tree.map(lambda _: None, sys)
    in_axes = in_axes.tree_replace({k: 0 for k in fields})
    return sys_v, in_axes


def _selftest(n: int = 8):
    """Every field this function touches must come back BATCHED, n copies deep.

    checks/check_model.py runs the same probe on the per-episode draw and cannot
    run this one: it has to work on the robot, where there is no jax. So it lives
    here and is run by hand - `python -m env.randomize` from rl/ - whenever this
    file or model.build_spec() moves. It exists for one reason: `mu_load` shipped
    unrandomised for weeks and the only symptom was a shape.

    `dof_frictionloss` and `dof_armature` are the two fields most worth probing,
    because they are the two that arrived here from the per-episode draw and
    because a zero in either is not a narrower spread. A zero frictionloss is no
    Coulomb friction at all on a path whose law has handed the floor away
    (`tau_c_external`, model.py docstring 2); a zero armature is a rotor with no
    inertia in the term that dominates the knee's.
    """
    import numpy as np

    from env.walk import Walk

    ok = True

    def check(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'ok  ' if good else 'FAIL'} {name}{'  ' + detail if detail else ''}")

    env = Walk()
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    sys_v, in_axes = domain_randomize(env.sys, keys,
                                      joint_dofs=env.joint_dofs)

    vadr = np.asarray(env._vadr)
    ranges = model_mod.domain_ranges()
    p0 = model_mod.actuator.load(quiet=True)

    def probe(field, nominal, block_key, units):
        """Batched, non-zero and in range on the twelve joints; free joint untouched."""
        a = np.asarray(getattr(sys_v, field))
        base = np.asarray(getattr(env.sys, field))
        lo, hi = ranges[block_key[0]][block_key[1]]["range"]
        check(f"{field} is batched", a.shape == (n, env.sys.nv),
              f"{a.shape} vs ({n}, {env.sys.nv})")
        check(f"brax is told {field} is batched",
              getattr(in_axes, field) == 0)
        joints = a[:, vadr]
        check(f"{field} is non-zero on every joint of every environment",
              bool(joints.min() > 0.0), f"min {joints.min():.4g} {units}")
        check(f"{field} spans the range and nothing wider",
              bool(joints.min() >= lo * nominal - 1e-9
                   and joints.max() <= hi * nominal + 1e-9),
              f"{joints.min():.4g}..{joints.max():.4g} of "
              f"{lo * nominal:.4g}..{hi * nominal:.4g}")
        check(f"{field} differs between environments", bool(joints.std() > 0.0),
              f"sd {joints.std():.4g} {units}")
        free = np.delete(a, vadr, axis=1)
        free0 = np.delete(base, vadr)
        check(f"{field} leaves the free joint alone",
              bool(np.all(free == free0[None, :])),
              f"max |delta| {np.abs(free - free0[None, :]).max():.4g}")

    # the Coulomb floor: build_spec installs the fitted tau_c on the twelve
    # joints and 0 on the free joint, and this draw multiplies it.
    probe("dof_frictionloss", float(p0.tau_c),
          model_mod.FRICTIONLOSS_RANGE, "N*m")
    # the reflected rotor inertia, the other field that left the episode draw.
    probe("dof_armature", float(p0.J_m),
          model_mod.ARMATURE_RANGE, "kg*m^2")

    mass = np.asarray(sys_v.body_mass)
    check("body_mass is still batched", mass.shape == (n, env.sys.nbody))
    fr = np.asarray(sys_v.geom_friction)
    check("geom_friction is still batched", fr.shape[0] == n)

    print("  PASS" if ok else "  FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
