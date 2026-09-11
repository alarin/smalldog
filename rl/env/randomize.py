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

The split is not a matter of taste, and getting it wrong is silent. `J_m` was in
the per-episode draw for the whole of step 4 and did nothing: it went into an
`actuator.Params` field, and not one function on the training path reads it —
`duty`, `current`, `friction`, `motor_torque` all ignore it, because the inertia
belongs to the mass matrix and MuJoCo owns that. It is randomised HERE now, as
`dof_armature`, out of the same range.

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
    a_block, a_key = model_mod.ARMATURE_RANGE
    j_lo, j_hi = ranges[a_block][a_key]["range"]

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
        k_fr, k_m, k_c, k_p, k_j, k_up, k_h, k_d, k_xy = jax.random.split(key, 9)

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

        out = {"geom_friction": fr, "body_mass": mass, "body_ipos": ipos,
               "dof_armature": arm}

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
