"""
randomize.py — the randomisation that lives in the MODEL, for brax's
`randomization_fn`.

There are two randomisations in this tree and the split is not arbitrary:

  HERE, per ENVIRONMENT, fixed for the whole run: things that are fields of the
  MuJoCo model — foot friction, link masses, where the centre of mass actually
  is, where the procedural terrain boxes sit. brax vmaps the model over these,
  so every field touched here costs num_envs copies of itself in VRAM. That is
  why the heightfield is not among them: 446k points x 2048 environments is
  3.6 GB on a card with 8.

  IN model.sample_actuator_params, per EPISODE, carried in the env state: the
  servo, the pack, the bus. Those are not model fields — actuator.py computes
  the torque outside MuJoCo — so they can be resampled at every reset, which is
  better, and they cost twelve floats per environment.

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
                     n_boxes: int = 0):
    """brax randomization_fn: (sys, rng) -> (sys_v, in_axes).

    `rng` arrives with one key per environment. Returns the vmapped system and
    the in_axes tree telling brax which fields are batched.
    """
    ranges = ranges or model_mod.domain_ranges()
    fr_lo, fr_hi = ranges["contact"]["friction"]["range"]
    m_lo, m_hi = ranges["body"]["mass_scale"]["range"]
    c_lo, c_hi = ranges["body"]["com_offset_m_abs"]["range"]
    p_lo, p_hi = ranges["body"]["payload_kg_abs"]["range"]
    h_lo, h_hi = ranges["terrain"]["box_height_m_abs"]["range"]
    d_lo, d_hi = ranges["terrain"]["box_density"]["range"]

    # The boxes are the last n_boxes geoms in the model — model.build appends
    # them to the worldbody after everything the CAD generated.
    box_slice = slice(sys.ngeom - n_boxes, sys.ngeom) if n_boxes else None
    box_half_z = model_mod.BOX_HALF[2]
    patch = model_mod.BOX_PATCH_M

    def one(key):
        k_fr, k_m, k_c, k_p, k_h, k_d, k_xy = jax.random.split(key, 7)

        # -- foot friction. Sliding only; the torsional and rolling components
        #    of MuJoCo's friction triple are not what a printed foot varies in.
        fr = sys.geom_friction.at[:, 0].set(
            sys.geom_friction[:, 0] * jax.random.uniform(k_fr, (), minval=fr_lo, maxval=fr_hi))

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

        out = {"geom_friction": fr, "body_mass": mass, "body_ipos": ipos}

        if n_boxes:
            # Each box is raised to a random top height, or left buried. Density
            # is per environment: some environments are flat ground on purpose,
            # because the robot has to walk on a floor too.
            dens = jax.random.uniform(k_d, (), minval=d_lo, maxval=d_hi)
            up = jax.random.uniform(k_h, (n_boxes,), minval=0.0, maxval=1.0) < dens
            top = jax.random.uniform(k_h, (n_boxes,), minval=h_lo, maxval=h_hi)
            z = jnp.where(up, top - box_half_z, -box_half_z - 1.0)
            xy = jax.random.uniform(k_xy, (n_boxes, 2), minval=-patch, maxval=patch)
            pos = sys.geom_pos.at[box_slice, 0:2].set(xy)
            pos = pos.at[box_slice, 2].set(z)
            out["geom_pos"] = pos

        return out

    fields = jax.vmap(one)(rng)
    sys_v = sys.tree_replace({k: v for k, v in fields.items()})

    in_axes = jax.tree.map(lambda _: None, sys)
    in_axes = in_axes.tree_replace({k: 0 for k in fields})
    return sys_v, in_axes
