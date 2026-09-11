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

`tau_c` crossed that line on 2026-09-11 and it is the one entry here that is
placed by a constraint rather than by preference. The Coulomb floor has to be
MuJoCo's `dof_frictionloss` to stick at rest at all (model.py docstring 2,
PLAN.md 2b), and `dof_frictionloss` is a model field — so its spread has to be
drawn HERE, per environment and fixed for the run, instead of per episode as it
was. The physical reading is not worse: grease and preload are what this term
varies with, and they do not change between one episode and the next. What is
lost is variety per unit of wall-clock, and the cost in VRAM is nv floats.

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
                     n_boxes: int = 0, box_geoms=None):
    """brax randomization_fn: (sys, rng) -> (sys_v, in_axes).

    `rng` arrives with one key per environment. Returns the vmapped system and
    the in_axes tree telling brax which fields are batched.

    `box_geoms` is the box geom indices, from model.box_geom_ids(). Pass them.
    They are NOT the last n_boxes geoms — see that function for what the tail
    slice actually points at, and what it did when it was tried.
    """
    ranges = ranges or model_mod.domain_ranges()
    fr_lo, fr_hi = ranges["contact"]["friction"]["range"]
    m_lo, m_hi = ranges["body"]["mass_scale"]["range"]
    c_lo, c_hi = ranges["body"]["com_offset_m_abs"]["range"]
    p_lo, p_hi = ranges["body"]["payload_kg_abs"]["range"]
    tc_lo, tc_hi = ranges["actuator"]["tau_c"]["range"]
    h_lo, h_hi = ranges["terrain"]["box_height_m_abs"]["range"]
    d_lo, d_hi = ranges["terrain"]["box_density"]["range"]

    if n_boxes and box_geoms is None:
        raise ValueError(
            "n_boxes without box_geoms. The indices have to come from "
            "model.box_geom_ids(mj_model, n_boxes); this function used to "
            "guess them as the tail of the geom array and the guess was "
            "wrong — it selected the rear-right leg.")
    box_idx = jnp.asarray(box_geoms, dtype=int) if n_boxes else None
    box_half_z = model_mod.BOX_HALF[2]
    patch = model_mod.BOX_PATCH_M

    def one(key):
        k_fr, k_m, k_c, k_p, k_h, k_d, k_xy, k_fl = jax.random.split(key, 8)

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

        # -- the servo's Coulomb friction, which lives in MuJoCo because only a
        #    stick-slip constraint can hold a joint at rest. Multiplicative on
        #    the nominal model.build_spec() installed (the fitted tau_c), so the
        #    free joint's six dofs — frictionloss 0 there — stay 0 whatever is
        #    drawn, and no dof index has to be hard-coded to protect them.
        fl = sys.dof_frictionloss * jax.random.uniform(
            k_fl, (sys.nv,), minval=tc_lo, maxval=tc_hi)

        out = {"geom_friction": fr, "body_mass": mass, "body_ipos": ipos,
               "dof_frictionloss": fl}

        if n_boxes:
            # Each box is raised to a random top height, or left buried. Density
            # is per environment: some environments are flat ground on purpose,
            # because the robot has to walk on a floor too.
            dens = jax.random.uniform(k_d, (), minval=d_lo, maxval=d_hi)
            up = jax.random.uniform(k_h, (n_boxes,), minval=0.0, maxval=1.0) < dens
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
    here and is run by hand — `python -m env.randomize` from rl/ — whenever this
    file or model.build_spec() moves. It exists for one reason: `mu_load` shipped
    unrandomised for weeks and the only symptom was a shape.

    `dof_frictionloss` is the field most worth probing now, because it carries
    the Coulomb floor for the whole training path (model.py docstring 2), and a
    zero there is not a narrower spread — it is no friction at all.
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
    sys_v, in_axes = domain_randomize(env.sys, keys)

    fl = np.asarray(sys_v.dof_frictionloss)
    vadr = np.asarray(env._vadr)
    check("dof_frictionloss is batched", fl.shape == (n, env.sys.nv),
          f"{fl.shape} vs ({n}, {env.sys.nv})")
    check("brax is told it is batched", in_axes.dof_frictionloss == 0)

    nominal = float(model_mod.actuator.load(quiet=True).tau_c)
    lo, hi = model_mod.domain_ranges()["actuator"]["tau_c"]["range"]
    joints = fl[:, vadr]
    check("the floor is non-zero on every joint of every environment",
          bool(joints.min() > 0.0), f"min {joints.min():.4f} N*m")
    check("it spans the tau_c range and nothing wider",
          bool(joints.min() >= lo * nominal - 1e-9
               and joints.max() <= hi * nominal + 1e-9),
          f"{joints.min():.4f}..{joints.max():.4f} of "
          f"{lo * nominal:.4f}..{hi * nominal:.4f}")
    check("environments differ", bool(joints.std() > 0.0),
          f"sd {joints.std():.4f} N*m")
    free = np.delete(fl, vadr, axis=1)
    check("the free joint is left alone", bool(np.all(free == 0.0)),
          f"max {free.max():.4g}")

    mass = np.asarray(sys_v.body_mass)
    check("body_mass is still batched", mass.shape == (n, env.sys.nbody))
    fr = np.asarray(sys_v.geom_friction)
    check("geom_friction is still batched", fr.shape[0] == n)

    print("  PASS" if ok else "  FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
