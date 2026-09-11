# CLAUDE.md — rl/

The RL walking policy: trained in MuJoCo/MJX, exported to ONNX, run on the robot at 50 Hz
to replace the analytic IK and the hand-written trot in `ros2/smalldog_walker`.
`README.md` here is the human-facing plan; read it before writing code. **Read
`../3d/CLAUDE.md` too** — its rules about source and output apply across the repo, and this
tree sits downstream of all of them.

## Environment

```bash
uv sync --extra cpu      # mac: CPU JAX, seconds-long smoke tests
uv sync --extra cuda     # WSL2 + RTX 3070: training (../WSL.md is that machine's runbook)
uv sync --extra fit      # scipy + matplotlib, for the bench fit
uv run python checks/check_model.py
```

Scripts run in place (`package = false`); run everything from `rl/`. `check_model.py` and
`imu_placement.py` need only `mujoco` and `numpy`, so they run on any machine including the
robot. Do not add CadQuery here — this tree reads the generated description. `uv.lock` is
versioned deliberately: a pinned resolution is the point when three machines share one tree.

## Invariants

- **`ros2/smalldog_description` is read-only from here.** MJCF, meshes and
  `robot_params.json` are output of `3d/mini_dog.py`. A mass, an inertia, a limit or an
  axis that is wrong in simulation is wrong in the CAD — fix it there, regenerate
  (`../3d/CLAUDE.md` step 6). Never patch an XML here to make training behave.
- **Training-only model changes go through `mujoco.MjSpec` at load time**, in `model.py` —
  the fitted actuator, backlash, per-environment randomisation, procedural terrain.
  `checks/imu_placement.py` shows the mechanism.
- **`actuator.py` is the one copy of the ST3215 law.** `robot/bench/fit_bam.py` fits its
  parameters into `params/st3215.json`; `model.py` installs them. After editing it, run
  `python -c "import actuator; actuator._selftest()"`.
- **`params/st3215.json` is a fit, not a solved servo.** Read `p.source`: the analytic
  passes read only freeswing/hold/holdbi, and its derived free speed is 5.90 rad/s against
  the 3.86 the hardware does, because the real servo stops at a firmware plateau the law
  does not carry (PLAN.md 3c). Do not narrow `params/domain_rand.json` on the strength of
  a fit of ONE servo — those ranges are the spread across twelve.
- **Units cross a boundary here.** `3d/` is mm; MJCF/URDF are SI. `robot_params.json` names
  the unit in every key (`hip_xyz_mm`, `joint_limits_rad`, `total_mass_kg`). Read the suffix.
- **The three joint-limit ladders mean three different things**: `joint_limits_rad` (the
  ROM scan's hard limits), `joint_soft_limits_rad` (what the action space and the safety
  clip scale off), `joint_velocity_limit` (the no-load speed; the achievable ceiling is
  `joint_rate_ceiling_rad_s`). `check_model.py` reports all of them.
- **A randomisation axis is not alive because it is drawn.** Three were drawn, correctly
  shaped, and dead: `J_m` went into a `Params` field nothing on the training path reads,
  the bus delay was truncated to whole ticks so every draw was 0, and the per-episode draw
  was per-run because brax calls `env.reset` once. `model.EPISODE_DRAW` is the one
  declaration both the check and the env walk; `check_model.py` asks whether the law's
  output MOVES across each range. Add an axis by adding a row there.
- **Randomise the field the PHYSICS reads, and that is sometimes a MuJoCo field.** `J_m` is
  `dof_armature`; `tau_c` is `dof_frictionloss`, because the law's Coulomb term
  `(tau_c + mu_load·|tau_t|)·tanh(w/v_eps)` is zero at rest and cannot hold a standing
  joint, so `model.build_spec()` installs the fitted `tau_c` there and every MuJoCo caller
  passes `tau_c_external=True` so the floor is applied once. Model fields can only be
  randomised through brax's `randomization_fn`, so both are drawn **per environment** in
  `env/randomize.py`, not per episode — less variety per wall-clock, but rotor inertia and
  grease do not change between episodes anyway. `check_model.py` asserts both are absent
  from the episode draw and at their nominal in the compiled model; `python -m
  env.randomize` asserts the draw on top comes back batched (needs jax, so it cannot live
  in `check_model.py`, which has to run on the robot).
- **Run `checks/check_model.py` before training against a changed model.** It is not
  `export_sim.py --check` (is the model still the robot?); it asks whether the model is fit
  to train against. Non-zero exit is a failure.
- **MJX takes the heightfield; it does not take the logs.** The robot is MJX-clean by
  construction (meshes `contype="0" conaffinity="0"`; collision set 4 capsules + 4 spheres +
  11 boxes); the one unsupported pair is `(mjGEOM_CYLINDER, mjGEOM_BOX)` — the course's two
  logs. So the heightfield is a legitimate training surface, the course is `eval.py`-only,
  and the default stays flat plus procedural boxes because only those randomise per
  environment. `mjx.put_model` is the arbiter; never resolve a raise by editing the scene.
- **8 GB of VRAM sets `num_envs`** (budget 1024–2048) and JAX preallocates 75 % by default:
  set `XLA_PYTHON_CLIENT_MEM_FRACTION`. An OOM surfaces as an XLA allocation error.
- `runs/` is checkpoints and video — gitignored. A result worth keeping is a number in a
  commit message or a file in `params/`.

## Verifying a change

1. `python -c "import actuator; actuator._selftest()"` if `actuator.py` moved.
2. `python checks/check_model.py` and `--terrain`, both exit 0.
3. On the mac, the CPU extra runs the same code — smoke-test before the 3070 spends an hour.
4. `eval.py` is the honest number: deterministic rollouts and a sim-to-sim pass in vanilla
   MuJoCo, not the MJX environment reporting on itself.

## Re-baselines

Each of these changed the ENVIRONMENT: a checkpoint from before cannot be compared with one
from after, and the next run is a **retrain, not a fine-tune**. All landed 2026-09-09…11;
no policy trained before them is worth keeping.

- **The IMU site moved** (7.6 mm up, `3d/`): a different observation.
- **The fitted actuator and the measured stall torque** went into the model; the torque
  ceiling handed to MuJoCo is 7.5 N·m, derived (5.0 was silently clipping 13 % of
  `(k_u, u_bat)` draws).
- **Joint limits widened to the CAD ROM scan** (roll/pitch 1.5708, knee 1.9199; soft
  1.4508 / 1.4508 / 1.7999, from 0.78 / 1.18 / 1.73): a different action space.
- **An episode is now an episode**: `Walk.step` redraws the command, the servo/pack/bus
  draw, the observation history and the foot positions on `info["episode_done"]`. Before,
  ~15 % of environments stood still for the whole run, each episode began with four frames
  of the previous fallen robot, and `foot_slip` was charged an 85× spike on step one. Also
  fixed: the box-height curriculum reached 12 of its declared 22 mm (one key drew both
  coin and height), the bus delay was 0 on every draw (now one tick late on ~32 %), and
  push magnitude and gap shared a key.
- **Friction at rest** (`frictionloss` = `tau_c`, above): torque off from the stance the base
  fell 107.7 mm before and 65.6 after; residual joint speed holding the stance 0.0033 →
  0.0009 rad/s.

## Notes

- On macOS the passive viewer (`--view`) needs `mjpython`, as `ros2/tools/view.sh`
  documents. `--shot` needs a GL backend and says so rather than dying.
- MuJoCo caches a heightfield by file name within a process: a seed sweep writes one file
  per seed. Those files are scratch; do not commit them.
