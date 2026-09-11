# CLAUDE.md — rl/

The RL walking policy: trained in MuJoCo/MJX, exported to ONNX, run on the robot
at 50 Hz to replace the analytic IK and the hand-written trot in
`ros2/smalldog_walker`. `README.md` here is the human-facing plan and the step
numbering; read it before writing code.

**Read `../3d/CLAUDE.md` too.** Its rules about what is source and what is output
apply across the whole repo, and this tree sits downstream of all of them.

## Environment

```bash
uv sync --extra cpu      # mac: CPU JAX, seconds-long smoke tests
uv sync --extra cuda     # WSL2 + RTX 3070: training
uv sync --extra fit      # scipy + matplotlib, for the bench fit
uv run python checks/check_model.py
```

Scripts run in place (`package = false`) — there is no wheel to install. Run
everything from `rl/`.

On the WSL2 box, `../WSL.md` is the runbook for that machine: the venv, the card,
USB passthrough, and which of the other trees' checks can be run there at all.

`check_model.py` and `imu_placement.py` need only `mujoco` and `numpy`, which are
base dependencies, so they run on any of the three machines including the robot.

## The machine split

The CAD lives on the mac; training happens on the WSL2 box; the robot is a third
machine. **The git repository is the only thing that crosses between them** —
there is no shared filesystem, no rsync, no scp of a checkpoint. If a result has
to reach another machine, it gets committed. Corollary: `uv.lock` is versioned
deliberately, because a pinned resolution is the whole point when three machines
share one tree.

Do not add CadQuery to this tree. Nothing here needs to build geometry; it reads
the already-generated description.

## Invariants — do not break these

- **`ros2/smalldog_description` is read-only from here.** MJCF, meshes and
  `robot_params.json` are output of `3d/mini_dog.py` via
  `smalldog_description/scripts/generate_model.py`. A mass, an inertia, a joint
  limit or an axis that is wrong in simulation is wrong in the CAD — fix it there
  and regenerate, following `../3d/CLAUDE.md` step 6. Never patch an XML here to
  make training behave.
- **Training-only model changes go through `mujoco.MjSpec` at load time**, in
  `model.py` — a fitted actuator, backlash, per-environment randomisation, a
  procedural terrain. `checks/imu_placement.py` shows the mechanism. Nothing
  hand-tuned may land in a committed XML, because the committed XMLs are not ours.
- **`actuator.py` is the one copy of the ST3215 law.** `robot/bench/fit_bam.py`
  fits its parameters into `params/st3215.json`; `model.py` installs them into
  MuJoCo. Neither side keeps a second copy of the equations. After editing it,
  run `python -c "import actuator; actuator._selftest()"`.
- **`params/st3215.json` is a FIT** — `fit_bam.py`, 44 runs at three voltages,
  2026-09-10 — so `actuator.load()` no longer warns. That is not the same as
  solved, and the two things it does not cover are both in its `source` string:
  the analytic passes read only freeswing/hold/holdbi, so ten of the trajectories
  reach the fit only under `--refine`; and its derived free speed is 5.90 rad/s
  against the 3.86 the hardware does, because the real servo stops at a firmware
  plateau the law does not carry (PLAN.md 3c). Read `p.source`, and do not narrow
  `params/domain_rand.json` on the strength of a fit of ONE servo — those ranges
  are the spread across twelve.
- **Units cross a boundary here.** `3d/` is millimetres everywhere; MJCF and
  URDF are SI (metres, kg, radians). `robot_params.json` names the unit in every
  key — `hip_xyz_mm`, `l_thigh_mm`, `joint_limits_rad`, `stance_base_height_m`,
  `total_mass_kg`. Read the suffix; do not assume.
- **The three joint-limit ladders mean three different things** —
  `joint_limits_rad` (the ROM scan's hard limits), `joint_soft_limits_rad` and
  `joint_velocity_limit`. `check_model.py` reports all three because conflating
  them is the easy mistake. An action space clipped to the hard limits is not the
  same policy as one clipped to the soft ones.
- **A randomisation axis is not alive because it is drawn.** Three of them were
  drawn, correctly shaped, and dead: `J_m` went into an `actuator.Params` field
  no function on the training path reads (the inertia is `dof_armature`), the bus
  delay was drawn in seconds and truncated to whole ticks so all 2000 draws came
  back 0, and the whole per-episode draw was per-RUN because brax calls
  `env.reset` once. `checks/check_model.py` asks the second question now — does
  the law's output MOVE across this range — and `model.EPISODE_DRAW` is the one
  declaration both the check and the env walk. Add an axis by adding a row there,
  and expect the check to tell you if nothing consumes it.
- **Run `checks/check_model.py` before training against a changed model.** It is
  not a second copy of `export_sim.py --check`, which asks whether the model is
  still the robot. This one asks whether the model is fit to train against: the
  actuator's dominant inertia is a guess, the contact parameters the solver
  applies are not the ones the model asked for, and the limit ladders disagree.
  A non-zero exit is a failure, not a warning.
- **MJX takes the heightfield; it does not take the logs.** Measured against the
  locked resolution, not assumed: the robot is MJX-clean by construction (all
  meshes `contype="0" conaffinity="0"`, collision set 4 capsules + 4 spheres + 11
  boxes), `scene.xml` and `type="hfield"` both go to the GPU, and the single
  unsupported thing is `(mjGEOM_CYLINDER, mjGEOM_BOX)` — the two obstacle-course
  logs against the robot's collision boxes. So the heightfield is a legitimate
  training surface, the course is `eval.py`-only, and the default stays flat plus
  procedural boxes because only those randomise per environment. `mjx.put_model`
  is the arbiter if the lock moves; never resolve a raise by editing the scene.
  See README, "What MJX can collide".
- **8 GB of VRAM sets `num_envs`, and JAX preallocates 75% of it by default.**
  Budget 1024–2048 environments and set `XLA_PYTHON_CLIENT_MEM_FRACTION`. An OOM
  here surfaces as an XLA allocation error, not as anything about the model.
- `runs/` is checkpoints and rollout video — gitignored, and it stays that way.
  A result worth keeping is a number in a commit message or a file in `params/`,
  not a 2 GB directory.

## Verifying a change

1. `python -c "import actuator; actuator._selftest()"` if `actuator.py` moved.
2. `python checks/check_model.py` and `--terrain`, both exit 0.
3. On the mac, the CPU extra runs the same code — use it for a smoke test before
   pushing something the 3070 will spend an hour on.
4. `eval.py` is the honest number: deterministic rollouts and a sim-to-sim pass in
   vanilla MuJoCo, not the MJX training environment reporting on itself.

## Re-baselines

Every entry here is a change to the ENVIRONMENT, which means checkpoints from
before it cannot be compared with ones from after it and the next run is a
retrain rather than a fine-tune. `eval.py` numbers move with them.

- **2026-09-11 — the joint limits widened to the CAD ROM scan.** `ros2/.../
  generate_model.py` carried a hand-typed `J_LIM` of roll 0.90 / pitch 1.30 /
  knee 1.85 rad with a comment claiming it came from the CAD scan; it did not.
  `3d/out/bom.json` reads ±90° / ±90° / ±110°, so the model this tree loads now
  ships roll/pitch **1.5708** and knee **1.9199**, with
  `joint_soft_limits_rad` at 1.4508 / 1.4508 / 1.7999 (was 0.78 / 1.18 / 1.73).
  `model.py`'s action range and the safety clip both scale off those, so the
  policy's action space is a different space — **retrain, do not fine-tune**.
  `checks/check_model.py` stays 0 FAIL and its limit ladder now reads soft <
  MuJoCo stop <= CAD ROM with the real scan in the CAD ROM column. Nothing else
  moved: same 2.493 kg, same `MJ_*`, same meshes.

- **2026-09-11 — the episode boundary, and four dead randomisation axes.** No
  reward weight, no observation and no CAD moved; what moved is that an episode
  is now an episode. `Walk.step` redraws the command, the servo/pack/bus draw,
  the observation history and the foot positions on
  `info["episode_done"]`, which brax's `EpisodeWrapper` writes and which includes
  truncation. Before it, all of those were drawn once at step zero and held for
  the whole run: ~15 % of environments stood still for 60 M steps, the "per
  episode" draw was per environment, each new episode began with four frames of
  the previous fallen robot in its observation, and `foot_slip` was charged an
  85x spike on the first step of every episode (measured, CPU, four envs). Also
  in the same pass: the box-height curriculum reached 12.1 mm of its declared
  22 mm because one key drew both the raise coin and the height; the bus delay
  was 0 ticks on every draw and now runs one tick late on ~32 % of episodes from
  the measured `params/bus_timing.json`; `J_m` moved from the inert per-episode
  draw to `dof_armature`; the push magnitude and the gap to the next push shared
  a key, so the hardest shove was always followed by the longest wait; and the
  torque ceiling handed to MuJoCo went 5.0 -> 7.5 N*m, derived, because 13.2 % of
  `(k_u, u_bat)` draws asked for more than 5.0 and were silently clipped.

## Notes

- On macOS the passive viewer (`--view`) needs `mjpython`, exactly as
  `ros2/tools/view.sh` documents. `--shot` needs a GL backend — `MUJOCO_GL=egl`
  with a GPU, `osmesa` for software — and says so rather than dying.
- MuJoCo caches a heightfield by file name within a process, so a seed sweep must
  write one file per seed. Those files are scratch: do not commit them.
