# PLAN.md — what to build next, and why

Three things are missing — **a battery, an IMU, and the Orange Pi is not installed** — and
they gate different work. Steps 4–6 need nothing that is not already on the desk; 7–10
wait on hardware. Steps 1–3 are done and their results live in the trees, not here.

## Where things stand

| tree | state |
|---|---|
| `3d/` | done and verified; the ladder in `3d/CLAUDE.md` is green on the current tree |
| `ros2/` | generated from the CAD, trots in sim, terrain and course regressions baselined |
| `rl/` | code complete, checks green, **never trained in earnest** |
| `robot/` | walks tethered on the bench at 2.55 kg (1 kg plate as ballast); every servo number below is measured on one unit |

## Done — where the result lives

| step | result | write-up |
|---|---|---|
| 1 servo identification | friction 0.19 N·m + 0.28 per N·m carried; stiffness 40.9 N·m/rad at 12 V; total speed-proportional 1.37 N·m·s/rad (`b_v` is not separable from back-EMF on this bench — use the total) | `robot/README.md`, "Two things the fit cannot find" |
| 2 `mu_load` in `actuator.py` | `tau_c` 0.184, `J_m` 0.0165, `mu_load` 0.286 | `robot/README.md`, "The load-dependent friction term" |
| 2b static friction | `simulate()` is Karnopp; the training path gets stick from MuJoCo `frictionloss` = `tau_c`; the refit that followed was **not** adopted (mixed result) | `rl/CLAUDE.md`, "Re-baselines" |
| 2c stall torque | **4.50 N·m at 12 V** on a scale; both the register route (7.4) and the datasheet (2.94) were wrong | `robot/README.md`, "The stall torque" |
| 3 fitted constants into the CAD | `MJ_KP` 40.9, `MJ_FRICTIONLOSS` 0.184, `MJ_DAMPING` 1.37, `MJ_ARMATURE` 0.0165 | `3d/mini_dog.py` section 4; `3d/CLAUDE.md` baseline |
| 3b gait rate limit | limits against the achievable 3.15 rad/s ceiling, not the no-load speed | `ros2/README.md`, "Gait" |
| 3c no-load speed | **3.86 rad/s**, a firmware profile cap: open loop the motor does 5.0 (`--pwm`, reading A); 4.50 N·m stands | `robot/README.md`, "The no-load speed" |

## What each missing part gates

**The battery is not just runtime.** The sim trot is hypersensitive to mass and to where it
sits (`3d/CLAUDE.md`), so every gait number tuned with a plate is tuned against a stand-in.
And supply voltage changes the servo: stiffness is 3.44 N·m/rad per volt, so a 3S pack
(12.6 → 9.9 V) costs 21 % of it over a discharge. That is a gait variable and belongs in
domain randomisation.

**The IMU gates the feedback half of everything.** The walker levels the body and holds
heading on it; in sim that took the obstacle course from 1 obstacle cleared to 5, and none
of it has run on hardware. `rl/`'s observation includes the IMU, so no policy is deployable
until one exists. The mount is decided (`IMU_*` in `mini_dog.py`, checked by
`rl/checks/imu_placement.py`), just not populated.

**The Pi can wait.** The 50 Hz loop already closes from the mac over CDC; what the Pi buys
— ROS 2 on the robot, the LiDAR, the camera, the ONNX policy — is all untethered concerns.

---

## 3c, closed: the plateau is the position loop's profile (A)

`noload_speed.py --pwm` ran on the second unit: open loop the speed is linear in duty
through 1000 (5.0 rad/s at 12.1 V, k_e 2.39), while position mode on the same unit stops
at 3.88 with the register flat at 2500 counts/s. So 4.50 N·m stands, and what
`rl/actuator.py` still lacks is a rate cap on the *goal* — a state per joint, not a duty
cap. What `--pwm` does not prove is that the loop applies full duty at zero speed: the torque
rig at `TORQUE_LIMIT` 800 and 1000 is that check, next rig session.

## 4. Randomise over the pack

**Done:** `mu_load` is drawn per joint per episode over the same `[0.40, 2.20]` as `tau_c`;
`check_model.py` asserts every per-unit servo parameter comes back batched. Voltage is
measured rather than guessed.

**Done:** four units through `holdbi` and `speed` at 12 V (`robot/README.md`, "Unit to
unit"). Electrical side within 5 %; friction floor 0.64–1.01× and load slope 0.88–1.33× of
the nominal. `tau_c`, `mu_load` and `kp` in `rl/params/domain_rand.json` are `measured`
ranges now — 2.2× narrower than the guess on tau_c. A policy trained before this saw a
wider world than exists; nothing needs retraining for it.

## 5. Settle the traction question before touching a sole

Two bench runs, no code. The sim has already been asked and cannot answer: four alternative
soles were built and measured (`ros2/README.md`, "The foot's contact patch") and none
bought anything, because MuJoCo contact is rigid and Coulomb friction ignores area. On
hardware the foot is a true R = 13 hemisphere with a 2.5 mm Hertz patch at 1.3 MPa, and the
sim ships μ = 1.2 where TPU on a bench is 0.3–0.5. Whether the robot still *slips* after
the gait fit (`robot/README.md`) is an inference, not a measurement.

1. **Dust the runway** (talc or flour over the ~40 cm). A planted foot leaves a dot, a
   sliding one a streak; streak length is slip per step. Read it per foot — the plate sits
   forward, so front duty is 0.68 against the rear's 0.40.
2. **A/B the surface**: same gait, same start mark, same seconds, bare bench against a
   rubber mat. Distance grows → the deficit is slip and sole work pays. Unchanged → traction
   is not where the speed goes. Shorten the run rather than let one arm hit the end.
3. **Only if 1–2 say yes:** cross-leg consistency from the encoders — two planted diagonal
   feet each imply a body velocity through FK, and their disagreement is slip. Use measured
   positions, not commanded (tracking error peaks at 35°).

What each answer costs: the friction coefficient is `MJ_FOOT_FRICTION` in `mini_dog.py`
section 4 plus the ladder — cheap. Geometry is a full ladder and a BOM line, and two things
to know first: the shin is 34.6° off vertical under load (46.7° worst), so a flat normal to
the shin lands on its edge — rake it or use a large-radius crown; and the terrain
heightfield cannot compare feet at `CELL_MM` = 12 (a ⌀17.7 face is smaller than one cell),
so `--rough` is the arm that can. Choose a shape with coupons on a tilt plate, not with feet.

## 6. Train

Everything that blocked it is resolved: the two sims agree on servo strength (4.50), the
training path has friction at rest, `check_model.py` is 0 FAIL. The WSL2 box does this
alone. Do not expect to deploy — that needs the IMU. Step 3b's gait re-tune is for the
hand-tuned regression harness, not a blocker: RL learns its own gait.

---

## 7. When the IMU arrives

Mount at the `IMU_*` site, solder to the pads (headerless — `3d/ref/imu/README.md`), re-run
`rl/checks/imu_placement.py` against the real mount, then close the walker's loop on
hardware for the first time. Cheaper than the battery, and it unblocks both the walker's
levelling/heading hold and the deployability of anything `rl/` produces.

## 8–10. When the battery arrives

8. **Replace the ballast with the pack** and re-measure every gait figure once, with a
   control run beside it — `3d/CLAUDE.md`'s mass-cliff warning applies as written.
9. **Install the Orange Pi** and move the 50 Hz loop onto it. Re-run `bus_probe.py` there
   first: the transaction cost is a property of the machine.
10. **LiDAR and camera** — sensors for autonomy, not walking, and both need the Pi. The
    model side is done: measured point rate, cone read out of the compiled model by both
    consumers.

---

**Throughout:** every model change goes through the ladder in `3d/CLAUDE.md` in one pass,
and every bench number goes into the repo *and* `ST3215_STS3215_measured_parameters.md`.
