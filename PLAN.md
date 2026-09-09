# PLAN.md — what to build next, and why

Written 2026-09-08, the day the bench produced its first real actuator numbers.

Three things are missing — **a battery, an IMU, and the Orange Pi is not installed** — and
they gate different work. This plan is ordered so that the missing hardware blocks as
little as possible: steps 1–6 need nothing that is not already on the desk, and they are
the ones that make the rest worth doing.

`rl/README.md` numbers the RL track 1–7 and that numbering is untouched here. These are
project steps, not RL steps.

## Where things actually stand

| tree | state |
|---|---|
| `3d/` | **done and verified.** The ladder in `3d/CLAUDE.md` is green on the current tree; `MAC.md` has the record |
| `ros2/` | generated from the CAD, trots in sim, terrain and course regressions baselined |
| `rl/` | code complete, **never trained in earnest** — `params/st3215.json` is still vendor priors and `train_ppo.py` says so on every run |
| `robot/` | the robot walks, tethered, on the bench at 2.55 kg with a 1 kg plate for ballast |
| `robot/bench` | three-voltage identification set captured, **plus the two differential ladders of step 1 (2026-09-09)**; friction is now measured rather than fitted, and the electrical half is one model term short — step 2 |

## What each missing part gates

**The battery is not just runtime.** Two things hang off it that are easy to miss:

- **Mass, and where it sits.** The design mass is 2.499 kg; the robot is 1.55 kg and gets
  to 2.55 with a cast-iron plate. `3d/CLAUDE.md` measures the flat trot as hypersensitive
  to mass *and* to where the mass sits — 11 g moved the sim trot 778 → 597 mm, and the same
  11 g at the nose gave 547. A plate resting on the deck is not a pack bolted into the
  cradle, so every gait number tuned today is tuned against a stand-in.
- **Voltage, which we now know changes the servo.** The bench measured position-loop
  stiffness at **28.8 N·m/rad at 12.1 V and 24.5 at 10.1 V** — linear in supply. A 3S pack
  runs 12.6 → 9.9 V, so the robot gets ~25 % softer as it discharges. That is a gait
  variable, not a housekeeping detail, and it belongs in domain randomisation.

**The IMU gates the feedback half of everything.** The walker levels the body and holds
heading on it; in sim that feedback took the obstacle course from 1 obstacle cleared to 5.
None of it has ever run on hardware. And `rl/`'s observation vector includes the IMU — so a
policy trained now cannot be deployed at all until one exists. `rl/checks/imu_placement.py`
has already done the analysis and it is not marginal: ω × (ω × r) + α × r reaches **9.0 m/s²
— 42° of apparent tilt** — for a board out by the Pi, against 25° on the centreline. The
mounting point is a model constant (`IMU_*` in `mini_dog.py` section 3) and the slot is
3.6 mm with the board 2.8 of it, so this is decided, just not populated.

**The Pi can wait, and "no battery, so no need" is the right call.** A tethered Pi adds
nothing over a tethered mac: the 50 Hz loop already closes from the mac over CDC and
`bus_probe` measured the transaction cost there. What the Pi buys is ROS 2 on the robot,
the LiDAR, the camera, and running the ONNX policy — all of which are untethered concerns.
Installing it earlier costs mass in the wrong place and debugging surface for no gain.

---

# Steps 1–6 — nothing missing, do these now

## 1. Finish the servo identification — **DONE 2026-09-09**

Both runs are in `bench/data/` at 8 / 10 / 12 V, 1.066 kg on the 90 mm arm:
`sweep.py --traj holdbi` (the hold ladder up and then down, so every angle is reached
from both sides) and `--traj speed` (the same triangle at 20 / 10 / 5 / 2 / 1 s, i.e.
0.1 / 0.2 / 0.4 / 1.0 / 2.0 rad/s). `bench/hysteresis.py` reads them; the full write-up
is `robot/README.md`, "Two things the fit cannot find", and the numbers are also in
`ST3215_STS3215_measured_parameters.md`.

**The bidirectional ladder delivered what it promised, and more than was asked.**
Friction at a hold is **0.19 N·m + 0.28 per N·m of load carried**, voltage-invariant to
6 % and 16 % across a 1.5× supply range — the load-dependent term, measured directly for
the first time rather than inferred from a slope ratio. It also caught a 40 % error
nobody was looking for: the **position-loop stiffness is 40.9 N·m/rad at 12 V, not 28.8**.
A one-directional ladder bills the friction band to elasticity, so the servo is both
stiffer and rougher than the old reading said, and the two errors hide each other.

**The speed ladder worked and its premise did not.** It supplies steady speeds, five of
them, and pins the total speed-proportional torque to 1.37 N·m·s/rad within 5 %. But that
total is `b_v + k_t·k_e`, back-EMF alone accounts for 1.31 of it, and the remainder is
0.06 ± 0.07 — consistent with zero. **This bench cannot split them and no further runs
will.** Back-EMF and viscous friction each cost a motor voltage proportional to ω and
neither depends on supply, so the voltage sweep that separates every other electrical
term does nothing here; splitting them needs the motor current, which `PRESENT_CURRENT`
gives only as `d²U/R` at a 6.5 mA LSB. The paragraph below that says "only steady speeds
can" supply `b_v` was half right: steady speeds give the sum, and the sum is what a
simulator needs anyway. **Use the total and stop trying to split it.**

Two things this turned up that were not on anyone's list:

- **A run no pass consumes changes no parameter.** The new data was captured, `fit_bam.py`
  re-run, and every fitted parameter came back identical to five decimals to a fit on the
  old data alone — the analytic passes select on `freeswing` and `hold`, so `holdbi` and
  `speed` were read only under `--refine`, and not at all without it. `check_identifiable`
  now names every trajectory in the directory that no analytic pass reads. **Do not read
  an unchanged fit as agreement.**
- **The servo runs out of authority at ~1.8 rad/s without the duty pinning** — 0.74 rad of
  position error at a commanded 2.0 rad/s, at all three voltages. So it is not the motor's
  ceiling. `D_COEF` is 32 and `actuator.py` models `kd = 0`, which would do exactly this,
  but so would an internal output clamp; this data does not separate them. Worth one small
  trajectory when someone is next at the bench, and worth knowing now because 1.8 rad/s is
  inside where the 50 Hz policy commands.

## 2. Give `actuator.py` a load-dependent friction term — **DONE 2026-09-09**

`Params.mu_load`, N·m of friction per N·m crossing the gearbox, applied in
`actuator.friction()` and in `simulate()`'s integrator; `fit_bam.seed_from_holdbi`
measures it from the bidirectional ladder and it is now the FIRST analytic pass,
because the free swing subtracts `tau_c` and had been using half the real value.

**What it moved.** Two numbers, both in the direction the ladder predicted, and
both because `tau_c` was wrong rather than because the new term is doing work:

| | before | after |
|---|---|---|
| `tau_c` | 0.084 | **0.184** |
| `J_m` | 0.0240 | **0.0165** |
| `mu_load` | — | **0.286** |
| position RMS | 8.96° | 8.84° |
| current RMS | 1.25 A | 1.01 A |

**What it did NOT move, and this refutes a claim in the step-1 write-up.** `--refine`
still diverges, and identically: `kp` pinned at its bound (1887 against 2017 before),
`k_e` 11.3 against 11.8, an implied no-load speed of 1.06 rad/s against 1.02. Missing
friction was not the cause. Nor did `k_u` move — the fitted stall is still 4.23 N·m — see
2c, where cancelling friction properly makes it *worse*. A fit railing `kp` while `k_e`
grows to compensate is reaching for torque the electrical model cannot supply, which is
the same shape as 2c and probably the same cause.

`tau_c` and `J_m` are not independent here: the free swing gets `J_m` from
`(m·g·r·sinq − tau_c)/acc`, so doubling `tau_c` nearly halves the driving torque
and `J_m` with it. **`J_m` = 0.0165 is the number step 3 should carry**, not the
0.024–0.042 this plan quoted before — that range was computed against a `tau_c`
of 0.08. It is still 2× `MJ_ARMATURE`'s 0.008, so step 3's argument stands, but
the size of the correction has changed and step 3 should not quote the old range.

The `tau_c` result is worth more than the number: the bidirectional ladder and
the free-swing fit are two independent routes that had disagreed by 2×, and they
now agree at **0.184 vs 0.186**. That is the cross-check this parameter never had.

### 2b. The model has no static friction, and that is the next real defect

`actuator._sign()` is `tanh(w/v_eps)`, which is exactly **zero at rest**. So the
simulated servo has no stiction at all: a hold approached from below and from
above settles at the same duty (measured on the model: half-difference 0.00 and
0.03 V, against the real servo's 0.29). The consequences are concrete:

- **`mu_load` is measurable on hardware but not recoverable from the model's own
  output.** `fit_bam --selftest` now says so and says why — it is a statement
  about the model, not about the experiment, which is the opposite of every other
  entry in that list. The term is still live wherever the joint MOVES, which is
  where it takes load off the fit; it is only the rest case that is missing.
- `ST3215_STS3215_measured_parameters.md` already records that "small commanded
  corrections below the breakaway do not move the joint at all" and puts stiction
  at 0.23–0.35 N·m. The model cannot produce that behaviour today.

The fix is a friction that can hold at ω = 0 — the standard form is Karnopp's:
below a velocity threshold, friction opposes the net applied torque up to a
ceiling, rather than being proportional to a smoothed sign. **It is not a
one-liner and it should not be bolted on without care**: `simulate()` has every
torque to hand and can do it directly, but `rl/env/walk.py` and `rl/eval.py` call
`motor_torque()` with MuJoCo owning the load, so there the honest route is
probably MuJoCo's own `frictionloss` — which `rl/model.py` currently sets to zero
on the grounds that `actuator.py` supplies friction. Decide that before writing
code, and keep the two consumers on one law.

### 2c. The torque constant is 2.2× the datasheet, and it is NOT friction

This plan assumed the inflated `k_u` (fitted stall 4.23 N·m against a spec 2.94)
was missing friction being absorbed. **It is not.** Cancelling friction properly
makes it worse, not better — the friction-cancelled paired holds give an implied
stall of **7.4 N·m**. Two channels that share only the gravity anchor agree:

| route | torque constant | implied stall @ 12 V |
|---|---|---|
| friction-cancelled paired holds, current channel | 2.394 N·m/A | 7.40 |
| duty channel × R (`hysteresis.py`) | 2.382 N·m/A | 7.37 |
| unidirectional holds (the old route) | 1.368 N·m/A | 4.23 |
| vendor spec | 1.089 N·m/A | 2.94 |

The two measurements agree to 0.3 % with each other and disagree with the vendor
by 2.2×. The servo cannot produce 7.4 N·m, so something is mis-scaled, and the
prime suspect is already on the "not measured" list in
`ST3215_STS3215_measured_parameters.md`: **`PRESENT_CURRENT`'s 6.5 mA LSB is
confirmed only against vendor numbers, never against a shunt.** `PRESENT_LOAD`'s
per-mille scaling is the other candidate — if 1000 is not 100 % duty, every
`d²U/R` correction in `fit_bam.py` inherits it.

Two things follow. First, **`tau_c` and `mu_load` are immune** — both are ratios
of quantities in the same register units, converted through a gravity anchor that
is known exactly, so a constant scale error cancels out of both. Second, the way
to settle it is an **external shunt or an INA226 on the supply**, which costs a
part and an evening and would pin `R`, `k_u` and the efficiency at once. Until
then, do not read the fitted stall as a torque the robot has.

### 2c-i. QUEUED: the rig is designed and sliced, waiting on the printer

**Picked up 2026-09-09, parked on print time. Everything needed to resume cold is
here — nothing about this depends on remembering the conversation it came from.**

`3d/torque_rig.py` is the fixture, and `3d/out/gcode/torque_rig.3mf` is sliced for
the Qidi: two parts on one plate, 5 walls, 40 % infill, **9 h 01 m, 224 g**, no
support generated. Re-slice with

```bash
cd 3d && .venv/bin/python torque_rig.py        # -> out/torque/{step,stl} + the numbers
.venv/bin/python tools/slice_orca.py --machine "Qidi Q2 0.4 nozzle - Copy" \
    --process "0.20mm Standard @Qidi Q2 - Copy" --filament "QIDI НИТ petg черный" \
    --name torque_rig --walls 5 --infill 40% torque_frame torque_arm
```

**Hardware to have ready:** M6 × 40 and two M6 nuts (the adjustable anvil), 4 × M3 × 6
into the driven hub, 2 × M3 × 10 set screws for the thrust clamp, and the 2 kg kitchen
scale the rig is sized around (28 mm to its platform — `SCALE_H`; re-measure and re-run
`torque_rig.py` if it is a different scale).

**Do this before the first push, and nothing else first:** set `TORQUE_LIMIT` to **350**
of 1000 and read it back. It is the only thing between a 2 kg scale and a servo that may
turn out to be the 7.4 N·m one — at full duty two of the three candidates put 2.5–4.4 kg
through it. `torque_rig.py` prints the full protocol, the three failure modes and the
arithmetic behind the cap; run it and read what it says rather than working from memory.

**What comes back, and where it goes.** Fit a line through `(d·U, τ)` and through
`(i, τ)`. The first slope is `k_u` in N·m per volt with no register scaling in it, so
`k_u × 12` is the stall; the second is `k_t` in N·m/A, which pins `PRESENT_CURRENT`'s
6.5 mA LSB. Then:

- if the stall is not 2.94, `SERVO_STALL_NM` in `mini_dog.py` section 4 changes, and with
  it the ROS 2 model's `forcerange` — which is `3d/CLAUDE.md` steps 5 and 6 again;
- either way **the `rl/` vs `ros2/` servo-strength disagreement gets settled**, which is
  the second of the three things blocking training (see below);
- `ST3215_STS3215_measured_parameters.md` carries the ⚠ warning box that this is
  unresolved. Take it out, or replace it with the answer — it is the public claim.

## 3. Push the fitted numbers into the CAD and re-baseline — **DONE 2026-09-09**

All four constants in `mini_dog.py` section 4, then the whole `3d/CLAUDE.md` ladder in the
same pass. `MJ_KP` 25 → **40.9**, `MJ_FRICTIONLOSS` 0.02 → **0.184**, `MJ_DAMPING` 0.12 →
**1.37**, `MJ_ARMATURE` 0.008 → **0.0165**. The full record is in `MAC.md`; step 6's
baselines are re-stated in `3d/CLAUDE.md`.

Two of those are not the parameter this plan named. `MJ_KP` moved because the stiffness
was itself wrong by 40 % (step 1), and `MJ_FRICTIONLOSS` moved because MuJoCo's
`frictionloss` is a proper stick-slip constraint and is therefore **the only place in this
project where static friction can be modelled at all** — `rl/actuator.py` cannot, by step
2b. `MJ_DAMPING` is the *total* speed-proportional torque rather than `b_v`, because a
`position` actuator has no back-EMF to put the rest in.

**Nothing else moved, and that is checked rather than assumed.** `export_sim.py --check`
reads the same 2.488 kg, 187 mm stand height and camera axis; the ROS 2 regeneration moved
`mujoco/defaults.xml` and nothing else; `fea.py` was skipped because it reads no `MJ_*`,
the constants are absent from `out/bom.json`, and neither mass nor geometry changed.

**Every gait distance fell by a quarter to a third, and it is the fix working.**

| | control (old `MJ_*`) | after |
|---|---|---|
| flat trot | 556.6 mm | **487.0 mm** |
| terrain, seeds 7…12 | 520 ±67 mm | **340 ±39 mm** |
| course, seeds 7 / 8 / 9 | 1/7, 5/7, 4/7 | **2/7, 2/7, 0/7** — all upright |

Same mass to the gram, so this is not the mass cliff, and the terrain sweep moved 180 mm
against a 32 mm standard error — the first re-baseline in this project that is decisively
not one distribution. The number that settles the interpretation is the joint's speed
ceiling, `(forcerange − frictionloss)/damping`: **24.3 rad/s before against 2.01 after**,
where the servo's vendor no-load speed is 4.71 and the bench measured ~1.8 under the 1 kg
arm. The old model let every leg swing five times faster than the servo can turn with no
load at all, and the hand-tuned gait had settled into that headroom.

So the sim was flattering the robot in the one dimension a walker spends most, and the
distances that just fell were never real. **Do not put `MJ_DAMPING` back.**

### 3b. The gait needs re-tuning against the honest joint

Not started. `ros2/tools/standalone_sim.py`'s hand-tuned trot is now commanding swing
speeds the servo does not have, which is why the distances fell; re-tuning is what gets
them back, and this time they will mean something. Do this before reading any further
terrain or course number as evidence about geometry.

It also re-dates the `rl/` work: a policy trained against the old model learned to spend
joint speed that does not exist, so anything trained before today **retrains rather than
fine-tunes**. That is the same conclusion the IMU move reached earlier the same day for an
unrelated reason, and the two compound.

## 4. Randomise over the pack

`rl/params/domain_rand.json`, with the voltage range now measured rather than guessed.

**Randomise `mu_load` too, and this is now a measured gap rather than a suggestion.**
Verified on the WSL2 box, 2026-09-09, on the current tree: `model.sample_actuator_params`
has no entry for it, `domain_ranges()` has none, and `walk.py`'s `_params()` replaces nine
fields and leaves `mu_load` as the scalar off `_p0` — live, `_params().mu_load` has shape
`()` while `_params().tau_c` beside it is `(8, 12)` with a 0.33 spread. So training today
runs **one shared gearbox friction across every environment while the Coulomb term next to
it is randomised**, which is the wrong way round: this plan's own argument below is that
gearbox friction is *exactly* the parameter that varies unit to unit, and it is now the
larger of the two terms (0.29 N·m per N·m carried, against a 0.18 N·m floor). The jax
registration is already correct — `mu_load` flattens as data, and a `vmap` over a batched
`Params` was confirmed to give four distinct per-environment torques — so this is one
entry in `domain_rand.json` plus one line in `sample_actuator_params`, not plumbing.

**Why:** step 3 gives one servo at one voltage. The robot will run a pack that sags from
12.6 to 9.9 V, and we now know what that does across three measured points rather than
two — stiffness is linear in supply at **3.44 N·m/rad per volt** (28.1 / 34.4 / 40.9 at
8 / 10 / 12 V), so the pack's range costs 21 % of it, and torque per unit duty falls with
it. Randomise the friction too: it is the term that does *not* move with voltage, so it
stays put while everything around it shifts. Randomising over it is what makes a policy survive a discharge instead
of only working on a full pack. This is also the cheapest insurance against the one-sample
problem: every number in `ST3215_STS3215_measured_parameters.md` came from **one servo**,
and gearbox friction is exactly the parameter that varies unit to unit.

## 5. Settle the traction question before touching a sole

Two runs on the bench, no code, nothing that is not already on the desk. This one is out
of the actuator chain above — it does not wait on steps 1–4 and they do not wait on it.

**Why it is open at all:** the two trees disagree about the foot, and both of them are
right about their own robot.

- `ros2/README.md`, "The foot's contact patch": four alternative soles were built through
  `MjSpec` and measured against the unchanged control — a truncated flat `pad`, a `tripod`
  of three lobes, a sprung `ankle`, and the dome with its torsion switched on. **None of
  them bought anything**: flat ground all within ±21 mm of the 781 mm control, rough
  ground equal or worse against a ±94…±252 mm seed spread, and the passive ankle
  unambiguously negative at every spring rate. The reason is in the same section —
  MuJoCo's contact is rigid, Coulomb friction does not depend on area, and the compliance
  that actually gives is the servo's `kp = 25`, not the foot.
- `robot/README.md`, "The feet are the wrong shape for the floor": on hardware `foot()` is
  a true hemisphere, R = 13, no flat. At 6.25 N per foot Hertz gives a patch of **2.5 mm
  diameter, 4.8 mm², at 1.3 MPa**. Rubber friction carries an adhesion term that scales
  with *real* contact area, so this is not the textbook case where area drops out — a
  point grips worse than a pad of the same material. And the geom ships
  `friction="1.2 …"` where TPU on a bare bench is realistically 0.3–0.5, so **the sim
  assumes 2.5–4× the grip the robot has.**

So the sim cannot answer this, and it has already been asked. What is not measured is
whether the robot still slips *now*: the gait fit in `252e1c6` took it from 0.067 m/s
against 0.20 commanded to **0.12–0.15 against 0.14**, and `robot/README.md` reads landing
on the commanded speed as the slip having gone with the drag. That is an inference, not a
measurement, and it is the one worth spending an evening on before any geometry moves.

**The measurement, in order:**

1. **Dust the runway.** Talc, chalk or flour over the ~40 cm of travel. A planted foot
   leaves a dot, a sliding one leaves a streak, and the streak's length is slip per step
   in millimetres — the same quantity the sim reports as `skid mm/foot`. Read it **per
   foot**: the ballast plate sits forward, front duty is 0.68 against the rear's 0.40–0.43,
   so the two ends of the robot are not doing the same thing.
2. **A/B the surface.** The same gait, the same start mark, the same seconds, on bare
   bench against a rubber mat or paper. Distance grows on the grippy one → the deficit is
   slip, μ is the binding constraint, and sole work pays. Distance unchanged → traction is
   not where the speed is going, and the sole becomes a wear question rather than a
   performance one. Note the runway is the limit here, not the robot: 3 s already travels
   350–450 mm out of ~40 cm, so shorten the run rather than let one arm hit the end. The
   bench has no non-slip surface today; that is a consumable to buy, not a blocker.
3. **Only if 1–2 say yes: cross-leg consistency, from the encoders alone.** In a trot the
   two diagonal stance feet each imply a body velocity through forward kinematics, and if
   both are planted the two must agree. The disagreement is a slip signal that needs no
   instrument, and integrating stance-foot velocity over a run against the measured
   distance gives the total. Use **measured** positions, not commanded — tracking error
   peaks at 0.61 rad / 35°, so the commanded path is not where the leg was, and
   `PRESENT_POSITION` is trustworthy while driving (0 impossible jumps in 321 samples)
   unlike the temperature byte. Backlash and leg compliance land in the same residual, so
   read it comparatively (mat against bench, front against rear), not as an absolute.

**What each answer costs.** The friction coefficient is a real defect either way and it is
cheap: `MJ_FOOT_FRICTION` in `mini_dog.py` section 4, measured rather than inherited, read
by both exporters, then `3d/CLAUDE.md` steps 4–6. Nobody chose 1.2. The geometry is the
expensive one — a truncation is one parameter in `foot()` but a full ladder plus a BOM line
if `FOOT_CB_Z` moves, and it belongs in `MAC.md`'s queue. Two things to know before
cutting a flat: the shin is **34.6° off vertical while it carries load** (46.7° worst), so
a flat normal to the shin axis lands on its edge — rake it to the loaded angle or use a
large-radius crown, which buys area over R = 13 without an edge. And the terrain
heightfield **cannot compare feet** at `CELL_MM = 12` — a ⌀17.7 face is smaller than one
cell and collides with prism walls, which is how `tripod` once stood still with 4 kN
through its touch sensors. `--rough` is the arm that can.

If the sole does change, the way to choose the shape is coupons, not feet: printed pucks
at the real normal force (3.8 N/foot standing at 1.55 kg, ~7.6 N in the trot; 6.1 and
12.3 N at the design 2.5 kg), measuring μ on a tilt plate, the real patch off a carbon
print, and the behaviour of the same coupon tilted 35°. Vary one thing at a time — flat
diameter, shore, tread, wall count.

## 6. Train

**Why:** the whole `rl/` tree exists and has never been run in anger, because training
against vendor priors was correctly judged to be wasted work. After steps 1–4 it is not.
The WSL2 box does this alone; the mac and the robot are not involved.

Train and evaluate in sim now. **Do not expect to deploy** — that needs the IMU, below.

### Not ready yet, and the checklist is short — asked and answered 2026-09-09

Three things, none of them large, and two are decisions about which number goes where
rather than new work:

1. **`rl/` has no static friction at all.** `model.py` zeroes MuJoCo's `damping` and
   `frictionloss` because `actuator.py` supplies them, but `actuator.py`'s friction is
   `tanh(w/v_eps)` — exactly zero at rest (step 2b). So a standing or stancing robot in
   `rl/` feels **zero** joint friction while the real servo has 0.18–0.35 N·m and the
   ROS 2 model now has 0.184. At ~0.3–0.5 N·m of knee torque in stance that is a third
   to a half of the load. Step 3 made this gap *wider* rather than narrower: both sims
   were wrong together at 0.02 before, and now `ros2/` is right and `rl/` is at zero.
2. **The two sims disagree about servo strength by 44 %.** `rl/`'s emergent stall is
   `k_u × 12` = 4.23 N·m; `ros2/` clamps the same joint at the datasheet's 2.94. Not a
   bug in `model.py` — its ±5 N·m ceiling is a documented NaN guard — but a consequence
   of 2c that nobody had costed: it decides how strong the servo is *in training*.
   **2c-i is the measurement that settles it**, and it is queued on the printer.
3. **Step 4 is not done.** No randomisation over pack voltage, and `mu_load` is not
   randomised at all — confirmed live on the WSL2 box, `_params().mu_load` is shape `()`
   while `_params().tau_c` beside it is `(8, 12)` with a 0.33 spread.

Two things that are **not** blockers, so they do not get used as reasons to wait: step
3b's gait re-tune is for the hand-tuned `standalone_sim` trot, which is a regression
harness — RL learns its own gait. And the missing IMU blocks deployment, not training,
exactly as this step already says.

---

# Step 7 — when the IMU arrives

## 7. Wire the IMU, then close the walker's loop on hardware for the first time

Mount at the `IMU_*` site, solder to the pads (the slot is 3.6 mm and the board with
headers does not fit), then re-run `rl/checks/imu_placement.py` against the real mount.

**Why this before the battery:** it is the cheaper part, the analysis is already done and
waiting, and it unblocks two independent things at once — the walker's body levelling and
heading hold, which have only ever run in simulation, and the deployability of anything
`rl/` produces. Training without it is speculative; training after it is testable.

The first hardware result to look for is the one the sim already predicts: terrain feedback
is worth a great deal more than flat-ground performance suggests.

---

# Steps 8–10 — when the battery arrives

## 8. Replace the ballast with the pack, and re-tune the gait against real mass

**Why:** the 1 kg plate is a stand-in that sits where you put it. The pack has a cradle, a
position and a centre of mass the CAD already knows about. Every gait figure measured at
2.55 kg with a plate should be re-measured once, deliberately, with a control run beside it
— `3d/CLAUDE.md`'s warning about reading a distance change as a geometry regression applies
exactly as written.

## 9. Install the Orange Pi and move the 50 Hz loop onto it

**Why now and not earlier:** everything the Pi adds is untethered. `robot/runtime` is
already controller-agnostic and machine-agnostic; moving it is a deployment step, not a
development one. Re-run `bus_probe.py` on the Pi first — the transaction cost is a property
of the machine, and the 50 Hz budget was measured on the mac.

## 10. LiDAR and camera

**Why last:** both are sensors for autonomy, not for walking, and both need the Pi. The
model side is already done and already honest — `lidar.py` carries the *measured* point
rate (62 341/s at 12 Hz, against a catalogue 21 600) and both sim consumers read the cone
out of the compiled model rather than a config file.

---

## The one thing to keep doing throughout

Every model change goes through the ladder in `3d/CLAUDE.md` — FEA, `export_sim.py
--check`, and the ROS 2 regeneration — in the same pass. The ROS 2 tree is the easy one to
forget: it lives outside the repo, nothing imports it, and its meshes keep rendering
happily with whatever geometry they were baked from.

And when a bench number changes, it goes in **two** places: the repo, and
`ST3215_STS3215_measured_parameters.md`, which is the public reference for people who will
never build this robot.
