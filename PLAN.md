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
| `robot/bench` | three-voltage identification set captured; the mechanical half is fitted and cross-validated, the electrical half is one model term short |

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

## 1. Finish the servo identification

Two runs, both small trajectory edits, both about five minutes per voltage at the bench.

- **A bidirectional hold ladder.** Every angle approached from above *and* from below. At a
  static hold friction helps carry the load and its sign flips with approach direction, so
  the half-difference in duty is friction at that load and the mean is the honest motor
  torque. Today's ladder hits 0.6 rad both ways by accident and that one row is the only
  place the hysteresis is visible.
- **A triangle speed ladder** — the same triangle at periods 20 / 10 / 5 / 2 / 1 s, giving
  0.1 / 0.2 / 0.4 / 1.0 / 2.0 rad/s.

**Why:** `b_v` is still pinned at its lower bound and `k_u` is inflated — the fit implies a
stall of 4.23 N·m against a spec 2.94. The free swing cannot supply `b_v` because at this
friction level the arm does not oscillate, so there is no decay envelope to separate
Coulomb from viscous; only steady speeds can. And the inflated torque constant is the
missing load-dependent friction coming out somewhere: at a hold, friction *helps*, so a
model that cannot represent friction growing with load has nowhere to put the surplus
except the motor.

## 2. Give `actuator.py` a load-dependent friction term

**Why:** it is the largest thing the model is missing, and three independent measurements
now agree it is real — the hold ladder's torque-vs-duty slope is 1.38× steeper than `k_u·U`
predicts at both voltages, the free swings dissipate 62–65 % of the applied torque, and
`actuator.py`'s own `eta = k_u·R/k_e` from the vendor specs says 57 %. Its law is
`tau_c·sign(w) + b_v·w`, both independent of load. Until that term exists, step 1's numbers
have nowhere to land and every fit will keep inflating `k_u` to absorb them.

Note the asymmetry when writing it: forward-driving loses ~28 % of motor torque,
back-driving ~62 %. Same gearbox, different direction.

## 3. Push the fitted numbers into the CAD and re-baseline

`MJ_ARMATURE`, `MJ_DAMPING`, `MJ_FRICTIONLOSS` in `mini_dog.py` section 4, then
`3d/CLAUDE.md` steps 5 and 6 in the same pass.

**Why:** `MJ_ARMATURE` is 0.008 and the bench says **0.024–0.042**. That is the dominant
term in the joint's dynamics — `check_model.py` already measured it at ~73× the knee link's
own inertia — so a 3× error in it is not a refinement, it is a different robot. Expect
every number in step 6 to move and re-baseline them deliberately rather than reading the
change as a regression.

`3d/CLAUDE.md` says these constants "become its initial guess — never a second opinion
sitting beside it" once the bench fits the actuator. This is that moment. One correction to
make while you are there: that file calls `MJ_ARMATURE` = 0.008 "not an estimate … the
ST3215's reflected rotor inertia". It was never measured — `actuator.py` marks its whole
block `fitted: false` — and the bench says it is several times low.

## 4. Randomise over the pack

`rl/params/domain_rand.json`, with the voltage range now measured rather than guessed.

**Why:** step 3 gives one servo at one voltage. The robot will run a pack that sags from
12.6 to 9.9 V, and we now know what that does — stiffness falls 25 %, and torque per unit
duty falls with it. Randomising over it is what makes a policy survive a discharge instead
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
