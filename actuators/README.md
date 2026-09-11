# actuators — bench tests for a 5010 + printed gearbox

**This is not about the dog in this repository.** It is the next one, and it
lives here only because the lessons it is built from are here — nothing in
`3d/`, `ros2/`, `rl/` or `robot/` imports it, generates it, or is affected by it,
and it has no parts, no code and no place in any build.

The ST3215 is a closed bus servo; a 5010 is a bare BLDC turning a gearbox you
printed yourself, and almost every measurement changes character because of
that. This file says which tests to run, in what order, and which
traps `smalldog` already paid for so this one does not pay again.

Nothing here is written yet. It is a test plan, not a result.

---

## What changes when you leave the bus servo behind

| | ST3215 (smalldog) | 5010 + printed gearbox |
|---|---|---|
| current | a register with an **unverified LSB**, unusable below ~0.2 A | a shunt you own — measure it directly |
| torque command | `TORQUE_LIMIT`, a per-mille duty cap of unknown meaning | commanded current, in amps |
| position loop | the vendor's, unknowable, `kp` in register units | yours |
| gearbox | sealed, metal, efficiency inferred | printed, visible, and **it will creep, wear and deflect** |
| the hard part | finding out what the registers meant | finding out what the plastic does over time |

**The single biggest win: torque becomes directly measurable.** `smalldog` spent a
year with a stall torque wrong by 53 % because every route to it ran through
registers that were mis-scaled by 1.64x, and it took a kitchen scale to settle.
With a current shunt and a known `k_t`, torque is `k_t * i` and you can check it
against a scale on day one. **Do that on day one.**

**The single biggest new risk: the gearbox is plastic.** Everything a metal
gearbox lets you assume — that the ratio is exact, that backlash is constant,
that stiffness is infinite, that today's efficiency is next month's — is now a
thing to measure, and to re-measure after running.

---

## Part 0 — the ratio is a measurement problem, not a catalogue choice

**You cannot choose the gearbox until the motor has been on a scale.** This is
the first real decision and it is downstream of exactly two numbers from Part 1:
`k_t` (1.3) and the current the motor will take continuously without cooking
(1.6). Their product is the continuous torque, and **continuous is the number
that sizes a leg**, not peak — a standing robot holds torque indefinitely.

The ratio is squeezed from both ends, and the squeeze is the design:

| pushes the ratio UP | pushes it DOWN |
|---|---|
| joint torque needed to stand, walk and survive a landing | joint speed the gait needs |
| gearbox losses (Part 2.4) eat delivered torque | backdrivability (2.5) — high ratio kills it |
| | backlash and compliance multiply at the output |
| | reflected inertia goes as ratio², and it already dominated smalldog's leg |

**For scale, from the dog in this repository:** the ST3215 makes **4.50 N·m** at
the joint through a 1:345 reduction, and its usable ceiling is **3.15 rad/s**.
Those two numbers are what a working 1.55 kg dog actually used, so they are the
honest starting target — and both were *measured here*, not taken from a
datasheet, which is the only reason they can be trusted as a target at all.

A 5010 is a very different starting point. Torque is scarce and speed is
abundant: a gimbal-wound 5010 makes something like a few tenths of a N·m
continuously, while its no-load speed at 12 V is hundreds of rad/s. So the
arithmetic usually runs **torque-first** — pick the ratio that delivers the joint
torque you need, then check that the speed that survives it still clears the
gait, which it will by a wide margin. That is the opposite of smalldog, where
speed was the binding constraint and the gait spent a year asking for more than
the joint had.

**Reflected inertia is the trap on this axis.** It scales with ratio², and
`check_model.py` already found the ST3215's rotor inertia to be **151× the knee
link's own**. At 1:345 that is expected; the point is that a QDD actuator is
chosen partly to *escape* it, and a ratio picked purely for torque can hand it
straight back. Compute `J_m * ratio²` against the link inertia for every
candidate ratio before committing, because it decides how fast the leg can
reverse — which is what a trot is.

**So the order is: 1.3 and 1.6 first, then this decision, then everything else.**
Parts 2 and 3 characterise a gearbox that already exists; this is the step that
decides which one to build.

## Part 1 — the bare motor, no gearbox

Do these before the gearbox exists. Every one of them is contaminated later by
gearbox friction, and several become unmeasurable.

**1.1 Pole pairs.** Count electrically: drive a slow open-loop electrical
revolution and count mechanical revolutions. A 5010 is usually 14 poles / 7 pole
pairs, but *count it* — every later electrical angle depends on it and a wrong
count looks like a working motor with bad torque ripple.

**1.2 Phase resistance and inductance.** R from a DC current injection into two
phases (not an ohmmeter — lead and contact resistance is the same order as the
winding). L from the current rise time or an LCR meter. Sets your current-loop
gains and the thermal model.

**1.3 Torque constant `k_t`, against a scale. THE FIRST TEST, AND THE ONE THE
GEARBOX CHOICE WAITS ON.** Lock the rotor, command a known current, measure force
at a known radius. Sweep current, fit the **slope**, not a single point. This is
the number everything else is checked against, it is the number `smalldog` got
wrong, and together with 1.6 it is what Part 0 needs to pick a ratio. Do it
before anything is designed around it.

**1.4 Back-EMF constant `k_e`.** Spin the motor with a drill at a measured speed,
scope the open-circuit line voltage. **In SI, `k_e` (V·s/rad) and `k_t` (N·m/A)
are the same number** — so 1.3 and 1.4 are an independent cross-check of each
other, and disagreement means one of your measurements is wrong rather than the
motor being unusual. `smalldog` never had this check and it is why a bad `k_t`
survived.

**1.5 Cogging torque.** Zero current, rotate slowly, log torque against angle.
A gimbal motor is chosen partly for low cogging; with a reduction it is
multiplied by the ratio at the output, so measure it before you decide the ratio
is high enough to ignore.

**1.6 Thermal envelope. The other half of the ratio decision.** Current against
steady-state winding temperature, to the limit you are willing to run. This sets
*continuous* torque, which is a completely different number from peak and is the
one that sizes a leg — a standing robot holds torque indefinitely, and a 5010 has
no heatsink but its own stator. Quote both: peak for landings, continuous for
standing, and design the ratio to the continuous one. Note the same effect `smalldog` measured on the ST3215: torque decays
4–7 % over 8 seconds of holding, because the winding warms and resistance rises.
On a current-controlled drive that decay **disappears** — the loop holds current
regardless of R — which is a real advantage, but it moves the limit from torque
to temperature. Log temperature, not just current.

---

## Part 2 — the printed gearbox alone, motor removed

Drive the input with something you can measure — a second motor, a torque wrench,
a hand crank on a lever. The point is to characterise the gearbox *without* the
motor's own friction in the number.

**2.1 Ratio, counted not assumed.** Mark input and output, turn, count. A
cycloidal drive's ratio is easy to get wrong by one.

**2.2 Backlash.** Lock the input, apply a small torque to the output both ways,
measure the angle swept. Do it **at several output positions** — a printed
gearbox is not uniform around a revolution, and the worst position is the one
that matters. Record the number; it is the baseline for 2.7.

**2.3 Friction: breakaway and running, and make it load-dependent.**
This is the test `smalldog` learned the hard way. Friction is **not** one
constant: it is `tau_c + mu_load * |tau_transmitted|`, and on the ST3215 the
load-dependent term was the *larger* of the two (0.29 N·m per N·m carried against
a 0.18 N·m floor). A gearbox characterised at one load tells you almost nothing
about another.
So: measure breakaway torque at the output at **several transmitted loads**, and
fit the two coefficients. The bidirectional hold ladder is the technique — walk
to each angle from both directions; the half-difference is the friction at that
load and the mean is the driving torque without it.

**2.4 Efficiency, forward and back-driving, and they are different.** Input
torque × input speed against output torque × output speed. `smalldog` measured
~28 % lost forward-driving against ~62 % back-driving on the ST3215 — most of
which turned out to be the fixed friction term measured against two very
different loads, not a genuinely asymmetric gearbox. Watch for the same artefact:
compare at *equal transmitted torque*, or you will invent an asymmetry.

**2.5 Backdrivability.** The output torque needed to turn the input, with the
motor open-circuit. For a legged robot this is a design requirement, not a
curiosity: it decides whether the leg can absorb an impact or has to fight it.
Quote it as a torque, and also as a ratio against 2.3's breakaway.

**2.6 Torsional stiffness.** Lock the input, apply known output torque, measure
output deflection. Printed teeth deflect, and the number goes straight into any
model that has a joint spring. Expect it to be soft enough to matter.

**2.7 Wear, and this is the one a metal gearbox never needed.** Run a duty cycle
— a few hours of representative reversing load — then **re-run 2.2 and 2.3**.
Backlash growth and friction change over time is the thing that decides whether a
printed gearbox is a prototype or a robot. Repeat at intervals and plot it.

**2.8 Creep and temperature.** Hold a sustained load and watch the output angle
over minutes, at room temperature and warm. Plastic creeps; a gearbox that holds
a stance for an hour is a different problem from one that takes a step.

---

## Part 3 — the assembled actuator

**3.1 Output torque against commanded current.** The end-to-end number:
`k_t * ratio * efficiency`. Check it against Part 1's `k_t` and Part 2's ratio
and efficiency — if the product does not match the direct measurement, one of the
three is wrong, and that reconciliation is the whole point of testing the pieces
separately first.

**3.2 No-load speed against voltage.** Terminal speed is where the torque
ceiling meets the damping. `smalldog` shipped a gait that commanded 4.00 rad/s at
a joint whose real ceiling was 3.15, because the limit was derived from a vendor
no-load figure nobody had checked — **31.7 % of commanded samples asked for speed
the joint did not have.** Measure it; do not inherit it.

**3.3 Step response and bandwidth.** Closed-loop, at several amplitudes. Small
steps show the friction band; large ones show the torque ceiling.

**3.4 Thermal, assembled.** The gearbox is now a blanket over the motor, and it
is plastic. Find the continuous torque that keeps the *gearbox* below the
temperature where 2.8's creep gets worse — that may bind before the winding does.

---

## The traps, carried forward from smalldog

Each of these cost real time there. They are cheap to avoid here.

- **Read a scale held, not pulsed.** A short burst reads the arm's inertia and
  the scale's own filter ringing on top of the static force — measured at a
  roughly *constant* ~130 g offset, which corrupts the slope as well as the
  level. Hold for several seconds and read it settled.
- **The encoder cannot see the fixture move.** It measures rotor against stator.
  If the whole motor turns in its mount, or the stand rocks, the encoder reads
  perfectly steady while your torque number is wrong. Design the fixture
  self-reacting — the load loop closing inside the fixture — or clamp it and
  verify the clamp separately.
- **A scale needs its own feet on a flat surface.** Bridging it across a narrow
  beam distorts the load cell's mounting and reads several percent off.
- **Measure the slope, never an endpoint.** Every good number in `smalldog` came
  from a ladder and a fit; every bad one came from a single reading.
- **Measure several units.** Manufacturing spread is what domain randomisation
  needs, and no amount of precision on one unit reveals it. This matters more
  here, not less: printed gearboxes vary between *prints*, let alone between
  designs.
- **One constant, one home.** `smalldog` shipped the servo mass, the joint feel,
  the foot contact, the stall torque and the IMU position each duplicated across
  two files, and every one of them silently diverged. Decide where a number lives
  before there are two consumers.

---

## Fixtures needed

- **A torque fixture** — a lever at a known radius onto a scale. Make it
  self-reacting (a C-frame that closes its own load loop) and put the scale flat
  on its own feet inside it. `smalldog/3d/torque_rig.py` is the worked example
  including what was wrong with it.
- **A pendulum stand** for gravity-anchored tests — the arm hangs at the zero
  position so `m·g·r·sin(q)` is the known load. `smalldog/3d/bench_rig.py`.
  Gravity is the one torque reference that needs no calibration, which is why
  friction was the one family of numbers smalldog never got wrong.
- **A current shunt** on the phase or the DC bus, and a scope. This is the
  instrument smalldog did not have and most needed.
- **A supply whose voltage you can set**, for three points.
- **A thermocouple or IR thermometer** on the stator and on the gearbox.
- **A second motor or a torque wrench** to drive the gearbox in Part 2.

---

## Order

1. **1.3 and 1.6 first of all** — `k_t` on a scale and the thermal limit. Their
   product is continuous torque, and nothing about the gearbox can be chosen
   without it (Part 0).
2. The rest of Part 1, before the gearbox exists.
3. Reconcile 1.3 against 1.4. Do not proceed until `k_t == k_e`.
4. **Choose the ratio** (Part 0), and check `J_m * ratio²` against the link
   inertia while you do.
5. Part 2 on the first printed gearbox, and again on a second print of the same
   design — print-to-print spread is a real axis here.
6. Part 3.
7. 2.7 and 2.8 on a schedule, forever. They are the only tests here whose answer
   changes with time, which makes them the only ones that can invalidate a design
   after it looks finished.
