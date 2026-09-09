# Feetech / Waveshare ST3215 (STS3215) — measured parameters

Bench measurements of a real **ST3215** serial-bus servo, 12 V, 1:345 gearbox, the
30 kg·cm variant. Published so that the next person does not have to buy the servo and
build a test stand to find out that the datasheet is optimistic in some places, silent in
others, and that one register does not mean what it looks like it means.

**Measured 2026-09-08. One servo, one sample.** Nothing here is a datasheet value unless
it says so. Where a number disagrees with the vendor's, both are given.

Search terms, so this page is findable: *ST3215 parameters, STS3215 parameters, ST3215
torque constant, ST3215 armature inertia, ST3215 friction, ST3215 present current wrong,
STS3215 PRESENT_CURRENT register 69, ST3215 MuJoCo armature damping frictionloss, ST3215
system identification, ST3215 backlash, SO-100 SO-101 STS3215 servo model, LeRobot servo
parameters, ST3215 gearbox efficiency.*

---

## TL;DR — the table

All values are **at the output shaft** (i.e. after the 1:345 reduction), SI units.

| quantity | measured | vendor / prior | how |
|---|---|---|---|
| Terminal resistance `R` | **≈ 4.0 Ω** (3.84 – 4.4) | 4.44 (12 V ÷ 2.7 A) | four routes, below |
| Torque constant `k_t` | **1.12 N·m/A** (1.17 on the clean branch) | 1.09 (2.94 N·m ÷ 2.7 A) | hold ladder, corrected motor current |
| Back-EMF constant `k_e` | **2.13 V·s/rad** | 2.55 (12 ÷ 4.71) | blocked-output stall, across speeds |
| No-load speed | **5.64 rad/s @ 12 V** | 4.71 | derived from `k_e` |
| **Gearbox efficiency η** | **0.44 – 0.53** | 0.43 (implied by the 3 specs) | `k_t / k_e` |
| Reflected inertia `J_m` | **0.028 – 0.042 kg·m²** | 0.008 was a guess | two arms, free swing |
| Coulomb friction `τ_c` | **0.16 – 0.19 N·m** | — | two ladders, both approaches |
| Viscous friction `b_v` | **0.06 ± 0.07 N·m·s/rad** — *unresolved* | — | speed ladder minus back-EMF |
| Total speed-proportional torque | **1.37 N·m·s/rad** (`b_v + k_t·k_e`) | — | speed ladder, 3 voltages |
| Static breakaway `τ_s` | **0.23 – 0.35 N·m** | — | bracketed, hold + release |
| Load-dependent friction | **+0.28 N·m per N·m carried** | not modelled anywhere | bidirectional hold ladder |
| Back-driving loss | **≈ 62 %** of applied torque | — | free-swing energy balance |
| Position-loop stiffness | **40.9 N·m/rad @ 12 V**<br>**34.4 @ 10 V, 28.1 @ 8 V** | — | droop, friction cancelled |
| Backlash | **0.50°** at the output | ≤ 0.5° | reversal hysteresis |
| Position dead zone | **0.36°** | `CW_DEAD`/`CCW_DEAD` = 1 count | reversal |

Control registers every number above is conditional on — a fit is only valid for the gains
the servo had at the time:

```
P_COEF 32   D_COEF 32   I_COEF 0   MODE 0 (position)   ACCELERATION 0
CW_DEAD 1   CCW_DEAD 1  STARTUP_FORCE 16   TORQUE_LIMIT 1000   MAX_TORQUE 1000
PROTECTION_CURRENT 310  OVERLOAD_TORQUE 80  PROTECTIVE_TORQUE 20  OFFSET 85
```

---

## The one that will cost you a day: `PRESENT_CURRENT` is **supply** current

Register **69 (0x45)**, LSB **6.5 mA** — both as documented. What is *not* documented is
that it reports the current drawn from the **bus supply**, not the current through the
motor. Behind a PWM half-bridge those differ by the duty cycle, and at the low duties a
servo actually holds at, they differ by a lot.

Holding **0.865 N·m**, this servo reported **0.084 A**. A motor making that torque carries
about **0.5 A**. It looks like a broken sensor. It is not:

```
I_motor    = d · U / R           (a PWM'd brushed motor at duty d)
I_reported = d · I_motor = d² · U / R
```

so

```
    I_motor = PRESENT_CURRENT / d              d = PRESENT_LOAD / 1000
    R       = d² · U / PRESENT_CURRENT
```

**The proof.** Invert it over a ladder of holding angles — duty from `PRESENT_LOAD`,
seven points across a 7× range of duty — and it returns a constant it was never told:

| supply | `R` recovered | spread |
|---|---|---|
| 12.10 V | **4.31 Ω** | 4.29 / 4.29 / 4.29 / 4.44 / 4.36 / 4.33 |
| 10.09 V | **4.41 Ω** | 4.01 … 4.87 |

against the 4.44 Ω the vendor's locked-rotor spec implies. Two supplies 20 % apart agree
to 2 %. The `d²` law is right.

Three further routes, all agreeing, none of which knew the others' answer:

| route | `R` |
|---|---|
| Blocked output at full duty (`w` = 0, so `I = U/R` exactly) | **4.0 / 4.1 Ω** at 8 and 10 V |
| Regression over 20 936 stationary samples at any duty | **3.98 Ω** |
| Regression over 374 genuinely saturated samples | **3.84 Ω** |

**Consequences for anyone fitting a model to this servo:**

- Regressing torque against this register raw regresses against `d²` instead of `d`. It
  will not converge, and it will not tell you why — you get an unphysical non-zero-torque
  intercept at zero current.
- Divide by duty first, and both `R` and `k_t` land within 8 % of the vendor's numbers.
- The 6.5 mA LSB is **correct**. The bug is in interpretation, not scaling.

---

## Electrical

`R` and `k_t` above. The efficiency is the interesting one.

For any DC motor `k_t` [N·m/A] and `k_e` [V·s/rad] are the same number. Through a gearbox
they are not: `k_e` at the output is `n·k_e,motor` but `k_t` at the output is
`η·n·k_t,motor`, so their ratio **is** the efficiency:

```
η = k_t / k_e = 1.12 / 2.55 = 0.44
```

The three vendor specs independently imply `η = k_u·R/k_e = 0.43`. **A 1:345 spur stack at
~43 % is genuinely that lossy** — if you have been assuming a servo delivers its stall
torque per amp, it does not.

`k_e` is measured too, from the blocked-output runs where the duty is pinned and the
current then falls with speed as `I = U/R − (k_e/R)·ω`: **2.13 V·s/rad**, against the 2.55
the vendor's no-load speed implies. Using the measured `k_e` gives η = 1.12/2.13 = **0.53**;
using the vendor's, 0.44. Somewhere in 0.44–0.53, and nowhere near 1.

**One number here is NOT trustworthy: the torque constant read off a hold ladder alone.**
Fitting these runs with a model whose friction is only `τ_c·sign(ω) + b_v·ω` returns
`k_u·R` = 1.90 N·m/A and an implied stall of **5.92 N·m — twice the spec**. That is not a
strong servo, it is the missing load-dependent friction term coming out somewhere. At a
static hold friction *helps* carry the load, so a model that cannot represent friction
growing with load has nowhere to put the surplus except the motor. If you fit this servo,
expect an inflated torque constant until you add that term.

---

## Friction — and why one number is not enough

Every number below comes from a **difference between two runs of the same trajectory**,
not from a fit. That matters more than it sounds: a static hold does not settle at zero
net torque, it settles wherever friction happens to balance the rest, so the duty it holds
depends on which side the joint arrived from. Walk a ladder of angles once and every rung
sits somewhere inside that band with nothing in the log to say where; walk it up and then
down and half the difference *is* the friction while the mean is the torque without it.
The same trick on a triangle at five periods gives the speed dependence. Both ladders are
in `robot/bench/sweep.py` as `holdbi` and `speed`, and `robot/bench/hysteresis.py` prints
the table below from the raw CSVs in one command.

| | 8 V | 10 V | 12 V |
|---|---|---|---|
| effective torque constant, N·m per volt of motor drive | 0.636 | 0.611 | 0.596 |
| friction mobilised at a hold, unloaded, N·m | 0.192 | 0.181 | 0.186 |
| **…per N·m of load carried** | **0.286** | **0.295** | **0.251** |
| kinetic Coulomb extrapolated to zero load, N·m | 0.161 | 0.160 | 0.168 |
| total speed-proportional torque, N·m·s/rad | 1.403 | 1.364 | 1.334 |

The right-hand four rows are friction and a torque constant, so they **must not** move with
supply voltage, and they do not — 5–16 % across a 1.5× range, which is the error bar on the
whole method. That invariance is the check that these are physical quantities and not
artefacts of the loop.

**Coulomb `τ_c` ≈ 0.16 – 0.19 N·m**, and that is about **twice** what indirect methods give.
The two ladders agree with each other closely — 0.187 N·m from the static band, 0.163 from
the moving one — while a free-swing energy balance, a one-directional hold intercept and a
full trajectory fit all land at 0.08–0.09. The direct measurement is the one to believe:
each of the indirect three has somewhere else to put friction (`k_u`, `b_v`, the inertia)
and the differential measurement does not. The older numbers are kept below for anyone
comparing methods rather than servos.

| method | `τ_c` | direct? |
|---|---|---|
| Bidirectional hold ladder, three voltages | **0.187** | yes |
| Speed ladder intercept, three voltages | **0.163** | yes |
| Release acceleration at ω = 0, two arms | 0.125 | partly |
| Energy balance over two free swings | 0.086 | no |
| Hold-ladder intercept, one approach only | 0.079 (12 V), 0.070 (10 V) | no |
| Full trajectory fit | 0.085 | no |

**Load-dependent friction — the term most models are missing, now measured rather than
inferred.** Friction grows with the torque being transmitted, as gear-tooth normal forces
do: **+0.28 N·m per N·m carried**, on top of the 0.19 N·m floor, so at the 0.88 N·m the
long arm asks for, friction is 0.43 N·m — half the load again. This supersedes the
"+38 % of motor torque" that earlier editions of this document inferred from a
torque-vs-duty slope ratio; that route reads a one-directional ladder, which is exactly
the measurement the approach direction contaminates. Expressed against motor torque
instead of load it is about +22 %, so the direction of the correction is downward.

**Stiction is separate and larger**: 0.23–0.35 N·m breakaway. Note that the 0.19 N·m above
is not the same number and is not in conflict with it — a hold settles as soon as friction
balances, so what it mobilises is bounded by breakaway rather than equal to it.
Consequences you will meet:

- With a 1 kg mass on a 45 mm arm the servo holds anywhere in a **±14° band** around
  vertical, torque off. *Where it comes to rest is not where it hangs* — do not calibrate a
  zero from it.
- Small commanded corrections below the breakaway do not move the joint at all.

**Viscous friction `b_v` is NOT resolved here, and no amount of bench time on this rig will
resolve it.** What the speed ladder measures cleanly is the *total* torque proportional to
speed — **1.37 N·m·s/rad**, to 5 % across three voltages. That total is `b_v + k_t·k_e`, and
back-EMF alone accounts for 1.31 of it (`k_e` = 2.13 × `k_t_eff` = 0.614), leaving
**0.06 ± 0.07** for viscous friction: consistent with zero, and consistent with the 0.09–0.115
the free swings imply. The reason is structural rather than a shortage of data — back-EMF
and viscous friction each cost a motor voltage proportional to ω, and **neither depends on
supply**, so the three-voltage sweep that separates every other electrical term does nothing
at all here. Splitting them needs the motor current, and `PRESENT_CURRENT` gives that only
as `d²·U/R` at a 6.5 mA LSB. **Use the total.** For a simulator it is the total that sets
how the joint resists being moved, and putting all of it in one term or the other changes
nothing a walking robot can feel.

**Back-driving is asymmetric**: the free swing, arm falling under gravity, loses **~62 %** of
the applied torque, against ~28 % forward-driving. Both are the same 1:345 stack; a single
`τ_c·sign(ω) + b_v·ω` law cannot express either that or the load dependence above.


---

## Mechanical

**Reflected inertia `J_m` = 0.028 – 0.042 kg·m²** at the output. This is the number to put
in MuJoCo's `armature`. Common priors of 0.008 are **3–5× too low**.

The spread is honest: 0.028 from a full trajectory fit, 0.037–0.042 from two-arm free-swing
release. Two arms whose `J_load` differ by only 2× cannot pin it better; a wider lever
would.

Sanity check for how dominant this is: at 1:345 the reflected rotor inertia is **~73×** a
typical small link's own inertia. In a legged robot it, not the leg, sets the joint
dynamics.

**Backlash 0.50°** at the output (0.0087 rad), from hysteresis at a loaded reversal.
Matches the vendor's "≤ 0.5°".

**The encoder sits AFTER the gearbox.** Torque off, rock the horn against the play, and
`PRESENT_POSITION` moves. This is worth knowing before you model the servo: the position
you read over the bus is the *true output angle*, so the backlash is a hole in the torque
path and not in the measurement — a controller can see its own play. It also confirms the
vendor's "360° magnetic encoder, 360/4096", which only makes sense with the sensor on the
output shaft. Thirty seconds to check, and it changes how you wire up observations.

---

## Control loop

The servo runs its own position loop; you are configuring it, not replacing it.

**Steady-state stiffness scales with supply voltage:**

| supply | stiffness | droop |
|---|---|---|
| 12 V | 40.9 N·m/rad | 1.40 °/N·m |
| 10 V | 34.4 N·m/rad | 1.67 °/N·m |
| 8 V | 28.1 N·m/rad | 2.04 °/N·m |

Linear in supply — 3.44 N·m/rad per volt — exactly as `duty = kp_reg·err` then
`τ = k_u·U·duty` predicts. **Your robot gets softer as its battery drains**, and a fixed
`kp` in a simulator cannot represent that. Over a 3S pack's 12.6 → 9.9 V that is a 21 %
change in stiffness with nothing else altered.

**These are ~40 % stiffer than earlier editions of this document said** (28.8 N·m/rad at
12.1 V, 24.5 at 10.1), and the older numbers were wrong rather than merely different. A
one-directional hold ladder measures the total standing offset from target and calls it
droop, but only part of that offset is elastic — the rest is the friction band, which does
not restore and is not stiffness. Approaching every angle from both sides and taking the
mean cancels the band and leaves the elastic part alone. The correction matters because it
runs the wrong way for anyone matching a simulator: the servo is stiffer *and* has more
friction than the single-approach reading suggests, and the two errors hide each other in
any test that only looks at where the joint ends up.

For anyone matching a MuJoCo `position` actuator to this servo at 12 V: `kp ≈ 41 N·m/rad`
at `P_COEF` 32, **with a friction term** — without one, 25–29 is the value that reproduces
the observed droop, and it will be too soft the moment the joint moves.

**Dead zone 0.36°** at the output.

---

## What is NOT measured here

Stated so nobody quotes an absence as a value:

- **`k_e` independently** — needs a no-load free-running speed test. Currently the vendor's
  4.71 rad/s at 12 V.
- **Absolute current calibration** — the 6.5 mA LSB is confirmed only against vendor
  numbers, not against an external shunt or INA226. Everything scales with it.
- **Temperature dependence** — every run above sat between 30 and 32 °C. Winding
  resistance rises with temperature and this does not capture that.
- **Sample-to-sample spread** — one servo. Gearbox friction in particular is exactly the
  kind of parameter that varies between units.
- **The `b_v` / `k_e` split** — see the friction section. Only the sum is measured, and this
  is a structural limit of a bench with no external ammeter, not a gap that more runs close.
- **Why the loop runs out of authority above ~1.8 rad/s** — the duty never pins, so it is
  not the motor's ceiling. `D_COEF` is 32 and unmodelled, which would do exactly this, but
  a run that separates a derivative term from an internal output clamp has not been taken.
(The encoder question used to be listed here. It is answered — see below.)

---

## Method

One servo held with its axis horizontal in a rigid fixture, driving a rigid arm with a
weighed 1.053 kg disc bolted at a known radius. Two arms — 45 mm and 90 mm — because a
single arm cannot separate inertia from friction. Logged at 200 Hz over a 1 Mbaud bus.

Trajectories: free release from horizontal (inertia and friction, no motor torque);
a ladder of held angles (static torque balance — the cleanest measurement here); **the same
ladder walked up and then down, so every angle is reached from both sides** (friction, and
the stiffness with friction removed); steps; slow triangles through zero under load
(backlash); **the same triangle at five periods from 20 s down to 1 s, i.e. 0.1 to 2.0 rad/s**
(the speed dependence — nothing else in the set holds a steady non-zero speed); reversals of
shrinking amplitude (dead zone, stiction); a 0.2–8 Hz chirp (held out for validation).

The two bold entries are what the friction section rests on, and they were added after the
first full pass because a fit over the rest of the set could not find friction: it has three
other places to put it and no way to tell them apart. **A differential measurement beats a
fit here, and it costs ten minutes per voltage.**

One caution on the fast end of the speed ladder: at a commanded 2.0 rad/s this servo does
not track — it reaches 1.8 rad/s at 0.74 rad of position error, and it does so without the
duty ever pinning, so it is not a torque ceiling in the usual sense. That row is the
velocity ceiling, not a steady speed, and it is excluded from the regressions above.

**Run the whole set at three supply voltages.** 12 / 10 / 8 V here. It is what makes the
electrical parameters identifiable, and it is also the only check available on whether a
number is physical at all: friction and a torque constant must not move with supply, so a
row that does is a measurement problem. Every friction figure above is quoted at all three
for that reason.

What three voltages do **not** buy is the `b_v` / `k_e` split, and this document said for a
while that they did. At a steady speed both cost a motor voltage proportional to ω and
neither depends on supply, so they remain the same column of the regressor at eight volts
as at twelve. Only the current separates them.

Raw CSVs, the drive code and the fitting code are in this repository under `robot/bench/`.
Each CSV carries its own metadata header — supply voltage, arm mass and radius, arm
inertia, encoder zero, and the full control-register set the run is conditional on.

## Corrections welcome

If you measure a different value on your servo, that is interesting and worth an issue —
particularly for the friction numbers and `J_m`, which are the ones most likely to vary
between units and the ones with the widest error bars here.
