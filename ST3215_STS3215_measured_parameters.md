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
| Coulomb friction `τ_c` | **0.08 – 0.09 N·m** | — | four independent methods |
| Viscous friction `b_v` | **≈ 0.115 N·m·s/rad** | — | two-arm free-swing energy |
| Static breakaway `τ_s` | **0.23 – 0.35 N·m** | — | bracketed, hold + release |
| Load-dependent friction | **+38 % of motor torque** | not modelled anywhere | hold ladder, both voltages |
| Back-driving loss | **≈ 62 %** of applied torque | — | free-swing energy balance |
| Position-loop stiffness | **28.8 N·m/rad @ 12.1 V**<br>**24.5 N·m/rad @ 10.1 V** | — | steady-state droop |
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

**Coulomb `τ_c` ≈ 0.08–0.09 N·m**, from four methods that do not share assumptions:

| method | `τ_c` |
|---|---|
| Energy balance over two free swings, solving `τ_c` and `b_v` together | 0.086 |
| Hold-ladder intercept (gravity torque extrapolated to zero duty) | 0.079 (12 V), 0.070 (10 V) |
| Release acceleration at ω = 0, two arms | 0.125 |
| Full trajectory fit | 0.085 |

It is voltage-independent, as friction must be. **Watch out for the single-arm trap:** an
energy balance over free-swing drops on *one* arm gives ≈ 0.28 N·m, because it bills the
viscous loss to Coulomb. A slow heavy swing and a fast light one fit the same decay. Two
arms at different speeds split them, and **the viscous half is the larger one** — 0.43 N·m
at 4.7 rad/s against 0.09 of Coulomb.

**Stiction is separate and ~3× larger**: 0.23–0.35 N·m. Consequences you will meet:

- With a 1 kg mass on a 45 mm arm the servo holds anywhere in a **±14° band** around
  vertical, torque off. *Where it comes to rest is not where it hangs* — do not calibrate a
  zero from it.
- Small commanded corrections below the breakaway do not move the joint at all.

**Load-dependent friction — the term most models are missing.** Friction is not constant;
it grows with the torque being transmitted, as gear-tooth normal forces do. The hold
ladder's torque-vs-duty slope is **1.38× steeper** than `k_u·U` predicts, at both voltages
(1.40 at 12 V, 1.37 at 10 V), i.e. friction ≈ `τ_c + 0.38·τ_motor`.

And it is **asymmetric**: back-driving the gearbox (the free swing, arm falling under
gravity) loses **~62 %** of the applied torque, against ~28 % forward-driving. Both are the
same 1:345 stack; a single `τ_c·sign(ω) + b_v·ω` law cannot express either one.

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
| 12.10 V | 28.8 N·m/rad | 1.99 °/N·m |
| 10.09 V | 24.5 N·m/rad | 2.33 °/N·m |

Ratio 1.176 against a voltage ratio of 1.199 — linear, exactly as `duty = kp_reg·err` then
`τ = k_u·U·duty` predicts. **Your robot gets softer as its battery drains**, and a fixed
`kp` in a simulator cannot represent that.

For anyone matching a MuJoCo `position` actuator to this servo at 12 V: `kp ≈ 25–29
N·m/rad` is a good number at `P_COEF` 32.

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
(The encoder question used to be listed here. It is answered — see below.)

---

## Method

One servo held with its axis horizontal in a rigid fixture, driving a rigid arm with a
weighed 1.053 kg disc bolted at a known radius. Two arms — 45 mm and 90 mm — because a
single arm cannot separate inertia from friction. Logged at 200 Hz over a 1 Mbaud bus.

Trajectories: free release from horizontal (inertia and friction, no motor torque);
a ladder of held angles (static torque balance — the cleanest measurement here); steps;
slow triangles through zero under load (backlash); reversals of shrinking amplitude (dead
zone, stiction); a 0.2–8 Hz chirp (held out for validation).

**Run the whole set at three supply voltages.** At one voltage the back-EMF damping and
the viscous friction enter every equation as the same coefficient of ω, and no amount of
data separates them. 12 / 10 / 8 V here.

Raw CSVs, the drive code and the fitting code are in this repository under `robot/bench/`.
Each CSV carries its own metadata header — supply voltage, arm mass and radius, arm
inertia, encoder zero, and the full control-register set the run is conditional on.

## Corrections welcome

If you measure a different value on your servo, that is interesting and worth an issue —
particularly for the friction numbers and `J_m`, which are the ones most likely to vary
between units and the ones with the widest error bars here.
