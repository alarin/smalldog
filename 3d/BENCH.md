# The servo bench — printed parts

`../robot/README.md`, "The bench, in order", specifies the identification bench and then
says **"Build it first. The parts list is in the project notes"**. This is that parts list,
as CAD.

```bash
.venv/bin/python servo_bench.py            # → out/bench/{step,stl}, out/bench/bom.json
.venv/bin/python servo_bench.py --render   # … and out/bench/view_*.png
```

Every servo dimension, fit, clearance, nut pocket, density and mass in
[servo_bench.py](servo_bench.py) is imported from `mini_dog.py`. Nothing is re-typed: the
bench holds the servo in `md.sleeve()` driving `md.fork()` on the two stock aluminium hubs,
so what it identifies is **this robot's joint** — the printed compliance and the printed
backlash included — and not a servo in a vice.

## What the protocol requires of the hardware

Three things in `robot/bench` fix this geometry. Nothing else about it is free.

**1. q = 0 is the arm hanging straight down.** `rl/actuator.pendulum_load` is
`−m·g·r·sin(q)` with `q_zero_is_down=True`, and `fit_bam.py` regresses measured current
against `m·g·r·sin(q)`. So the joint axis is horizontal and the arm is a pendulum. A bench
that put q = 0 at the horizontal would need cos, and every torque constant off it would be
wrong by however far the mounting was out.

**2. The arm has to swing about that, ±89°.** `sweep.py --qmax` defaults to 1.4 rad and
aborts at `|q| > qmax + 0.15`. The stand has to clear that or the *stand* becomes the
travel limit instead of the software.

That is what makes this a **portal**. The long arm and its payload reach 178 mm, so the
whole lower half-plane out to that radius belongs to the arm and there is nowhere to put
structure except above the axis — which is exactly where the servo's own case already
points, and where the sleeve's far end sits at r = 38.1, outside the fork's 34.0 mm sweep.
Two leg plates outboard of the fork in y carry that head down to the base, straddling the
arm. An L-shaped stand cannot do this: its upright would have to stand in the arm's own
swept disc. Measured on the real solids, the arm gets **±110°** against the ±89° needed.

**3. Two arms, and the short one carries as little as will still move the joint.**
`fit_bam.py` gets `J_m = J_tot − J_load`, a difference of two numbers, and declines the
free swing outright when `m·g·r·sin(q₀) ≤ tau_c` — *"this arm cannot move the joint. Use a
longer one"*. Both bounds are checked every run against the servo's own priors in
`../rl/params/st3215.json`, so the arms are sized against the servo rather than by eye.

Note which way that trade runs: `J_load` grows as r² and the gravity torque only as r, so
**shortening the arm is quadratically better than lightening it**. Hence a fairly heavy
payload on a 50 mm station rather than a light one further out.

## The number this file exists to produce

`sweep.py` takes `--mass`, `--radius`, `--arm-inertia` and `fit_bam.py` uses them as

```
mgr    = mass · g · radius                  the gravity torque
J_load = arm_inertia + mass · radius²       the load inertia
```

Note what is **not** in `mgr`: the printed arm's own weight. It is a pendulum too — on the
long arm it is 7 % of the total first moment at the outer station and 20 % at the inner one
— and that bias lands straight on the torque constant, where no amount of data can see it,
because the fit is *told* the torque rather than measuring it.

Both inputs are free, so `servo_bench.py` makes both exactly right and prints them:

```
mass        := m_payload + m_arm · r_com / r_station      an EFFECTIVE mass, same torque
arm_inertia := J_total_about_axis − mass · r_station²     whatever is left over
```

Write `--arm-inertia` with an **equals sign**, as in the table below. It can be negative,
and argparse reads a leading minus as the start of the next option — `--arm-inertia
-1.0e-04` fails with *"expected one argument"*. Both signs are tested against `sweep.py
--dry-run`.

`arm_inertia` can come out **negative** and that is not a bug: the arm's own mass sits at a
smaller radius than the payload, so it buys more gravity torque per unit of inertia than a
point mass at the station does, and the correction has to give the inertia back. What
matters is that `mgr` and `J_load` both come out right, and they do.

### Copy these

From the last run. `--volts` is yours — run every driven trajectory at **three** of them.

| arm | station | in the cup | for | arguments |
|---|---|---|---|---|
| `arm_short` | 50 mm | 0.25 kg | `--traj freeswing` | `--mass 0.3239 --radius 0.050 --arm-inertia=1.246e-04` |
| `arm_short` | 90 mm | 0.25 kg | `--traj freeswing` | `--mass 0.3031 --radius 0.090 --arm-inertia=3.087e-05` |
| `arm_long` | 100 mm | 0.25 kg | `--traj hold`/`step`/`chirp`/… | `--mass 0.3262 --radius 0.100 --arm-inertia=9.599e-05` |
| `arm_long` | 150 mm | 0.40 kg | `--traj hold`/`step`/`chirp`/… | `--mass 0.4598 --radius 0.150 --arm-inertia=-1.028e-04` |

and what the checks say about them:

| config | m·g·r | drive at release | J_load | share of J_load+J_m |
|---|---|---|---|---|
| short @ 50 | 0.159 N·m | +0.098 N·m | 9.34 × 10⁻⁴ kg·m² | 10 % |
| short @ 90 | 0.268 N·m | +0.199 N·m | 2.49 × 10⁻³ kg·m² | 24 % |
| long @ 100 | 0.320 N·m | +0.248 N·m | 3.36 × 10⁻³ kg·m² | 30 % |
| long @ 150 | 0.676 N·m | +0.580 N·m | 1.02 × 10⁻² kg·m² | 56 % |

The two free-swing rows differ in **both** J and torque, which is the point of running two:
one fall cannot separate inertia from Coulomb friction, because a heavier J with less
friction fits the same curve. The share column only matters for those two — a `hold` run
wants a big load and does not care, which is the whole reason there are two arms.

**These numbers are only as good as the printed masses.** `servo_bench.py` estimates them
as solid volume × `md.PRINT_FILL_MEAN`, a mean over other parts. **Weigh both arms and the
filled cup** and re-derive, or accept a few percent of error on the torque scale.

## Printed BOM

| part | qty | for | material / settings | print bbox (mm) | est. mass |
|---|---|---|---|---|---|
| `bench_frame` | 1 | bench | PETG/ASA, 5 walls, 30 % gyroid | 196 × 132 × 258 | 575 g |
| `arm_short` | 1 | bench | PETG/ASA, 5 walls, 40 % — **weigh it** | 28 × 116 × 46 | 52 g |
| `arm_long` | 1 | bench | PETG/ASA, 5 walls, 40 % — **weigh it** | 28 × 176 × 46 | 71 g |
| `mass_cup` | 1 | bench | PETG/ASA, 4 walls, 40 % | 56 × 56 × 46 | 27 g |
| `enc_magnet_cap` | 1 | optional | PETG/ASA, 4 walls, 60 % | 26 × 27 × 3 | 2 g |
| `enc_bridge` | 1 | optional | PETG/ASA, 4 walls, 40 % | 40 × 74 × 14 | 23 g |
| `leg_tower` | 1 | leg rig | PETG/ASA, 5 walls, 40 % | 92 × 116 × 54 | 161 g |
| `foot_plate` | 1 | leg rig | PETG/ASA, 4 walls, 40 % | 96 × 96 × 14 | 89 g |
| `cell_riser` | 1 | leg rig | PETG/ASA, 4 walls, 60 % | 33 × 23 × 16 | 12 g |

**The bench itself is ~725 g**, the optional encoder pair 24 g, the leg rig 262 g. Every
bbox fits a 256 mm bed; the frame is the only tall print at 258 mm.

`bench_frame` is 575 g of that and it is deliberate — it is the structure, the footprint
and the ballast at once, and the frame is far more over-strength than it needs to be
(inter-layer SF ≈ 91, **0.027° of flex at stall**) precisely so that it is not what you are
identifying. The legs are plates rather than boxes because the joint's torque bends them
*in their own plane*, which is a plate's stiff direction; a shelled box of the same outline
cost 300 g and bought nothing.

Print `mini_dog.py`'s own `servo_gauge` first, as for the robot: it is what says a real
ST3215 slides into this sleeve.

## Buy

| item | qty | for | note |
|---|---|---|---|
| ST3215 + both stock aluminium hubs | 1 (3+ for a spread) | bench | every fitted parameter is per servo and they differ |
| Waveshare URT-1, or an ESP32 + a TTL half-duplex driver | 1 | bench | set the FTDI latency timer to **1** — it ships at 16 ms and that alone eats the control tick |
| **adjustable** supply, 9–13 V, ≥ 3 A, with a current readout | 1 | bench | one of the two things `robot/README.md` says decides the quality of the fit. ≥ 3 A because a stalled ST3215 draws 2.7 A |
| known payload ~0.25 kg and ~0.4 kg, or shot/nuts | 2 | bench | what goes in `mass_cup` — the cup holds 90 cm³, about 0.42 kg of steel shot or 0.63 kg of lead |
| scale, 1 g or better | 1 | bench | for the payloads **and** both printed arms |
| M6 × 60 + M6 nut + 2 washers (payload → station) | 2 | bench | |
| M2.5 × 6 / × 7 (fork → the two hubs, as the robot) | 4 / 4 | bench | |
| M3 × 10 + M3 nut (sleeve thrust clamp) | 2 | bench | and not longer |
| M4 × 30 + M4 nut + washer (base plate → board) | 6 | bench | stable on its own weight at full load (0.55 N·m restoring against 0.23) but a free swing is a dynamic reaction |
| plywood/MDF ≥ 400 × 250 × 18 + 2 G-clamps | 1 | bench | the stand is printed; the ground is not |
| cable tie, 2.5 mm | 6 | bench | servo lead off the head, one over the cup's mouth |
| M2.5 × 8 (magnet cap + fork arm → driven hub) | 4 | optional | 2.5 of cap and 4.0 of arm to clear, 2.5 of hub to thread into. **Not longer** — past the hub it bottoms on the case |
| AS5600 breakout + ⌀6 × 2.5 **diametric** N35 magnet | 1 | optional | an axial magnet of the same size reads as a constant and looks like a dead sensor. Board outline **verify** |
| M3 × 16 + M3 nut (bridge → the +y leg) | 2 | optional | |
| straight-bar load cell, 5 kg, + HX711 | 1 | leg rig | outline **verify** |
| M4 × 20 / M5 × 16 / M4 × 25 / M6 × 35 | 2/2/2/4 | leg rig | cell, riser, board, tower |

**Power the servo from the supply directly**, not through the URT-1 and not through USB.
Share the ground and nothing else.

### Why the supply has to be adjustable

At one voltage the back-EMF damping and the viscous friction are the same column of the
regressor and no amount of data separates them; `fit_bam.py` checks for this and says so
rather than returning a confident number. Run every driven trajectory at three voltages —
`robot/README.md`'s own examples use 12.6 and 11.1, i.e. a 3S pack's range.

## The encoder pair is optional

`robot/bench` fits everything off the servo's own telemetry, which is the point of fitting
what the robot can actually observe: a parameter the robot cannot see is a parameter the
policy cannot use. So `enc_magnet_cap` and `enc_bridge` are **not required**.

They are here for the one question the servo cannot answer about itself. `--traj rock` asks
whether Present Position moves when you rock the arm against the play; if it does *not*,
only an external angle says whether the **joint** moved — that is, whether the encoder sits
before the gearbox or the joint is simply locked. `rl/actuator.py`'s `enc_after_backlash`
default assumes the former and step 4's observation wiring depends on it.

The bridge is referenced to the **leg**, not the base plate, so what the frame does under
load is largely common mode between the sensor and the servo case. Its mounting holes are
slots in y: set the gap by sliding it until the AS5600's AGC register sits mid-range, which
is the only honest way to set a magnetic air gap. Nominal is 2.5 mm magnet-to-IC, inside
the 0.5–3 mm the part wants; the slot gives ±1.5 mm and its inboard end is a hard stop
against the cap.

## Assembly, and the order that matters

1. **`servo_gauge` first** (`mini_dog.py`'s) — a real ST3215 has to slide into the sleeve
   and the ⌀14 pattern has to line up.
2. Bolt `bench_frame` to the board through six M4.
3. Two M3 nuts into the thrust lug's channels → servo up into the sleeve from underneath,
   connector into the head and the lead out of the top → both thrust bolts in, evenly,
   until the case stops rocking. Same order as a leg.
4. Both aluminium hubs onto the servo, then the arm onto the hubs: M2.5 × 6 driven side,
   M2.5 × 7 passive side.
5. Payload into `mass_cup`, cup over the beam at the station, one M6 × 60 through both.
   **Weigh the filled cup.** The mouth faces along the joint axis, which is horizontal at
   every swing angle, so nothing can fall out however far the arm goes over — the tie over
   the mouth is a dust cover, not a retainer.
6. *Optional:* magnet into `enc_magnet_cap` (press fit; the fork arm backs it up, so
   nothing relies on glue), cap onto the **+y** fork arm swapping that arm's four M2.5 × 6
   for M2.5 × 8 → two M3 nuts into the +y leg's channels, which open on its outer face →
   `enc_bridge` on the leg's inner face, two M3 × 16 through its slots.

### Setting `--centre`, which is not a detail

`--centre` is the encoder count at q = 0, and q = 0 has to be **true vertical**, because
the whole torque scale is `m·g·r·sin(q)`. Two degrees of mounting error is 3.5 % on the
first `hold` point and it biases every electrical parameter downstream of it.

Do not eyeball it. Torque off, release the arm from about +30°, let it settle, read Present
Position; release it from −30°, read it again; **average the two**. The stiction band
brackets true vertical and the average sits in the middle of it. That also gives you the
band's width for free, which is a first look at the same `tau_c` the free swing will fit.

## Checks that are failures, not warnings

`servo_bench.py` ends with these, in the same spirit as `mini_dog.py`'s `body clear:` line.
Each exists because something it now catches was actually wrong while this was being built:

- **`frame clear:`** — `bench_frame` and `enc_bridge` are bolted into one rigid stand and
  nothing else notices when they occupy the same solid. It caught the bridge reaching up
  into the head beam.
- **`swing`** — the free travel of each arm against the frame, on real solids, compared
  against `sweep.py`'s own abort angle. It runs against `bench_frame(hollow=False)`, the
  frame's *silhouette*, because a per-angle interference scan is static and an arm can
  otherwise appear inside a lightening hole it could never reach through.
- **`arm clearance:`** — the long arm and its loaded carrier against the base plate at the
  bottom of the swing, measured off the real solids rather than from a nominal length.
- **the two arm bounds** — `m·g·r·sin(q₀) > 2·tau_c` on every config, and
  `J_load < J_m` on the free-swing ones, both against `../rl/params/st3215.json`. These are
  the ones that decide whether the fit works at all.
- **`cell floats:`** — leg rig. `foot_plate` and `cell_riser` must not share solid; a
  bending-beam cell whose ends are bridged reads a fraction of the force and looks exactly
  like one that works.
- **`flange bolts … clear`** — leg rig. The tower's bolt nuts against the hip-roll fork's
  swept circle. The flange does not exist in the robot's own model, so `mini_dog.py`'s ROM
  scan cannot see it.

## The leg rig is not part of `robot/bench`

`leg_tower` + `foot_plate` + `cell_riser` hold **a whole leg** at the robot's own geometry
over a 5 kg load cell. That protocol is single-servo throughout, so this is an addition:
somewhere to check a fitted `rl/actuator.py` against a real leg, where the knee carries the
shin's inertia while the hip is somewhere a single-joint bench never put it. It is also the
only place the servo's reported load can be calibrated against real newtons, which is what
`smalldog_walker/contact.py` wants for foot contact and nobody has.

The cradle is `mini_dog.roll_module()` itself, moved onto a flange, so the leg hangs off
exactly the structure it hangs off on the robot. Bolt the flange to a board or a 2020
upright with the leg past the edge; the roll axis wants to be **187 mm** above the plate's
top face, with the plate at x +24, y +76 from the flange — off the *posed* solid, not from
`FOOT_Z`, which is 13 mm out because the stance pose swings the foot up and forward.

Build it after the servo bench, or not at all.

## What is unconfirmed

Marked **verify** in `servo_bench.py` and genuinely so: the AS5600 breakout outline and
hole pitch (`ENC_PCB`, `ENC_PCB_HOLES`), and the load cell's bar and its bolt pattern and
threads (`CELL_*`).

Two constants also live twice, for the same reason `mini_dog.py`'s LiDAR scan pattern does
— nothing in `3d/` can import `robot/` (it pulls in pyserial) and nothing in `robot/` can
import CadQuery: `SWEEP_QMAX`, `SWEEP_ABORT` and `SWEEP_START` here mirror `sweep.py`'s
`--qmax`, its travel abort and `traj_freeswing(start=)`. **Change both, in the same pass.**
