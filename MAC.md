# The mac's queue — CAD changes that have never been run

Three machines share this repository and nothing else (`rl/CLAUDE.md`, "The machine
split"):

| machine | what it is for |
|---|---|
| **this one**: a mac | the CAD — `3d/`, and everything generated out of it |
| Windows + WSL2, RTX 3070 | training — `rl/`, plus the pure-Python MuJoCo sim. See [`WSL.md`](WSL.md) |
| an Orange Pi 5 Pro | the robot — `robot/runtime`, step 7 |

`WSL.md` is a permanent runbook for its machine. **This file is not**: it is a queue. Five
commits changed `3d/` from a session that had no CadQuery, so nothing below has been
through a single boolean. Two of them change printed geometry. Work the ladder in
`3d/CLAUDE.md`, "Verifying a change" — the order is load-bearing — and delete the sections
here as they come back green.

## What is queued, and what each one risks

| commit | what it did | printed geometry? |
|---|---|---|
| `93f9e93` | clamp screw is now headless; `thrust_bolts()` into all three `rom_scan` calls; `thrust_clear()` | no — hardware and checks only |
| `12e9ef1` | the hub is tapped **M3**, not M2.5: fork clearance holes ⌀2.9 → ⌀3.4 | **yes**, all 12 forks |
| `a52f843` | `fork_access()` — a driver probe on each fork screw's axis | no — a check only |
| `7e30e75` | `fork_access_bores()` — ⌀6 through the tray front wall and the gusset skin | **yes**, `chassis_bottom` |
| `b4d7e4a` | the fastener BOM, recounted off the geometry | no — docs |

All of it is on `claude/equipment-delivery-timeline-4gp7lq`.

## Step 0 — take the baseline first, or step 4 is worthless

`fea.py` is a comparison, not an absolute: the number that matters is whether the
inter-layer SF **dropped**, and the changes are already committed, so the "before" has to
come out of git. `b4d7e4a` is the last commit whose `mini_dog.py` is the old one.

```bash
git worktree add /tmp/smalldog-base b4d7e4a
cd /tmp/smalldog-base/3d && ~/smalldog/3d/.venv/bin/python fea.py --all | tee /tmp/fea-before.txt
```

The worktree has no `.venv` of its own — run the main checkout's interpreter from inside
it, which is why the paths above are absolute. Keep `/tmp/fea-before.txt`.

## Step 1 — `.venv/bin/python mini_dog.py`

Every part valid, and then read four lines. Two are new and two must not have moved.

**New, and they are the point of the whole batch:**

```
clamp clear: M3 x 10 set screw reaches r = 21.57 vs the spine's 23.0 (+1.43 mm)
fork access: all six arms break out within 7.5 mm (roll/passive through @6 bores)
```

`r = 21.57` is the lug, not the screw — headless, the screw comes to 20.96 and the lug is
the binding term again, which is the design intent restored. A `!! THRUST CLAMP` line means
`THRUST_HEAD_*` disagrees with the fastener; a `!! FORK ACCESS roll/passive` line means the
bores did not land where the screws are, and that is the one to debug first because the
legs cannot be bolted on without them.

**Must be unchanged:**

```
hip_roll   free  -90 ..  +90        body clear: ...
hip_pitch  free  -90 ..  +90        imu clear: ...
knee       free -110 .. +110        foot bolt: ...  gps clear: ...  lidar clear: ...
```

The ROM is the one to watch. `thrust_bolts()` is now in the static side of all three scans,
so if a clamp screw fouls anything the ROM is where it shows — and the ROM feeds
`joint_limits_rad` in `bom.json`, which feeds the gait's clamps and `robot/runtime`'s soft
limits. A moved ROM is not a cosmetic change.

Total mass should come back **~1.2 g lighter** than 2.496 kg: 96 hub holes going ⌀2.9 → ⌀3.4
is 0.95 cm³, plus a little from the access bores.

## Step 2 — bboxes in the same table

Nothing here moved a part's outline. Any bbox change is a bug, not a result.

## Step 3 — `.venv/bin/python render.py`, and actually look

Look at the hip specifically: four ⌀6 bores now break through the tray's front wall on the
outboard side of each roll cradle, and the outboard one of the four sits 0.2 mm off the
tray's side wall. If it has eaten into the side wall's outer skin, that is a tangency and
it will read as a knife edge in the render. It should not — the bores stop exactly on the
tray's inner face for that reason — but this is the cheapest place to catch it.

## Step 4 — `.venv/bin/python fea.py --all`, against `/tmp/fea-before.txt`

**This is the gate.** Judge on the **inter-layer** SF, the second of the `SF xy / z` pair.

The change under test is the hub holes in the fork arms, and `hip_bracket_A` / `thigh_A` /
`shin_A` are all fixed at exactly those holes (`hub_clamp()`), so the case is well posed.
Two effects that point opposite ways, which is why this is measured and not argued:

* the ligaments shrink — 2.35 → 2.10 mm from the hole to the arm's central bore, 5.05 →
  4.80 mm out to the rim;
* the bearing area grows — 10.0 → 12.0 mm² per screw, so bearing stress on the PETG under
  the same joint torque falls 17 %, and the screw goes M2.5 → M3 in shear.

Neutral-to-better is the expectation. **Any drop in inter-layer SF is a regression** — fix
it or report the before/after explicitly, per `3d/CLAUDE.md`.

### The gap in this step, and it is a real one

`fea.py --all` covers `shin_A`, `thigh_A`, `hip_bracket_A` and **nothing else**.
`part_specs()` has no entry for `chassis_bottom`, so **the access bores are not covered by
any FEA case in this repository** — and they are cut into the root gusset's inner skin and
the tray's front wall, which is the leg's root load path.

Two honest ways forward, and it is a judgement call:

1. add a `chassis_bottom` spec to `part_specs()` — fixed at the deck bosses, loaded at the
   roll sleeve — and run it before and after. It is the right answer and it is an hour;
2. accept it and watch the printed part at the hip, on the grounds that the bores sit in a
   2.8 mm internal diaphragm well inside a 63 × 30.7 mm block whose outer walls carry the
   load. Plausible, unmeasured.

Do not print four chassis and find out.

## Step 5 — `.venv/bin/python export_sim.py --check`

Rebuilds `out/sim/` and stands the robot up in MuJoCo for 3 s. Pass is
`4 feet down, upright +1.00` and a base height near the CAD stance. The masses moved by
~1.2 g so the numbers will not be bit-identical to the last run; the stance should be.

## Step 6 — regenerate the ROS 2 description, then the consumers

```bash
cd ros2 && ../3d/.venv/bin/python smalldog_description/scripts/generate_model.py
```

Masses and inertias moved, so `robot_params.json` moves with them, and three things read
it:

```bash
cd rl   && python checks/check_model.py && python checks/check_model.py --terrain
cd ros2 && ../3d/.venv/bin/python tools/standalone_sim.py --headless
cd robot && python runtime/walk.py --dry-run --profile
```

The last one is the new 50 Hz runtime and needs no hardware — it reads the same
`robot_params.json` and will say so if the joint order or the limits moved under it. Its
own selftests (`runtime/calib.py`, `runtime/safety.py`, `runtime/loop.py`, each
`--selftest`) are pure stdlib and run anywhere.

## Step 7 — the bench fixture, which also changed

```bash
cd 3d && .venv/bin/python bench_rig.py
```

`hub_face()` follows the hub to M3, so its own asserts (bore, hubs, swing clearance) have
to pass again. Not part of the robot — no FEA, no sim consumer — but `robot/bench` cannot
run without it.

## Hardware notes that go with this batch

* **Thrust clamp: M3 × 10 set screws**, not cap screws. A cap head reaches r = 24.2 into an
  annulus the fork spine sweeps from 23.0 and binds the joint over ±2…38°. Found on the
  bench, then reproduced in the arithmetic.
* **Hub screws are M3 × 6 driven / M3 × 7 passive.** Ceilings 6.5 and 7.15; past the hub
  they bottom on the case and the vendor FAQ says that burns servos. A legs plate already
  printed does **not** need reprinting — the hole only grew, so drill the eight per joint
  out to 3.3–3.4.
* **The hip's inboard fork screws are fitted from inside the tray**, through the ⌀6 bores,
  with a **ball-end** hex key tilted inboard — deck off, before the battery goes in. There
  is no straight run to the outboard one of the four and there cannot be; see
  `3d/README.md`, "Reaching the fork screws".
