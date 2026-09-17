# Self-charging

An idea, not a spec. Nothing here is built; every number is a first estimate unless it
points at a file.

## The question

Can the dog charge itself? Today the only thing that leaves the tray is one XT30 that a
hand plugs in (`POWER.md`, the tree). `ros2/BEHAVIOUR.md` already lists "don't die" —
battery low → walk to the charger corner, lie down — and its "done when: plugged in" is
the part with no hardware.

## Verdict

Possible, and the design is already most of the way there. It needs a dock with two
contacts, two pads on the robot's belly, and the walk-to-dock behaviour. Nothing on the
robot's power tree changes; the dock is the bench supply with a different plug.

## Why the current tree allows it

- **The BMS is same-port on purpose** (`POWER.md`, *The BMS is same-port*). Current pushed
  into the fused P+ node from outside goes through both FET banks, so charging while the
  Pi and the LiDAR run is protected. The dock is exactly the bench-supply case that decided
  same-port.
- **The charge branch has its own 7.5 A fuse.** Dock contacts go in parallel with the
  XT30 on that branch. Charge current stays ≤ 5 A.
- **"I am charging" is free.** Every servo reports bus voltage each tick
  (`PRESENT_VOLTAGE`, read by `robot/runtime/safety.py`). On the dock it rises to the
  charger's 12.6 V. `volt_max` is 13.2, so a 12.6 V CC/CV source does not trip it.
- **The dock is a landmark.** The L2 sees a small V-shaped target; Nav2 gets the robot to
  ~0.3 m; the last 20 cm is dead reckoning into a funnel.

## What to build

### Dock

- A 12.6 V CC/CV module (3 A is enough) off a laptop brick. The BMS protects, it does not
  charge: the CC/CV stage lives in the dock, never on the robot.
- Two spring contacts in a printed funnel. The funnel takes ±3 cm and ±10° of arrival
  error; the LiDAR gives better than that at 0.3 m.
- **Contacts are dead until the robot is on them.** An ID resistor across the robot's pads
  (or a third sense pad) closes a relay in the dock. Exposed 12.6 V pads on a floor are
  otherwise a short waiting for a coin.
- Asymmetric pads (two sizes, or offset from the centreline) so polarity cannot flip.

### Robot

- **Two pads on the belly**, wired to the XT30 branch behind the 7.5 A fuse.
- The rear wall is out: the roll servos sit 9.5 mm behind it and the XT30 already takes
  the only free band (`3d/README.md`, *End-wall openings*).
- The belly works because the dog **lies down** on the dock: no alignment in z, and lying
  is already the sleep pose. **verify** what the lowest point is when lying — it has to be
  the tray floor, not a folded shin or the low cable window (`PANEL_LOW`, z = −19, on the
  centreline at both ends — pads go outboard of it).
- Pads are a CAD change: two pockets in the tray floor plus a `pad_clear()` probe against
  the folded legs. Their mass goes into `ELECTRONICS_KG`, weighed, not estimated.

### Software

1. Publish the battery: bus voltage → state of charge (the open item in
   `ros2/BEHAVIOUR.md`). A 3S pack reads 12.6 V full and 9.9 V empty, and sags at every
   footfall, so read it standing still or filter over seconds.
2. "don't die": trigger below ~10.5 V at rest; walk to the dock; lie down; torque-off.
   Done when bus voltage ≥ 12.4 V for 5 s. If the voltage has not risen 10 s after lying
   down, stand up, back off 20 cm, try once more, then give up and call for a hand.
3. Charge time: 3S2P of P42A ≈ 8.4 Ah, ~93 Wh. At 3 A with the servos off it is about
   3 h from empty; the Pi + LiDAR take ~1 A of the 3.

## What not to do

- **Wireless (Qi).** 15 W into 93 Wh is 6+ hours, and the coils need ±10 mm — worse than
  contacts on every axis, and the coil and shield are mass on the belly.
- **The robot plugging its own XT30.** ~1 mm of alignment and a real push; there is no
  manipulator, and the connector's 0.8 mm lip is sized for unplug force only.
- **A fuse in the shared negative.** The two-fuse rule in `POWER.md` stands: the 7.5 A
  fuse sits in the charge branch's positive, the 30 A MIDI in the trunk.

## Open

- Does the YH2204A-class BMS balance at all, and at what current? (verify) If it is weak,
  a balance charge with the module out over the JST-XH still happens by hand now and then.
- Lowest point when lying (verify, on the floor with the real robot).
- Contact material: brass pads on the robot, spring pogo pins on the dock, or the reverse.
  Pogo pins on the moving part collect dirt.
- Is `volt_max` 13.2 enough margin against the charger's CV setpoint plus the fuse's drop
  read from the wrong side? Measure on the bench before it goes in the loop.

## Review questions

