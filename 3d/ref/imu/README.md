# BMI088 breakout

The outline and the overall thickness are measured off the board in hand; the rest is
still the nominal for a generic BMI088 breakout. Treat the file the way
`ref/camera/README.md` asks you to treat its flagged readings: a dimension that measures
different is a dimension to correct here and re-run, never one to shave in `mini_dog.py`.

| what | value | in `mini_dog.py` | confidence |
|---|---|---|---|
| PCB outline | 20.5 × 10.6 mm | `IMU_BOARD[0:2]` | measured |
| PCB thickness | 1.6 mm | `IMU_BOARD[2]` | 1.6 is the industry default; low risk |
| component stack over the PCB | 1.0 mm (2.6 overall, measured) | `IMU_STACK` | measured; see the header note |
| mounting | no holes — double-sided tape, ~0.5 mm | `IMU_TAPE` | **verify** the tape |
| mass | 3 g | `IMU_KG` | **verify** — weigh it |

## The header is not optional to leave off

`IMU_STACK` = 1.0 mm is a **headerless** board: the BMI088 package, its passives and
nothing else. A 2.54 mm pin header is 8.5 mm tall on its own and does not fit: the board
lies face up on the Orange Pi case's bottom plate, under the Pi, in the 8 mm between
plate and PCB (`imu_clear()` reads 5.4 mm over the package, 1.2 mm to the case's spacer).

Solder the six wires (VCC, GND, SDA/SDI, SCL/SCK, and the two chip selects — the BMI088
is two devices, accelerometer and gyroscope, on one bus) directly to the pads, and run
them to the Pi's own I2C pads on its underside — they are 40 mm away, on the same side of
the board. The component face looks **up**, at the Pi.

## Where it sits

`imu_in_case_top.png` — the Orange Pi's case seen from above through the board: the IMU
taped face up on the bottom plate, its edges 9.25 / 29.75 mm from the case's micro-SD
face and 14.95 / 25.55 mm from the side face opposite the 40-pin header. Drawn from the
model (`IMU_X, IMU_Y` = 0, +15); the M.2 band on it is the vendor drawing's, not measured.

## Orientation

The model places the sensor's package at `imu_xyz()` and gives the `imu` site no rotation:
the board's axes are the robot's. Mount it with +X forward and +Z up. A board fitted
rotated is not a calibration problem to fix in software — it is a different robot from the
one both simulators export, which is the whole reason the mounting point is a constant in
`mini_dog.py` and not a note in a launch file.
