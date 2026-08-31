# BMI088 breakout — transcribed, not measured

The IMU is the one payload on this robot that arrived in the model before it arrived on
the bench. Every number here is a **nominal** for a generic 20 × 15 mm BMI088 breakout,
transcribed from vendor listings rather than measured off the part in hand, and the model
is built on it. Treat the whole file the way `ref/camera/README.md` asks you to treat its
two flagged readings: a dimension that measures different is a dimension to correct here
and re-run, never one to shave in `mini_dog.py`.

| what | nominal | in `mini_dog.py` | confidence |
|---|---|---|---|
| PCB outline | 20.0 × 15.0 mm | `IMU_BOARD[0:2]` | **verify** |
| PCB thickness | 1.6 mm | `IMU_BOARD[2]` | 1.6 is the industry default; low risk |
| component stack over the PCB | 1.2 mm | `IMU_STACK` | **verify**, and see the header note |
| mounting holes | 2 × M2.5, 15.0 mm apart on the long axis | `IMU_HOLE_P` | **verify** |
| mass | 3 g | `IMU_KG` | **verify** — weigh it |

## The header is not optional to leave off

`IMU_STACK` = 1.2 mm is a **headerless** board: the BMI088 package, its passives and
nothing else. A 2.54 mm pin header is 8.5 mm tall on its own and does not fit in this bay
by a factor of three — the slot the board lives in is 3.6 mm from the battery pack's top
to the deck's underside, and the board and its components already spend 2.8 of that.

Solder the six wires (VCC, GND, SDA/SDI, SCL/SCK, and the two chip selects — the BMI088
is two devices, accelerometer and gyroscope, on one bus) directly to the pads, on the
side the model puts them: the component face looks **down**, at the battery. Fit a header
and the deck will not close.

## Orientation

The model places the sensor's package at `imu_xyz()` and gives the `imu` site no rotation:
the board's axes are the robot's. Mount it with +X forward and +Z up. A board fitted
rotated is not a calibration problem to fix in software — it is a different robot from the
one both simulators export, which is the whole reason the mounting point is a constant in
`mini_dog.py` and not a note in a launch file.
