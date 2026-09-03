"""runtime/ — the 50 Hz control loop and the safety layer. Step 7.

The three pieces are separate because they fail separately:

    calib.py    which servo is which joint, where its zero is, which way it turns
    safety.py   the limits, and the one place that decides to cut torque
    loop.py     the tick: read feedback, ask a controller, write goals, hold 50 Hz
    walk.py     the CLI that wires the trot gait to all of the above

`loop.Runtime` does not know what a trot is. It takes a callable
`source(dt, feedback) -> 12 joint angles` and does the same thing whether that is
`smalldog_walker`'s analytic gait (today) or the ONNX policy out of `rl/` (step 6,
once there is an IMU and a servo fit). That boundary is the point of the file: the
safety layer and the bus bookkeeping must not be written twice, once per
controller.

Nothing here needs a robot. Every module has a `--selftest` or a `--dry-run` that
drives `feetech.loopback`, so the loop's arithmetic is exercised on every run
rather than on the day the servos arrive.
"""
