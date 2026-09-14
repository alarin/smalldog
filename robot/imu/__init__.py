"""imu/ — the BMI088 on the deck, and the one filter that turns it into what the
policy reads: projected gravity, body rates, proper acceleration.

`bmi088.py` is the driver (SPI, two chip selects) and `Attitude` the complementary
filter. `robot/runtime/policy.py` is the consumer. Nothing here knows about the
servos, and nothing in `runtime/` reads a register of the IMU.
"""
