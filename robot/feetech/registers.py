"""
registers.py — the Feetech SMS/STS control table, in one place.

Transcribed from the ST3215 documentation and the SCServo Arduino library that
3d/ref/st3215_wiki.html points at. It is a transcription, not a measurement, so
treat it the way this repository treats every other vendor number: verify it
before building on it. `python -m robot.feetech.bus --dump <id>` prints every
register below against a live servo, and a value that cannot be right (a
temperature of 200, a voltage of 0.3) means the address is wrong for your
firmware, not that the servo is broken.

Two encoding traps, both of which produce plausible-looking wrong numbers:

  * **Byte order.** The STS/SMS family is little-endian; the older SCS family is
    big-endian at the same addresses. A big-endian read of position 2048 gives
    2048 as well — the trap only bites off-centre, which is where you stop
    looking. `Bus(endian=...)` exists for this; the default is STS.

  * **Sign-magnitude, and not all at the same bit.** Speed, load, current and
    offset are NOT two's complement, and the trap has two floors. Read one as
    two's complement and a small negative speed comes back as ~32768 — that is
    the obvious half. The second is that the sign is at bit 15 for speed and
    current but at **bit 10 for load** and **bit 11 for offset**, because those
    two registers are narrower than 16 bits: load is per-mille of torque and
    offset is +-2047. Assume bit 15 for all of them and the wrong ones do not
    fail, they just come back positive and plausible — a load of -388 reads as
    +1412 and an offset of -1997 reads as +4045. `SIGN_BIT` below is the map,
    and three of its five entries are measured rather than transcribed.

Units are the servo's own. Conversion to SI lives in `Servo`, not here, so this
file stays a pure transcription.
"""

# ------------------------------------------------------- EPROM (persistent)
ID                  = 5
BAUD_RATE           = 6
RETURN_DELAY        = 7
RESPONSE_LEVEL      = 8
MIN_ANGLE_LIMIT     = 9      # 2
MAX_ANGLE_LIMIT     = 11     # 2
MAX_TEMPERATURE     = 13
MAX_VOLTAGE         = 14
MIN_VOLTAGE         = 15
MAX_TORQUE          = 16     # 2
PHASE               = 18
UNLOADING_CONDITION = 19
LED_ALARM_CONDITION = 20
P_COEF              = 21
D_COEF              = 22
I_COEF              = 23
STARTUP_FORCE       = 24     # 2  "punch": the minimum drive outside the dead zone
CW_DEAD             = 26
CCW_DEAD            = 27
PROTECTION_CURRENT  = 28     # 2
ANGULAR_RESOLUTION  = 30
OFFSET              = 31     # 2, sign-magnitude
MODE                = 33     # 0 position, 1 velocity, 2 pwm, 3 step
PROTECTIVE_TORQUE   = 34
PROTECTION_TIME     = 35
OVERLOAD_TORQUE     = 36
SPEED_P             = 37
OVERCURRENT_TIME    = 38
SPEED_I             = 39

# --------------------------------------------------------- SRAM (volatile)
TORQUE_ENABLE       = 40
ACCELERATION        = 41
GOAL_POSITION       = 42     # 2
GOAL_TIME           = 44     # 2
GOAL_SPEED          = 46     # 2
TORQUE_LIMIT        = 48     # 2
LOCK                = 55
PRESENT_POSITION    = 56     # 2
PRESENT_SPEED       = 58     # 2, sign-magnitude
PRESENT_LOAD        = 60     # 2, sign-magnitude
PRESENT_VOLTAGE     = 62
PRESENT_TEMPERATURE = 63
ASYNC_WRITE_FLAG    = 64
SERVO_STATUS        = 65
MOVING              = 66
PRESENT_CURRENT     = 69     # 2, sign-magnitude

#: One contiguous read from PRESENT_POSITION covers everything the bench and the
#: runtime want, current included — 15 bytes, one round trip per servo instead
#: of six. The gap at 67..68 is read and thrown away; that is far cheaper than
#: the extra transactions.
FEEDBACK_START, FEEDBACK_LEN = PRESENT_POSITION, 15

#: Sign-magnitude registers, mapped to **which bit carries the sign**. It is not
#: bit 15 for all of them, and assuming so is how a -388 load reads back as +1412.
#: Three of these were measured on a real ST3215 on 2026-09-07, not transcribed:
#:
#:   PRESENT_SPEED  bit 15. Driving one knee each way gave raw 1200 and 33968,
#:                  and 33968 - 32768 = 1200.
#:   PRESENT_LOAD   bit 10. The same two moves gave raw 1412 and 392, and
#:                  1412 - 1024 = 388. The magnitude is the vendor's 0..1000
#:                  per-mille of torque, so it never needs more than ten bits.
#:   OFFSET         bit 11, range +-2047. Twelve servos centred by their own
#:                  middle-position command came back reading 3049, 4045, 3995
#:                  and 3900 where the arithmetic predicted -1001, -1997, -1947
#:                  and -1852: 2048 apart, every one.
#:
#: GOAL_SPEED and PRESENT_CURRENT are still the vendor's word — **verify**. The
#: current one is nearly harmless because `Servo.decode` takes its absolute
#: value; the goal-speed one would be a sign error on every write that used it.
SIGN_BIT = {PRESENT_SPEED: 0x8000, GOAL_SPEED: 0x8000, PRESENT_CURRENT: 0x8000,
            PRESENT_LOAD: 0x400, OFFSET: 0x800}
#: The keys alone, for anything that only asks "is this register signed at all".
SIGN_MAGNITUDE = set(SIGN_BIT)

#: Width in bytes; anything not listed is one byte.
WIDTH = {MIN_ANGLE_LIMIT: 2, MAX_ANGLE_LIMIT: 2, MAX_TORQUE: 2, STARTUP_FORCE: 2,
         PROTECTION_CURRENT: 2, OFFSET: 2, GOAL_POSITION: 2, GOAL_TIME: 2,
         GOAL_SPEED: 2, TORQUE_LIMIT: 2, PRESENT_POSITION: 2, PRESENT_SPEED: 2,
         PRESENT_LOAD: 2, PRESENT_CURRENT: 2}

#: Everything the fit must be told, because the fit is only valid for these
#: settings and the robot must then run with the same ones. sweep.py stamps
#: them into every csv and fit_bam.py refuses to merge runs that disagree.
CONTROL_REGISTERS = ["P_COEF", "D_COEF", "I_COEF", "STARTUP_FORCE", "CW_DEAD",
                     "CCW_DEAD", "MODE", "ACCELERATION", "GOAL_SPEED",
                     "TORQUE_LIMIT", "MAX_TORQUE", "PROTECTION_CURRENT",
                     "OVERLOAD_TORQUE", "PROTECTIVE_TORQUE", "OFFSET",
                     "MIN_ANGLE_LIMIT", "MAX_ANGLE_LIMIT", "ANGULAR_RESOLUTION"]

# ------------------------------------------------------------ instructions
PING, READ, WRITE, REG_WRITE, ACTION, RESET = 0x01, 0x02, 0x03, 0x04, 0x05, 0x06
SYNC_READ, SYNC_WRITE = 0x82, 0x83
BROADCAST_ID = 0xFE

# ------------------------------------------------------------- conversions
COUNTS_PER_TURN = 4096
#: LSB of Present Current. **verify** against the INA226 on the bench: the
#: figure quoted for this family is 6.5 mA, and the whole electrical half of the
#: fit is scaled by it, so it is not a number to inherit from a forum post.
CURRENT_LSB_A = 0.0065
VOLTAGE_LSB_V = 0.1
#: Present Speed is in counts/s on STS. **verify**: some firmware reports
#: 0.732 rpm units instead, which is a factor of 50 and very visible.
SPEED_LSB_COUNTS_PER_S = 1.0
