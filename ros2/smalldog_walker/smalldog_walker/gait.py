"""Trot gait generator for SmallDog. Pure Python.

Diagonal pairs move together: (FL, RR) and (FR, RL), 180 deg out of phase,
duty 0.5. Foot targets are produced in base_link coordinates, then handed to
LegKinematics per leg.

Command: body velocity (vx, vy, wz) in the base frame + body height.

Terrain feedback is optional and additive: call `feedback()` with the body attitude and
the foot contacts and the same trot closes three small loops on top of the open-loop
profile (attitude levelling, contact-triggered touchdown, probing for ground that is not
where it was expected).  Supply nothing - or let the sensors drop out - and it degrades
back to exactly the blind gait, which is what it was before.
"""
import math
from .leg_kinematics import LegKinematics

LEGS = ("fl", "fr", "rl", "rr")
PHASE = {"fl": 0.0, "rr": 0.0, "fr": 0.5, "rl": 0.5}

class TrotGait:
    def __init__(self, params):
        p = params
        self.legs = list(p["legs"])
        self.joint_names = list(p["joint_names"])
        mm = 1e-3
        self.hip = {l: [v * mm for v in p["hip_xyz_mm"][l]] for l in self.legs}
        self.kin = {}
        for l in self.legs:
            dy, dz = p["hip_to_pitch_mm"][l][1] * mm, p["hip_to_pitch_mm"][l][2] * mm
            self.kin[l] = LegKinematics(dy, dz, p["l_thigh_mm"] * mm, p["l_shin_mm"] * mm,
                                        p["foot_r_mm"] * mm)
        self.foot_r = p["foot_r_mm"] * mm
        # nominal foot placement in the base frame, directly under the pitch axis
        self.nominal = {}
        for l in self.legs:
            hx, hy, hz = self.hip[l]
            dy = p["hip_to_pitch_mm"][l][1] * mm
            self.nominal[l] = [hx, hy + dy, 0.0]          # z filled from body_height

        # ---- reachable band of one leg, measured from the hip-pitch axis -------------
        # the knee soft limit sets how short the leg can get, full extension how long
        l1, l2 = p["l_thigh_mm"] * mm, p["l_shin_mm"] * mm
        self.limits = dict(p.get("joint_soft_limits_rad", p["joint_limits_rad"]))
        self.hard_limits = dict(p["joint_limits_rad"])
        knee_max = self.limits["knee"]
        self.d_min = math.sqrt(max(1e-9, l1*l1 + l2*l2 + 2*l1*l2*math.cos(knee_max)))
        self.d_max = (l1 + l2) * 0.97
        self.pitch_dz = abs(p["hip_to_pitch_mm"][self.legs[0]][2] * mm)

        # tunables
        self.period = 0.45              # s, one full gait cycle at a walk...
        self.period_min = 0.30          # ... shortened towards this as the command rises
        self.stride_max = 0.045         # m of ground per half cycle, sets the shortening
        self.max_step = 0.060           # m, fore/aft half-stride clamp
        self.max_step_y = 0.030         # m, lateral half-stride clamp (roll range is small)
        self.stand_smooth = 0.12
        self.swing_height = 0.022
        self._body_height = 0.158
        self.body_height = self._body_height          # runs the setter -> clamps
        self.max_joint_rate = p.get("joint_velocity_limit", 4.7) * 0.85

        # ---- terrain feedback: all off until feedback() is actually called ----------
        # tuned over 12 terrain seeds, because one seed's distance is noise: the blind
        # trot spreads 490..700 mm over the same settings.  Judge a change on the mean
        # AND the spread — holding the spread down is most of what this buys.
        self.level_kp    = 1.4     # foot travel per rad of body tilt, per m of lever arm
        self.level_kd    = 0.04    # ... and per rad/s of body rate
        self.level_tau   = 0.30    # s, low-pass on the attitude: slope, not the trot's rock
        self.level_max   = 0.020   # m, clamp on the attitude term at one foot
        self.touch_from  = 0.55    # ignore contact before this: that is the foot leaving
        self.touch_hold  = 0.005   # s of contact before it counts as a landing
        self.touch_band  = 0.008   # m of early contact written off as servo lag
        self.gz_max     = 0.035    # m, how far above nominal a landing may be believed
        self.gz_tau     = 0.25     # s, decay of that memory once the foot is in the air
        self.fb_timeout = 0.20     # s without feedback() -> fall back to the blind gait
        # heading hold.  The command is in body axes, so a body that has been turned walks
        # straight ahead *of itself* and along an arc through the world; nothing measured
        # that until now.  Flat ground hides it (1.8 deg over 1.2 m); on relief the feet
        # slip on the slopes and it compounds - 1.65 m off the centreline in 25 s.
        self.yaw_kp     = 1.5      # rad/s of turn per rad of heading error
        self.yaw_kd     = 0.05     # ... and per rad/s of measured yaw rate, damping only
        self.yaw_max    = 0.5      # rad/s, clamp on the correction
        self.yaw_cmd_eps = 0.02    # rad/s of commanded turn above which the hold lets go
        # start from the model's mechanical zero so the first command is ramped too
        self._q_prev = [0.0] * len(self.joint_names)

        self.t = 0.0
        self._phase = 0.0               # cycles, integrated directly: the period moves
        self._q = {l: (0.0, p["stance_rad"]["pitch"], p["stance_rad"]["knee"]) for l in self.legs}
        self._moving = 0.0
        self._roll = self._pitch = self._wx = self._wy = 0.0
        self._roll_f = self._pitch_f = 0.0
        self._yaw = self._wzg = 0.0
        self._yaw_ref = None            # latched heading; None = not holding one
        self._contact = {l: False for l in self.legs}
        self._gz      = {l: 0.0 for l in self.legs}    # ground height under each foot, vs nominal
        self._lift    = {l: 0.0 for l in self.legs}    # that foot's current swing lift
        self._touched = {l: False for l in self.legs}  # has this swing found ground yet
        self._airborne = {l: False for l in self.legs}
        self._con_t   = {l: 0.0 for l in self.legs}    # s this foot has felt the current contact
        self._fb_age = 1e9

    # ------------------------------------------------------------------
    @property
    def body_height(self):
        return self._body_height

    @body_height.setter
    def body_height(self, h):
        """keep the whole stride inside the leg's reachable band, apex included."""
        lo = self.d_min + self.swing_height + self.pitch_dz
        hi = math.sqrt(max(1e-9, self.d_max**2 - self.max_step**2)) + self.pitch_dz
        if lo > hi:                       # swing too tall for this leg: shrink it
            self.swing_height = max(0.0, hi - self.d_min - self.pitch_dz)
            lo = self.d_min + self.swing_height + self.pitch_dz
        self._body_height = min(hi, max(lo, float(h)))

    def reach_info(self):
        return dict(d_min=self.d_min, d_max=self.d_max,
                    height_min=self.d_min + self.swing_height + self.pitch_dz,
                    height_max=math.sqrt(max(1e-9, self.d_max**2 - self.max_step**2))
                               + self.pitch_dz,
                    body_height=self._body_height, swing_height=self.swing_height)

    # ------------------------------------------------------------------
    def feedback(self, quat=None, gyro=None, contact=None):
        """Hand the gait what the robot can feel.  Every argument is optional.

        quat     (w, x, y, z), body orientation in the world frame — MuJoCo's `imu_quat`
                 sensor, or a sensor_msgs/Imu `orientation`.
        gyro     (wx, wy, wz) rad/s in the body frame.  All three are used: wx and wy
                 damp the levelling, wz damps the heading hold.
        contact  {leg: bool}, "is this foot loaded".  In sim that is the `{leg}_contact`
                 touch sensor; on the real robot it is whatever answers the same question
                 — the knee servo's load reading, most likely.  The gait wants the boolean
                 and nothing else, so the two can never disagree about units.

        Stop calling this and the gait blends back to open loop within `fb_timeout`; that
        is the same path as never calling it, so a dropped IMU degrades instead of lying.
        """
        if quat is not None:
            w, x, y, z = (float(v) for v in quat)
            # z-up frame, so these are "left side up" and "nose down", both positive.
            self._roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            self._pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
            self._yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        if gyro is not None:
            self._wx, self._wy = float(gyro[0]), float(gyro[1])
            self._wzg = float(gyro[2])
        if contact is not None:
            self._contact = {l: bool(contact.get(l, False)) for l in self.legs}
        self._fb_age = 0.0

    def _level(self, nx, ny):
        """Foot z offset that pushes the body back level, for a foot at (nx, ny).

        A small body rotation (roll, pitch) lifts the corner at (nx, ny) by
        roll*ny - pitch*nx — the z part of omega x r, in a z-up frame where a positive
        roll raises the +Y side and a positive pitch drops the nose.  A corner that is
        high by h is put back by retracting that leg by h, so the correction carries the
        same sign as the displacement, not the opposite one.  (Getting the roll term
        backwards makes a divergent loop: the trot rolled onto its back inside a second,
        every time, on flat ground as readily as on rough.)

        It acts on the *filtered* attitude, over `level_tau` — most of what the IMU sees
        is the trot's own rocking at the gait frequency, which is the gait working, not an
        error.  Chasing that instead of the slope underneath it holds the body beautifully
        level and costs nearly half the forward speed: 793 -> 429 mm on flat ground, where
        there is no slope to correct at all.  The rate term is small and only damps.
        """
        dz = ((self.level_kp * self._roll_f + self.level_kd * self._wx) * ny
              - (self.level_kp * self._pitch_f + self.level_kd * self._wy) * nx)
        return _clamp(dz, self.level_max)

    def _heading(self, wz_cmd, live):
        """Extra yaw rate that puts the body back on the heading it was told to hold.

        The gait commands body velocity in body axes: `fx = -(vx - wz*ny)` turns with the
        robot.  So a robot that has been knocked 20 deg off keeps walking straight ahead
        of *itself* and curves through the world, and nothing in the open-loop profile can
        notice.  Rough ground supplies the knock — the feet slip on the slopes — which is
        why the drift is 1.8 deg on the flat and 1.65 m of sideways travel on relief.

        The reference is latched, not commanded: whatever heading the robot had when it
        last started walking straight is the one it keeps.  It is dropped the moment the
        operator asks for a turn, so the loop never fights a deliberate one, and dropped
        again whenever feedback goes stale, so a blackout does not end with the robot
        snapping back to a heading it held a minute ago.

        Sim yaw is truth; hardware yaw is not.  MuJoCo's imu_quat is the real orientation,
        but the robot has no magnetometer, so its yaw is an integrated gyro and drifts.
        This holds a straight line over a run, which is what it is for — it is not an
        absolute bearing and must not be sold as one.
        """
        if not live or self._moving < 0.5 or abs(wz_cmd) > self.yaw_cmd_eps:
            self._yaw_ref = None
            return 0.0
        if self._yaw_ref is None:
            self._yaw_ref = self._yaw
            return 0.0
        e = _wrap(self._yaw_ref - self._yaw)
        return _clamp(self.yaw_kp * e - self.yaw_kd * self._wzg, self.yaw_max)

    def _ground(self, l, dt, live, swing, a):
        """Update this leg's idea of where the ground is, and return it (m, vs nominal).

        One decision per footfall.  Ground that came *up* announces itself as contact
        partway through the swing, and the foot then stands where it touched instead of
        carrying on with a sine that would peel it straight back off the hillside.  Once
        it has landed the number is frozen until the next lift-off; while the foot is
        still on its way over, the last footfall is forgotten, because the next one is a
        stride away and the field is not correlated over that distance.

        Ground that *dropped* is deliberately not handled here — the body sags into it and
        `_level` picks up the attitude.  A symmetric version, where a foot that felt
        nothing kept reaching downwards, was written, tuned and measured, and it never
        paid: on the shipping heightfield it does not fire at all (the swing always finds
        ground on the way down), and every trigger loose enough to fire fired on flat
        ground too.  Phase-gated it reached on every stride and cut the flat trot from
        793 mm to 151 mm; clock-gated at 0.62 periods, the same.  Widened to a full period
        it went quiet everywhere, including at +-60 mm of relief, where switching it off
        was worth +7 mm and one less fall.  Do not re-add it without a terrain that has an
        actual step-down in it and a measurement that says it helps.

        "Landed" is a debounced contact past `touch_from`, not any contact at all.  A trot
        at this period keeps a stance foot off the ground about 60 % of its nominal stance
        even on a flat floor: the servos lag the profile and the diagonal pair trades
        support.  The same lag sets `touch_from` — the foot is still on its way off the
        floor a third of the way into the commanded swing (measured: it leaves at s ~ 0.6
        and lands at s ~ 0.05, a tenth of a cycle behind the profile), and counting that
        as an early landing latched the front legs 5 mm high for good and cost 350 mm on
        a flat floor.
        """
        fade = self._gz[l] * min(1.0, dt / max(self.gz_tau, 1e-3))
        if not live:                       # no sensors: nothing is known, so hold nothing
            self._gz[l] -= fade
        elif self._touched[l]:
            pass                           # standing where it landed, mid-swing or not
        elif self._con_t[l] > self.touch_hold and (not swing or a > self.touch_from):
            # touch_band is the servo's own lag: on flat ground the foot always makes
            # contact a little before the profile reaches the plane, and latching that
            # would walk the body a few mm further up on every stride.
            self._touched[l] = True
            self._gz[l] = _clamp(self._gz[l] + max(0.0, self._lift[l] - self.touch_band),
                                 self.gz_max)
        elif swing:
            self._gz[l] -= fade
        return self._gz[l]

    # ------------------------------------------------------------------
    def leg_phase(self, leg):
        """where this leg is in its cycle, 0..1. Stance below 0.5, swing above."""
        return (self._phase + PHASE[leg]) % 1.0

    def period_for(self, speed):
        """Gait period for a commanded speed — shorter the faster it is asked to go.

        A fixed period means the stride grows with the command, and past about 0.25 m/s
        the trot spends long enough on one diagonal, far enough from the middle, that
        rough ground tips it. The terrain feedback does not save it — it levels the body,
        it cannot shorten the time the body is unsupported. Falls out of 7 terrain seeds,
        6 s each, with the feedback on:

            period       0.26  0.28  0.30  0.32  0.34  0.37  0.40  0.45  0.50
            0.20 m/s        0     0     0     0     0     0     0     0     0
            0.30 m/s        0     0     0     0     1     0     0     3     2
            0.45 m/s        1     0     0     0     2     2     6     5     6

        Monotone, not a resonance: shorter is better the faster it goes, and at a walk it
        makes no difference at all. `stride_max` is set so the schedule lands in the
        zero column at every speed and saturates at `period` by 0.20 m/s, which is where
        everything else was tuned.

        Note this is the *commanded* speed, not the achieved one. It has to be: the
        period sets the stride, so reading it back off the body would close a loop
        through the very thing it sets.
        """
        if speed <= 1e-6:
            return self.period
        return min(self.period, max(self.period_min, 2.0 * self.stride_max / speed))

    # ------------------------------------------------------------------
    def foot_targets(self, dt, vx, vy, wz):
        """returns {leg: (x, y, z)} in base_link coordinates."""
        speed = math.hypot(vx, vy) + abs(wz) * 0.25
        target_moving = 1.0 if speed > 1e-3 else 0.0
        k = min(1.0, dt / max(self.stand_smooth, 1e-3))
        self._moving += (target_moving - self._moving) * k
        period = self.period_for(speed)
        if self._moving > 1e-3:
            self.t += dt
            # integrate the phase, not the clock: `period` changes with the command, and
            # s = t / period would jump the whole gait mid-stride every time it did
            self._phase = (self._phase + dt / max(period, 1e-3)) % 1.0
        else:
            self.t = 0.0
            self._phase = 0.0

        # feedback is only as good as it is fresh; stale sensors mean the blind gait
        self._fb_age += dt
        live = self._fb_age <= self.fb_timeout
        if not live:
            self._roll = self._pitch = self._wx = self._wy = self._wzg = 0.0
            self._contact = {l: False for l in self.legs}
        # the correction steers; it must not also shorten the gait period, so it lands
        # after `speed` and `period` are settled and only reaches the foot velocities
        wz = wz + self._heading(wz, live)
        k = min(1.0, dt / max(self.level_tau, 1e-3))
        self._roll_f += (self._roll - self._roll_f) * k
        self._pitch_f += (self._pitch - self._pitch_f) * k

        half = 0.5 * period                            # stance duration
        out = {}
        for l in self.legs:
            nx, ny, _ = self.nominal[l]
            # foot velocity in the base frame = -(v + w x r)
            fx = -(vx - wz * ny)
            fy = -(vy + wz * nx)
            dx = _clamp(fx * half * 0.5, self.max_step)
            dy = _clamp(fy * half * 0.5, self.max_step_y)

            s = self.leg_phase(l)
            swing = s >= 0.5
            a = (s - 0.5) / 0.5 if swing else s / 0.5
            if swing and not self._airborne[l]:        # lift-off: this footfall is over
                self._touched[l] = False
            self._airborne[l] = swing

            self._con_t[l] = self._con_t[l] + dt if self._contact[l] else 0.0
            gz = self._ground(l, dt, live, swing, a)
            if not swing:                              # stance: foot travels with the ground
                px, py = nx + dx * (2 * a - 1), ny + dy * (2 * a - 1)
                self._lift[l] = 0.0
            else:                                      # swing: return + lift
                px, py = nx + dx * (1 - 2 * a), ny + dy * (1 - 2 * a)
                # once the foot has found ground the lift is over, whatever the phase says:
                # carrying on with the sine would peel it straight back off the hillside
                self._lift[l] = (0.0 if self._touched[l] else
                                 self.swing_height * math.sin(math.pi * a) * self._moving)
            pz = -self.body_height + gz + self._lift[l] + self._level(nx, ny)
            m = self._moving
            out[l] = (nx + (px - nx) * m, ny + (py - ny) * m, pz)
        return out

    def joint_targets(self, dt, vx, vy, wz):
        """returns a list of joint angles ordered like self.joint_names."""
        feet = self.foot_targets(dt, vx, vy, wz)
        q = {}
        for l in self.legs:
            hx, hy, hz = self.hip[l]
            fx, fy, fz = feet[l]
            local = (fx - hx, fy - hy, fz - hz)
            try:
                q[l] = self.kin[l].ik(local, prev=self._q[l])
            except ValueError:
                q[l] = self._q[l]
            self._q[l] = q[l]
        order = {"roll": 0, "pitch": 1, "knee": 2}
        out = []
        for n in self.joint_names:
            leg, kind = n.split("_")
            out.append(_clamp(q[leg][order[kind]], self.limits[kind]))
        # rate-limit to the servo's own speed so start-up never asks for a step change
        step = self.max_joint_rate * max(dt, 1e-4)
        out = [pv + _clamp(v - pv, step) for v, pv in zip(out, self._q_prev)]
        self._q_prev = list(out)
        return out

def _wrap(a):
    """angle to (-pi, pi].  Without this the hold takes the long way round at +-180 deg."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _clamp(v, lim):
    return max(-lim, min(lim, v))
