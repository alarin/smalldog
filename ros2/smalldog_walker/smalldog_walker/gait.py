"""Trot gait generator for SmallDog. Pure Python.

Diagonal pairs move together: (FL, RR) and (FR, RL), 180 deg out of phase,
duty 0.5. Foot targets are produced in base_link coordinates, then handed to
LegKinematics per leg.

Command: body velocity (vx, vy, wz) in the base frame + body height.
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
        self.period = 0.45              # s, one full gait cycle
        self.max_step = 0.060           # m, fore/aft half-stride clamp
        self.max_step_y = 0.030         # m, lateral half-stride clamp (roll range is small)
        self.stand_smooth = 0.12
        self.swing_height = 0.022
        self._body_height = 0.158
        self.body_height = self._body_height          # runs the setter -> clamps
        self.max_joint_rate = p.get("joint_velocity_limit", 4.7) * 0.85
        # start from the model's mechanical zero so the first command is ramped too
        self._q_prev = [0.0] * len(self.joint_names)

        self.t = 0.0
        self._q = {l: (0.0, p["stance_rad"]["pitch"], p["stance_rad"]["knee"]) for l in self.legs}
        self._moving = 0.0

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
    def foot_targets(self, dt, vx, vy, wz):
        """returns {leg: (x, y, z)} in base_link coordinates."""
        speed = math.hypot(vx, vy) + abs(wz) * 0.25
        target_moving = 1.0 if speed > 1e-3 else 0.0
        k = min(1.0, dt / max(self.stand_smooth, 1e-3))
        self._moving += (target_moving - self._moving) * k
        if self._moving > 1e-3:
            self.t += dt
        else:
            self.t = 0.0

        half = 0.5 * self.period                       # stance duration
        out = {}
        for l in self.legs:
            nx, ny, _ = self.nominal[l]
            # foot velocity in the base frame = -(v + w x r)
            fx = -(vx - wz * ny)
            fy = -(vy + wz * nx)
            dx = _clamp(fx * half * 0.5, self.max_step)
            dy = _clamp(fy * half * 0.5, self.max_step_y)

            s = (self.t / self.period + PHASE[l]) % 1.0
            if s < 0.5:                                # stance: foot travels with the ground
                a = s / 0.5
                px, py = nx + dx * (2 * a - 1), ny + dy * (2 * a - 1)
                pz = -self.body_height
            else:                                      # swing: return + lift
                a = (s - 0.5) / 0.5
                px, py = nx + dx * (1 - 2 * a), ny + dy * (1 - 2 * a)
                pz = -self.body_height + self.swing_height * math.sin(math.pi * a) * self._moving
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

def _clamp(v, lim):
    return max(-lim, min(lim, v))
