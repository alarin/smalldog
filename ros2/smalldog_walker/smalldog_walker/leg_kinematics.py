"""Analytic 3-DOF leg kinematics for SmallDog. Pure Python — no ROS imports,
so it can be unit-tested and reused by the standalone MuJoCo sim.

Leg frame = the hip-roll joint frame: origin on the roll axis, axes parallel to
base_link at zero pose. +X forward, +Y left, +Z up.

Chain:  roll (about X)  ->  pitch (about Y)  ->  knee (about Y)
The pitch axis sits at (0, dy, dz) from the roll axis; thigh and shin have no
lateral offset, so the whole thigh/shin plane is carried by the roll joint.
"""
import math

class LegKinematics:
    def __init__(self, dy, dz, l_thigh, l_shin, foot_r=0.0):
        self.dy, self.dz = float(dy), float(dz)
        self.l1, self.l2 = float(l_thigh), float(l_shin)
        self.foot_r = float(foot_r)
        self.reach_max = self.l1 + self.l2
        self.reach_min = abs(self.l1 - self.l2) + 1e-4

    # ------------------------------------------------------------------ FK
    def fk(self, q):
        q1, q2, q3 = q
        u = -self.l1 * math.sin(q2) - self.l2 * math.sin(q2 + q3)
        w = -self.l1 * math.cos(q2) - self.l2 * math.cos(q2 + q3)
        y0, z0 = self.dy, self.dz + w                     # in the rolled frame
        c, s = math.cos(q1), math.sin(q1)
        return (u, y0 * c - z0 * s, y0 * s + z0 * c)

    # ------------------------------------------------------------------ IK
    def ik(self, p, prev=None):
        """foot position (x, y, z) in the leg frame -> (roll, pitch, knee) rad.
        Raises ValueError if the point is unreachable."""
        px, py, pz = p
        r = math.hypot(py, pz)
        if r < 1e-9:
            raise ValueError("foot target on the roll axis")
        ratio = self.dy / r
        if abs(ratio) > 1.0:
            raise ValueError(f"foot target unreachable in roll: |dy/r|={abs(ratio):.3f}")
        alpha = math.atan2(pz, py)
        cand = [alpha + math.acos(ratio), alpha - math.acos(ratio)]
        ref = prev[0] if prev else 0.0
        q1 = min(cand, key=lambda a: abs(_wrap(a - ref)))
        q1 = _wrap(q1)

        c, s = math.cos(q1), math.sin(q1)
        w = (-py * s + pz * c) - self.dz                  # foot z relative to the pitch axis
        u = px
        d = math.hypot(u, w)
        d = min(max(d, self.reach_min), self.reach_max - 1e-6)
        c3 = (d * d - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
        c3 = min(1.0, max(-1.0, c3))
        q3 = math.acos(c3)                                # knee always bends one way
        q2 = math.atan2(-u, -w) - math.atan2(self.l2 * math.sin(q3),
                                            self.l1 + self.l2 * math.cos(q3))
        return (q1, _wrap(q2), q3)

def _wrap(a):
    while a > math.pi:  a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a
