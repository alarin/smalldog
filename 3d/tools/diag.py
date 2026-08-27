import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mini_dog as md, cadquery as cq
th = md.thigh(); hb = md.hip_bracket(); srv = md.mv(md.servo_dummy(), md.PITCH_LOC)
parts = {"bracket": hb, "pitch_servo": srv}
for a in (-50,-40,-30,-20,-10,0,10,20,30,40):
    m = md.rot_pitch(th, a)
    msgs=[]
    for nm,p in parts.items():
        try: inter = m.val().intersect(p.val())
        except Exception: continue
        v = inter.Volume()
        if v > 1.0:
            b = inter.BoundingBox()
            msgs.append("%s %.0fmm3 @ x%.0f..%.0f y%.0f..%.0f z%.0f..%.0f"%(nm,v,b.xmin,b.xmax,b.ymin,b.ymax,b.zmin,b.zmax))
    print("pitch %+4d : %s" % (a, "  |  ".join(msgs) if msgs else "clear"))
