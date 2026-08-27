import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mini_dog as md
hb = md.hip_bracket(); ch = md.chassis_bottom()
i = hb.val().intersect(ch.val())
print("volume %.1f mm3" % i.Volume())
b = i.BoundingBox()
print("bbox x %.2f..%.2f  y %.2f..%.2f  z %.2f..%.2f" % (b.xmin,b.xmax,b.ymin,b.ymax,b.zmin,b.zmax))
for s in i.Solids():
    bb = s.BoundingBox()
    print("  piece %.1f mm3  x %.1f..%.1f y %.1f..%.1f z %.1f..%.1f" %
          (s.Volume(), bb.xmin,bb.xmax,bb.ymin,bb.ymax,bb.zmin,bb.zmax))
