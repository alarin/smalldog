import cadquery as cq, math
from collections import defaultdict
imp = cq.importers.importStep("ref/ROBOTIC_DOG_-STEP/ROBOTIC DOG.step")
sol = imp.solids().vals()
print("solids:", len(sol))
tot = imp.val().BoundingBox()
print("ASSEMBLY BBOX x %.1f..%.1f (%.1f)  y %.1f..%.1f (%.1f)  z %.1f..%.1f (%.1f)"%(
 tot.xmin,tot.xmax,tot.xlen,tot.ymin,tot.ymax,tot.ylen,tot.zmin,tot.zmax,tot.zlen))
groups=defaultdict(list)
for s in sol:
    b=s.BoundingBox()
    key=(round(s.Volume(),0), round(b.xlen,2), round(b.ylen,2), round(b.zlen,2))
    groups[key].append((round(b.xmin+b.xlen/2,2), round(b.ymin+b.ylen/2,2), round(b.zmin+b.zlen/2,2)))
print("\n=== distinct solid types (vol, bbox dims) sorted by count*vol ===")
for k in sorted(groups, key=lambda k:-len(groups[k])*k[0])[:28]:
    print("  n=%3d vol=%10.0f bbox %7.2f x %7.2f x %7.2f" % (len(groups[k]),k[0],k[1],k[2],k[3]))
    if len(groups[k])<=14:
        for c in sorted(groups[k]): print("        centre",c)
