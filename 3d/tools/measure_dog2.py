import cadquery as cq, math
from collections import defaultdict
imp = cq.importers.importStep("ref/ROBOTIC_DOG_-STEP/ROBOTIC DOG.step")
sol = imp.solids().vals()
axes=[]
for s in sol:
    v=s.Volume()
    if not (560<v<570 or 750<v<765): continue
    for f in s.Faces():
        if f.geomType()!="CYLINDER": continue
        ad=f._geomAdaptor().Cylinder()
        if abs(ad.Radius()-9.6)>0.05: continue
        d=ad.Axis().Direction(); p=ad.Axis().Location()
        b=f.BoundingBox()
        c=((b.xmin+b.xmax)/2,(b.ymin+b.ymax)/2,(b.zmin+b.zmax)/2)
        n=(round(d.X(),3),round(d.Y(),3),round(d.Z(),3))
        # project centre onto axis-perpendicular coords
        axes.append((round(v,0), n, tuple(round(x,2) for x in c)))
        break
seen=set(); out=[]
for a in axes:
    if a in seen: continue
    seen.add(a); out.append(a)
print("hub plates found:", len(out))
for a in sorted(out, key=lambda a:(a[2][0], a[2][1], a[2][2])):
    print("  vol=%6.0f axis=%s centre=%s" % a)
