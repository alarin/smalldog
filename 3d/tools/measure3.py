import cadquery as cq, math
from collections import defaultdict
imp = cq.importers.importStep("ref/ST3215-3D/ST3215.step")
sol = sorted(imp.solids().vals(), key=lambda s:-s.Volume())
print("=== HUB PLATES ===")
for idx in (6,7):
    s = sol[idx]; b = s.BoundingBox()
    print("solid",idx,"vol %.1f"%s.Volume(), "y %.3f..%.3f"%(b.ymin,b.ymax))
    for f in s.Faces():
        if f.geomType()=="PLANE" and abs(abs(f.normalAt().y)-1)<1e-6 and f.Area()>10:
            fb=f.BoundingBox()
            print("   plane y=%7.3f area=%7.1f  x %7.2f..%7.2f z %7.2f..%7.2f"%(fb.ymin,f.Area(),fb.xmin,fb.xmax,fb.zmin,fb.zmax))
    hs=[]
    for f in s.Faces():
        if f.geomType()=="CYLINDER":
            ad=f._geomAdaptor().Cylinder(); d=ad.Axis().Direction()
            if abs(abs(d.Y())-1)>1e-4: continue
            fb=f.BoundingBox()
            hs.append((round(ad.Radius(),2), round((fb.xmin+fb.xmax)/2,2), round((fb.zmin+fb.zmax)/2,2)))
    for h in sorted(set(hs)): print("   cyl r=%.2f at x=%.2f z=%.2f  (r_from_axis=%.2f)"%(h[0],h[1],h[2],math.hypot(h[1]+25.5,h[2])))
print()
print("=== CASE OUTLINE: cylinders with axis along Y, r<4, r_from_axis>12 (corners/screws) ===")
seen=defaultdict(list)
for idx in (0,1,2):
    s=sol[idx]
    for f in s.Faces():
        if f.geomType()!="CYLINDER": continue
        ad=f._geomAdaptor().Cylinder(); d=ad.Axis().Direction()
        if abs(abs(d.Y())-1)>1e-4: continue
        r=ad.Radius()
        if r>4.2 or r<0.8: continue
        fb=f.BoundingBox()
        seen[(round(r,2), round((fb.xmin+fb.xmax)/2,2), round((fb.zmin+fb.zmax)/2,2))].append((idx,round(fb.ymin,2),round(fb.ymax,2)))
for k in sorted(seen, key=lambda k:(k[1],k[2])):
    print("  r=%5.2f x=%8.2f z=%8.2f  %s"%(k[0],k[1],k[2],sorted(set(seen[k]))[:3]))
