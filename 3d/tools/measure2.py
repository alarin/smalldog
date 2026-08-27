import cadquery as cq
from collections import defaultdict
imp = cq.importers.importStep("ref/ST3215-3D/ST3215.step")
sol = imp.solids().vals()
sol.sort(key=lambda s:-s.Volume())
for i,s in enumerate(sol):
    b=s.BoundingBox()
    print("solid %d vol=%9.1f  x %8.3f..%8.3f  y %8.3f..%8.3f  z %8.3f..%8.3f" %
          (i,s.Volume(),b.xmin,b.xmax,b.ymin,b.ymax,b.zmin,b.zmax))
print()
case = sol[0]
print("=== CASE (largest solid) ===")
# planar faces normal to Y
for f in case.Faces():
    if f.geomType()!="PLANE": continue
    n=f.normalAt()
    if abs(abs(n.y)-1)<1e-6:
        b=f.BoundingBox()
        if f.Area()>30:
            print("  planeY y=%8.3f area=%8.1f  x %7.2f..%7.2f z %7.2f..%7.2f" %
                  (b.ymin,f.Area(),b.xmin,b.xmax,b.zmin,b.zmax))
print()
print("=== through-holes parallel to Y (whole assembly) ===")
g=defaultdict(list)
for s in sol:
    for f in s.Faces():
        if f.geomType()!="CYLINDER": continue
        ad=f._geomAdaptor().Cylinder(); d=ad.Axis().Direction()
        if abs(abs(d.Y())-1)>1e-4: continue
        r=ad.Radius()
        if r>5.5: continue
        b=f.BoundingBox()
        g[(round(r,2), round((b.xmin+b.xmax)/2,2), round((b.zmin+b.zmax)/2,2))].append((round(b.ymin,2),round(b.ymax,2)))
for k in sorted(g, key=lambda k:(k[1],k[2])):
    ys=g[k]
    print("  r=%5.2f  x=%8.2f z=%8.2f   y-ranges %s" % (k[0],k[1],k[2], sorted(set(ys))[:4]))
