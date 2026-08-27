import cadquery as cq, math
from collections import defaultdict
imp = cq.importers.importStep("ref/ST3215-3D/ST3215.step")
sol = sorted(imp.solids().vals(), key=lambda s:-s.Volume())
print("=== side-wall threaded holes (axis perpendicular to servo axis Y) ===")
g=defaultdict(list)
for i in (0,1,2):
    for f in sol[i].Faces():
        if f.geomType()!="CYLINDER": continue
        ad=f._geomAdaptor().Cylinder(); d=ad.Axis().Direction()
        if abs(d.Y())>1e-3: continue
        r=ad.Radius()
        if r>2.2: continue
        b=f.BoundingBox()
        ax = "X" if abs(d.X())>0.9 else ("Z" if abs(d.Z())>0.9 else "?")
        g[(round(r,2),ax,round((b.xmin+b.xmax)/2,2),round((b.ymin+b.ymax)/2,2),round((b.zmin+b.zmax)/2,2))].append(i)
for k in sorted(g,key=lambda k:(k[1],k[3],k[2])):
    print("  r=%4.2f ax=%s  x=%7.2f y=%7.2f z=%7.2f   solids %s"%(k[0],k[1],k[2],k[3],k[4],sorted(set(g[k]))))
print()
print("=== case outer corner fillets (planes normal to X/Z, extremes) ===")
for i in (0,1,2):
    b=sol[i].BoundingBox()
    print(" solid",i,"x %.3f..%.3f y %.3f..%.3f z %.3f..%.3f"%(b.xmin,b.xmax,b.ymin,b.ymax,b.zmin,b.zmax))
