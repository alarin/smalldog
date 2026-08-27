import cadquery as cq
from collections import defaultdict
imp = cq.importers.importStep("ref/ST3215-3D/ST3215.step")
solids = imp.solids().vals()
print("solids:", len(solids))
sh = imp.val()
bb = sh.BoundingBox()
print("BBOX  x %.3f..%.3f (%.3f)  y %.3f..%.3f (%.3f)  z %.3f..%.3f (%.3f)" % (
    bb.xmin,bb.xmax,bb.xlen, bb.ymin,bb.ymax,bb.ylen, bb.zmin,bb.zmax,bb.zlen))
print("volume %.1f mm3" % sh.Volume())
faces = sh.Faces()
print("faces:", len(faces))
kinds = defaultdict(int)
for f in faces: kinds[f.geomType()] += 1
print(dict(kinds))
# cylinders: report axis dir, radius, centre
cyls = defaultdict(list)
for f in faces:
    if f.geomType() != "CYLINDER": continue
    s = f._geomAdaptor()
    cy = s.Cylinder()
    r = cy.Radius()
    ax = cy.Axis()
    d = ax.Direction(); p = ax.Location()
    key = (round(r,3), round(abs(d.X()),3), round(abs(d.Y()),3), round(abs(d.Z()),3))
    cyls[key].append((round(p.X(),3), round(p.Y(),3), round(p.Z(),3), round(f.Area(),1)))
for k in sorted(cyls, key=lambda k:-k[0]):
    v = cyls[k]
    print("CYL r=%7.3f axis=(%.2f,%.2f,%.2f) n=%d" % (k[0],k[1],k[2],k[3],len(v)))
    for item in sorted(set(v))[:12]:
        print("      at", item)
