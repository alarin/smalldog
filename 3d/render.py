import sys, math, os
import cadquery as cq
import vtk
from vtk.util.colors import *

def render(shapes_colors, out, cam, size=(1400,1000), up=(0,0,1)):
    ren = vtk.vtkRenderer(); ren.SetBackground(1,1,1)
    for shp, col in shapes_colors:
        pd = shp.toVtkPolyData(0.08, 0.2, True)
        m = vtk.vtkPolyDataMapper(); m.SetInputData(pd)
        a = vtk.vtkActor(); a.SetMapper(m)
        a.GetProperty().SetColor(*col)
        a.GetProperty().SetInterpolationToPhong()
        a.GetProperty().SetSpecular(0.25); a.GetProperty().SetSpecularPower(30)
        ren.AddActor(a)
        e = vtk.vtkFeatureEdges(); e.SetInputData(pd)
        e.BoundaryEdgesOn(); e.FeatureEdgesOn(); e.SetFeatureAngle(35)
        e.ManifoldEdgesOff(); e.NonManifoldEdgesOff()
        em = vtk.vtkPolyDataMapper(); em.SetInputConnection(e.GetOutputPort())
        ea = vtk.vtkActor(); ea.SetMapper(em); ea.GetProperty().SetColor(0.1,0.1,0.1)
        ea.GetProperty().SetLineWidth(1.1); ren.AddActor(ea)
    rw = vtk.vtkRenderWindow(); rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren); rw.SetSize(*size)
    c = ren.GetActiveCamera(); c.SetPosition(*cam); c.SetFocalPoint(0,0,-60); c.SetViewUp(*up)
    ren.ResetCamera(); c.Zoom(1.25)
    l = vtk.vtkLight(); l.SetPosition(400,-600,900); l.SetIntensity(0.9); ren.AddLight(l)
    rw.Render()
    w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(rw); w2i.Update()
    wr = vtk.vtkPNGWriter(); wr.SetFileName(out); wr.SetInputConnection(w2i.GetOutputPort()); wr.Write()
    print("wrote", out)

if __name__ == "__main__":
    import mini_dog as md
    hb, th, sh, ft = md.build()
    sc = []
    grey=(0.62,0.65,0.70); org=(0.90,0.58,0.16); wht=(0.85,0.86,0.88); blk=(0.16,0.16,0.18)
    alu=(0.72,0.76,0.82)
    sc.append((md.PARTS["chassis_bottom"][0].val(), grey))
    sc.append((md.PARTS["chassis_top"][0].val(), grey))
    sc.append((md.PARTS["lidar_mount"][0].val(), grey))
    sc.append((md.PARTS["gps_mount"][0].val(), grey))
    sc.append((md.PARTS["camera_mount"][0].val(), grey))
    sc.append((md.camera_module().val(), blk))
    srv = [md.mv(md.servo_dummy(), L) for _, L in md.JOINTS]
    hub = [md.mv(md.hubs(), L) for _, L in md.JOINTS]
    for f in (lambda w: w, md.mirY, md.mirX, lambda w: md.mirX(md.mirY(w))):
        sc.append((f(md.posed(hb,"hip")).val(), org))
        sc.append((f(md.posed(th,"thigh")).val(), wht))
        sc.append((f(md.posed(sh,"shin")).val(), wht))
        sc.append((f(md.posed(ft,"shin")).val(), blk))
        for i,k in enumerate(("hip","thigh","shin")):
            sc.append((f(md.posed(srv[i],k)).val(), blk))
            sc.append((f(md.posed(hub[i],k)).val(), alu))
    render(sc, "out/view_iso.png",  (520,-620,320))
    render(sc, "out/view_front.png",(900,-10,10))
    render(sc, "out/view_side.png", (10,-900,10))
