#!/usr/bin/env python3
"""
Retro EDF hover rocket - preliminary parametric CAD (CadQuery).

Coordinates: Z up, Z=0 is the ground under the feet. All dimensions in mm.
Run:  python rocket.py            -> writes ./out/{stl,step,viewer}
Everything that depends on a purchased part is a parameter at the top.
MEASURE YOUR FAN before printing the aft body: EDF_OD / EDF_LEN drive the bore.
"""
import base64
import json
import math
import os
import struct

import cadquery as cq
import numpy as np
from scipy.interpolate import PchipInterpolator

# ------------------------------------------------------------------ parameters
EDF_OD = 98.0        # fan housing outer diameter (90 mm class)  << measure
EDF_ID = 90.0        # fan duct inner diameter
EDF_LEN = 80.0       # housing length seated in the bore          << measure
EDF_HUB = 38.0       # hub / motor diameter (for swept-area maths only)
FIT = 0.2            # radial clearance on sliding fits
WALL_AFT = 1.6       # aft body skin (4 perimeters @ 0.4)
WALL = 1.2           # mid body + nose skin (3 perimeters @ 0.4)

Z_EXIT = 100.0       # nozzle exit plane above ground (~1 fan diameter)
Z_SEAT = 197.0       # bottom face of the fan housing
Z_J1 = Z_SEAT + EDF_LEN + 10.0   # aft / mid joint (spigot clamps the fan)
Z_BULK = 370.0       # battery bulkhead (top of inlet plenum)
Z_J2 = 465.0         # mid / nose joint
Z_TIP = 600.0        # nose tip  -> overall height

VANE_Z = 126.0       # vane pivot axis height
VANE_AZ = [45, 135, 225, 315]
LEG_AZ = [0, 120, 240]
TUNNEL_AZ = 180      # wire tunnel azimuth
WIN_AZ = [30 + 60 * k for k in range(6)]   # inlet windows (pillars under legs)

LEG_T = 14.0         # leg plate thickness
R_FOOT = 175.0       # foot circle radius
SPIGOT = 10.0        # joint overlap length

CHK_Z0, CHK_ROW, CHK_ROWS, CHK_COLS = 365.0, 50.0, 3, 6   # checker band

R_FAN = EDF_OD / 2 + FIT + WALL_AFT     # outer skin radius over the fan
LEG_BOLTS = [(R_FAN + 6.5, 180.0), (R_FAN + 6.5, 212.0)]   # (radius, z) of the M3 clevis bolts

# Outer mould line: (z, radius). PCHIP keeps the fan zone exactly cylindrical.
PROFILE = [
    (Z_EXIT, 44.5), (140, 45.5), (175, 46.8), (190, 49.0),
    (Z_SEAT, R_FAN), (Z_J1, R_FAN),
    (330, 54.6), (372, 56.0), (410, 55.4), (Z_J2, 49.5),
    (500, 45.0), (530, 38.5), (560, 26.0), (585, 11.0), (Z_TIP, 0.0),
]
_p = PchipInterpolator([p[0] for p in PROFILE], [p[1] for p in PROFILE])


def ro(z):
    """Outer skin radius at height z."""
    return float(_p(z))


# ------------------------------------------------------------------ helpers
def _zs(z0, z1, step=5.0):
    n = max(int(abs(z1 - z0) / step), 2) + 1
    return np.linspace(z0, z1, n)


def solid_under(fn, z0, z1):
    """Solid of revolution between the axis and r = fn(z)."""
    pts = [(max(float(fn(v)), 0.0), float(v)) for v in _zs(z0, z1)]
    wp = cq.Workplane("XZ").moveTo(0, z0).lineTo(*pts[0])
    wp = wp.spline(pts[1:], includeCurrent=True)
    if pts[-1][0] > 1e-6:
        wp = wp.lineTo(0, z1)
    return wp.close().revolve(360, (0, 0, 0), (0, 1, 0))


def ring(poly):
    """Revolve a closed (r, z) polygon about the Z axis."""
    return cq.Workplane("XZ").polyline(poly).close().revolve(360, (0, 0, 0), (0, 1, 0))


def shell(z0, z1, wall):
    outer = solid_under(ro, z0, z1)
    inner = solid_under(lambda z: ro(z) - wall, z0 - 1, z1 + 1)
    return outer.cut(inner)


def plate(poly, half):
    """Extrude an (x, z) polygon symmetrically +-half in Y."""
    return cq.Workplane("XZ").polyline(poly).close().extrude(half, both=True)


def rotz(obj, deg):
    return obj.rotate((0, 0, 0), (0, 0, 1), deg)


def radial_cyl(z, r0, r1, dia):
    """Cylinder with its axis along +X at height z, from x=r0 to x=r1."""
    return (cq.Workplane("YZ").workplane(offset=r0).center(0, z)
            .circle(dia / 2).extrude(r1 - r0))


def y_hole(x, z, dia, half=30):
    return cq.Workplane("XZ").center(x, z).circle(dia / 2).extrude(half, both=True)


# ------------------------------------------------------------------ aft body
def make_aft():
    body = shell(Z_EXIT, Z_J1, WALL_AFT)
    bore = R_FAN - WALL_AFT

    # fan seat: ledge at the fan duct ID, 45 deg underside (prints unsupported)
    body = body.union(ring([
        (bore + 0.6, Z_SEAT), (EDF_ID / 2, Z_SEAT), (EDF_ID / 2, Z_SEAT - 1.5),
        (bore + 0.6, Z_SEAT - 1.5 - (bore + 0.6 - EDF_ID / 2)),
    ]))

    core = solid_under(lambda z: ro(z) - 0.8, Z_EXIT - 5, Z_J1 + 5)
    cavity = solid_under(lambda z: ro(z) - WALL_AFT, Z_EXIT - 5, Z_J1 + 5)

    # leg clevises: two cheeks per leg, 45 deg lead-in underneath
    ck = [(40, 150), (R_FAN + 12, 150 + R_FAN + 12 - 40), (R_FAN + 12, 214),
          (R_FAN - 1, 248), (40, 248)]      # top edge fairs into the leg's upper edge
    cheeks = plate(ck, LEG_T / 2 + FIT + 3).cut(plate(ck, LEG_T / 2 + FIT)).cut(core)
    bolts = y_hole(*LEG_BOLTS[0], 3.2).union(y_hole(*LEG_BOLTS[1], 3.2))
    cheeks = cheeks.cut(bolts)
    for az in LEG_AZ:
        body = body.union(rotz(cheeks, az))

    # wire tunnel: motor phases + servo leads run past the fan outside the bore
    to = [(40, 146), (R_FAN + 7, 146 + R_FAN + 7 - 40), (R_FAN + 7, Z_J1), (40, Z_J1)]
    ti = [(40, 148), (R_FAN + 5.4, 148 + R_FAN + 5.4 - 40),
          (R_FAN + 5.4, Z_J1 - 1.2), (40, Z_J1 - 1.2)]
    channel = plate(ti, 7.5)
    tunnel = plate(to, 9.0).cut(channel).cut(core)
    body = body.union(rotz(tunnel, TUNNEL_AZ))
    # slot through the skin so wires can enter/leave the bore under and over the fan
    slot = plate([(40, Z_SEAT - 14), (R_FAN + 2, Z_SEAT - 14),
                  (R_FAN + 2, Z_J1 - 1.2), (40, Z_J1 - 1.2)], 7.5)
    body = body.cut(rotz(slot, TUNNEL_AZ))
    # open the bottom of the tunnel for the servo leads
    entry = plate([(ro(160) + 1.0, 150), (R_FAN + 5.4, 150), (R_FAN + 5.4, 176),
                   (ro(160) + 1.0, 176)], 6.0)
    body = body.cut(rotz(entry.translate((0, 0, -8)), TUNNEL_AZ))

    # vane pivot bosses
    rv = ro(VANE_Z)
    boss = radial_cyl(VANE_Z, 20, rv + 4.5, 11.0).cut(cavity)
    hole = radial_cyl(VANE_Z, 20, rv + 10, 6.4)
    for az in VANE_AZ:
        body = body.union(rotz(boss, az)).cut(rotz(hole, az))

    # 45 deg conical joint face (self-centring, no overhang on the mating part)
    body = body.cut(ring([(46, Z_J1 - 4.95), (54, Z_J1 + 3.05), (54, Z_J1 + 6),
                          (46, Z_J1 + 6)]))
    # joint screws
    for az in (30, 150, 270):
        body = body.cut(rotz(radial_cyl(Z_J1 - SPIGOT / 2, 40, 60, 2.6), az))
    return body


# ------------------------------------------------------------------ mid body
def make_mid():
    body = shell(Z_J1, Z_J2, WALL)
    bore = R_FAN - WALL_AFT
    so = bore - FIT            # spigot outer radius
    si = so - 2.0
    # spigot into the aft bore; its lower face clamps the fan housing
    body = body.union(ring([
        (si, Z_J1 - SPIGOT), (so, Z_J1 - SPIGOT), (so, Z_J1 - (R_FAN - so)),
        (R_FAN, Z_J1), (R_FAN - 0.8, Z_J1 + 1.0), (R_FAN - 0.8, Z_J1 + 6.0),
        (si, Z_J1 + 6.0 - (R_FAN - 0.8 - si)),
    ]))
    # notch in the spigot over the wire tunnel
    body = body.cut(rotz(plate([(40, Z_J1 - SPIGOT - 1), (so + 0.5, Z_J1 - SPIGOT - 1),
                                (so + 0.5, Z_J1 - 1.9), (40, Z_J1 - 1.9)], 7.5),
                         TUNNEL_AZ))

    # stiffening ribs behind the window pillars
    for az in range(0, 360, 60):
        rib = (cq.Workplane("XY").workplane(offset=Z_J1 + 6)
               .center(ro(Z_J1) - WALL - 1.6, 0).rect(4.4, 2.4).extrude(Z_BULK - 12 - Z_J1 - 6))
        body = body.union(rotz(rib, az))

    # bulkhead ledge, 45 deg underside
    rw = ro(Z_BULK) - WALL
    r_led = ro(Z_J2) - WALL - 3.3
    body = body.union(ring([
        (rw + 0.6, Z_BULK), (r_led, Z_BULK), (r_led, Z_BULK - 1.5),
        (rw + 0.6, Z_BULK - 1.5 - (rw + 0.6 - r_led)),
    ]))

    # inlet windows: hexagonal so they print without support either way up
    hw = 18.5
    win = (cq.Workplane("YZ").polyline([(0, 298), (hw, 310), (hw, 346), (0, 358),
                                        (-hw, 346), (-hw, 310)]).close().extrude(75))
    for az in WIN_AZ:
        body = body.cut(rotz(win, az))

    # top joint: conical face + screw holes
    rt = ro(Z_J2)
    body = body.cut(ring([(rt - 4.8, Z_J2 - 4.95), (rt + 3.2, Z_J2 + 3.05),
                          (rt + 3.2, Z_J2 + 6), (rt - 4.8, Z_J2 + 6)]))
    for az in (30, 150, 270):
        body = body.cut(rotz(radial_cyl(Z_J1 - SPIGOT / 2, 40, 60, 2.6), az))
        body = body.cut(rotz(radial_cyl(Z_J2 - SPIGOT / 2, 40, 60, 2.6), az))
    return body


# ------------------------------------------------------------------ nose
def make_nose():
    outer = solid_under(ro, Z_J2, Z_TIP)
    zc = Z_J2
    while ro(zc) - WALL > 3.0:
        zc += 0.5
    inner = solid_under(lambda z: ro(z) - WALL, Z_J2 - 1, zc)
    body = outer.cut(inner)
    rt = ro(Z_J2)
    so = rt - WALL - FIT
    si = so - 1.6
    body = body.union(ring([
        (si, Z_J2 - SPIGOT), (so, Z_J2 - SPIGOT), (so, Z_J2 - (rt - so)),
        (rt, Z_J2), (rt - 0.6, Z_J2 + 1.0), (rt - 1.0, Z_J2 + 5.0),
        (si, Z_J2 + 5.0 - (rt - 1.0 - si)),
    ]))
    for az in (30, 150, 270):
        body = body.cut(rotz(radial_cyl(Z_J2 - SPIGOT / 2, 30, 60, 2.6), az))
    return body


# ------------------------------------------------------------------ bulkhead
def make_bulkhead():
    r = ro(Z_J2) - WALL - 0.7          # must pass through the mid body top opening
    d = cq.Workplane("XY").workplane(offset=Z_BULK).circle(r).extrude(3.0)
    # battery strap slots, cable pass-through, corner locators for a 50x50 pack
    for x in (-27.5, 27.5):
        d = d.cut(cq.Workplane("XY").workplane(offset=Z_BULK - 1).center(x, 0)
                  .rect(4, 24).extrude(6))
    d = d.cut(cq.Workplane("XY").workplane(offset=Z_BULK - 1).center(0, 35)
              .rect(24, 11).extrude(6))
    for x in (-28, 28):
        for y in (-28, 28):
            if y > 0:
                continue        # keep the cable side clear
            d = d.union(cq.Workplane("XY").workplane(offset=Z_BULK + 3).center(x, y)
                        .rect(5, 5).extrude(10))
    return d


# ------------------------------------------------------------------ legs
def make_leg():
    outer = [(44, 240), (80, 222), (120, 188), (152, 146), (R_FOOT, 104)]
    inner = [(160, 40), (156, 62), (138, 96), (112, 126), (84, 144), (60, 150), (44, 150)]
    wp = cq.Workplane("XZ").moveTo(*outer[0]).spline(outer[1:], includeCurrent=True)
    wp = wp.lineTo(R_FOOT + 3, 40).lineTo(*inner[0])
    wp = wp.spline(inner[1:], includeCurrent=True).close()
    leg = wp.extrude(LEG_T / 2, both=True)
    try:
        leg = leg.faces(">Y or <Y").edges().fillet(5.0)
    except Exception as e:   # keep going with square edges if OCC refuses
        print("  leg fillet failed, using chamfer:", str(e)[:60])
        leg = leg.faces(">Y or <Y").edges().chamfer(3.0)

    pod_pts = [(9.5, 12), (16, 38), (13.5, 68), (6.5, 92), (0, 106)]
    pod = (cq.Workplane("XZ").moveTo(0, 10).lineTo(6, 10)
           .spline(pod_pts, includeCurrent=True).close()
           .revolve(360, (0, 0, 0), (0, 1, 0)).translate((R_FOOT, 0, 0)))
    leg = leg.union(pod)
    # socket for the TPU foot
    leg = leg.cut(cq.Workplane("XY").workplane(offset=9).center(R_FOOT, 0)
                  .circle(3.1).extrude(13))
    # trim root to the body skin, drill clevis bolts
    leg = leg.cut(solid_under(lambda z: ro(z) + 0.3, Z_EXIT - 5, Z_J1 + 5))
    leg = leg.cut(y_hole(*LEG_BOLTS[0], 3.2)).cut(y_hole(*LEG_BOLTS[1], 3.2))
    return leg


def make_foot():
    f = cq.Workplane("XY").circle(9).extrude(10).faces("<Z").edges().fillet(3)
    f = f.union(cq.Workplane("XY").workplane(offset=10).circle(2.9).extrude(11))
    return f.translate((R_FOOT, 0, 0))


# ------------------------------------------------------------------ vanes
def make_vane():
    r_tip = ro(VANE_Z) - WALL_AFT - 2.2
    v = plate([(6, VANE_Z - 25), (r_tip, VANE_Z - 25), (r_tip, VANE_Z + 11),
               (6, VANE_Z + 11)], 1.5)
    try:
        v = v.edges("|X").chamfer(1.1, 4.0)
    except Exception:
        pass
    v = v.union(radial_cyl(VANE_Z, 4, ro(VANE_Z) + 13, 6.0))
    return v


# ------------------------------------------------------------------ checker split
def split_checker(part, zlo, zhi):
    """Return (red, white) bodies; white = checker cells inside [zlo, zhi]."""
    white = None
    step = 360.0 / CHK_COLS
    for j in range(CHK_ROWS):
        z0, z1 = CHK_Z0 + j * CHK_ROW, CHK_Z0 + (j + 1) * CHK_ROW
        z0, z1 = max(z0, zlo), min(z1, zhi)
        if z1 - z0 < 1:
            continue
        cell = (cq.Workplane("XZ").polyline([(30, z0), (80, z0), (80, z1), (30, z1)])
                .close().revolve(step, (0, 0, 0), (0, 1, 0)))
        for i in range(CHK_COLS):
            if (i + j) % 2:
                continue
            c = rotz(cell, i * step)
            white = c if white is None else white.union(c)
    w = part.intersect(white)
    r = part.cut(white)
    return r, w


# ------------------------------------------------------------------ ghosts (bought parts)
def ghosts():
    g = {}
    edf = (cq.Workplane("XY").workplane(offset=Z_SEAT).circle(EDF_OD / 2)
           .circle(EDF_ID / 2).extrude(EDF_LEN))
    hub = cq.Workplane("XY").workplane(offset=Z_SEAT - 30).circle(EDF_HUB / 2).extrude(EDF_LEN + 30)
    spin = (cq.Workplane("XZ").polyline([(0, Z_SEAT + EDF_LEN), (EDF_HUB / 2, Z_SEAT + EDF_LEN),
                                         (0, Z_SEAT + EDF_LEN + 22)]).close()
            .revolve(360, (0, 0, 0), (0, 1, 0)))
    g["EDF 90 mm"] = edf.union(hub).union(spin)
    g["Battery 6S 5000"] = (cq.Workplane("XY").workplane(offset=Z_BULK + 3)
                            .rect(50, 50).extrude(155))
    g["ESC"] = (cq.Workplane("XY").workplane(offset=Z_BULK - 26)
                .rect(84, 38).extrude(12))
    g["Flight controller"] = (cq.Workplane("XY").workplane(offset=Z_BULK + 3 + 158)
                              .rect(36, 36).extrude(8))
    g["GPS"] = (cq.Workplane("XY").workplane(offset=Z_BULK + 3 + 172)
                .rect(25, 25).extrude(8))
    servo = (cq.Workplane("YZ").workplane(offset=ro(VANE_Z) + 6)
             .center(0, VANE_Z + 5.25).rect(12.5, 23).extrude(27))
    s = None
    for az in VANE_AZ:
        r = rotz(servo, az)
        s = r if s is None else s.union(r)
    g["Vane servos x4"] = s
    return g


# ------------------------------------------------------------------ build
def build():
    print("building aft body");   aft = make_aft()
    print("building mid body");   mid = make_mid()
    print("building nose");       nose = make_nose()
    print("building bulkhead");   bulk = make_bulkhead()
    print("building leg");        leg = make_leg()
    foot = make_foot()
    vane = make_vane()
    print("splitting checker")
    mid_r, mid_w = split_checker(mid, Z_J1 - SPIGOT - 1, Z_J2)
    nose_r, nose_w = split_checker(nose, Z_J2, Z_TIP)
    return dict(aft=aft, mid=mid, nose=nose, bulk=bulk, leg=leg, foot=foot, vane=vane,
                mid_r=mid_r, mid_w=mid_w, nose_r=nose_r, nose_w=nose_w)


RED, WHITE, DARK, GREY = (0.82, 0.10, 0.08), (0.96, 0.95, 0.92), (0.16, 0.16, 0.18), (0.6, 0.6, 0.62)


def mesh_of(wp, tol=0.25, ang=0.35):
    verts, tris = [], []
    for s in wp.vals():
        v, t = s.tessellate(tol, ang)
        off = len(verts)
        verts += [(p.x, p.y, p.z) for p in v]
        tris += [(a + off, b + off, c + off) for a, b, c in t]
    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.uint32)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "out")
    for d in ("stl", "step", "viewer"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    P = build()

    def vol(k):
        return sum(s.Volume() for s in P[k].vals()) / 1000.0   # cm3

    def area(k):
        return sum(s.Area() for s in P[k].vals()) / 100.0      # cm2

    PLA = 1.24
    # thin shells print solid; legs print as 2 perimeters + sparse infill
    leg_g = (area("leg") * 0.09 + vol("leg") * 0.08) * PLA
    rows = [
        ("Aft body (fan mount, nozzle, leg clevises)", 1, vol("aft") * PLA, "aft_body.stl"),
        ("Mid body (inlets, battery bay)", 1, vol("mid") * PLA, "mid_body.stl"),
        ("Nose cone", 1, vol("nose") * PLA, "nose_cone.stl"),
        ("Battery bulkhead", 1, vol("bulk") * PLA, "bulkhead.stl"),
        ("Leg (print as two halves)", 3, leg_g, "leg_half_a.stl + leg_half_b.stl"),
        ("Thrust vane", 4, vol("vane") * PLA, "vane.stl"),
        ("Foot pad (TPU)", 3, vol("foot") * 1.21, "foot_pad.stl"),
    ]
    total = sum(q * m for _, q, m, _ in rows)

    # ---- STL, print orientation
    def stl(wp, name):
        cq.exporters.export(wp, os.path.join(out, "stl", name), tolerance=0.03,
                            angularTolerance=0.15)

    stl(P["aft"].translate((0, 0, -Z_EXIT)), "aft_body.stl")
    stl(P["mid"].translate((0, 0, -(Z_J1 - SPIGOT))), "mid_body.stl")
    stl(P["mid_r"].translate((0, 0, -(Z_J1 - SPIGOT))), "mid_body_colour1_red.stl")
    stl(P["mid_w"].translate((0, 0, -(Z_J1 - SPIGOT))), "mid_body_colour2_white.stl")
    stl(P["nose"].translate((0, 0, -(Z_J2 - SPIGOT))), "nose_cone.stl")
    stl(P["nose_r"].translate((0, 0, -(Z_J2 - SPIGOT))), "nose_cone_colour1_red.stl")
    stl(P["nose_w"].translate((0, 0, -(Z_J2 - SPIGOT))), "nose_cone_colour2_white.stl")
    stl(P["bulk"].translate((0, 0, -Z_BULK)), "bulkhead.stl")
    big = cq.Workplane("XY").box(600, 300, 800, centered=(True, False, True))
    ha = P["leg"].intersect(big)                       # y >= 0 half
    hb = P["leg"].cut(big)                             # y <= 0 half
    stl(ha.rotate((0, 0, 0), (1, 0, 0), 90), "leg_half_a.stl")
    stl(hb.rotate((0, 0, 0), (1, 0, 0), -90), "leg_half_b.stl")
    stl(P["vane"].translate((0, 0, -VANE_Z)).rotate((0, 0, 0), (1, 0, 0), 90), "vane.stl")
    stl(P["foot"].translate((-R_FOOT, 0, 0)), "foot_pad.stl")

    # ---- STEP assembly
    G = ghosts()
    a = cq.Assembly(name="retro_edf_rocket")
    a.add(P["aft"], name="aft_body", color=cq.Color(*RED))
    a.add(P["mid_r"], name="mid_body_red", color=cq.Color(*RED))
    a.add(P["mid_w"], name="mid_body_white", color=cq.Color(*WHITE))
    a.add(P["nose_r"], name="nose_red", color=cq.Color(*RED))
    a.add(P["nose_w"], name="nose_white", color=cq.Color(*WHITE))
    a.add(P["bulk"], name="bulkhead", color=cq.Color(*GREY))
    for i, az in enumerate(LEG_AZ):
        a.add(rotz(P["leg"], az), name=f"leg_{i+1}", color=cq.Color(*RED))
        a.add(rotz(P["foot"], az), name=f"foot_{i+1}", color=cq.Color(*DARK))
    for i, az in enumerate(VANE_AZ):
        a.add(rotz(P["vane"], az), name=f"vane_{i+1}", color=cq.Color(*GREY))
    a.save(os.path.join(out, "step", "rocket_assembly.step"))
    for k, n in (("aft", "aft_body"), ("mid", "mid_body"), ("nose", "nose_cone"),
                 ("leg", "leg"), ("bulk", "bulkhead"), ("vane", "vane")):
        cq.exporters.export(P[k], os.path.join(out, "step", n + ".step"))

    # ---- viewer meshes: one binary blob + manifest
    items = [
        ("Aft body", "aft", P["aft"], RED, 0, None),
        ("Mid body", "mid", P["mid_r"], RED, 1, None),
        ("Mid body", "mid", P["mid_w"], WHITE, 1, None),
        ("Nose cone", "nose", P["nose_r"], RED, 2, None),
        ("Nose cone", "nose", P["nose_w"], WHITE, 2, None),
        ("Bulkhead", "bulk", P["bulk"], GREY, 1.5, None),
    ]
    for az in LEG_AZ:
        items.append(("Legs", "legs", rotz(P["leg"], az), RED, 0, az))
        items.append(("Legs", "legs", rotz(P["foot"], az), DARK, 0, az))
    for az in VANE_AZ:
        items.append(("Vanes", "vanes", rotz(P["vane"], az), GREY, -0.6, None))
    for name, wp in G.items():
        lvl = {"EDF 90 mm": 0.5, "Vane servos x4": -0.3, "ESC": 1.2}.get(name, 1.5)
        items.append((name, "ghost", wp, (0.25, 0.45, 0.75), lvl, None))

    blob, manifest = bytearray(), []
    for label, group, wp, col, lvl, az in items:
        v, t = mesh_of(wp)
        manifest.append(dict(label=label, group=group, color=[round(c, 3) for c in col],
                             level=lvl, az=az, voff=len(blob), vn=int(len(v))))
        blob += v.tobytes()
        manifest[-1].update(toff=len(blob), tn=int(len(t)))
        blob += t.tobytes()
    fsa = math.pi * ((EDF_ID / 2) ** 2 - (EDF_HUB / 2) ** 2)
    exit_r = ro(Z_EXIT) - WALL_AFT
    win_area = 6 * (37 * 36 + 37 * 12)
    meta = dict(
        height=Z_TIP, max_dia=round(2 * max(ro(z) for z in np.linspace(Z_EXIT, Z_TIP, 500)), 1),
        foot_circle=2 * R_FOOT, fsa=round(fsa), exit_pct=round(100 * math.pi * exit_r ** 2 / fsa),
        inlet_pct=round(100 * win_area / fsa), printed_g=round(total),
        rows=[dict(name=n, qty=q, each=round(m, 1), file=f) for n, q, m, f in rows],
        stations=dict(exit=Z_EXIT, vane=VANE_Z, seat=Z_SEAT, j1=Z_J1, bulk=Z_BULK, j2=Z_J2, tip=Z_TIP),
    )
    with open(os.path.join(out, "viewer", "meshes.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(out, "viewer", "manifest.json"), "w") as f:
        json.dump(dict(parts=manifest, meta=meta), f)

    print(f"\n{'part':46s} qty   g each")
    for n, q, m, _ in rows:
        print(f"{n:46s} {q:3d} {m:8.1f}")
    print(f"{'printed total (PLA @1.24)':46s}     {total:8.1f}")
    print("meta:", json.dumps({k: v for k, v in meta.items() if k not in ('rows',)}))
    print("viewer blob: %.2f MB, %d tris" % (len(blob) / 1e6, sum(m['tn'] for m in manifest)))


if __name__ == "__main__":
    main()
