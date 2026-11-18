#!/usr/bin/env python3
"""
Checker rocket, rev D - ducted-fan hover version (CadQuery).

Same traced outer shape as rev C, scaled so a 90 mm ducted fan fits inside it:
  - fan in the lower checker, breathing through a ring gap hidden on a checker row line
  - the tail is opened up into the exhaust duct, with four steering vanes and servo mounts
  - battery in the widest part of the body, flight electronics above it
  - two forward-looking camera ports in the nose (stereo pair) and a GPS seat in the tip

Coordinates: Z up, Z=0 is the ground under the feet. All dimensions in mm.
MEASURE YOUR FAN, SERVOS AND CAMERAS before printing: FAN, SERVO and CAM below drive the fits.

    python rocket.py          # writes ./out/{stl,step,viewer}
"""
import argparse
import json
import math
import os

import cadquery as cq
import numpy as np
from scipy.interpolate import PchipInterpolator

# ------------------------------------------------------------------ parameters
P = dict(
    H=900.0,       # overall height, ground to nose tip
    D=118.0,       # maximum body diameter (reference is H / 7.75; 118 gives the fan 8 mm of margin)
    STANCE=376.0,  # foot circle diameter (reference: 3.19 x D)
    ROWS=5, COLS=4,
)
FAN = dict(od=98.0, id=90.0, length=80.0, hub=38.0)             # 90 mm class EDF housing
SERVO = dict(L=22.6, W=12.2, ears=27.8, above_ears=6.8, depth=22.7, shaft_off=5.2)   # 9 g class
CAM = dict(eye_y=25.0, port=16.0, lens=9.0, plate_x=10.0)       # stereo pair: 50 mm baseline
BATTERY = (50.0, 50.0, 155.0)                                    # 6S 5000 mAh class

# Outer mould line, normalised (z / H, r / (D/2)). Above the fins this is the traced
# reference shape; below them the tail is kept wide because it is the fan's exhaust duct.
SHAPE = dict(
    profile=[(0.1667, 0.72), (0.235, 0.745), (0.30, 0.765), (0.36, 0.785), (0.415, 0.80),
             (0.467, 0.835), (0.501, 0.885), (0.55, 0.94), (0.60, 0.98), (0.673, 1.0),
             (0.742, 0.965), (0.776, 0.92), (0.811, 0.85), (0.845, 0.795), (0.88, 0.69),
             (0.914, 0.545), (0.948, 0.37), (0.975, 0.20), (1.0, 0.0)],
    t_exit=0.1667,    # nozzle exit (bottom of the vane ring)
    t_tail=0.235,     # where the fins' arch meets the body; vane ring joins here
    t_chk0=0.415,     # checker band bottom
    t_chk1=0.845,     # checker band top = nose joint
)
LEG = dict(
    t_root_top=0.412, t_pod_tip=0.25, t_pod_max=0.085, t_pod_bot=0.008, t_cut=0.211,
    pod_r=0.38, fin_w=0.085,
    outer_ctrl=(0.642, 0.39), inner_ctrl=(0.516, 0.074),
    pod=[(0.26, 0.0), (0.80, 0.09), (0.97, 0.22), (1.0, 0.34), (0.93, 0.52), (0.76, 0.69),
         (0.50, 0.84), (0.22, 0.94), (0.0, 1.0)],
)

WD, W, FIT, LAP = 1.6, 1.0, 0.15, 8.0     # duct wall, upper skin wall, sliding fit, lap joint length
GAP, LIP = 26.0, 3.2                        # intake ring gap height, fan retaining lip
LEG_AZ = [0, 120, 240]
VANE_AZ = [45, 135, 225, 315]
STRUT_AZ = [45, 135, 225, 315]
FRONT_AZ = 180                              # the cameras look this way
WIRE_AZ = 60
SECT_K = 12
CHK_PHASE = 22.5                            # checker cell edges kept off the model's symmetry planes and seams
FC_X = -10.0                                # flight controller sits behind the camera upright


def derive():
    g = globals()
    H, D = P["H"], P["D"]
    g["H"], g["R"] = H, D / 2
    g["_p"] = PchipInterpolator([t for t, _ in SHAPE["profile"]], [r for _, r in SHAPE["profile"]])
    g["Z_EXIT"], g["Z_TAIL"] = SHAPE["t_exit"] * H, SHAPE["t_tail"] * H
    g["Z_CHK0"], g["Z_CHK1"] = SHAPE["t_chk0"] * H, SHAPE["t_chk1"] * H
    g["ROW_H"] = (Z_CHK1 - Z_CHK0) / P["ROWS"]
    g["R_FOOT"], g["POD_R"], g["FIN_W"] = P["STANCE"] / 2, LEG["pod_r"] * D, LEG["fin_w"] * D
    g["RBORE"] = FAN["od"] / 2 + 0.2

    # lowest station wide enough for the fan, then snap the intake gap up to a checker row line
    z = Z_CHK0
    while ro(z) < RBORE + WD and z < 0.673 * H:
        z += 0.25
    bounds = [Z_CHK0 + k * ROW_H for k in range(P["ROWS"] + 1)]
    ok = [b for b in bounds if b >= z + FAN["length"] + LIP - 0.01 and b < 0.70 * H]
    if not ok:
        raise SystemExit("The fan does not fit this body: increase D")
    g["Z_GAP0"] = ok[0]
    g["Z_GAP1"] = Z_GAP0 + GAP
    g["Z_FTOP"] = Z_GAP0 - LIP
    g["Z_SEAT"] = Z_FTOP - FAN["length"]
    g["Z_J1"] = Z_GAP0 - ROW_H if Z_GAP0 - ROW_H > Z_SEAT + 3 else Z_SEAT + 5   # tail duct / fan ring joint
    g["Z_J2"] = Z_CHK1
    g["Z_AV"] = Z_J2 - 16.0                    # avionics plate
    g["Z_EYE"] = Z_J2 + 20.0                   # camera ports
    g["Z_LEGCUT"] = LEG["t_cut"] * H

    # servo bracket on the vane ring: fixes the vane pivot height
    g["R_VR"] = ro(Z_EXIT + 30)
    g["R_PO"] = R_VR + 20.5                    # outer face of the servo plate
    g["Z_BR0"] = Z_EXIT
    g["Z_PB"] = Z_BR0 + (R_PO - R_VR + 1.0)
    g["Z_PT"] = Z_PB + SERVO["ears"] + 5.0
    g["Z_SC"] = (Z_PB + Z_PT) / 2
    g["Z_VANE"] = Z_SC - SERVO["shaft_off"]
    assert Z_PT < Z_TAIL - 1, "servo bracket does not fit under the tail joint"
    g["STRUT_R"] = ro(Z_GAP1) - 7.0
    z_rt = LEG["t_root_top"] * H
    g["LEG_SCREW_Z"] = [Z_TAIL + f * (z_rt - Z_TAIL) for f in (0.25, 0.55, 0.85)]


def ro(z):
    return float(R * _p(min(max(z / H, SHAPE["t_exit"]), 1.0)))


# ------------------------------------------------------------------ helpers
def _zs(z0, z1, step=4.0):
    n = max(int(abs(z1 - z0) / step), 3) + 1
    return np.linspace(z0, z1, n)


def solid_under(fn, z0, z1):
    pts = [(max(float(fn(v)), 0.0), float(v)) for v in _zs(z0, z1)]
    wp = cq.Workplane("XZ").moveTo(0, z0).lineTo(*pts[0])
    wp = wp.spline(pts[1:], includeCurrent=True)
    if pts[-1][0] > 1e-6:
        wp = wp.lineTo(0, z1)
    return wp.close().revolve(360, (0, 0, 0), (0, 1, 0))


def ring(poly):
    return cq.Workplane("XZ").polyline(poly).close().revolve(360, (0, 0, 0), (0, 1, 0))


def band(fn_out, thick, z0, z1):
    zs = _zs(z0, z1, 2.0)
    return ring([(fn_out(z), z) for z in zs] + [(fn_out(z) - thick, z) for z in zs[::-1]])


def shell(z0, z1, wall):
    return solid_under(ro, z0, z1).cut(solid_under(lambda z: ro(z) - wall, z0 - 1, z1 + 1))


def rotz(obj, deg):
    return obj.rotate((0, 0, 0), (0, 0, 1), deg)


def vcyl(x, y, z0, z1, r):
    return cq.Workplane("XY").workplane(offset=z0).center(x, y).circle(r).extrude(z1 - z0)


def xcyl(y, z, x0, x1, r):
    """Cylinder along +X from x0 to x1."""
    return cq.Workplane("YZ").workplane(offset=x0).center(y, z).circle(r).extrude(x1 - x0)


def box(x0, x1, y0, y1, z0, z1):
    return (cq.Workplane("XY").workplane(offset=z0).center((x0 + x1) / 2, (y0 + y1) / 2)
            .rect(x1 - x0, y1 - y0).extrude(z1 - z0))


def xz_plate(poly, y0, y1):
    """Extrude an (x, z) polygon from y0 to y1."""
    p = cq.Workplane("XZ").polyline(poly).close().extrude(-(y1 - y0))
    return p.translate((0, y0, 0))


def qbez(p0, p1, p2, n):
    t = np.linspace(0, 1, n + 1)[:, None]
    return ((1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)).tolist()


# ------------------------------------------------------------------ fins and pods
def leg_curves():
    z_rt, z_pt, z_pm = LEG["t_root_top"] * H, LEG["t_pod_tip"] * H, LEG["t_pod_max"] * H
    p0, p2 = (ro(z_rt) - 1.0, z_rt), (R_FOOT, z_pt)
    p1 = (p0[0] + LEG["outer_ctrl"][0] * (p2[0] - p0[0]), p0[1] + LEG["outer_ctrl"][1] * (p2[1] - p0[1]))
    q0, q2 = (R_FOOT - 0.9 * POD_R, z_pm), (ro(Z_TAIL) - 1.0, Z_TAIL)
    q1 = (q2[0] + LEG["inner_ctrl"][0] * (q0[0] - q2[0]), Z_TAIL - LEG["inner_ctrl"][1] * (Z_TAIL - z_pm))
    return (p0, p1, p2), (q0, q1, q2)


def _interp(pts, z):
    if z <= pts[0][1]:
        return pts[0][0]
    for a, b in zip(pts, pts[1:]):
        if z <= b[1]:
            f = (z - a[1]) / (b[1] - a[1]) if b[1] > a[1] else 0.0
            return a[0] + f * (b[0] - a[0])
    return pts[-1][0]


def leg_slices():
    outer, inner = leg_curves()
    out = qbez(*outer, 40)[::-1]
    inn = qbez(*inner, 24)
    z_rt, z_pt = LEG["t_root_top"] * H, LEG["t_pod_tip"] * H
    zs = [p[1] for p in inn] + [Z_TAIL + (z_rt - Z_TAIL) * i / 18 for i in range(1, 19)]
    res = []
    for z in zs:
        xo = R_FOOT if z <= z_pt else _interp(out, z)
        xi = _interp(inn, z) if z < Z_TAIL else inn[-1][0]
        res.append((z, xi, xo, FIN_W))
    return res


def sect_pts(xi, xo, w, K=SECT_K):
    s0 = min(max(w / (xo - xi), 0.08), 0.45)
    pts = []
    for i in range(K + 1):
        s = (i / K) ** 1.5
        f = math.sqrt(max(1 - (1 - s / s0) ** 2, 0.0)) if s < s0 else 1 - ((s - s0) / (1 - s0)) ** 1.6
        pts.append((xi + s * (xo - xi), w * max(f, 0.0)))
    return pts


def pod_profile():
    zb, zt = LEG["t_pod_bot"] * H, LEG["t_pod_tip"] * H
    return [(r * POD_R, zb + s * (zt - zb)) for r, s in LEG["pod"]]


def make_leg():
    """One fin with its pod, trimmed to the body, as (upper, lower) printable pieces."""
    wires = []
    for z, xi, xo, w in leg_slices():
        up = sect_pts(xi, xo, w)
        e1 = cq.Edge.makeSpline([cq.Vector(x, y, z) for x, y in up])
        e2 = cq.Edge.makeSpline([cq.Vector(x, -y, z) for x, y in reversed(up)])
        wires.append(cq.Wire.assembleEdges([e1, e2]))
    fin = cq.Workplane("XY").newObject([cq.Solid.makeLoft(wires, True)])
    pp = pod_profile()
    pod = (cq.Workplane("XZ").moveTo(0, pp[0][1]).lineTo(*pp[0])
           .spline(pp[1:], includeCurrent=True).close()
           .revolve(360, (0, 0, 0), (0, 1, 0)).translate((R_FOOT, 0, 0)))
    leg = fin.union(pod)
    leg = leg.cut(solid_under(lambda z: ro(z) + 0.25, Z_EXIT - 5, Z_J1 + 5))     # saddle to the duct
    leg = leg.cut(vcyl(R_FOOT, 0, pp[0][1] - 1, pp[0][1] + 9, 2.1))              # foot pad socket
    for z in LEG_SCREW_Z:                                                        # M3 heat-set inserts
        leg = leg.cut(xcyl(0, z, ro(z) - 2, ro(z) + 9.5, 2.0))
    # alignment pins across the print split
    x_in = _interp(qbez(*leg_curves()[1], 24), Z_LEGCUT)
    for x in (R_FOOT, (x_in + R_FOOT) / 2):
        leg = leg.cut(vcyl(x, 0, Z_LEGCUT - 10, Z_LEGCUT + 10, 1.55))
    big = box(-50, 400, -200, 200, Z_LEGCUT, 2000)
    return leg, leg.intersect(big), leg.cut(big)


def make_foot():
    zb = LEG["t_pod_bot"] * H
    f = cq.Workplane("XY").circle(0.26 * POD_R).extrude(zb)
    return f.union(vcyl(0, 0, zb, zb + 8, 1.95)).translate((R_FOOT, 0, 0))


# ------------------------------------------------------------------ body sections
def lap_cut(z0, z1):
    """Remove the outer half of the wall between z0 and z1 (this part keeps the inner half)."""
    return band(lambda z: ro(z) + 0.6, 0.6 + WD / 2 + 0.05, z0, z1)


def make_vane_ring():
    body = shell(Z_EXIT, Z_TAIL, WD).union(band(ro, WD / 2, Z_TAIL - 1.0, Z_TAIL + LAP))
    cavity = solid_under(lambda z: ro(z) - WD, Z_EXIT - 2, Z_TAIL + LAP + 2)
    core = solid_under(lambda z: ro(z) - 0.8, Z_EXIT - 2, Z_TAIL + LAP + 2)
    rv = ro(Z_VANE)
    hw = SERVO["W"] / 2
    # servo bracket: two cheeks with a 45 deg underside, and a plate the servo ears screw to
    cheek = [(rv - 1.0, Z_BR0), (R_PO, Z_PB), (R_PO, Z_PT), (R_PO - 8.0, Z_PT), (rv - 1.0, Z_PT - 12.0)]   # top corner relieved for the fins
    br = xz_plate(cheek, hw, hw + 2.4).union(xz_plate(cheek, -hw - 2.4, -hw)).cut(core)
    plate = box(R_PO - 2.0, R_PO, -hw - 2.4, hw + 2.4, Z_PB, Z_PT)
    plate = plate.cut(box(R_PO - 3, R_PO + 1, -hw, hw, Z_SC - SERVO["L"] / 2, Z_SC + SERVO["L"] / 2))
    for dz in (-SERVO["ears"] / 2, SERVO["ears"] / 2):
        plate = plate.cut(xcyl(0, Z_SC + dz, R_PO - 4, R_PO + 2, 0.8))
    boss = xcyl(0, Z_VANE, 20, rv + 4.0, 5.5).cut(cavity)
    hole = xcyl(0, Z_VANE, 20, rv + 8.0, 3.2)
    for az in VANE_AZ:
        body = body.union(rotz(br.union(plate), az)).union(rotz(boss, az)).cut(rotz(hole, az))
    return body


def make_tail_duct():
    body = shell(Z_TAIL, Z_J1, WD)
    ri = ro(Z_SEAT) - WD
    # fan seat: ledge at the fan duct diameter, 45 deg underside
    body = body.union(ring([(ri + 0.6, Z_SEAT), (FAN["id"] / 2, Z_SEAT), (FAN["id"] / 2, Z_SEAT - 1.5),
                            (ri + 0.6, Z_SEAT - 1.5 - (ri + 0.6 - FAN["id"] / 2))]))
    body = body.cut(lap_cut(Z_J1 - LAP, Z_J1 + 0.5)).cut(lap_cut(Z_TAIL - 0.5, Z_TAIL + LAP))
    for az in LEG_AZ:                                  # M3 clearance for the leg screws
        for z in LEG_SCREW_Z:
            body = body.cut(rotz(xcyl(0, z, ro(z) - 6, ro(z) + 2, 1.7), az))
    return body


def make_fan_ring():
    body = shell(Z_J1, Z_GAP0, WD).union(band(ro, WD / 2, Z_J1 - LAP, Z_J1 + 1.0))
    ri = lambda z: ro(z) - WD
    zb = Z_FTOP - 6.0
    za = zb - (ri(zb) - RBORE) - 0.5
    # ring that centres the top of the fan and holds it down against its own thrust
    back = [(ri(z) + 0.5, z) for z in np.linspace(Z_GAP0, za, 8)]
    lip = ring([(RBORE, zb), (RBORE, Z_FTOP), (RBORE - 2.2, Z_FTOP + 2.2), (RBORE - 2.2, Z_GAP0)] + back)
    lip = lip.cut(rotz(box(RBORE - 4, R + 5, -8, 8, za - 2, Z_GAP0 + 1), WIRE_AZ)
                  .intersect(solid_under(lambda z: ro(z) - WD, Z_J1, Z_GAP0 + 2)))    # motor wire notch
    body = body.union(lip)
    outer = solid_under(lambda z: ro(z) - 0.3, Z_J1, Z_GAP1)     # struts sit a hair inside the mould line
    strut = box(RBORE - 3.3, R + 5, -3.0, 3.0, Z_GAP0 - 2.0, Z_GAP1).intersect(outer)
    strut = strut.cut(vcyl(STRUT_R, 0, Z_GAP1 - 9, Z_GAP1 + 1, 2.0))               # M3 heat-set insert
    for az in STRUT_AZ:
        body = body.union(rotz(strut, az))
    return body


def make_upper():
    body = shell(Z_GAP1, Z_J2, W)
    floor = solid_under(lambda z: ro(z) - 0.5, Z_GAP1, Z_GAP1 + 2.5)
    for az in STRUT_AZ:
        a = math.radians(az)
        floor = floor.cut(vcyl(STRUT_R * math.cos(a), STRUT_R * math.sin(a), Z_GAP1 - 1, Z_GAP1 + 4, 1.7))
    for x in (-BATTERY[0] / 2 - 4, BATTERY[0] / 2 + 4):                              # battery strap
        floor = floor.cut(box(x - 2, x + 2, -13, 13, Z_GAP1 - 1, Z_GAP1 + 4))
    floor = floor.cut(box(-15, 15, 32, 40, Z_GAP1 - 1, Z_GAP1 + 4))                  # ESC cooling slot
    floor = floor.cut(vcyl(0, -38, Z_GAP1 - 1, Z_GAP1 + 4, 6.0))                     # wiring
    body = body.union(floor)
    rw = ro(Z_AV) - W
    r_led = ro(Z_J2) - W - FIT - 1.2 - 3.5
    body = body.union(ring([(rw + 0.6, Z_AV), (r_led, Z_AV), (r_led, Z_AV - 1.5),
                            (rw + 0.6, Z_AV - 1.5 - (rw + 0.6 - r_led))]))
    return body


def make_nose():
    outer = solid_under(ro, Z_J2, H)
    zc = Z_J2
    while ro(zc) - W > 3.0:
        zc += 0.5
    body = outer.cut(solid_under(lambda z: ro(z) - W, Z_J2 - 1, zc))
    for y in (-CAM["eye_y"], CAM["eye_y"]):                                          # camera ports
        body = body.cut(rotz(xcyl(y, Z_EYE, 5, R + 10, CAM["port"] / 2), FRONT_AZ))
    return body


def make_avionics():
    """Drop-in plate above the battery: flight controller on the plate, cameras on the upright."""
    rp = ro(Z_J2) - W - FIT - 1.2 - 0.8
    plate = vcyl(0, 0, Z_AV, Z_AV + 2.5, rp)
    for dx in (-15.25, 15.25):                         # 30.5 mm flight-controller pattern
        for dy in (-15.25, 15.25):
            plate = plate.cut(vcyl(FC_X + dx, dy, Z_AV - 1, Z_AV + 4, 1.6))
    plate = plate.cut(vcyl(0, -rp + 9, Z_AV - 1, Z_AV + 4, 6.0))                     # cable hole
    px = CAM["plate_x"]
    room = solid_under(lambda z: ro(z) - W - 3.0, Z_AV, Z_EYE + 24)   # clears the nose shoulder ring
    wall = box(px, px + 2.5, -45, 45, Z_AV, Z_EYE + 16).intersect(room)
    for y in (-CAM["eye_y"], CAM["eye_y"]):
        wall = wall.cut(xcyl(y, Z_EYE, px - 1, px + 4, CAM["lens"] / 2))
        for dy in (-10.5, 10.5):                       # 21 x 12.5 mm camera-board hole pattern
            for dz in (-6.25, 6.25):
                wall = wall.cut(xcyl(y + dy, Z_EYE + dz, px - 1, px + 4, 0.9))
    gus = [(px, Z_AV + 2.5), (px - 22, Z_AV + 2.5), (px, Z_AV + 21)]
    for y in (-22, 22):
        wall = wall.union(xz_plate(gus, y - 1.2, y + 1.2))
    # built facing +X, then turned so the cameras look out of the front
    return plate.union(wall)        # modelled facing +X; main() turns it to FRONT_AZ


def make_gps_seat():
    r = 21.0
    d = vcyl(0, 0, 0, 2.0, r).cut(box(-13, 13, -13, 13, 1.0, 3.0)).cut(vcyl(0, 9, -1, 3, 2.5))
    z = Z_J2
    while ro(z) - W > r + 0.3:
        z += 0.5
    return d.translate((0, 0, z - 2.0)), z - 2.0


def make_coupler(zj, wall, down, up):
    eps = 2.0
    grows = ro(zj + eps) > ro(zj - eps)
    r_j = ro(zj) - wall - FIT

    def r_out(z):
        conform = (z < zj and grows) or (z > zj and not grows)
        return ro(z) - wall - FIT if conform else r_j

    return band(r_out, 1.2, zj - down, zj + up)


def make_vane():
    r_tip = ro(Z_VANE) - WD - 2.0
    v = xz_plate([(5, Z_VANE - 28), (r_tip, Z_VANE - 28), (r_tip, Z_VANE + 12), (5, Z_VANE + 12)], -1.5, 1.5)
    try:
        v = v.edges("|X").chamfer(1.1, 4.0)
    except Exception:
        pass
    return v.union(xcyl(0, Z_VANE, 3, ro(Z_VANE) + 9.0, 3.0))


# ------------------------------------------------------------------ bought parts (shown in the viewer only)
def ghosts():
    g = {}
    f = FAN
    edf = vcyl(0, 0, Z_SEAT, Z_FTOP, f["od"] / 2).cut(vcyl(0, 0, Z_SEAT - 1, Z_FTOP + 1, f["id"] / 2))
    hub = vcyl(0, 0, Z_SEAT - 28, Z_FTOP, f["hub"] / 2)
    spin = (cq.Workplane("XZ").polyline([(0, Z_FTOP), (f["hub"] / 2, Z_FTOP), (0, Z_FTOP + 20)]).close()
            .revolve(360, (0, 0, 0), (0, 1, 0)))
    g["Ducted fan, 90 mm"] = edf.union(hub).union(spin)
    bx, by, bz = BATTERY
    zf = Z_GAP1 + 2.5
    g["Battery, 6S"] = box(-bx / 2, bx / 2, -by / 2, by / 2, zf, zf + bz)
    g["Speed controller"] = box(-19, 19, 30, 42, zf + 2, zf + 87)
    g["Flight controller"] = box(FC_X - 18, FC_X + 18, -18, 18, Z_AV + 6, Z_AV + 14)
    px = CAM["plate_x"]
    cams = None
    for y in (-CAM["eye_y"], CAM["eye_y"]):
        c = box(px - 9, px, y - 12.5, y + 12.5, Z_EYE - 12, Z_EYE + 12).union(xcyl(y, Z_EYE, px, px + 12, 4.0))
        cams = c if cams is None else cams.union(c)
    g["Stereo cameras"] = cams
    g["Depth and companion board"] = box(px - 16, px - 11, -30, 30, Z_AV + 24, Z_AV + 54)
    _, zg = make_gps_seat()
    g["GPS"] = box(-12.5, 12.5, -12.5, 12.5, zg + 1, zg + 9)
    s = SERVO
    servo = box(R_PO - s["above_ears"], R_PO - s["above_ears"] + s["depth"], -s["W"] / 2, s["W"] / 2,
                Z_SC - s["L"] / 2, Z_SC + s["L"] / 2).union(xcyl(0, Z_VANE, R_PO - s["above_ears"] - 5, R_PO, 2.4))
    sv = None
    for az in VANE_AZ:
        r = rotz(servo, az)
        sv = r if sv is None else sv.union(r)
    g["Vane servos"] = sv
    return g


# ------------------------------------------------------------------ checker split
def split_checker(part, zlo, zhi):
    """Split a part into (red, white) bodies along the checker cells.
    Done one cell at a time: a single boolean against all cells at once can fail silently."""
    red, white = part, []
    step = 360.0 / P["COLS"]
    v0 = vol(part)
    for j in range(P["ROWS"]):
        z0, z1 = max(Z_CHK0 + j * ROW_H, zlo), min(Z_CHK0 + (j + 1) * ROW_H, zhi)
        if z1 - z0 < 1:
            continue
        cell = (cq.Workplane("XZ").polyline([(2, z0), (R + 20, z0), (R + 20, z1), (2, z1)])
                .close().revolve(step, (0, 0, 0), (0, 1, 0)))
        for i in range(P["COLS"]):
            if (i + j) % 2:
                continue
            c = rotz(cell, CHK_PHASE + i * step)
            w = part.intersect(c).solids().vals()
            if not w:
                print("  WARNING: checker cell row %d col %d came back empty" % (j, i))
            white += w
            red = red.cut(c)
    if not white:
        return part, None
    wwp = cq.Workplane("XY").newObject(white)
    lost = abs(vol(red) + vol(wwp) - v0)
    if lost > 0.02 * v0:
        print("  WARNING: checker split lost %.1f cm3 of %.1f" % (lost, v0))
    print("  %d white cells" % len(white))
    return red, wwp


# ------------------------------------------------------------------ build + export
RED, WHITE, DARK, GREY, BLUE = (0.84, 0.09, 0.06), (0.97, 0.96, 0.92), (0.13, 0.13, 0.14), (0.62, 0.62, 0.64), (0.22, 0.42, 0.72)


def vol(wp):
    return sum(s.Volume() for s in wp.vals()) / 1000.0


def area(wp):
    return sum(s.Area() for s in wp.vals()) / 100.0


def valid(wp):
    return all(s.isValid() for s in wp.vals())


def mesh_of(wp, tol=0.25, ang=0.3):
    verts, tris = [], []
    for s in wp.vals():
        v, t = s.tessellate(tol, ang)
        off = len(verts)
        verts += [(p.x, p.y, p.z) for p in v]
        tris += [(a + off, b + off, c + off) for a, b, c in t]
    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    derive()
    here = os.path.dirname(os.path.abspath(__file__))
    out = a.out or os.path.join(here, "out")
    for d in ("stl", "step", "viewer"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
        for f in os.listdir(os.path.join(out, d)):
            os.remove(os.path.join(out, d, f))
    front = lambda wp: rotz(wp, FRONT_AZ)      # parts modelled facing +X, cameras look along FRONT_AZ

    print("vane ring");   vring = make_vane_ring()
    print("tail duct");   tail = make_tail_duct()
    print("fan ring");    fring = make_fan_ring()
    print("upper body");  upper = make_upper()
    print("nose");        nose = make_nose()
    print("avionics");    avio = front(make_avionics())
    gps, z_gps = make_gps_seat()
    cpl = make_coupler(Z_J2, W, 10.0, 12.0)
    vane = make_vane()
    foot = make_foot()
    print("legs");        leg, leg_up, leg_lo = make_leg()
    print("checker")
    tail_r, tail_w = split_checker(tail, Z_TAIL, Z_J1)
    fring_r, fring_w = split_checker(fring, Z_J1, Z_GAP0)
    upper_r, upper_w = split_checker(upper, Z_GAP1, Z_J2)
    parts = dict(vane_ring=vring, tail_duct=tail, fan_ring=fring, upper_body=upper, nose=nose,
                 avionics_plate=avio, gps_seat=gps, nose_shoulder=cpl, vane=vane, leg_upper=leg_up,
                 leg_lower=leg_lo, foot_pad=foot)
    for n, wp in parts.items():
        bb = cq.Compound.makeCompound(wp.vals()).BoundingBox()
        print(f"  {n:15s} solids={len(wp.solids().vals())} valid={valid(wp)} "
              f"size {bb.xlen:.0f} x {bb.ylen:.0f} x {bb.zlen:.0f}")

    PLA = 1.24
    leg_wall, leg_fill = 0.045, 0.03          # one perimeter + 3 % infill: these are big hollow shapes
    lg = lambda wp: (area(wp) * leg_wall + vol(wp) * leg_fill) * PLA
    rows = [
        dict(key="nose", name="Nose, with camera ports", qty=1, g=vol(nose) * PLA, tall=H - Z_J2, file="nose.stl", light=True),
        dict(key="upper", name="Upper body (battery bay)", qty=1, g=vol(upper) * PLA, tall=Z_J2 - Z_GAP1, file="upper_body.stl", light=True),
        dict(key="fring", name="Fan ring, with intake struts", qty=1, g=vol(fring) * PLA, tall=Z_GAP1 - Z_J1 + LAP, file="fan_ring.stl", light=False),
        dict(key="tail", name="Tail duct", qty=1, g=vol(tail) * PLA, tall=Z_J1 - Z_TAIL, file="tail_duct.stl", light=False),
        dict(key="vring", name="Vane ring, with servo mounts", qty=1, g=vol(vring) * PLA, tall=Z_TAIL + LAP - Z_EXIT, file="vane_ring.stl", light=False),
        dict(key="leg_up", name="Leg, upper half", qty=3, g=lg(leg_up), tall=LEG["t_root_top"] * H - Z_LEGCUT, file="leg_upper.stl", light=True),
        dict(key="leg_lo", name="Leg, lower half with pod", qty=3, g=lg(leg_lo), tall=Z_LEGCUT - LEG["t_pod_bot"] * H, file="leg_lower.stl", light=True),
        dict(key="vane", name="Steering vane", qty=4, g=vol(vane) * PLA, tall=3, file="vane.stl", light=False),
        dict(key="avio", name="Avionics plate with camera upright", qty=1, g=vol(avio) * PLA, tall=Z_EYE + 16 - Z_AV, file="avionics_plate.stl", light=False),
        dict(key="cpl", name="Nose shoulder ring", qty=1, g=vol(cpl) * PLA, tall=22, file="nose_shoulder.stl", light=False),
        dict(key="gps", name="GPS seat", qty=1, g=vol(gps) * PLA, tall=2, file="gps_seat.stl", light=False),
        dict(key="foot", name="Foot pad (TPU)", qty=3, g=vol(foot) * 1.21, tall=LEG["t_pod_bot"] * H + 8, file="foot_pad.stl", light=False),
    ]
    total = sum(r["qty"] * r["g"] for r in rows)
    light = sum(r["qty"] * r["g"] for r in rows if r["light"])

    def stl(wp, name):
        cq.exporters.export(wp, os.path.join(out, "stl", name), tolerance=0.03, angularTolerance=0.15)

    down = lambda wp, z: wp.translate((0, 0, -z))
    flip = lambda wp, z: wp.translate((0, 0, -z)).rotate((0, 0, 0), (1, 0, 0), 180)
    stl(down(nose, Z_J2), "nose.stl")
    stl(down(upper, Z_GAP1), "upper_body.stl")
    stl(down(fring, Z_J1 - LAP), "fan_ring.stl")
    stl(down(tail, Z_TAIL), "tail_duct.stl")
    stl(down(vring, Z_EXIT), "vane_ring.stl")
    for nm, r_, w_, z in (("upper_body", upper_r, upper_w, Z_GAP1), ("fan_ring", fring_r, fring_w, Z_J1 - LAP),
                          ("tail_duct", tail_r, tail_w, Z_TAIL)):
        if w_ is not None:
            stl(down(r_, z), nm + "_colour1_red.stl")
            stl(down(w_, z), nm + "_colour2_white.stl")
    stl(down(leg_up, Z_LEGCUT).translate((-R_FOOT, 0, 0)), "leg_upper.stl")          # cut face on the bed
    stl(flip(leg_lo, Z_LEGCUT).translate((-R_FOOT, 0, 0)), "leg_lower.stl")          # cut face on the bed, foot up
    stl(down(vane, Z_VANE).rotate((0, 0, 0), (1, 0, 0), 90), "vane.stl")
    stl(down(avio, Z_AV), "avionics_plate.stl")
    stl(down(cpl, Z_J2 - 10), "nose_shoulder.stl")
    stl(down(gps, z_gps), "gps_seat.stl")
    stl(foot.translate((-R_FOOT, 0, 0)), "foot_pad.stl")

    asm = cq.Assembly(name="checker_rocket_rev_d")
    def add(wp, name, col):
        if wp is not None:
            asm.add(wp, name=name, color=cq.Color(*col))
    add(nose, "nose", RED); add(upper_r, "upper_red", RED); add(upper_w, "upper_white", WHITE)
    add(fring_r, "fan_ring_red", RED); add(fring_w, "fan_ring_white", WHITE)
    add(tail_r, "tail_duct_red", RED); add(tail_w, "tail_duct_white", WHITE)
    add(vring, "vane_ring", RED); add(avio, "avionics_plate", GREY); add(cpl, "nose_shoulder", GREY); add(gps, "gps_seat", GREY)
    for i, az in enumerate(LEG_AZ):
        add(rotz(leg_up, az), f"leg_upper_{i+1}", RED); add(rotz(leg_lo, az), f"leg_lower_{i+1}", RED)
        add(rotz(foot, az), f"foot_{i+1}", DARK)
    for i, az in enumerate(VANE_AZ):
        add(rotz(vane, az), f"vane_{i+1}", GREY)
    asm.export(os.path.join(out, "step", "rocket_assembly.step"))
    for n, wp in parts.items():
        cq.exporters.export(wp, os.path.join(out, "step", n + ".step"))

    # ---- viewer: exact meshes of every printed part, plus the bought parts
    G = ghosts()
    for k in ("Stereo cameras", "Depth and companion board", "Flight controller"):
        G[k] = front(G[k])
    items = [
        ("nose", "Nose", nose, RED, 4.0, None), ("gps", "GPS seat", gps, GREY, 4.6, None),
        ("cpl", "Nose shoulder ring", cpl, GREY, 3.5, None), ("avio", "Avionics plate", avio, GREY, 3.2, None),
        ("upper", "Upper body", upper_r, RED, 2.4, None), ("upper", "Upper body", upper_w, WHITE, 2.4, None),
        ("fring", "Fan ring", fring_r, RED, 1.5, None), ("fring", "Fan ring", fring_w, WHITE, 1.5, None),
        ("tail", "Tail duct", tail_r, RED, 0.7, None), ("tail", "Tail duct", tail_w, WHITE, 0.7, None),
        ("vring", "Vane ring", vring, RED, 0.0, None),
    ]
    for az in LEG_AZ:
        items += [("leg_up", "Leg, upper half", rotz(leg_up, az), RED, 0.7, az),
                  ("leg_lo", "Leg, lower half", rotz(leg_lo, az), RED, 0.0, az),
                  ("foot", "Foot pad", rotz(foot, az), DARK, -0.25, az)]
    for az in VANE_AZ:
        items.append(("vane", "Steering vane", rotz(vane, az), GREY, -0.5, None))
    glevel = {"Ducted fan, 90 mm": 1.1, "Battery, 6S": 2.9, "Speed controller": 2.9, "Flight controller": 3.3,
              "Stereo cameras": 3.3, "Depth and companion board": 3.3, "GPS": 4.7, "Vane servos": 0.0}
    for name, wp in G.items():
        items.append(("ghost", name, wp, BLUE, glevel[name], None))
    blob, man = bytearray(), []
    for key, label, wp, col, lvl, az in items:
        if wp is None:
            continue
        v, t = mesh_of(wp)
        man.append(dict(key=key, label=label, color=list(col), level=lvl, az=az, voff=len(blob), vn=int(len(v))))
        blob += v.tobytes()
        man[-1].update(toff=len(blob), tn=int(len(t)))
        blob += t.tobytes()

    fsa = math.pi * ((FAN["id"] / 2) ** 2 - (FAN["hub"] / 2) ** 2)
    exit_r = ro(Z_EXIT) - WD
    intake = (2 * math.pi * ro(Z_GAP0 + GAP / 2) - len(STRUT_AZ) * 6.0) * GAP
    meta = dict(
        H=H, D=2 * R, stance=P["STANCE"], ratio=round(H / (2 * R), 2), fsa=round(fsa),
        exit_area=round(math.pi * exit_r ** 2), exit_pct=round(100 * math.pi * exit_r ** 2 / fsa),
        intake_pct=round(100 * intake / fsa), printed=round(total), printed_light=round(light),
        rows=[dict(key=r["key"], name=r["name"], qty=r["qty"], g=round(r["g"], 1), tall=round(r["tall"]),
                   file=r["file"], light=r["light"]) for r in rows],
        stations=[[round(H), "Nose tip"], [round(z_gps), "GPS seat"], [round(Z_EYE), "Camera ports, facing forward"],
                  [round(Z_J2), "Nose joint and checker top"], [round(Z_AV), "Avionics plate"],
                  [round(Z_GAP1), "Battery bay floor"], [round(Z_GAP0), "Intake ring gap, %d mm tall" % GAP],
                  [round(Z_SEAT), "Fan seat"], [round(Z_J1), "Fan ring joint"], [round(Z_CHK0), "Checker bottom, fins start"],
                  [round(Z_TAIL), "Vane ring joint"], [round(Z_VANE), "Vane pivots and servos"],
                  [round(Z_EXIT), "Nozzle exit"], [0, "Ground"]],
    )
    with open(os.path.join(out, "viewer", "meshes.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(out, "viewer", "manifest.json"), "w") as f:
        json.dump(dict(parts=man, meta=meta), f)

    print(f"\nH={H:.0f}  D={2*R:.0f}  H/D={H/(2*R):.2f}  stance={P['STANCE']:.0f}")
    print(f"fan seat z={Z_SEAT:.1f}, intake gap {Z_GAP0:.1f}-{Z_GAP1:.1f}, vane pivot z={Z_VANE:.1f}, exit z={Z_EXIT:.0f}")
    print(f"intake {meta['intake_pct']} % of fan swept area, nozzle {meta['exit_pct']} %")
    print(f"{'part':38s} qty  g each  print height")
    for r in rows:
        print(f"{r['name']:38s} {r['qty']:3d} {r['g']:7.1f} {r['tall']:8.0f}")
    print(f"{'printed total (PLA)':38s}     {total:7.1f}   (of which skins/legs that could be LW-PLA: {light:.0f})")
    print("viewer blob %.2f MB, %d triangles" % (len(blob) / 1e6, sum(m['tn'] for m in man)))
    return parts


if __name__ == "__main__":
    main()
