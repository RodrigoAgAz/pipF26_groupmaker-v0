#!/usr/bin/env python3
"""
Project Haddock, rev E - ducted-fan hover rocket (CadQuery).

The traced checker-rocket shape, scaled so a 90 mm ducted fan fits inside it. Rev E moves
the fan down into the tail, between the fins:
  - the intake ring gap now sits on the line where the red tail meets the checker
  - the exhaust duct below the fan is short, and the lower body is slightly fatter to hold the fan
  - above the intake is a power bay (speed controller), then the battery on a tray, then the
    flight electronics, with two forward camera ports and a GPS seat low in the nose (the tip is empty)
  - four steering vanes and their servo mounts hang below the tail

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
    D=118.0,       # maximum body diameter (reference is H / 7.75)
    STANCE=376.0,  # foot circle diameter (reference: 3.19 x D)
    ROWS=5, COLS=4,
)
FAN = dict(od=98.0, id=90.0, length=80.0, hub=38.0)             # 90 mm class EDF housing
SERVO = dict(L=22.6, W=12.2, ears=27.8, above_ears=6.8, depth=22.7, shaft_off=5.2)   # 9 g class
CAM = dict(eye_y=25.0, port=16.0, lens=9.0, plate_x=10.0)       # stereo pair: 50 mm baseline
BATTERY = (50.0, 50.0, 155.0)                                    # 6S 5000 mAh class

# Outer mould line, normalised (z / H, r / (D/2)). From the widest point up this is the traced
# reference shape. Below it the body is kept wide enough to hold the fan between the fins.
SHAPE = dict(
    profile=[(0.1667, 0.72), (0.235, 0.775), (0.285, 0.838), (0.315, 0.864), (0.415, 0.868),
             (0.467, 0.874), (0.501, 0.89), (0.55, 0.94), (0.60, 0.98), (0.673, 1.0),
             (0.742, 0.965), (0.776, 0.92), (0.811, 0.85), (0.845, 0.795), (0.88, 0.69),
             (0.914, 0.545), (0.948, 0.37), (0.975, 0.20), (1.0, 0.0)],
    t_exit=0.1667,    # nozzle exit (bottom of the vane ring)
    t_tail=0.235,     # where the fins' arch meets the body; vane ring joins here
    t_chk0=0.415,     # checker band bottom = intake gap
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
GAP, LIP_MAX = 30.0, 3.2                    # intake ring gap height, room for the fan retaining lip
LEG_AZ = [0, 120, 240]
VANE_AZ = [45, 135, 225, 315]
STRUT_AZ = [45, 135, 225, 315]
JOINT_AZ = [30, 150, 270]                   # screws between the power bay and the battery tray
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
    z = Z_TAIL
    while ro(z) < RBORE + WD and z < 0.673 * H:
        z += 0.25
    bounds = [Z_CHK0 + k * ROW_H for k in range(P["ROWS"] + 1)]
    ok = [b for b in bounds if b >= z + FAN["length"] + LIP_MAX - 0.01 and b < 0.70 * H]
    if not ok:
        raise SystemExit("The fan does not fit this body: increase D")
    g["Z_GAP0"] = ok[0]
    g["Z_GAP1"] = Z_GAP0 + GAP
    g["LIP"] = (ro(Z_GAP0) - WD) - (RBORE - 2.0)       # 45 deg retaining lip under the intake ring
    assert 0.5 < LIP <= LIP_MAX, "fan retaining lip out of range: %.2f" % LIP
    g["Z_FTOP"] = Z_GAP0 - LIP
    g["Z_SEAT"] = Z_FTOP - FAN["length"]
    g["Z_JM"] = [b for b in bounds if b >= Z_GAP1 + 100][0]     # power bay / upper body joint
    g["Z_TRAY"] = Z_JM + 12.0                                   # underside of the battery tray
    g["Z_BATT"] = Z_TRAY + 2.5                                  # battery stands here
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
    g["STRUT_R"] = RBORE - 2.2
    # leg screws all sit below the fan, where there is room for their heads inside the duct;
    # that is also where a landing pulls the fin away from the body. A nub locates the top.
    g["LEG_SCREW_Z"] = [Z_TAIL + 12, (Z_TAIL + Z_SEAT) / 2, Z_SEAT - 12]
    g["LEG_NUB_Z"] = Z_SEAT + 0.75 * FAN["length"]


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


def polar(r, az):
    a = math.radians(az)
    return r * math.cos(a), r * math.sin(a)


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
    """One fin with its pod, trimmed to the body, as (whole, upper, lower) printable pieces."""
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
    leg = leg.cut(solid_under(lambda z: ro(z) + 0.25, Z_EXIT - 5, Z_GAP0 + 5))   # saddle to the duct
    leg = leg.cut(vcyl(R_FOOT, 0, pp[0][1] - 1, pp[0][1] + 9, 2.1))              # foot pad socket
    for z in LEG_SCREW_Z:                                                        # M3 heat-set inserts
        leg = leg.cut(xcyl(0, z, ro(z) - 2, ro(z) + 9.5, 2.0))
    leg = leg.cut(xcyl(0, LEG_NUB_Z, ro(LEG_NUB_Z) - 2, ro(LEG_NUB_Z) + 3.9, 2.25))   # pocket for the locating nub
    x_in = _interp(qbez(*leg_curves()[1], 24), Z_LEGCUT)                         # alignment pins across the print split
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
    cheek = [(rv - 1.0, Z_BR0), (R_PO, Z_PB), (R_PO, Z_PT), (R_PO - 8.0, Z_PT), (rv - 1.0, Z_PT - 12.0)]
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
    """Lower body between the fins: the fan drops in from the top and sits on a ledge."""
    body = shell(Z_TAIL, Z_GAP0, WD)
    ri = ro(Z_SEAT) - WD
    body = body.union(ring([(ri + 0.6, Z_SEAT), (FAN["id"] / 2, Z_SEAT), (FAN["id"] / 2, Z_SEAT - 1.5),
                            (ri + 0.6, Z_SEAT - 1.5 - (ri + 0.6 - FAN["id"] / 2))]))
    # top keeps the outer half of the wall (the intake ring's skirt drops inside it);
    # bottom keeps the inner half (the vane ring's skirt slips over it)
    body = body.cut(band(lambda z: ro(z) - WD / 2 + 0.05, WD, Z_GAP0 - LAP, Z_GAP0 + 0.5))
    body = body.cut(lap_cut(Z_TAIL - 0.5, Z_TAIL + LAP))
    # wire channel past the fan: a slot through the wall under a small external blister
    rr = ro(Z_GAP0 - 20)
    z0 = Z_SEAT - 16
    blister = xz_plate([(rr - 1.0, z0), (rr + 6.5, z0 + 7.5), (rr + 6.5, Z_GAP0), (rr - 1.0, Z_GAP0)], -9, 9)
    channel = xz_plate([(rr - 2.6, z0 + 3.5), (rr + 5.0, z0 + 11.1), (rr + 5.0, Z_GAP0 + 1), (rr - 2.6, Z_GAP0 + 1)], -7.5, 7.5)
    body = body.union(rotz(blister, WIRE_AZ)).cut(rotz(channel, WIRE_AZ))
    body = body.cut(rotz(xcyl(0, z0 + 16, rr + 3, rr + 9, 3.0), WIRE_AZ))       # servo leads enter here
    for az in LEG_AZ:
        for z in LEG_SCREW_Z:                              # M3 clearance for the leg screws
            body = body.cut(rotz(xcyl(0, z, ro(z) - 6, ro(z) + 2, 1.7), az))
        body = body.union(rotz(xcyl(0, LEG_NUB_Z, ro(LEG_NUB_Z) - 0.6, ro(LEG_NUB_Z) + 3.2, 2.0), az))
    return body


def make_intake_ring():
    """Closes the top of the tail duct: holds the fan down, and its struts carry everything above."""
    rr = ro(Z_GAP0)
    ri, rl = rr - WD, RBORE - 2.0
    body = ring([(ri, Z_GAP0 - LAP), (rr - WD / 2 - 0.05, Z_GAP0 - LAP), (rr - WD / 2 - 0.05, Z_GAP0), (rr, Z_GAP0),
                 (rr, Z_GAP0 + 2.2), (rl, Z_GAP0 + 2.2), (rl, Z_GAP0), (ri, Z_FTOP)])
    outer = solid_under(lambda z: ro(z) - 0.3, Z_GAP0 - 1, Z_GAP1)     # struts sit a hair inside the mould line
    strut = xz_plate([(rl, Z_GAP0 + 1.0), (rl - 3.7, Z_GAP0 + 4.7), (rl - 3.7, Z_GAP1), (R + 5, Z_GAP1),
                      (R + 5, Z_GAP0 + 1.0)], -3.0, 3.0).intersect(outer)
    strut = strut.cut(vcyl(STRUT_R, 0, Z_GAP1 - 9, Z_GAP1 + 1, 2.0))    # M3 heat-set insert
    for az in STRUT_AZ:
        body = body.union(rotz(strut, az))
    body = body.cut(rotz(box(ri - 1.5, rr + 9, -7.5, 7.5, Z_GAP0 - LAP - 1, Z_GAP0 + 3.2), WIRE_AZ))   # wires pass here
    return body


def make_power_bay():
    """First section above the intake. Its floor is the intake roof and bolts to the struts."""
    body = shell(Z_GAP1, Z_JM, W)
    floor = solid_under(lambda z: ro(z) - 0.5, Z_GAP1, Z_GAP1 + 2.5)
    for az in STRUT_AZ:
        x, y = polar(STRUT_R, az)
        floor = floor.cut(vcyl(x, y, Z_GAP1 - 1, Z_GAP1 + 4, 1.7))
    floor = floor.cut(box(-15, 15, 32, 40, Z_GAP1 - 1, Z_GAP1 + 4))                  # speed controller cooling slot
    x, y = polar(36, WIRE_AZ - 22)
    floor = floor.cut(vcyl(x, y, Z_GAP1 - 1, Z_GAP1 + 4, 6.5))                       # motor and servo leads
    body = body.union(floor)
    for az in JOINT_AZ:
        body = body.cut(rotz(xcyl(0, Z_JM - 6, 30, R + 5, 1.4), az))
    return body


def joint_ring(zj, wall, down, up, thick=1.2):
    """Ring across a body joint. Each half follows the bore if the bore narrows away from the
    joint, and is a plain cylinder if it widens, so it can always be inserted."""
    eps = 2.0
    grows = ro(zj + eps) > ro(zj - eps)
    r_j = ro(zj) - wall - FIT

    def r_out(z):
        conform = (z < zj and grows) or (z > zj and not grows)
        return ro(z) - wall - FIT if conform else r_j

    return band(r_out, thick, zj - down, zj + up), r_j


def make_battery_tray():
    """Joint ring between the power bay and the upper body, capped to carry the battery."""
    ringp, r_j = joint_ring(Z_JM, W, 12.0, Z_TRAY - Z_JM, thick=2.0)
    cap = vcyl(0, 0, Z_TRAY - 0.5, Z_BATT, r_j)
    for x in (-BATTERY[0] / 2 - 4, BATTERY[0] / 2 + 4):                              # battery strap
        cap = cap.cut(box(x - 2, x + 2, -13, 13, Z_TRAY - 1, Z_BATT + 1))
    cap = cap.cut(vcyl(0, -38, Z_TRAY - 1, Z_BATT + 1, 7.0))                         # battery leads
    body = ringp.union(cap)
    for az in JOINT_AZ:
        body = body.cut(rotz(xcyl(0, Z_JM - 6, 30, R + 5, 1.0), az))                 # pilot holes for the joint screws
    return body


def make_upper():
    body = shell(Z_JM, Z_J2, W)
    rw = ro(Z_AV) - W
    r_led = ro(Z_J2) - W - FIT - 1.2 - 3.5
    return body.union(ring([(rw + 0.6, Z_AV), (r_led, Z_AV), (r_led, Z_AV - 1.5),
                            (rw + 0.6, Z_AV - 1.5 - (rw + 0.6 - r_led))]))


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
    """Drop-in plate above the battery: flight controller on the plate, cameras on the upright.
    Modelled facing +X; main() turns it to FRONT_AZ."""
    rp = ro(Z_J2) - W - FIT - 1.2 - 0.8
    plate = vcyl(0, 0, Z_AV, Z_AV + 2.5, rp)
    for dx in (-15.25, 15.25):                         # 30.5 mm flight-controller pattern
        for dy in (-15.25, 15.25):
            plate = plate.cut(vcyl(FC_X + dx, dy, Z_AV - 1, Z_AV + 4, 1.6))
    plate = plate.cut(vcyl(0, -rp + 9, Z_AV - 1, Z_AV + 4, 6.0))                     # cable hole
    px = CAM["plate_x"]
    room = solid_under(lambda z: ro(z) - W - 3.0, Z_AV, Z_EYE + 24)                  # clears the nose shoulder ring
    wall = box(px, px + 2.5, -45, 45, Z_AV, Z_EYE + 16).intersect(room)
    for y in (-CAM["eye_y"], CAM["eye_y"]):
        wall = wall.cut(xcyl(y, Z_EYE, px - 1, px + 4, CAM["lens"] / 2))
        for dy in (-10.5, 10.5):                       # 21 x 12.5 mm camera-board hole pattern
            for dz in (-6.25, 6.25):
                wall = wall.cut(xcyl(y + dy, Z_EYE + dz, px - 1, px + 4, 0.9))
    gus = [(px, Z_AV + 2.5), (px - 22, Z_AV + 2.5), (px, Z_AV + 21)]
    for y in (-22, 22):
        wall = wall.union(xz_plate(gus, y - 1.2, y + 1.2))
    return plate.union(wall)


def make_gps_seat():
    """GPS shelf low in the nose, just above the camera upright, so the tip itself stays empty."""
    z0 = Z_AV + 60.0                               # clears the upright and the boards behind it
    r = ro(z0 + 2.0) - W - 0.25                    # wedges in the tapering nose at this height
    d = vcyl(0, 0, 0, 2.0, r).cut(box(-13, 13, -13, 13, 1.0, 3.0)).cut(vcyl(0, 16.5, -1, 3, 2.5))
    for az in (45, 135, 225, 315):                 # lightening holes
        x, y = polar(r - 9.0, az)
        d = d.cut(vcyl(x, y, -1, 3, 5.5))
    return d.translate((0, 0, z0)), z0


def make_vane():
    r_tip = ro(Z_VANE) - WD - 2.0
    v = xz_plate([(5, Z_VANE - 28), (r_tip, Z_VANE - 28), (r_tip, Z_VANE + 12), (5, Z_VANE + 12)], -1.5, 1.5)
    try:
        v = v.edges("|X").chamfer(1.1, 4.0)
    except Exception:
        pass
    return v.union(xcyl(0, Z_VANE, 3, ro(Z_VANE) + 9.0, 3.0))


# ------------------------------------------------------------------ bought parts
def component_layout():
    """Where every bought part sits. Drives the collision check and the viewer's drawings.
    Items under 'nose frame' are given facing +X and turned to FRONT_AZ."""
    zp = Z_GAP1 + 2.5          # power bay floor
    px = CAM["plate_x"]
    _, zg = make_gps_seat()
    return dict(
        front_az=FRONT_AZ, wire_az=WIRE_AZ, vane_az=VANE_AZ, gap0=Z_GAP0, gap1=Z_GAP1, power_floor=zp, z_av=Z_AV,
        tray=Z_TRAY,
        fan=dict(z0=Z_SEAT, z1=Z_FTOP, od=FAN["od"], id=FAN["id"], hub=FAN["hub"]),
        battery=dict(sx=BATTERY[0], sy=BATTERY[1], sz=BATTERY[2], z0=Z_BATT),
        esc=dict(x0=-19, x1=19, y0=30, y1=42, z0=zp + 2, z1=zp + 87),
        pm=dict(x0=-12.5, x1=12.5, y0=-40, y1=-33, z0=zp + 5, z1=zp + 26),
        # nose frame
        fc=dict(x=FC_X, size=36, z0=Z_AV + 8.5, z1=Z_AV + 16),
        rx=dict(x0=-20, x1=0, y0=24.5, y1=36.5, z0=Z_AV + 2.5, z1=Z_AV + 7.5),
        cam=dict(px=px, wall=2.5, eye_y=CAM["eye_y"], z=Z_EYE),
        depth=dict(x0=px - 16, x1=px - 11, y0=-30, y1=30, z0=Z_AV + 24, z1=Z_AV + 54),
        rid=dict(x0=px - 23, x1=px - 19, y0=-15, y1=15, z0=Z_AV + 27, z1=Z_AV + 45),
        gps=dict(z=zg + 2.0, size=25),
        servo=dict(r_po=R_PO, z_c=Z_SC, z_vane=Z_VANE, **SERVO),
    )


def ghosts():
    """Simple envelopes of the bought parts, in world position, for collision checking."""
    L = component_layout()
    front = lambda wp: rotz(wp, FRONT_AZ)
    bx = lambda d: box(d["x0"], d["x1"], d["y0"], d["y1"], d["z0"], d["z1"])
    g = {}
    f = L["fan"]
    edf = vcyl(0, 0, f["z0"], f["z1"], f["od"] / 2).cut(vcyl(0, 0, f["z0"] - 1, f["z1"] + 1, f["id"] / 2))
    g["fan"] = edf.union(vcyl(0, 0, f["z0"] - 28, f["z1"] + 18, f["hub"] / 2))
    b = L["battery"]
    g["battery"] = box(-b["sx"] / 2, b["sx"] / 2, -b["sy"] / 2, b["sy"] / 2, b["z0"], b["z0"] + b["sz"])
    g["esc"] = bx(L["esc"])
    g["pm"] = bx(L["pm"])
    c = L["fc"]
    g["fc"] = front(box(c["x"] - 18, c["x"] + 18, -18, 18, c["z0"], c["z1"]))
    g["rx"] = front(bx(L["rx"]))
    k = L["cam"]
    cams = None
    for y in (-k["eye_y"], k["eye_y"]):
        one = box(k["px"] - 4, k["px"], y - 12.5, y + 12.5, k["z"] - 12, k["z"] + 12).union(
            xcyl(y, k["z"], k["px"], k["px"] + 13, 4.0))
        cams = one if cams is None else cams.union(one)
    g["cams"] = front(cams)
    g["depth"] = front(bx(L["depth"]))
    g["rid"] = front(bx(L["rid"]))
    g["gpsmod"] = box(-12.5, 12.5, -12.5, 12.5, L["gps"]["z"], L["gps"]["z"] + 7)
    s_ = SERVO
    rt = R_PO - s_["above_ears"]
    servo = box(rt, rt + s_["depth"], -s_["W"] / 2, s_["W"] / 2, Z_SC - s_["L"] / 2, Z_SC + s_["L"] / 2)
    sv = None
    for az in VANE_AZ:
        r_ = rotz(servo, az)
        sv = r_ if sv is None else sv.union(r_)
    g["servo"] = sv
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
RED, WHITE, DARK, GREY = (0.84, 0.09, 0.06), (0.97, 0.96, 0.92), (0.13, 0.13, 0.14), (0.62, 0.62, 0.64)


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


def build():
    """All printed parts in flight position."""
    front = lambda wp: rotz(wp, FRONT_AZ)
    leg, leg_up, leg_lo = make_leg()
    gps, z_gps = make_gps_seat()
    cpl, _ = joint_ring(Z_J2, W, 10.0, 12.0)
    return dict(vring=make_vane_ring(), tail=make_tail_duct(), iring=make_intake_ring(), power=make_power_bay(),
                tray=make_battery_tray(), upper=make_upper(), nose=make_nose(), avio=front(make_avionics()),
                gps=gps, cpl=cpl, vane=make_vane(), foot=make_foot(), leg=leg, leg_up=leg_up, leg_lo=leg_lo), z_gps


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

    print("building parts")
    B, z_gps = build()
    print("checker")
    power_r, power_w = split_checker(B["power"], Z_GAP1, Z_JM)
    upper_r, upper_w = split_checker(B["upper"], Z_JM, Z_J2)
    for n, wp in B.items():
        if n == "leg":
            continue
        bb = cq.Compound.makeCompound(wp.vals()).BoundingBox()
        print(f"  {n:8s} solids={len(wp.solids().vals())} valid={valid(wp)} size {bb.xlen:.0f} x {bb.ylen:.0f} x {bb.zlen:.0f}")

    PLA = 1.24
    leg_wall, leg_fill = 0.045, 0.03          # one perimeter + 3 % infill: these are big hollow shapes
    lg = lambda wp: (area(wp) * leg_wall + vol(wp) * leg_fill) * PLA
    v = lambda k: vol(B[k]) * PLA
    rows = [
        dict(key="nose", name="Nose, with camera ports", qty=1, g=v("nose"), tall=H - Z_J2, file="nose.stl", light=True),
        dict(key="upper", name="Upper body (battery bay)", qty=1, g=v("upper"), tall=Z_J2 - Z_JM, file="upper_body.stl", light=True),
        dict(key="tray", name="Battery tray and joint ring", qty=1, g=v("tray"), tall=Z_BATT - Z_JM + 12, file="battery_tray.stl", light=False),
        dict(key="power", name="Power bay", qty=1, g=v("power"), tall=Z_JM - Z_GAP1, file="power_bay.stl", light=True),
        dict(key="iring", name="Intake ring, with struts", qty=1, g=v("iring"), tall=Z_GAP1 - Z_GAP0 + LAP, file="intake_ring.stl", light=False),
        dict(key="tail", name="Tail duct (holds the fan)", qty=1, g=v("tail"), tall=Z_GAP0 - Z_TAIL, file="tail_duct.stl", light=False),
        dict(key="vring", name="Vane ring, with servo mounts", qty=1, g=v("vring"), tall=Z_TAIL + LAP - Z_EXIT, file="vane_ring.stl", light=False),
        dict(key="leg_up", name="Leg, upper half", qty=3, g=lg(B["leg_up"]), tall=LEG["t_root_top"] * H - Z_LEGCUT, file="leg_upper.stl", light=True),
        dict(key="leg_lo", name="Leg, lower half with pod", qty=3, g=lg(B["leg_lo"]), tall=Z_LEGCUT - LEG["t_pod_bot"] * H, file="leg_lower.stl", light=True),
        dict(key="vane", name="Steering vane", qty=4, g=v("vane"), tall=3, file="vane.stl", light=False),
        dict(key="avio", name="Avionics plate with camera upright", qty=1, g=v("avio"), tall=Z_EYE + 16 - Z_AV, file="avionics_plate.stl", light=False),
        dict(key="cpl", name="Nose shoulder ring", qty=1, g=v("cpl"), tall=22, file="nose_shoulder.stl", light=False),
        dict(key="gps", name="GPS seat", qty=1, g=v("gps"), tall=2, file="gps_seat.stl", light=False),
        dict(key="foot", name="Foot pad (TPU)", qty=3, g=vol(B["foot"]) * 1.21, tall=LEG["t_pod_bot"] * H + 8, file="foot_pad.stl", light=False),
    ]
    total = sum(r["qty"] * r["g"] for r in rows)
    light = sum(r["qty"] * r["g"] for r in rows if r["light"])

    def stl(wp, name):
        cq.exporters.export(wp, os.path.join(out, "stl", name), tolerance=0.03, angularTolerance=0.15)

    down = lambda wp, z: wp.translate((0, 0, -z))
    flip = lambda wp, z: wp.translate((0, 0, -z)).rotate((0, 0, 0), (1, 0, 0), 180)
    stl(down(B["nose"], Z_J2), "nose.stl")
    stl(down(B["upper"], Z_JM), "upper_body.stl")
    stl(flip(B["tray"], Z_BATT), "battery_tray.stl")                               # cap on the bed
    stl(down(B["power"], Z_GAP1), "power_bay.stl")
    stl(down(B["iring"], Z_GAP0 - LAP), "intake_ring.stl")
    stl(down(B["tail"], Z_TAIL), "tail_duct.stl")
    stl(down(B["vring"], Z_EXIT), "vane_ring.stl")
    for nm, r_, w_, z in (("upper_body", upper_r, upper_w, Z_JM), ("power_bay", power_r, power_w, Z_GAP1)):
        if w_ is not None:
            stl(down(r_, z), nm + "_colour1_red.stl")
            stl(down(w_, z), nm + "_colour2_white.stl")
    stl(down(B["leg_up"], Z_LEGCUT).translate((-R_FOOT, 0, 0)), "leg_upper.stl")   # cut face on the bed
    stl(flip(B["leg_lo"], Z_LEGCUT).translate((-R_FOOT, 0, 0)), "leg_lower.stl")   # cut face on the bed, foot up
    stl(down(B["vane"], Z_VANE).rotate((0, 0, 0), (1, 0, 0), 90), "vane.stl")
    stl(down(B["avio"], Z_AV), "avionics_plate.stl")
    stl(down(B["cpl"], Z_J2 - 10), "nose_shoulder.stl")
    stl(down(B["gps"], z_gps), "gps_seat.stl")
    stl(B["foot"].translate((-R_FOOT, 0, 0)), "foot_pad.stl")

    asm = cq.Assembly(name="project_haddock_rev_e")

    def add(wp, name, col):
        if wp is not None:
            asm.add(wp, name=name, color=cq.Color(*col))
    add(B["nose"], "nose", RED); add(upper_r, "upper_red", RED); add(upper_w, "upper_white", WHITE)
    add(B["tray"], "battery_tray", GREY); add(power_r, "power_bay_red", RED); add(power_w, "power_bay_white", WHITE)
    add(B["iring"], "intake_ring", RED); add(B["tail"], "tail_duct", RED); add(B["vring"], "vane_ring", RED)
    add(B["avio"], "avionics_plate", GREY); add(B["cpl"], "nose_shoulder", GREY); add(B["gps"], "gps_seat", GREY)
    for i, az in enumerate(LEG_AZ):
        add(rotz(B["leg_up"], az), f"leg_upper_{i+1}", RED); add(rotz(B["leg_lo"], az), f"leg_lower_{i+1}", RED)
        add(rotz(B["foot"], az), f"foot_{i+1}", DARK)
    for i, az in enumerate(VANE_AZ):
        add(rotz(B["vane"], az), f"vane_{i+1}", GREY)
    asm.export(os.path.join(out, "step", "project_haddock_assembly.step"))
    for k, n in (("nose", "nose"), ("upper", "upper_body"), ("tray", "battery_tray"), ("power", "power_bay"),
                 ("iring", "intake_ring"), ("tail", "tail_duct"), ("vring", "vane_ring"), ("leg_up", "leg_upper"),
                 ("leg_lo", "leg_lower"), ("vane", "vane"), ("avio", "avionics_plate"), ("cpl", "nose_shoulder"),
                 ("gps", "gps_seat"), ("foot", "foot_pad")):
        cq.exporters.export(B[k], os.path.join(out, "step", n + ".step"))

    # ---- viewer: exact meshes of every printed part (bought parts are drawn by the page from the layout)
    items = [
        ("nose", B["nose"], RED, 4.4, None), ("gps", B["gps"], GREY, 4.9, None),
        ("cpl", B["cpl"], GREY, 4.0, None), ("avio", B["avio"], GREY, 3.7, None),
        ("upper", upper_r, RED, 3.0, None), ("upper", upper_w, WHITE, 3.0, None),
        ("tray", B["tray"], GREY, 2.6, None),
        ("power", power_r, RED, 1.9, None), ("power", power_w, WHITE, 1.9, None),
        ("iring", B["iring"], RED, 1.4, None), ("tail", B["tail"], RED, 0.6, None), ("vring", B["vring"], RED, 0.0, None),
    ]
    for az in LEG_AZ:
        items += [("leg_up", rotz(B["leg_up"], az), RED, 0.6, az), ("leg_lo", rotz(B["leg_lo"], az), RED, 0.0, az),
                  ("foot", rotz(B["foot"], az), DARK, -0.25, az)]
    for az in VANE_AZ:
        items.append(("vane", rotz(B["vane"], az), GREY, -0.5, None))
    blob, man = bytearray(), []
    for key, wp, col, lvl, az in items:
        if wp is None:
            continue
        vv, t = mesh_of(wp)
        man.append(dict(key=key, color=list(col), level=lvl, az=az, voff=len(blob), vn=int(len(vv))))
        blob += vv.tobytes()
        man[-1].update(toff=len(blob), tn=int(len(t)))
        blob += t.tobytes()

    fsa = math.pi * ((FAN["id"] / 2) ** 2 - (FAN["hub"] / 2) ** 2)
    exit_r = ro(Z_EXIT) - WD
    intake = (2 * math.pi * ro(Z_GAP0 + GAP / 2) - len(STRUT_AZ) * 6.0) * GAP
    meta = dict(
        name="Project Haddock", rev="E",
        H=H, D=2 * R, stance=P["STANCE"], ratio=round(H / (2 * R), 2), fsa=round(fsa),
        exit_pct=round(100 * math.pi * exit_r ** 2 / fsa), intake_pct=round(100 * intake / fsa),
        duct_len=round(Z_SEAT - Z_EXIT), printed=round(total), printed_light=round(light),
        layout=component_layout(),
        rows=[dict(key=r["key"], name=r["name"], qty=r["qty"], g=round(r["g"], 1), tall=round(r["tall"]),
                   file=r["file"], light=r["light"]) for r in rows],
    )
    with open(os.path.join(out, "viewer", "meshes.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(out, "viewer", "manifest.json"), "w") as f:
        json.dump(dict(parts=man, meta=meta), f)

    print(f"\nH={H:.0f}  D={2*R:.0f}  H/D={H/(2*R):.2f}  stance={P['STANCE']:.0f}")
    print(f"fan {Z_SEAT:.1f}-{Z_FTOP:.1f}, intake gap {Z_GAP0:.1f}-{Z_GAP1:.1f}, power bay to {Z_JM:.1f}, "
          f"battery from {Z_BATT:.1f}, vane pivot {Z_VANE:.1f}, exit {Z_EXIT:.0f}")
    print(f"intake {meta['intake_pct']} % of fan swept area, nozzle {meta['exit_pct']} %, duct below fan {meta['duct_len']} mm, "
          f"body at fan {2*ro(Z_SEAT+40):.1f} mm")
    print(f"{'part':38s} qty  g each  print height")
    for r in rows:
        print(f"{r['name']:38s} {r['qty']:3d} {r['g']:7.1f} {r['tall']:8.0f}")
    print(f"{'printed total (PLA)':38s}     {total:7.1f}   (of which skins/legs that could be LW-PLA: {light:.0f})")
    print("viewer blob %.2f MB, %d triangles" % (len(blob) / 1e6, sum(m['tn'] for m in man)))


if __name__ == "__main__":
    main()
