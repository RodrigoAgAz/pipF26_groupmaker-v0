#!/usr/bin/env python3
"""
Checker rocket, rev C - parametric CAD for a 3D-printed model rocket (CadQuery).

Rev C re-traces the outer shape from the reference photos: the body is about 7.75 times
as tall as it is wide, the fins are broad lens-section blades that start at the bottom of
the checker, and each ends in a large teardrop pod. The body is closed and carries a
standard hobby rocket motor in the tail.

Coordinates: Z up, Z=0 is the ground under the feet. All dimensions in mm.

    python rocket.py                                   # reference proportions
    python rocket.py --params '{"H":540,"D":70}'       # paste from the viewer's "Copy dimensions"

The same shape numbers (SHAPE, LEG below) drive the interactive viewer.
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
    H=600.0,       # overall height, ground to nose tip
    D=77.0,        # maximum body diameter            (reference: H / 7.75)
    STANCE=246.0,  # foot circle diameter             (reference: 3.19 x D)
    ROWS=5,        # checker rows
    COLS=4,        # checker columns
    MOTOR=29,      # motor mount size: 24 or 29
)

# Motor mount tube (cardboard liner) outer diameter, typical motor length and loaded mass.
MOTORS = {
    24: dict(tube_od=25.0, length=95.0, mass=65.0),
    29: dict(tube_od=30.7, length=124.0, mass=125.0),
}

# Outer mould line, normalised (z / H, r / (D/2)), measured from the reference silhouette.
SHAPE = dict(
    profile=[(0.235, 0.52), (0.27, 0.62), (0.33, 0.72), (0.415, 0.80), (0.467, 0.835),
             (0.501, 0.885), (0.55, 0.94), (0.60, 0.98), (0.673, 1.0), (0.742, 0.965),
             (0.776, 0.92), (0.811, 0.85), (0.845, 0.795), (0.88, 0.69), (0.914, 0.545),
             (0.948, 0.37), (0.975, 0.20), (1.0, 0.0)],
    t_tail=0.235,     # body tail (motor exit)
    t_j1=0.415,       # tail / mid joint, at the bottom of the checker
    t_chk0=0.415,     # checker band bottom
    t_chk1=0.845,     # checker band top
)
# Fin and pod geometry: heights as fractions of H, widths as fractions of D.
LEG = dict(
    t_root_top=0.412, t_pod_tip=0.25, t_pod_max=0.085, t_pod_bot=0.008,
    pod_r=0.38,                 # pod maximum radius / D
    fin_w=0.085,                # fin half thickness / D
    outer_ctrl=(0.642, 0.39),   # Bezier control of the fin's outer edge (fractions of its x and z span)
    inner_ctrl=(0.516, 0.074),  # Bezier control of the arch under the fin
    # pod outline: (radius / pod_r, fraction of pod height from the bottom)
    pod=[(0.26, 0.0), (0.80, 0.09), (0.97, 0.22), (1.0, 0.34), (0.93, 0.52), (0.76, 0.69),
         (0.50, 0.84), (0.22, 0.94), (0.0, 1.0)],
)

WALL_AFT, WALL, FIT, CPL = 1.2, 1.0, 0.15, 12.0
LEG_AZ = [0, 120, 240]
SECT_K = 12


def derive():
    """Fill module-level dimensions from P."""
    g = globals()
    H, D = P["H"], P["D"]
    g["H"], g["R"] = H, D / 2
    g["_p"] = PchipInterpolator([t for t, _ in SHAPE["profile"]], [r for _, r in SHAPE["profile"]])
    g["Z_TAIL"] = SHAPE["t_tail"] * H
    g["Z_J1"] = SHAPE["t_j1"] * H
    g["Z_CHK0"], g["Z_CHK1"] = SHAPE["t_chk0"] * H, SHAPE["t_chk1"] * H
    g["ROW_H"] = (Z_CHK1 - Z_CHK0) / P["ROWS"]
    g["Z_J2"] = Z_CHK0 + math.ceil(P["ROWS"] / 2) * ROW_H   # nose joint on a row boundary
    g["RB"] = MOTORS[P["MOTOR"]]["tube_od"] / 2 + FIT        # motor tube bore radius
    g["R_FOOT"] = P["STANCE"] / 2
    g["POD_R"] = LEG["pod_r"] * D
    g["FIN_W"] = LEG["fin_w"] * D
    if ro(Z_TAIL) < RB + 0.8:
        raise SystemExit("Tail is too narrow for a %d mm motor tube: need D >= %.0f mm"
                         % (P["MOTOR"], 2 * (RB + 0.8) / SHAPE["profile"][0][1]))


def ro(z):
    """Outer skin radius at height z."""
    return float(R * _p(min(max(z / H, SHAPE["t_tail"]), 1.0)))


# ------------------------------------------------------------------ helpers
def _zs(z0, z1, step=4.0):
    n = max(int(abs(z1 - z0) / step), 3) + 1
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
    return cq.Workplane("XZ").polyline(poly).close().revolve(360, (0, 0, 0), (0, 1, 0))


def band(fn_out, thick, z0, z1):
    zs = _zs(z0, z1, 2.0)
    pts = [(fn_out(z), z) for z in zs] + [(fn_out(z) - thick, z) for z in zs[::-1]]
    return ring(pts)


def shell(z0, z1, wall):
    return solid_under(ro, z0, z1).cut(solid_under(lambda z: ro(z) - wall, z0 - 1, z1 + 1))


def rotz(obj, deg):
    return obj.rotate((0, 0, 0), (0, 0, 1), deg)


def vcyl(x, y, z0, z1, r):
    return cq.Workplane("XY").workplane(offset=z0).center(x, y).circle(r).extrude(z1 - z0)


def qbez(p0, p1, p2, n):
    t = np.linspace(0, 1, n + 1)[:, None]
    return ((1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)).tolist()


# ------------------------------------------------------------------ fins and pods (mirrored in the viewer JS)
def leg_curves():
    z_rt, z_pt, z_pm = LEG["t_root_top"] * H, LEG["t_pod_tip"] * H, LEG["t_pod_max"] * H
    p0, p2 = (ro(z_rt) - 1.0, z_rt), (R_FOOT, z_pt)
    p1 = (p0[0] + LEG["outer_ctrl"][0] * (p2[0] - p0[0]), p0[1] + LEG["outer_ctrl"][1] * (p2[1] - p0[1]))
    q0, q2 = (R_FOOT - 0.9 * POD_R, z_pm), (ro(Z_TAIL) - 1.0, Z_TAIL)
    q1 = (q2[0] + LEG["inner_ctrl"][0] * (q0[0] - q2[0]), Z_TAIL - LEG["inner_ctrl"][1] * (Z_TAIL - z_pm))
    return (p0, p1, p2), (q0, q1, q2)


def _interp(pts, z):
    """Linear x(z) through [(x, z), ...] sorted by ascending z."""
    if z <= pts[0][1]:
        return pts[0][0]
    for a, b in zip(pts, pts[1:]):
        if z <= b[1]:
            f = (z - a[1]) / (b[1] - a[1]) if b[1] > a[1] else 0.0
            return a[0] + f * (b[0] - a[0])
    return pts[-1][0]


def leg_slices():
    """Horizontal slices of one fin: (z, inner x, outer x, half thickness)."""
    outer, inner = leg_curves()
    out = qbez(*outer, 40)[::-1]          # ascending z
    inn = qbez(*inner, 24)                # ascending z, pod up to tail
    z_rt, z_pt = LEG["t_root_top"] * H, LEG["t_pod_tip"] * H
    zs = [p[1] for p in inn] + [Z_TAIL + (z_rt - Z_TAIL) * i / 18 for i in range(1, 19)]
    res = []
    for z in zs:
        xo = R_FOOT if z <= z_pt else _interp(out, z)
        xi = _interp(inn, z) if z < Z_TAIL else inn[-1][0]
        res.append((z, xi, xo, FIN_W))
    return res


def sect_pts(xi, xo, w, K=SECT_K):
    """Upper half of a fin slice: rounded on the inside, sharp spine on the outside."""
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
    wires = []
    for z, xi, xo, w in leg_slices():
        up = sect_pts(xi, xo, w)
        e1 = cq.Edge.makeSpline([cq.Vector(x, y, z) for x, y in up])
        e2 = cq.Edge.makeSpline([cq.Vector(x, -y, z) for x, y in reversed(up)])
        wires.append(cq.Wire.assembleEdges([e1, e2]))
    # ruled loft: straight between slices, so it follows the traced curves without overshoot
    fin = cq.Workplane("XY").newObject([cq.Solid.makeLoft(wires, True)])
    pp = pod_profile()
    pod = (cq.Workplane("XZ").moveTo(0, pp[0][1]).lineTo(*pp[0])
           .spline(pp[1:], includeCurrent=True).close()
           .revolve(360, (0, 0, 0), (0, 1, 0)).translate((R_FOOT, 0, 0)))
    leg = fin.union(pod)
    leg = leg.cut(vcyl(R_FOOT, 0, pp[0][1] - 1, pp[0][1] + 9, 2.1))   # foot pad socket
    return leg


def make_foot():
    zb = LEG["t_pod_bot"] * H
    f = cq.Workplane("XY").circle(0.26 * POD_R).extrude(zb)
    f = f.union(vcyl(0, 0, zb, zb + 8, 1.95))
    return f.translate((R_FOOT, 0, 0))


# ------------------------------------------------------------------ tail with legs
def make_aft():
    zc = Z_J1 - CPL - 4                       # where the centring cone meets the skin
    rs = ro(zc) - WALL_AFT
    leg = make_leg()
    # fins are fused to the solid body first and the cavity is cut afterwards:
    # more robust than fusing lofted fins to a thin shell, and it leaves the inside clean
    body = solid_under(ro, Z_TAIL, Z_J1)
    for az in LEG_AZ:
        body = body.union(rotz(leg, az))
    body = body.cut(solid_under(lambda z: ro(z) - WALL_AFT, Z_TAIL - 1, Z_J1 + 1))

    # 45 deg cone: centres the motor tube and carries it to the skin. The tube passes through.
    z_ci = zc - (rs + 0.5 - RB)
    cone = ring([(rs + 0.5, zc), (RB - 0.3, z_ci), (RB - 0.3, z_ci - 2.0), (rs + 0.5, zc - 2.0)])
    sleeve = ring([(RB - 0.3, Z_TAIL), (RB + 1.2, Z_TAIL), (RB + 1.2, z_ci), (RB - 0.3, z_ci)])
    body = body.union(sleeve).union(cone)
    inside = solid_under(lambda z: ro(z) - 0.6, Z_TAIL, Z_J1)
    web = (cq.Workplane("XY").workplane(offset=Z_TAIL).center(RB + 0.6 + 40, 0)
           .rect(80, 4.0).extrude(z_ci - 2 - Z_TAIL).intersect(inside))
    for az in LEG_AZ:
        body = body.union(rotz(web, az))
    body = body.cut(vcyl(0, 0, Z_TAIL - 1, zc + 2, RB))      # motor tube bore, through the cone
    rh = (RB + 1.2 + rs) / 2                                  # shock cord holes through the cone
    for az in (165, 195):
        a = math.radians(az)
        body = body.cut(vcyl(rh * math.cos(a), rh * math.sin(a), zc - 14, zc + 3, 1.3))
    return body, leg, dict(sleeve_len=z_ci - Z_TAIL, cone_z=zc)


def make_mid():
    return shell(Z_J1, Z_J2, WALL)


def make_nose():
    outer = solid_under(ro, Z_J2, H)
    zc = Z_J2
    while ro(zc) - WALL > 3.0:
        zc += 0.5
    return outer.cut(solid_under(lambda z: ro(z) - WALL, Z_J2 - 1, zc))


def make_coupler(zj, wall, down, up, crossbar=False):
    """Joint ring. Each half follows the bore if the bore narrows away from the joint,
    and is a plain cylinder if the bore widens, so it can always be inserted."""
    eps = 2.0
    grows = ro(zj + eps) > ro(zj - eps)
    r_j = ro(zj) - wall - FIT

    def r_out(z):
        conform = (z < zj and grows) or (z > zj and not grows)
        return ro(z) - wall - FIT if conform else r_j

    c = band(r_out, 1.2, zj - down, zj + up)
    if crossbar:   # shock cord anchor across the bottom of the nose shoulder
        bar = (cq.Workplane("XY").workplane(offset=zj - down).rect(2 * r_j, 4.0).extrude(3.0)
               .intersect(vcyl(0, 0, zj - down - 1, zj - down + 4, r_out(zj - down) - 0.6)))
        c = c.union(bar)
    return c


# ------------------------------------------------------------------ checker split
def split_checker(part, zlo, zhi):
    white = None
    step = 360.0 / P["COLS"]
    for j in range(P["ROWS"]):
        z0, z1 = max(Z_CHK0 + j * ROW_H, zlo), min(Z_CHK0 + (j + 1) * ROW_H, zhi)
        if z1 - z0 < 1:
            continue
        cell = (cq.Workplane("XZ").polyline([(2, z0), (R + 20, z0), (R + 20, z1), (2, z1)])
                .close().revolve(step, (0, 0, 0), (0, 1, 0)))
        for i in range(P["COLS"]):
            if (i + j) % 2:
                continue
            c = rotz(cell, i * step)
            white = c if white is None else white.union(c)
    if white is None:
        return part, None
    return part.cut(white), part.intersect(white)


# ------------------------------------------------------------------ build + export
RED, WHITE, DARK, GREY = (0.87, 0.14, 0.08), (0.97, 0.96, 0.92), (0.13, 0.13, 0.14), (0.62, 0.62, 0.64)


def vol(wp):
    return sum(s.Volume() for s in wp.vals()) / 1000.0     # cm3


def area(wp):
    return sum(s.Area() for s in wp.vals()) / 100.0        # cm2


def mesh_of(wp, tol=0.2, ang=0.3):
    verts, tris = [], []
    for s in wp.vals():
        v, t = s.tessellate(tol, ang)
        off = len(verts)
        verts += [(p.x, p.y, p.z) for p in v]
        tris += [(a + off, b + off, c + off) for a, b, c in t]
    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", help="JSON object overriding H, D, STANCE, ROWS, COLS, MOTOR")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.params:
        P.update(json.loads(a.params))
    derive()
    here = os.path.dirname(os.path.abspath(__file__))
    out = a.out or os.path.join(here, "out")
    for d in ("stl", "step", "viewer"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
        for f in os.listdir(os.path.join(out, d)):
            os.remove(os.path.join(out, d, f))

    print("building tail + legs"); aft, leg, info = make_aft()
    print("building mid body");    mid = make_mid()
    print("building nose");        nose = make_nose()
    cpl1 = make_coupler(Z_J1, WALL_AFT, CPL, CPL)
    cpl2 = make_coupler(Z_J2, WALL, 18.0, CPL, crossbar=True)
    foot = make_foot()
    print("splitting checker")
    mid_r, mid_w = split_checker(mid, Z_J1, Z_J2)
    nose_r, nose_w = split_checker(nose, Z_J2, H)
    for n, wp in (("tail", aft), ("mid", mid), ("nose", nose), ("leg", leg)):
        ok = all(s.isValid() for s in wp.vals())
        print(f"  {n}: solids={len(wp.solids().vals())} valid={ok}")

    PLA = 1.24
    # skins print solid; fins and pods print as 2 perimeters + ~6 % infill
    leg_g = (area(leg) * 0.09 + vol(leg) * 0.06) * PLA
    aft_g = max(vol(aft) - 3 * vol(leg), 0) * PLA + 3 * leg_g
    reach = R_FOOT + POD_R
    rows = [
        dict(key="aft", name="Tail with legs and motor mount", qty=1, g=aft_g, tall=Z_J1 - LEG["t_pod_bot"] * H,
             file="tail_with_legs.stl"),
        dict(key="mid", name="Mid body (parachute bay)", qty=1, g=vol(mid) * PLA, tall=Z_J2 - Z_J1,
             file="mid_body.stl"),
        dict(key="nose", name="Nose", qty=1, g=vol(nose) * PLA, tall=H - Z_J2, file="nose.stl"),
        dict(key="cpl1", name="Joint ring, tail to mid (glued)", qty=1, g=vol(cpl1) * PLA, tall=2 * CPL,
             file="joint_ring_tail.stl"),
        dict(key="cpl2", name="Nose shoulder (glued into nose)", qty=1, g=vol(cpl2) * PLA, tall=18 + CPL,
             file="nose_shoulder.stl"),
        dict(key="foot", name="Foot pad (TPU)", qty=3, g=vol(foot) * 1.21, tall=LEG["t_pod_bot"] * H + 8,
             file="foot_pad.stl"),
    ]
    total = sum(r["qty"] * r["g"] for r in rows)

    def stl(wp, name):
        cq.exporters.export(wp, os.path.join(out, "stl", name), tolerance=0.03, angularTolerance=0.15)

    flip = lambda wp, z: wp.translate((0, 0, -z)).rotate((0, 0, 0), (1, 0, 0), 180)
    stl(flip(aft, Z_J1), "tail_with_legs.stl")            # prints joint-face down, feet up
    stl(mid.translate((0, 0, -Z_J1)), "mid_body.stl")
    stl(nose.translate((0, 0, -Z_J2)), "nose.stl")
    if mid_w is not None:
        stl(mid_r.translate((0, 0, -Z_J1)), "mid_body_colour1_red.stl")
        stl(mid_w.translate((0, 0, -Z_J1)), "mid_body_colour2_white.stl")
    if nose_w is not None:
        stl(nose_r.translate((0, 0, -Z_J2)), "nose_colour1_red.stl")
        stl(nose_w.translate((0, 0, -Z_J2)), "nose_colour2_white.stl")
    stl(cpl1.translate((0, 0, -(Z_J1 - CPL))), "joint_ring_tail.stl")
    stl(cpl2.translate((0, 0, -(Z_J2 - 18))), "nose_shoulder.stl")
    stl(foot.translate((-R_FOOT, 0, 0)), "foot_pad.stl")

    asm = cq.Assembly(name="checker_rocket")
    asm.add(aft, name="tail_with_legs", color=cq.Color(*RED))
    asm.add(mid_r, name="mid_red", color=cq.Color(*RED))
    if mid_w is not None:
        asm.add(mid_w, name="mid_white", color=cq.Color(*WHITE))
    asm.add(nose_r, name="nose_red", color=cq.Color(*RED))
    if nose_w is not None:
        asm.add(nose_w, name="nose_white", color=cq.Color(*WHITE))
    asm.add(cpl1, name="joint_ring_tail", color=cq.Color(*GREY))
    asm.add(cpl2, name="nose_shoulder", color=cq.Color(*GREY))
    for i, az in enumerate(LEG_AZ):
        asm.add(rotz(foot, az), name=f"foot_{i+1}", color=cq.Color(*DARK))
    asm.export(os.path.join(out, "step", "rocket_assembly.step"))
    for wp, n in ((aft, "tail_with_legs"), (mid, "mid_body"), (nose, "nose"),
                  (cpl1, "joint_ring_tail"), (cpl2, "nose_shoulder")):
        cq.exporters.export(wp, os.path.join(out, "step", n + ".step"))

    # ---- data for the interactive viewer
    ts = np.linspace(SHAPE["t_tail"], 1.0, 201)
    shape = dict(SHAPE, table=[[round(float(t), 5), round(float(_p(t)), 5)] for t in ts])
    data = dict(defaults=P, shape=shape, leg=LEG, motors=MOTORS, leg_az=LEG_AZ, sect_k=SECT_K,
                walls=dict(aft=WALL_AFT, body=WALL), cad=dict(
                    rows=[dict(r, g=round(r["g"], 1), tall=round(r["tall"], 1)) for r in rows],
                    total=round(total, 1)))
    with open(os.path.join(out, "viewer", "shape.json"), "w") as f:
        json.dump(data, f)

    # ---- exact CAD meshes, for checking renders
    items = [("aft", aft, RED), ("mid", mid_r, RED), ("nose", nose_r, RED),
             ("cpl1", cpl1, GREY), ("cpl2", cpl2, GREY)]
    if mid_w is not None:
        items.append(("mid", mid_w, WHITE))
    if nose_w is not None:
        items.append(("nose", nose_w, WHITE))
    items += [("foot", rotz(foot, az), DARK) for az in LEG_AZ]
    blob, man = bytearray(), []
    for group, wp, col in items:
        v, t = mesh_of(wp)
        man.append(dict(group=group, color=list(col), voff=len(blob), vn=int(len(v))))
        blob += v.tobytes()
        man[-1].update(toff=len(blob), tn=int(len(t)))
        blob += t.tobytes()
    with open(os.path.join(out, "viewer", "cad_meshes.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(out, "viewer", "cad_manifest.json"), "w") as f:
        json.dump(man, f)

    print(f"\nH={H:.0f}  D={2*R:.0f}  H/D={H/(2*R):.2f}  stance={P['STANCE']:.0f}  motor={P['MOTOR']} mm")
    print(f"{'part':38s} qty  g each  print height")
    for r in rows:
        print(f"{r['name']:38s} {r['qty']:3d} {r['g']:7.1f} {r['tall']:8.0f}")
    print(f"{'printed total (PLA)':38s}     {total:7.1f}")
    print("tail print footprint %.0f x %.0f mm; tail dia %.1f mm, bore %.1f mm, pod dia %.0f mm" %
          (2 * R_FOOT * math.sin(math.pi / 3) + 2 * POD_R, 1.5 * R_FOOT + 2 * POD_R,
           2 * ro(Z_TAIL), 2 * RB, 2 * POD_R))


if __name__ == "__main__":
    main()
