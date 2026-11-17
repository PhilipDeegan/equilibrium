"""Pair test (11-03, step 1): two self-held bodies settled together vs each alone (copied from
exp/2026/10/25/jigsaw/ring2.py, seed generalised to rings at x = --cx; diagnostics per body + the depletion between).

Original: Rule requirement test: content held by a PROPAGATING depletion (no external well).

Content: step() on the discrete-weight periodic BCC box (../defect/bcc.py), face coefficient W * g_f,
g = (1 - d) / (1 + d), d = clip(kappa P, dmin, dmax) per cell.
Depletion carrier P on cells: P_tt = cD^2 lap(P) + F (lap = same BCC W^2 Laplacian, F = local |face energy throughput|),
leapfrog; propagates at cD (bounded), settles to ~1/r outside a steady source. Sponge on both fields at the rim.
Preseed: P relaxed (damped) against the seed's throughput for --tpre ("the event already formed it"); kappa then set so
max kappa P = --depth. Optional background gradient (external pressure) on until --toff.
Requirements scored: hold (energy near the body), deform (lop = |m 1| / m 0), translate (centroid), recover (after toff).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "10/25/defect"))
from bcc import AX, OFF, W_HEX, W_SQ, Box, _sh, _shb  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=64)
p.add_argument("--R", type=float, default=10.0)
p.add_argument("--a", type=float, default=4.0)
p.add_argument("--q", type=float, default=4)
p.add_argument("--sense", type=float, default=1.0)
p.add_argument("--depth", type=float, default=0.5, help="max contraction at t 0 (sets kappa)")
p.add_argument("--dmax", type=float, default=0.6)
p.add_argument("--cD", type=float, default=1.0, help="carrier speed / content speed")
p.add_argument("--tpre", type=float, default=60.0)
p.add_argument("--grad", type=float, default=0.0)
p.add_argument("--ton", type=float, default=0.0)
p.add_argument("--toff", type=float, default=1e9)
p.add_argument("--ramp", type=float, default=0.0, help="smooth on / off over this time")
p.add_argument("--T", type=float, default=300.0)
p.add_argument("--dt", type=float, default=0.1)
p.add_argument("--scf", type=int, default=4, help="self-consistent preseed iterations")
p.add_argument("--A", type=float, default=1.0, help="seed amplitude (body energy ~ A^2)")
p.add_argument("--kfix", type=float, default=0.0, help="fixed kappa (no depth normalisation): depth then follows the body")
p.add_argument("--src", choices=["flux", "energy", "local", "mix"], default="flux", help="P source: |throughput| or local energy density (diagnostic)")
p.add_argument("--freeze", action="store_true", help="P frozen after preseed (fixed well)")
p.add_argument("--every", type=int, default=2)
p.add_argument("--cx", default="0", help="comma list of ring centres along x")
p.add_argument("--out", default="")
p.add_argument("--snap", default="", help="comma list of times for P / u profile snapshots")
args = p.parse_args()

n = args.n
N3 = (n, n, n)
idx = np.indices(N3).astype(float)
c0 = n / 2.0
xA = [idx[d] - c0 for d in AX]
xB = [idx[d] + 0.5 - c0 for d in AX]


def cyl(x):
    return np.hypot(x[0], x[1]), np.arctan2(x[1], x[0]), x[2]


CX = [float(c) for c in args.cx.split(",")]


def shifted(x, c):
    return [x[0] - c, x[1], x[2]]


rings = []
for c in CX:
    rA_, psA_, zA_ = cyl(shifted(xA, c))
    rB_, psB_, zB_ = cyl(shifted(xB, c))
    rings.append((np.exp(-((rA_ - args.R) ** 2 + zA_**2) / (2 * args.a**2)), psA_,
                  np.exp(-((rB_ - args.R) ** 2 + zB_**2) / (2 * args.a**2)), psB_))
rA, psA, zA = cyl(xA)
rB, psB, zB = cyl(xB)
GA = sum(r[0] for r in rings)
GB = sum(r[2] for r in rings)
q = args.q
omega = 2.0 * q / args.R * args.sense

b = Box(N3)
b.uA[:] = args.A * sum(r[0] * np.cos(q * r[1]) for r in rings)
b.uB[:] = args.A * sum(r[2] * np.cos(q * r[3]) for r in rings)


def rim(x):
    r = np.sqrt(sum(xx**2 for xx in x))
    return np.clip((r - (n / 2 - 8)) / 8, 0, 1) ** 2


sA, sB = rim(xA), rim(xB)
fA, fB = np.exp(-0.5 * args.dt * sA), np.exp(-0.5 * args.dt * sB)


def lap(PA, PB):
    LA = sum(W_HEX**2 * (_sh(PB, o) - PA) for o in OFF)
    LB = sum(W_HEX**2 * (_shb(PA, o) - PB) for o in OFF)
    for d in AX:
        LA = LA + W_SQ**2 * (np.roll(PA, -1, axis=d) + np.roll(PA, 1, axis=d) - 2 * PA)
        LB = LB + W_SQ**2 * (np.roll(PB, -1, axis=d) + np.roll(PB, 1, axis=d) - 2 * PB)
    return 0.5 * LA, 0.5 * LB  # 1/2: content wave speed (c 2 in cubic units) for cD = 1


def energy_density():
    EA, EB = 0.5 * b.uA**2, 0.5 * b.uB**2
    for j, o in enumerate(OFF):
        s = 0.25 * b.ph[j] ** 2
        EA += s
        EB += np.roll(s, shift=tuple(x - 1 for x in o), axis=AX)
    for d in AX:
        sa, sb = 0.25 * b.sqA[d] ** 2, 0.25 * b.sqB[d] ** 2
        EA += sa + np.roll(sa, 1, axis=d)
        EB += sb + np.roll(sb, 1, axis=d)
    return EA, EB


mixw = None


def local_g():
    eA = np.clip(kappa * PA + (gfac * bgA if grad_on else 0.0), -0.5, args.dmax)
    eB = np.clip(kappa * PB + (gfac * bgB if grad_on else 0.0), -0.5, args.dmax)
    return (1 - eA) / (1 + eA), (1 - eB) / (1 + eB)


def source():
    global mixw
    if args.src == "flux":
        return throughput()
    if args.src == "energy":
        return energy_density()
    FA, FB = throughput()
    if args.src == "local":
        gA, gB = local_g()
        return FA / gA, FB / gB
    EA, EB = energy_density()
    if mixw is None:
        mixw = max(EA.max(), EB.max()) / max(FA.max(), FB.max())
    return 0.5 * (EA + mixw * FA), 0.5 * (EB + mixw * FB)


def throughput():
    FA, FB = np.zeros(N3), np.zeros(N3)
    for j, o in enumerate(OFF):
        s = np.abs(b.fh[j] * b.ph[j] * 0.5 * (b.uA + _sh(b.uB, o)))
        FA += s
        FB += np.roll(s, shift=tuple(x - 1 for x in o), axis=AX)
    for d in AX:
        sa = np.abs(b.fsA[d] * b.sqA[d] * 0.5 * (b.uA + np.roll(b.uA, -1, axis=d)))
        sb = np.abs(b.fsB[d] * b.sqB[d] * 0.5 * (b.uB + np.roll(b.uB, -1, axis=d)))
        FA += sa + np.roll(sa, 1, axis=d)
        FB += sb + np.roll(sb, 1, axis=d)
    return FA, FB


cD2 = args.cD**2
PA, PB = np.zeros(N3), np.zeros(N3)
VA, VB = np.zeros(N3), np.zeros(N3)
kappa = 0.0
bgA, bgB = args.grad * xA[0], args.grad * xB[0]
grad_on = False
gfac = 1.0


def set_medium():
    eA = np.clip(kappa * PA + (gfac * bgA if grad_on else 0.0), -0.5, args.dmax)
    eB = np.clip(kappa * PB + (gfac * bgB if grad_on else 0.0), -0.5, args.dmax)
    gA, gB = (1 - eA) / (1 + eA), (1 - eB) / (1 + eB)
    b.fh = np.stack([W_HEX * 0.5 * (gA + _sh(gB, o)) for o in OFF])
    b.fsA = np.stack([W_SQ * 0.5 * (gA + np.roll(gA, -1, axis=d)) for d in AX])
    b.fsB = np.stack([W_SQ * 0.5 * (gB + np.roll(gB, -1, axis=d)) for d in AX])


def p_step(FA, FB, h, damp=0.0):
    global PA, PB, VA, VB
    LA, LB = lap(PA, PB)
    VA += h * (cD2 * LA + FA - damp * VA)
    VB += h * (cD2 * LB + FB - damp * VB)
    VA *= fA
    VB *= fB
    PA += h * VA
    PB += h * VB
    PA *= fA
    PB *= fB


w0 = 2.0 * q / args.R
for j, o in enumerate(OFF):
    pass
gc = (1 - args.depth) / (1 + args.depth)
omega = 2.0 * gc * q / args.R * args.sense
QA = args.A * sum(r[0] * np.sin(q * r[1]) for r in rings)
QB = args.A * sum(r[2] * np.sin(q * r[3]) for r in rings)


def seed_phi():
    for j, o in enumerate(OFF):
        b.ph[j] = -b.fh[j] * (_sh(QB, o) - QA) / omega
    for d in AX:
        b.sqA[d] = -b.fsA[d] * (np.roll(QA, -1, axis=d) - QA) / omega
        b.sqB[d] = -b.fsB[d] * (np.roll(QB, -1, axis=d) - QB) / omega


for it in range(args.scf):
    seed_phi()
    FA0, FB0 = source()
    for i in range(int(args.tpre / args.dt)):
        p_step(FA0, FB0, args.dt, damp=0.3)
    kappa = args.kfix if args.kfix > 0 else args.depth / max(PA.max(), PB.max())
    set_medium()
    print(f"  scf {it}: F max {max(FA0.max(), FB0.max()):.3e}  kappa {kappa:.3e}", flush=True)
seed_phi()
print(f"preseed: kappa {kappa:.3e}  kP max {kappa * PA.max():.3f}", flush=True)

E0 = b.energy()
cw = np.zeros(3)


def centroid(cen):
    m = sum((xx - cc) ** 2 for xx, cc in zip(xA, cen)) < (2 * args.R) ** 2
    w = b.uA**2 * m
    return np.array([(xx * w).sum() / w.sum() for xx in xA]), (b.uA**2 + 0)[m].sum() / (b.uA**2).sum()


def lopsided(cen):
    y = [xx - cc for xx, cc in zip(xA, cen)]
    rho = np.hypot(y[0], y[1])
    m = ((rho - args.R) ** 2 + y[2] ** 2) < (2.5 * args.a) ** 2
    w = (b.uA**2)[m]
    ps = np.arctan2(y[1], y[0])[m]
    return abs(np.sum(w * np.exp(-1j * ps))) / w.sum(), w.sum()



ix_line = (np.abs(xA[1]) < 0.5) & (np.abs(xA[2]) < 0.5)
xs = xA[0][ix_line]
order = np.argsort(xs)
xs = xs[order]
snaps = [float(s) for s in args.snap.split(",")] if args.snap else []
rec = {"xs": xs, "kappa": kappa, "CX": np.array(CX)}


def body_stats():
    out = []
    eu = b.uA**2
    tot = eu.sum()
    for c in CX:
        y = shifted(xA, c)
        m = sum(yy**2 for yy in y) < (1.6 * args.R) ** 2
        w = eu * m
        cen = [float((yy * w).sum() / w.sum()) for yy in y]
        out.append((float(w.sum() / tot), cen))
    return out


nst = int(args.T / args.dt)
print(f"n {n} R {args.R} a {args.a} q {q} cx {CX} depth {args.depth} T {args.T}", flush=True)
t0 = time.time()
for i in range(1, nst + 1):
    t = i * args.dt
    b.step(args.dt)
    b.uA *= fA
    b.uB *= fB
    if i % args.every == 0 and not args.freeze:
        FA, FB = source()
        p_step(FA, FB, args.every * args.dt)
        set_medium()
    if i % int(round(10 / args.dt)) == 0 or i == nst:
        bs = body_stats()
        dline = (kappa * PA[ix_line])[order]
        mid = float(np.interp(0.0, xs, dline))
        el = time.time() - t0
        print(f"  t {t:6.0f}  E/E0 {b.energy() / E0:.4f}  " + "  ".join(
            f"body{k} share {s:.3f} cen {c[0]:+5.2f} {c[1]:+5.2f} {c[2]:+5.2f}" for k, (s, c) in enumerate(bs)) +
            f"  d(x=0) {mid:.3f}  d max {kappa * PA.max():.3f}  (ETA {el / i * (nst - i) / 60:.1f} min)", flush=True)
    for s in snaps:
        if abs(t - s) < 0.5 * args.dt:
            rec[f"d_{s:g}"] = (kappa * PA[ix_line])[order]
            rec[f"u2_{s:g}"] = (b.uA[ix_line] ** 2)[order]
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **rec, PA=PA, PB=PB, VA=VA, VB=VB, uA=b.uA, uB=b.uB, ph=b.ph, sqA=b.sqA, sqB=b.sqB,
             args=str(vars(args)))
print(f"done {time.time() - t0:.0f}s", flush=True)
