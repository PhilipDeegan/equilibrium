"""Ring mode at c (q wavelengths per circuit = q own periods per transit) in a depletion-feedback medium.

Discrete-weight periodic BCC box (../defect/bcc.py), step() with face coefficient W * g_f, g = (1 - d) / (1 + d) per cell
(contraction: rate and hop both reduced, full-GR form), g_f = mean of the pair. Depletion d per cell relaxes over tau
toward min(dmax, kappa * F), F = sum over the cell's faces of |W g_f phi_f ubar_f| (local energy throughput, no vectors).
Energy H = 1/2 sum u^2 + 1/2 sum phi^2 is exact for any g(t); depletion itself costs nothing (v1 guess).
Sponge in the outer shell (u per cell, phi per face once: antisymmetry kept). Seed: u = A G(rho) cos(q psi), phi from
the travelling-wave relation dphi/dt = -W g du with omega from the measured ring speed.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "defect"))
from bcc import AX, OFF, W_HEX, W_SQ, Box, _sh  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=72)
p.add_argument("--R", type=float, default=18.0)
p.add_argument("--a", type=float, default=3.0)
p.add_argument("--q", type=float, default=4)
p.add_argument("--A", type=float, default=1.0)
p.add_argument("--kappa", type=float, default=0.0)
p.add_argument("--dmax", type=float, default=0.5)
p.add_argument("--T", type=float, default=600.0)
p.add_argument("--dt", type=float, default=0.1)
p.add_argument("--every", type=int, default=10)
p.add_argument("--save", default="")
p.add_argument("--well", type=float, default=0.0, help="held depression depth on the seed tube (forced pinch stand-in)")
p.add_argument("--ksig", type=float, default=0.0, help="signed depletion: contract where u > 0, expand where u < 0")
p.add_argument("--tsig", type=float, default=0.5, help="signed depletion relaxation time")
p.add_argument("--dmin", type=float, default=-0.3, help="most negative total d (expansion bound)")
p.add_argument("--grad", type=float, default=0.0, help="background depletion gradient along x (contracted toward +x)")
p.add_argument("--toff", type=float, default=1e9, help="time the gradient is switched off")
p.add_argument("--follow", action="store_true", help="held well re-centred on the content centroid (co-moving pinch)")
p.add_argument("--sense", type=float, default=1.0, help="+1 travels +psi, -1 travels -psi")
p.add_argument("--preseed", action="store_true", help="start with d in equilibrium with the seed (event already forced it)")
p.add_argument("--tau", type=float, default=0.0, help="depletion relaxation time (0 = one period)")
args = p.parse_args()

n = args.n
N3 = (n, n, n)
idx = np.indices(N3).astype(float)
c0 = n / 2.0
xA = [idx[d] - c0 for d in AX]
xB = [idx[d] + 0.5 - c0 for d in AX]


def cyl(x):
    rho = np.hypot(x[0], x[1])
    return rho, np.arctan2(x[1], x[0]), x[2]


def env(x):
    rho, psi, z = cyl(x)
    return np.exp(-((rho - args.R) ** 2 + z**2) / (2 * args.a**2)), psi


gA_, psiA = env(xA)
gB_, psiB = env(xB)
q = args.q
c_eff = 2.0
omega = c_eff * q / args.R
omega_s = args.sense * omega


def field(G, psi, phase):
    return args.A * G * np.cos(q * psi - phase)


b = Box(N3)
b.uA[:] = field(gA_, psiA, 0.0)
b.uB[:] = field(gB_, psiB, 0.0)
# phi(t=0) for u = Re(A G e^{i(q psi - w t)}): phi = Re(W (u_nb - u)(complex) / (i w)); use quadrature field
qA, qB = field(gA_, psiA, np.pi / 2), field(gB_, psiB, np.pi / 2)
for j, o in enumerate(OFF):
    b.ph[j] = -W_HEX * (_sh(qB, o) - qA) / omega_s
for d in AX:
    b.sqA[d] = -W_SQ * (np.roll(qA, -1, axis=d) - qA) / omega_s
    b.sqB[d] = -W_SQ * (np.roll(qB, -1, axis=d) - qB) / omega_s

r_edge = n / 2 - 8


def sponge(x):
    r = np.sqrt(sum(xx**2 for xx in x))
    s = np.clip((r - r_edge) / 8, 0, 1)
    return np.exp(-0.5 * args.dt * s**2)


fA, fB = sponge(xA), sponge(xB)
dA, dB = np.zeros(N3), np.zeros(N3)
wA, wB = args.well * gA_, args.well * gB_
sA_, sB_ = np.zeros(N3), np.zeros(N3)
bgA, bgB = args.grad * xA[0], args.grad * xB[0]
grad_on = args.grad != 0.0
cw = np.zeros(3)


def well_at(cen):
    def g(x):
        y = [x[0] - cen[0], x[1] - cen[1], x[2] - cen[2]]
        rho = np.hypot(y[0], y[1])
        return np.exp(-((rho - args.R) ** 2 + y[2] ** 2) / (2 * args.a**2))
    return args.well * g(xA), args.well * g(xB)


def centroid(cen):
    m = sum((xx - cc) ** 2 for xx, cc in zip(xA, cen)) < (2 * args.R) ** 2
    w = b.uA**2 * m
    return np.array([(xx * w).sum() / w.sum() for xx in xA])


def lopsided(cen):
    y = [xx - cc for xx, cc in zip(xA, cen)]
    rho = np.hypot(y[0], y[1])
    m = ((rho - args.R) ** 2 + y[2] ** 2) < (2.5 * args.a) ** 2
    w = (b.uA**2)[m]
    ps = np.arctan2(y[1], y[0])[m]
    return abs(np.sum(w * np.exp(-1j * ps))) / w.sum()
tau = args.tau if args.tau > 0 else 2 * np.pi / omega


def set_medium():
    ga, gb = (bgA, bgB) if grad_on else (0.0, 0.0)
    eA, eB = np.clip(wA + dA + sA_ + ga, args.dmin, args.dmax), np.clip(wB + dB + sB_ + gb, args.dmin, args.dmax)
    gA, gB = (1 - eA) / (1 + eA), (1 - eB) / (1 + eB)
    b.fh = np.stack([W_HEX * 0.5 * (gA + _sh(gB, o)) for o in OFF])
    b.fsA = np.stack([W_SQ * 0.5 * (gA + np.roll(gA, -1, axis=d)) for d in AX])
    b.fsB = np.stack([W_SQ * 0.5 * (gB + np.roll(gB, -1, axis=d)) for d in AX])


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


tube = lambda x: cyl(x)[0]
rhoA, _, zA = cyl(xA)
rhoB, _, zB = cyl(xB)
inA = (rhoA - args.R) ** 2 + zA**2 < (2.5 * args.a) ** 2
inB = (rhoB - args.R) ** 2 + zB**2 < (2.5 * args.a) ** 2
ringA = np.abs(np.hypot(rhoA - args.R, zA)) < 1.0


def diag():
    eA, eB = b.uA**2, b.uB**2
    tot = b.energy()
    ein = 0.5 * (eA[inA].sum() + eB[inB].sum()) * 2  # u-part x 2 ~ equipartition
    w = eA + 0.0
    rbar = (rhoA * w).sum() / w.sum()
    zrms = np.sqrt((zA**2 * w).sum() / w.sum())
    ps = psiA[inA]
    ur = b.uA[inA]
    ms = np.arange(0, 11)
    cm = np.abs([np.sum(ur * np.exp(-1j * m * ps)) for m in ms]) ** 2
    return tot, ein / tot, rbar, zrms, cm / (cm.sum() + 1e-300)


FA0, FB0 = throughput()
print(f"seed throughput max {max(FA0.max(), FB0.max()):.4e} -> kappa for saturation ~ {args.dmax / max(FA0.max(), FB0.max()):.3e}", flush=True)
if args.preseed and args.kappa > 0:
    dA[:] = np.minimum(args.dmax, args.kappa * FA0)
    dB[:] = np.minimum(args.dmax, args.kappa * FB0)
    set_medium()
if args.well > 0 or grad_on:
    set_medium()
E0 = b.energy()
nst = int(args.T / args.dt)
print(f"n {n} R {args.R} a {args.a} q {q} A {args.A} kappa {args.kappa} dmax {args.dmax} T {args.T} omega {omega:.4f} "
      f"transit {2 * np.pi * args.R / c_eff:.1f} E0 {E0:.4e}", flush=True)
t0 = time.time()
hist = []
for i in range(1, nst + 1):
    b.step(args.dt)
    b.uA *= fA
    b.uB *= fB
    if grad_on and i * args.dt >= args.toff:
        grad_on = False
        set_medium()
    if args.follow and i % args.every == 0:
        cw = centroid(cw)
        wA, wB = well_at(cw)
        if not (args.kappa > 0 or args.ksig > 0):
            set_medium()
    if (args.kappa > 0 or args.ksig > 0) and i % args.every == 0:
        if args.kappa > 0:
            FA, FB = throughput()
            r = args.every * args.dt / tau
            dA += r * (np.minimum(args.dmax, args.kappa * FA) - dA)
            dB += r * (np.minimum(args.dmax, args.kappa * FB) - dB)
        if args.ksig > 0:
            r = min(1.0, args.every * args.dt / args.tsig)
            sA_ += r * (args.ksig * b.uA - sA_)
            sB_ += r * (args.ksig * b.uB - sB_)
        set_medium()
    if i % int(round(10 / args.dt)) == 0:
        tot, fin, rbar, zrms, cm = diag()
        m = int(np.argmax(cm))
        hist.append((i * args.dt, tot / E0, fin, rbar, zrms, m, cm[int(round(q))] if q == int(q) else 0.0, dA.max()))
        el = time.time() - t0
        print(f"  t {i * args.dt:6.0f}  E/E0 {tot / E0:.4f}  in-tube {fin:.3f}  rbar {rbar:5.2f}  zrms {zrms:5.2f}  "
              f"cen {cw[0]:+6.2f} {cw[1]:+6.2f} {cw[2]:+6.2f}  lop {lopsided(cw):.3f}  m* {m}  P(m=2..6) {' '.join(f'{x:.2f}' for x in cm[2:7])}  dmax {dA.max():.3f}  (ETA {el / i * (nst - i) / 60:.1f} min)", flush=True)
h = np.array(hist)
print(f"RESULT q {q} A {args.A} kappa {args.kappa} ksig {args.ksig}  E/E0 {h[-1, 1]:.4f}  in-tube {h[-1, 2]:.3f}  rbar {h[-1, 3]:.2f}  "
      f"zrms {h[-1, 4]:.2f}  m* {int(h[-1, 5])}  dmax {h[-1, 7]:.3f}", flush=True)
if args.save:
    np.savez(args.save, hist=h, uA=b.uA, uB=b.uB, dA=dA, dB=dB)
