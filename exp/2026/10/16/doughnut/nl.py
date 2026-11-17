"""10-16 doughnut (nl.py: + thresholded saturating throughput: eps_hop = 1 + Delta s / (1 + s), s = max(0, u - u_th) / u_th,
u = cell energy density (hops + triangles), u_th = th x initial max; core slower -> self-lensing; wake below threshold
stays linear).
 Hellwarth-Nouchi flying doughnut (TE) as a source-free IC in the hop / triangle DEC sector (vacuum).
A = curl(z f), f = 1 / Q, Q = (q1 + i tau)(q2 - i sigma) + rho^2, tau = z - t, sigma = z + t; E = -dA/dt, B = curl A.
Real part used (complex only as the construction tool). a = E(mid).d on hops, b = C (A(mid).d) (exact loop sums).
Tracks centroid speed, transverse / axial size, poloidal vs toroidal B share, and agreement with the exact solution."""
import argparse
import time

import numpy as np
import scipy.sparse as sp

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=140)
p.add_argument("--q1", type=float, default=3.0)
p.add_argument("--q2", type=float, default=30.0)
p.add_argument("--h", type=float, default=1.0, help="refinement: lengths q1, q2 in units of h cells (h 2 = twice finer)")
p.add_argument("--T", type=float, default=40.0)
p.add_argument("--dt", type=float, default=0.25)
p.add_argument("--every", type=float, default=5.0)
p.add_argument("--Delta", type=float, default=0.0)
p.add_argument("--th", type=float, default=0.3)
p.add_argument("--plane", type=int, default=0)
p.add_argument("--z0", type=float, default=-20.0)
args = p.parse_args()
D, MIR = C.D_HAT, C.K_MIRROR
CAN = [4, 5, 6, 7, 9, 11, 13]
r, s2 = C.dec_star("dec")
ac = 2 / np.sqrt(3)
star1 = np.array([r * ac / 8 if k < 8 else ac / 8 for k in CAN])
star2 = s2 / ac
side = args.P / np.sqrt(3)
L = ac_ = C.AnchorCluster(cpp_box(args.P, n_threads=8))
pos, nb, N = L.pos, L.nb, L.N
wrapx = lambda x: (x + side / 2) % side - side / 2
rows, cols, vals, tri = [], [], [], []
ids = np.arange(N)
for t, (s, k, m) in enumerate(C._DEC_TRIS):
    for (cellv, kk, sg) in (("0", k, 1.0), ("k", m, 1.0), ("0", s, -1.0)):
        base = ids if cellv == "0" else nb[:, k]
        if kk in CAN:
            cell, c, se = base, CAN.index(kk), 1.0
        else:
            cell, c, se = nb[base, kk], CAN.index(MIR[kk]), -1.0
        rows.append(ids * 12 + t); cols.append(cell * 7 + c); vals.append(np.full(N, sg * se))
    tri.append((0.5 * np.cross(D[k], D[s]), (D[k] + D[s]) / 3))
Cm = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(12 * N, 7 * N))
CT = Cm.T.tocsr()
S1 = np.tile(star1, N)
HC = np.repeat(np.arange(N), 7)
HN = np.empty(7 * N, int)
for c_, k_ in enumerate(CAN):
    HN[c_::7] = nb[:, k_]
TCELL = np.repeat(np.arange(N), 12)


def ucell(a, b):
    return np.bincount(HC, 0.5 * S1 * a**2, N) + np.bincount(HN, 0.5 * S1 * a**2, N) * 0 + np.bincount(TCELL, 0.5 * star2 * b**2, N)
DC = D[CAN]
mid = (pos[:, None, :] + 0.5 * DC[None]).reshape(-1, 3)
dirs = np.tile(DC, (N, 1))
TA = np.array([t_[0] for t_ in tri]); TC = np.array([t_[1] for t_ in tri])
tcen = (pos[:, None, :] + TC[None]).reshape(-1, 3)
tarea = np.tile(TA, (N, 1))
q1, q2 = args.q1 * args.h, args.q2 * args.h


def fields(X, t):
    if args.plane:
        qz = 2 * np.pi * args.plane / side
        ph = qz * (X[:, 2] - t)
        A = np.stack([np.cos(ph), 0 * ph, 0 * ph], 1)
        E = np.stack([-qz * np.sin(ph), 0 * ph, 0 * ph], 1)
        return A, E
    x, y, z = X[:, 0], X[:, 1], X[:, 2] - args.z0 * args.h
    tau, sig = z - t, z + t
    Q = (q1 + 1j * tau) * (q2 - 1j * sig) + x**2 + y**2
    dQ = -2 * t - 1j * (q1 + q2)
    A = np.stack([-2 * y, 2 * x, 0 * x], 1) / Q[:, None]**2
    E = np.stack([-2 * y, 2 * x, 0 * x], 1) * (2 * dQ / Q**3)[:, None]
    Wd = np.ones(len(X))
    for c in range(3):
        u = np.abs(wrapx(X[:, c] - (args.z0 * args.h if c == 2 else 0))) / (side / 2)
        Wd *= np.where(u < 0.7, 1.0, np.cos(0.5 * np.pi * np.clip((u - 0.7) / 0.3, 0, 1))**2)
    return A.real * Wd[:, None], E.real * Wd[:, None]


FG = lambda x: x - (CT @ (star2 * (Cm @ x))) / S1 / 27.0


def exact_b(t):
    Xm = wrapx(mid)
    A, _ = fields(Xm, t)
    return Cm @ FG(np.sum(A * dirs, 1))


Xm = wrapx(mid)
A0, E0 = fields(Xm, 0.0)
a = FG(np.sum(E0 * dirs, 1))
b = Cm @ FG(np.sum(A0 * dirs, 1))
en = lambda: 0.5 * np.sum(S1 * a**2 * eps_of(a, b)) + 0.5 * star2 * np.sum(b**2)
Xt = wrapx(tcen)
rho_t = np.hypot(Xt[:, 0], Xt[:, 1])
phi_hat = np.stack([-Xt[:, 1], Xt[:, 0], 0 * rho_t], 1) / np.maximum(rho_t, 1e-9)[:, None]
nhat = tarea / np.linalg.norm(tarea, axis=1)[:, None]
tor_w = np.sum(nhat * phi_hat, 1)**2


def stats(t):
    e = b**2
    zc = np.sum(e * Xt[:, 2]) / e.sum()
    rr = np.sqrt(np.sum(e * rho_t**2) / e.sum())
    lz = np.sqrt(np.sum(e * (Xt[:, 2] - zc)**2) / e.sum())
    ex = exact_b(t)
    core = (rho_t < 0.5 * side / 2) & (np.abs(Xt[:, 2] - zc) < 0.5 * side / 2)
    err = np.linalg.norm((b - ex)[core]) / np.linalg.norm(ex[core])
    ee = ex**2
    global EXS
    EXS = (np.sum(ee * Xt[:, 2]) / ee.sum(), np.sqrt(np.sum(ee * rho_t**2) / ee.sum()))
    return zc, rr, lz, np.sum(e * tor_w) / e.sum(), err


U0 = ucell(a, b).max()
uth = args.th * U0


def eps_of(a, b):
    if not args.Delta:
        return 1.0
    u = ucell(a, b)
    uh = 0.5 * (u[HC] + u[HN])
    sx = np.maximum(0.0, uh - uth) / uth
    return 1.0 + args.Delta * sx / (1 + sx)


nst = int(round(args.T / args.dt))
ev = int(round(args.every / args.dt))
print(f"P {args.P} N {N} q1 {q1} q2 {q2} ring radius sqrt(q1 q2) {np.sqrt(q1 * q2):.2f} E0 {en():.6e}", flush=True)
print("     t   dE/E0    z_c   speed   rho_rms  len_z  toroidal_B_share  |b - exact|/|exact|", flush=True)
E00, prev, t0 = en(), None, time.time()
for st in range(nst + 1):
    t = st * args.dt
    if st % ev == 0:
        zc, rr, lz, tor, err = stats(t)
        spd = (zc - prev[1]) / (t - prev[0]) if prev else np.nan
        prev = (t, zc)
        print(f"  {t:5.1f}  {en() / E00 - 1:+.1e}  {zc:+7.2f}  {spd:.4f}  {rr:6.2f}  {lz:6.2f}  {tor:.3f}  {err:.3e}  exact z_c {EXS[0]:+7.2f} rho {EXS[1]:6.2f}  ({time.time() - t0:.0f} s)", flush=True)
    if st == nst:
        break
    b -= 0.5 * args.dt * (Cm @ a)
    ep = eps_of(a, b)
    a += args.dt * (CT @ (star2 * b)) / (S1 * ep)
    if st % ev == 0 and args.Delta:
        print(f"      nonlinear hops {np.mean(np.atleast_1d(ep) > 1):.2e}  max eps {np.max(ep):.3f}", flush=True)
    b -= 0.5 * args.dt * (Cm @ a)
