"""10-16 nonrad: imposed neutral rotating cores moving along +z at v in the hop / triangle DEC sector (vacuum).
Electric dipole P = p(t) g (J = dP/dt, bound charge only, conserved), toroidal dipole T: M = alpha sig^2 curl(T(t) g)
(J = curl M). p, T rotate in xy at omega (T leads by psi). Radiation = energy outside a co-moving sphere of radius Rf,
growth per unit distance travelled. Philip: core = rotating + translating charge structure, wake = its field."""
import argparse
import time

import numpy as np
import scipy.sparse as sp

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=140)
p.add_argument("--v", type=float, default=1.0)
p.add_argument("--rot", type=float, default=0.5)
p.add_argument("--sig", type=float, default=3.0)
p.add_argument("--wP", type=float, default=1.0, help="electric dipole weight")
p.add_argument("--alpha", type=float, default=0.0, help="toroidal dipole weight (units sig^2)")
p.add_argument("--psi", type=float, default=0.0, help="toroidal phase lead (deg)")
p.add_argument("--T", type=float, default=40.0)
p.add_argument("--dt", type=float, default=0.25)
p.add_argument("--Rf", type=float, default=12.0)
p.add_argument("--z0", type=float, default=-20.0)
p.add_argument("--anapole", action="store_true", help="P = dT/dt exactly (alpha sets T)")
p.add_argument("--ramp", type=float, default=5.0)
p.add_argument("--every", type=float, default=10.0)
args = p.parse_args()
D, MIR = C.D_HAT, C.K_MIRROR
CAN = [4, 5, 6, 7, 9, 11, 13]
r, s2 = C.dec_star("dec")
ac = 2 / np.sqrt(3)
star1 = np.array([r * ac / 8 if k < 8 else ac / 8 for k in CAN])
star2 = s2 / ac
side = args.P / np.sqrt(3)
L = C.AnchorCluster(cpp_box(args.P, n_threads=8))
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
    tri.append((D[k] + D[s]) / 3)
Cm = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(12 * N, 7 * N))
CT = Cm.T.tocsr()
S1 = np.tile(star1, N)
DC = D[CAN]
mid = (pos[:, None, :] + 0.5 * DC[None]).reshape(-1, 3)
dirs = np.tile(DC, (N, 1))
tcen = (pos[:, None, :] + np.array(tri)[None]).reshape(-1, 3)
FG = lambda x: x - (CT @ (star2 * (Cm @ x))) / S1 / 27.0
psi = np.radians(args.psi)


def src(t):
    if args.anapole:
        h = 1e-3
        Tm, _, _ = src_raw(t - h)
        Tp, _, _ = src_raw(t + h)
        T0, Mm, zs = src_raw(t)
        return FG((Tp - Tm) / (2 * h)), Mm, zs
    Pe, Mm, zs = src_raw(t)
    return Pe, Mm, zs


def src_raw(t):
    zs = args.z0 + args.v * t
    X = wrapx(mid - np.array([0.13, 0.07, zs]))
    g = np.exp(-np.sum(X**2, 1) / (2 * args.sig**2))
    ph = args.rot * t
    g = g * np.sin(0.5 * np.pi * np.clip(t / args.ramp, 0, 1))**2
    Pe = args.wP * g * (dirs[:, 0] * np.cos(ph) + dirs[:, 1] * np.sin(ph))
    Te = args.alpha * args.sig**2 * g * (dirs[:, 0] * np.cos(ph + psi) + dirs[:, 1] * np.sin(ph + psi))
    if args.anapole:
        return Te, star2 * (Cm @ FG(Te)), zs
    return FG(Pe), star2 * (Cm @ FG(Te)), zs


a = np.zeros(7 * N)
b = np.zeros(12 * N)
Xc = wrapx(pos)
HC = np.repeat(np.arange(N), 7)
TC = np.repeat(np.arange(N), 12)


def energies(zs):
    u = np.bincount(HC, 0.5 * S1 * a**2, N) + np.bincount(TC, 0.5 * star2 * b**2, N)
    dz = wrapx(pos[:, 2] - zs)
    slab = np.abs(dz) < args.Rf / 2
    behind = (dz < -args.Rf / 2) & (dz > -side / 2 + 2)
    rr = np.hypot(*wrapx(pos[:, :2] - np.array([0.13, 0.07])).T)
    global RHO
    RHO = np.sqrt(np.sum(u[slab] * rr[slab]**2) / max(u[slab].sum(), 1e-300))
    return u[behind].sum(), u[slab].sum()


nst = int(round(args.T / args.dt))
ev = int(round(args.every / args.dt))
print(f"P {args.P} N {N} v {args.v} rot {args.rot} sig {args.sig} wP {args.wP} alpha {args.alpha} psi {args.psi} Rf {args.Rf}", flush=True)
hist = []
t0 = time.time()
Pp, _, _ = src(-0.5 * args.dt)
for st in range(nst + 1):
    t = st * args.dt
    if st % ev == 0:
        zs = args.z0 + args.v * t
        Ef, En = energies(zs)
        hist.append((t, Ef, En))
        print(f"  t {t:5.1f}  E_behind {Ef:.4e}  E_slab {En:.4e}  rho_rms(slab) {RHO:.2f}  ({time.time() - t0:.0f} s)", flush=True)
    if st == nst:
        break
    b -= 0.5 * args.dt * (Cm @ a)
    Pn, Mm, _ = src(t + 0.5 * args.dt)
    a += args.dt * (CT @ (star2 * b - Mm)) / S1 - (Pn - Pp)
    Pp = Pn
    b -= 0.5 * args.dt * (Cm @ a)
h = np.array(hist)
late = h[h[:, 0] >= args.T / 2]
rate = np.polyfit(late[:, 0] * args.v, late[:, 1], 1)[0]
print(f"RESULT dE_behind / d(distance) {rate:.4e}   E_slab(end) {h[-1, 2]:.4e}   ratio {rate / h[-1, 2]:.4e}")
