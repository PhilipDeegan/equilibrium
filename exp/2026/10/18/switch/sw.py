"""10-18 switch: imposed neutral core moving along +z at v whose content switches electric <-> magnetic (Philip).
Electric dipole p(t) = wP cos(om t) x g on hops (J = dP/dt); magnetic dipole m(t) = wM cos(om t - psi) mdir g as
triangle magnetization (H = star2 b - star2 M.A_t). psi 90 = switching (energy alternates), psi 0 = in phase
(Huygens / Kerker when m || y). Leak = growth of energy behind the co-moving slab per unit distance."""
import argparse
import time

import numpy as np
import scipy.sparse as sp

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=140)
p.add_argument("--v", type=float, default=1.0)
p.add_argument("--om", type=float, default=0.5)
p.add_argument("--sig", type=float, default=3.0)
p.add_argument("--wP", type=float, default=1.0, help="electric dipole weight")
p.add_argument("--wM", type=float, default=0.0, help="magnetic dipole weight")
p.add_argument("--mdir", default="y", choices=["x", "y", "z"])
p.add_argument("--psi", type=float, default=90.0, help="magnetic phase lag (deg)")
p.add_argument("--T", type=float, default=40.0)
p.add_argument("--dt", type=float, default=0.25)
p.add_argument("--Rf", type=float, default=12.0)
p.add_argument("--z0", type=float, default=-20.0)
p.add_argument("--save", default="")
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
TV = np.array([0.5 * np.cross(D[k], D[m]) for (s_, k, m) in C._DEC_TRIS])
for B0 in np.eye(3):
    A0 = 0.5 * np.cross(B0, mid)
    fl = (Cm @ np.sum(A0 * dirs, 1)).reshape(N, 12)
    inner = np.all(np.abs(wrapx(pos)) < side / 4, 1)
    assert np.allclose(fl[inner], (TV @ B0)[None], atol=1e-9), "triangle area vectors"
AM = np.tile(TV @ np.eye(3)[{"x": 0, "y": 1, "z": 2}[args.mdir]], N)


def src(t):
    zs = args.z0 + args.v * t
    ramp = np.sin(0.5 * np.pi * np.clip(t / args.ramp, 0, 1))**2
    g = lambda x: ramp * np.exp(-np.sum(wrapx(x - np.array([0.13, 0.07, zs]))**2, 1) / (2 * args.sig**2))
    ph = args.om * t
    Pe = args.wP * np.cos(ph) * g(mid) * dirs[:, 0]
    Mm = star2 * args.wM * np.cos(ph - psi) * g(tcen) * AM
    return FG(Pe), Mm, zs


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
print(f"P {args.P} N {N} v {args.v} om {args.om} sig {args.sig} wP {args.wP} wM {args.wM} mdir {args.mdir} psi {args.psi} Rf {args.Rf}", flush=True)
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
dz = wrapx(pos[:, 2] - (args.z0 + args.v * args.T))
near = np.hypot(*wrapx(pos[:, :2] - np.array([0.13, 0.07])).T) < args.sig
zb = np.round(dz[near]).astype(int)
ax = (a.reshape(N, 7) * DC[None, :, 0]).sum(1)[near]
ay = (a.reshape(N, 7) * DC[None, :, 1]).sum(1)[near]
zz = np.arange(zb.min(), zb.max() + 1)
cnt = np.bincount(zb - zz[0])
prof = np.array([np.bincount(zb - zz[0], q) / np.maximum(cnt, 1) for q in (ax, ay)])
print("wake axis profile (z rel core: <a.x> <a.y>), every 4:")
for k in range(0, len(zz), 4):
    print(f"  {zz[k]:+4d}  {prof[0, k]:+.3e}  {prof[1, k]:+.3e}")
if args.save:
    np.savez_compressed(args.save, a=a, b=b, pos=pos, hist=h, zz=zz, prof=prof, args=str(vars(args)))
