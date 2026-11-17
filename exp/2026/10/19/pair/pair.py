"""10-19 two bodies + the space between (Philip: the space is part of the charge distribution; its holding has a floor;
offload bounded by total energy vs the impedance of free space). Linear hop / triangle DEC (gl.py with beta 0), per-hop
permittivity eps (n^2 inside a body, floor 1 elsewhere), absorbing outer shell. Bodies = dielectric spheres holding a
field by impedance mismatch (Z = Z0 / n); energy starts in A only. Snapshot "charge" of the gap = field energy in flight
between the bodies (cylinder of radius a between their surfaces)."""
import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=140)
p.add_argument("--a", type=float, default=5.0, help="body radius (cells)")
p.add_argument("--sep", type=float, default=20.0, help="centre separation (0 = A alone)")
p.add_argument("--nA", type=float, default=3.0)
p.add_argument("--nB", type=float, default=3.0)
p.add_argument("--ngap", type=float, default=1.0, help="index of the gap cylinder (floor 1 = vacuum)")
p.add_argument("--sig", type=float, default=0.5, help="seed width / a")
p.add_argument("--sponge", type=float, default=6.0, help="absorbing shell width (cells)")
p.add_argument("--T", type=float, default=200.0)
p.add_argument("--dt", type=float, default=0.25)
p.add_argument("--every", type=float, default=2.0)
p.add_argument("--save", default="")
args = p.parse_args()

D, MIR = C.D_HAT, C.K_MIRROR
CAN = [4, 5, 6, 7, 9, 11, 13]
r, s2 = C.dec_star("dec")
ac = 2 / np.sqrt(3)
star1 = np.array([r * ac / 8 if k < 8 else ac / 8 for k in CAN])
star2 = s2 / ac
side = args.P / np.sqrt(3)
L = C.AnchorCluster(cpp_box(args.P, n_threads=1))
pos, nb, N = L.pos, L.nb, L.N
wrap = lambda x: (x + side / 2) % side - side / 2
ids = np.arange(N)
rows, cols, vals = [], [], []
for t, (s, k, m) in enumerate(C._DEC_TRIS):
    for (cellv, kk, sg) in (("0", k, 1.0), ("k", m, 1.0), ("0", s, -1.0)):
        base = ids if cellv == "0" else nb[:, k]
        if kk in CAN:
            cell, c, se = base, CAN.index(kk), 1.0
        else:
            cell, c, se = nb[base, kk], CAN.index(MIR[kk]), -1.0
        rows.append(ids * 12 + t); cols.append(cell * 7 + c); vals.append(np.full(N, sg * se))
Cm = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(12 * N, 7 * N))
CT = Cm.T.tocsr()
DC = D[CAN]
X0 = np.array([0.13, 0.07, 0.11])
xc = wrap(pos - X0)
mid = wrap((pos[:, None, :] + 0.5 * DC[None]).reshape(-1, 3) - X0)
dirs = np.tile(DC, (N, 1))
TRI = [(D[k] + D[s]) / 3 for (s, k, m) in C._DEC_TRIS]
tcen = wrap((pos[:, None, :] + np.array(TRI)[None]).reshape(-1, 3) - X0)
hc = np.repeat(ids, 7)
hn = nb[hc, np.array(CAN)[np.tile(np.arange(7), N)]]
IE = sp.csr_matrix((np.full(14 * N, 0.5), (np.concatenate([hc, hn]), np.tile(np.arange(7 * N), 2))), shape=(N, 7 * N))
tc = np.repeat(ids, 12)
tk = np.tile([k for (s, k, m) in C._DEC_TRIS], N)
ts = np.tile([s for (s, k, m) in C._DEC_TRIS], N)
IB = sp.csr_matrix((np.full(36 * N, 1 / 3), (np.concatenate([tc, nb[tc, tk], nb[tc, ts]]), np.tile(np.arange(12 * N), 3))),
                   shape=(N, 12 * N))

cA = np.array([-args.sep / 2, 0, 0]) if args.sep > 0 else np.zeros(3)
cB = np.array([args.sep / 2, 0, 0])
inA = lambda x: np.sum((x - cA)**2, 1) < args.a**2
inB = lambda x: (np.sum((x - cB)**2, 1) < args.a**2) if args.sep > 0 else np.zeros(len(x), bool)
lo, hi = cA[0] + args.a, cB[0] - args.a
ingap = lambda x: (np.hypot(x[:, 1], x[:, 2]) < args.a) & (x[:, 0] > lo) & (x[:, 0] < hi) if args.sep > 0 else np.zeros(len(x), bool)
eps = np.ones(7 * N)
eps[ingap(mid)] = args.ngap**2
eps[inA(mid)] = args.nA**2
eps[inB(mid)] = args.nB**2
S1 = np.tile(star1, N) * eps

rr = np.max(np.abs(xc), 1)
edge = side / 2 - args.sponge
dmp = (lambda x: 0.3 * np.clip((np.max(np.abs(x), 1) - edge) / args.sponge, 0, 1)**2) if args.sponge > 0 else (lambda x: np.zeros(len(x)))
kd, kb = np.exp(-dmp(mid) * args.dt), np.exp(-dmp(tcen) * args.dt)

X = mid - cA
g = np.exp(-np.sum(X**2, 1) / (2 * (args.sig * args.a)**2))
al = np.sum((g[:, None] * np.array([0, 0, 1.0])) * dirs, 1)
b = Cm @ al
d = np.zeros(7 * N)
uE = lambda d: IE @ (0.5 * d**2 / S1)
uB = lambda b: IB @ (0.5 * star2 * b**2)
sc = 1 / np.sqrt((uE(d) + uB(b)).sum())
b *= sc
gE = lambda d: d / S1
gB = lambda b: star2 * b

mA, mB, mG = inA(xc), inB(xc), ingap(xc)
mS = (np.max(np.abs(xc), 1) > edge)
mO = ~(mA | mB | mG | mS)
nst, ev = int(round(args.T / args.dt)), int(round(args.every / args.dt))
print(f"P {args.P} N {N} a {args.a} sep {args.sep} nA {args.nA} nB {args.nB} ngap {args.ngap} sponge {args.sponge} "
      f"cells A {mA.sum()} B {mB.sum()} gap {mG.sum()} out {mO.sum()} T {args.T}", flush=True)
hist, absorbed, t0 = [], 0.0, time.time()
for st in range(nst + 1):
    if st % ev == 0:
        uc = uE(d) + uB(b)
        H = uc.sum()
        row = (st * args.dt, uc[mA].sum(), uc[mB].sum(), uc[mG].sum(), uc[mO].sum(), uc[mS].sum(), absorbed, H + absorbed)
        hist.append(row)
        print(f"  t {row[0]:6.1f}  A {row[1]:.4f}  B {row[2]:.4f}  gap {row[3]:.3e}  out {row[4]:.4f}  shell {row[5]:.4f}  "
              f"absorbed {row[6]:.4f}  total {row[7]:.6f}  ({time.time() - t0:.0f} s, ETA {(time.time() - t0) / max(st, 1) * (nst - st) / 60:.1f} min)",
              flush=True)
    if st == nst:
        break
    b -= 0.5 * args.dt * (Cm @ gE(d))
    d += args.dt * (CT @ gB(b))
    b -= 0.5 * args.dt * (Cm @ gE(d))
    H0 = (uE(d) + uB(b)).sum()
    d *= kd
    b *= kb
    absorbed += H0 - (uE(d) + uB(b)).sum()
h = np.array(hist)
iA = h[:, 1]
tA = h[np.argmax(iA < np.exp(-1) * iA[0]), 0] if (iA < np.exp(-1) * iA[0]).any() else np.nan
print(f"RESULT sep {args.sep} nA {args.nA} nB {args.nB} ngap {args.ngap}  A 1/e time {tA:.1f}  B max {h[:, 2].max():.4f} "
      f"at t {h[np.argmax(h[:, 2]), 0]:.1f}  gap mean {h[:, 3].mean():.3e} max {h[:, 3].max():.3e}  absorbed end {h[-1, 6]:.4f}  "
      f"total drift {h[-1, 7] / h[0, 7] - 1:+.1e}")
if args.save:
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.save, hist=h, args=str(vars(args)))
