"""10-18 glider search: hop / triangle DEC with a saturating per-cell energy, source-free seeds, look for what persists.
H = sum_cells f(uE) + f(uB), f(u) = beta^2 (sqrt(1 + 2 u / beta^2) - 1) (beta 0 = linear); uE, uB = cell shares of
hop energy d^2 / 2 S1 (1/2 to each end) and triangle energy star2 b^2 / 2 (1/3 to each vertex). Separable -> Stormer-
Verlet symplectic (energy bounded, no drift). Seeds: b = curl alpha, d = dual curl of C -> div d = div b = 0 exactly."""
import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=60)
p.add_argument("--fam", default="pack", choices=["pack", "circ", "mag", "ring", "hopf", "rand"])
p.add_argument("--sig", type=float, default=3.0)
p.add_argument("--amp", type=float, default=1.0, help="peak cell energy / beta^2")
p.add_argument("--beta", type=float, default=1.0, help="0 = linear (amp then sets peak cell energy directly)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--T", type=float, default=60.0)
p.add_argument("--dt", type=float, default=0.25)
p.add_argument("--every", type=float, default=5.0)
p.add_argument("--save", default="")
p.add_argument("--load", default="")
p.add_argument("--revcheck", action="store_true")
p.add_argument("--q", type=float, default=0.0, help="count quantum (0 = float); integer d, b with rounded kicks")
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
S1 = np.tile(star1, N)
DC = D[CAN]
mid = (pos[:, None, :] + 0.5 * DC[None]).reshape(-1, 3)
dirs = np.tile(DC, (N, 1))
TRI = [(D[k] + D[s]) / 3 for (s, k, m) in C._DEC_TRIS]
tcen = (pos[:, None, :] + np.array(TRI)[None]).reshape(-1, 3)
TV = np.array([0.5 * np.cross(D[k], D[m]) for (s, k, m) in C._DEC_TRIS])
TVt = np.tile(TV, (N, 1))

hc = np.repeat(ids, 7)
hj = np.tile(np.arange(7), N)
hn = nb[hc, np.array(CAN)[hj]]
IE = sp.csr_matrix((np.full(14 * N, 0.5), (np.concatenate([hc, hn]), np.tile(np.arange(7 * N), 2))), shape=(N, 7 * N))
tc = np.repeat(ids, 12)
tk = np.tile([k for (s, k, m) in C._DEC_TRIS], N)
ts = np.tile([s for (s, k, m) in C._DEC_TRIS], N)
IB = sp.csr_matrix((np.full(36 * N, 1 / 3), (np.concatenate([tc, nb[tc, tk], nb[tc, ts]]), np.tile(np.arange(12 * N), 3))),
                   shape=(N, 12 * N))
IET, IBT = IE.T.tocsr(), IB.T.tocsr()
beta2 = args.beta**2

rng = np.random.default_rng(args.seed)
X0 = np.array([0.13, 0.07, 0.11])


def field(x, fam):
    X = wrap(x - X0)
    sg = args.sig
    g = np.exp(-np.sum(X**2, 1) / (2 * sg**2))[:, None]
    k = 2.0 / sg
    ez = np.array([0, 0, 1.0])
    if fam in ("pack", "circ"):
        A = np.zeros_like(X)
        A[:, 0] = np.cos(k * X[:, 2])
        if fam == "circ":
            A[:, 1] = np.sin(k * X[:, 2])
        A = g * A
        return A, -np.cross(ez, A)
    if fam == "mag":
        return g * np.array([1.0, 0, 0]), 0 * X
    rho = np.hypot(X[:, 0], X[:, 1]) + 1e-12
    ph = np.stack([-X[:, 1] / rho, X[:, 0] / rho, 0 * rho], 1)
    gr = np.exp(-((rho - sg)**2 + X[:, 2]**2) / (2 * (0.6 * sg)**2))[:, None]
    if fam == "ring":
        return gr * ph, 0 * X
    if fam == "hopf":
        return gr * (ph + ez), 0 * X
    out = []
    for _ in range(2):
        F = np.zeros_like(X)
        for _ in range(6):
            kv = rng.normal(size=3)
            kv *= rng.uniform(0.5, 2.0) / sg / np.linalg.norm(kv)
            F += np.outer(np.cos(X @ kv + rng.uniform(0, 2 * np.pi)), rng.normal(size=3))
        out.append(g * F)
    return out[0], out[1]


A, _ = field(mid, args.fam)
_, Cd = field(tcen, args.fam)
al = np.sum(A * dirs, 1)
wt = star2 * np.sum(Cd * TVt, 1)
b = Cm @ al
d = CT @ wt
uE = lambda d: IE @ (0.5 * d**2 / S1)
uB = lambda b: IB @ (0.5 * star2 * b**2)
u0 = uE(d) + uB(b)
sc = np.sqrt(args.amp * (beta2 if beta2 > 0 else 1.0) / u0.max())
d *= sc
b *= sc
if args.q > 0:
    d = args.q * (CT @ np.rint(sc * wt / args.q))
    b = args.q * (Cm @ np.rint(sc * al / args.q))
if args.load:
    z = np.load(args.load)
    d, b = z["d"], z["b"]
fp = (lambda u: 1 / np.sqrt(1 + 2 * u / beta2)) if beta2 > 0 else (lambda u: np.ones_like(u))
f = (lambda u: beta2 * (np.sqrt(1 + 2 * u / beta2) - 1)) if beta2 > 0 else (lambda u: u)
gE = lambda d: (IET @ fp(uE(d))) * d / S1
gB = lambda b: (IBT @ fp(uB(b))) * star2 * b
H = lambda d, b: f(uE(d)).sum() + f(uB(b)).sum()

PE, PB = np.linalg.pinv(DC), np.linalg.pinv(TV)


def mom(d, b):
    E = (d / S1).reshape(N, 7) @ PE.T
    B = b.reshape(N, 12) @ PB.T
    return np.cross(E, B)


R = 2 * args.sig
lag = max(1, int(round(R / args.every)))
nst = int(round(args.T / args.dt))
ev = int(round(args.every / args.dt))
H0 = H(d, b)
d0, b0 = d.copy(), b.copy()
print(f"P {args.P} N {N} fam {args.fam} sig {args.sig} amp {args.amp} beta {args.beta} seed {args.seed} H0 {H0:.5e}", flush=True)
hist, frames = [], []
t0 = time.time()
for st in range(nst + 1):
    if st % ev == 0:
        uc = f(uE(d)) + f(uB(b))
        Ht = uc.sum()
        c0 = pos[np.argmax(uc)]
        w = np.sum(wrap(pos - c0)**2, 1) < R**2
        Xw = wrap(pos[w] - c0)
        cen = c0 + (uc[w, None] * Xw).sum(0) / uc[w].sum()
        frames.append(uc)
        reuse = np.nan
        if len(frames) > lag:
            prev = frames[-1 - lag]
            reuse = np.minimum(uc[w], prev[w]).sum() / uc[w].sum()
        out = ~w
        Eout = uc[out].sum()
        nz = int(np.count_nonzero(d[out[hc]])) + int(np.count_nonzero(b[out[tc]]))
        g = mom(d, b)
        Pt, Pw = g.sum(0), g[w].sum(0)
        hist.append((st * args.dt, Ht, uc[w].sum() / Ht, uc.max(), *cen, reuse))
        print(f"  t {st * args.dt:5.1f}  H/H0-1 {Ht / H0 - 1:+.2e}  Ewin/H {uc[w].sum() / Ht:.3f}  umax {uc.max():.3e}  "
              f"cen {cen[0]:+6.2f} {cen[1]:+6.2f} {cen[2]:+6.2f}  reuse {reuse:.2f}  Pz tot {Pt[2]:+.3e} win {Pw[2]:+.3e}  "
              f"Eout {Eout:.4e} nz_out {nz}  "
              f"({time.time() - t0:.0f} s)", flush=True)
        if len(frames) > lag + 1:
            frames.pop(0)
    if st == nst:
        break
    if args.q > 0:
        b -= args.q * (Cm @ np.rint(0.5 * args.dt * gE(d) / args.q))
        d += args.q * (CT @ np.rint(args.dt * gB(b) / args.q))
        b -= args.q * (Cm @ np.rint(0.5 * args.dt * gE(d) / args.q))
        continue
    b -= 0.5 * args.dt * (Cm @ gE(d))
    d += args.dt * (CT @ gB(b))
    b -= 0.5 * args.dt * (Cm @ gE(d))
if args.revcheck and args.q > 0:
    d1, b1 = d.copy(), b.copy()
    for st in range(nst):
        b += args.q * (Cm @ np.rint(0.5 * args.dt * gE(d) / args.q))
        d -= args.q * (CT @ np.rint(args.dt * gB(b) / args.q))
        b += args.q * (Cm @ np.rint(0.5 * args.dt * gE(d) / args.q))
    print(f"REVCHECK back to t 0: max|d - d0| {np.abs(d - d0).max():.3e}  max|b - b0| {np.abs(b - b0).max():.3e}")
    d, b = d1, b1
h = np.array(hist)
if len(h) > 2:
    late = h[h[:, 0] >= args.T / 2]
    cz = np.unwrap(late[:, 6] * 2 * np.pi / side) * side / (2 * np.pi)
    cxy = [np.unwrap(late[:, j] * 2 * np.pi / side) * side / (2 * np.pi) for j in (4, 5)]
    vel = np.array([np.polyfit(late[:, 0], q, 1)[0] for q in (*cxy, cz)])
    print(f"RESULT fam {args.fam} sig {args.sig} amp {args.amp} beta {args.beta} seed {args.seed}  Ewin/H end {h[-1, 2]:.3f} "
          f"min {h[:, 2].min():.3f}  |v| late {np.linalg.norm(vel):.3f}  reuse late {np.nanmean(late[:, 7]):.2f}  "
          f"umax end/0 {h[-1, 3] / h[0, 3]:.3f}  dH/H {np.abs(h[:, 1] / H0 - 1).max():.1e}")
if args.save:
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.save, d=d, b=b, pos=pos, hist=h, args=str(vars(args)))
if args.load:
    uc = f(uE(d)) + f(uB(b))
    c0 = pos[np.argmax(uc)]
    rr = np.sqrt(np.sum(wrap(pos - c0)**2, 1))
    o = np.argsort(rr)
    cum = np.cumsum(uc[o]) / uc.sum()
    print("energy within radius r of the hot spot: " + "  ".join(f"r{q} {cum[np.searchsorted(rr[o], q)]:.3f}" for q in (0.5, 1, 1.5, 2, 3, 4, 6)))
    print(f"cells: hot {uc.max():.3e}  fpE {fp(uE(d))[np.argmax(uc)]:.3f}  fpB {fp(uB(b))[np.argmax(uc)]:.3f}  beta^2 {beta2}")
    win = rr < 2.0
    hw = win[hc]
    tw = win[tc]
    a = np.where(hw, d / S1, 0)
    bb = np.where(tw, b, 0)
    ca = Cm @ a
    hb = CT @ (star2 * bb)
    print(f"Rayleigh omega^2 (linear operator, r < 2): E {np.dot(ca, star2 * ca) / np.dot(a, S1 * a):.2f}  "
          f"B {np.dot(hb, hb / S1) / np.dot(bb, star2 * bb):.2f}  (gapped modes 27)")
    print(f"E / B energy split in r < 2: {uE(d)[win].sum() / (uE(d)[win].sum() + uB(b)[win].sum()):.3f}")
