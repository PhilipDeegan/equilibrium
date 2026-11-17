"""10-16 Z_N clock: integer count n mod N per cell, integer leapfrog n+ = 2n - n- + round(I / q), I = 2 sum_hex o + sum_sq o,
o = wrapped count difference across a face. Exactly reversible (n- = 2n - n+ + Q(n)). Smooth limit: wave equation c^2 = 4 / q.
Valve test: box winding k along x (uniform current) + vortex ring radius R in the yz plane; slip = box winding change."""
import argparse
import time

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box, solid_angle

p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=72)
p.add_argument("--Nc", type=int, default=1 << 20)
p.add_argument("--q", type=int, default=0, help="0: pick from CFL (0.5 of the limit)")
p.add_argument("--k", type=int, default=0, help="box winding along x")
p.add_argument("--R", type=float, default=0.0, help="vortex ring radius (0: none)")
p.add_argument("--sgn", type=int, default=1, help="ring orientation vs the current")
p.add_argument("--ticks", type=int, default=2000)
p.add_argument("--every", type=int, default=100)
p.add_argument("--wave", type=float, default=0.0, help=">0: plane wave along z, amplitude in counts (dispersion check)")
p.add_argument("--kw", type=int, default=1, help="plane-wave mode number")
p.add_argument("--threads", type=int, default=8)
args = p.parse_args()
NC = args.Nc
side = args.P / np.sqrt(3)
D = C.D_HAT
ac = C.AnchorCluster(cpp_box(args.P, n_threads=args.threads))
pos, nb, N = ac.pos, ac.nb, ac.N
WI = np.array([2] * 8 + [1] * 6, np.int64)
wrapc = lambda d: (d + NC // 2) % NC - NC // 2
wrapx = lambda x: (x + side / 2) % side - side / 2
x = wrapx(pos)


def imb(n):
    return np.sum(WI[None, :] * wrapc(n[nb] - n[:, None]), 1)


v = np.random.default_rng(1).standard_normal(N)
for _ in range(60):
    v = -C.laplacian(ac, v) if False else -(np.sum(WI[None, :] * (v[nb] - v[:, None]), 1))
    lam = np.linalg.norm(v); v /= lam
q = args.q or int(np.ceil(2 * lam / 4 / 0.5))
print(f"P {args.P} N {N} side {side:.2f} Nc {NC} lam_max {lam:.2f} q {q} (CFL q >= {lam / 4:.2f}) c {np.sqrt(4 / q):.4f} cells/tick", flush=True)

tri = [(ka, kb, ks) for ka in range(8) for kb in range(8) for ks in range(8, 14) if np.allclose(D[kb] - D[ka], D[ks])]


def chain(start, f):
    ch = [start]
    for _ in range(args.P // 2 - 1):
        ch.append(nb[ch[-1], f])
    assert nb[ch[-1], f] == ch[0]
    return np.array(ch)


starts = [int(np.argmin(np.sum((x - [0, y, z])**2, 1))) for y in (-side / 4, 0, side / 4) for z in (-side / 4, 0, side / 4)]
XCH = [chain(s, 9) for s in starts]
bw = lambda n: [int(np.sum(wrapc(n[nb[c, 9]] - n[c]))) / NC for c in XCH]

th = 2 * np.pi * args.k * x[:, 0] / side
if args.R > 0:
    s_ = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    ring = np.stack([0 * s_, args.R * np.cos(s_), args.R * np.sin(s_)], 1) + np.array([0.13, 0.07, 0.11])
    shifts = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float) * side
    phase = lambda X: 0.5 * solid_angle(ring, X, shifts=shifts, n_threads=args.threads)
    rng = np.random.default_rng(0)
    jumps = []
    for ax in range(3):
        smp = (rng.random((400, 3)) - 0.5) * side
        smp[:, ax] = -side / 2
        hi = smp.copy()
        hi[:, ax] += side
        jumps.append(np.angle(np.mean(np.exp(1j * (phase(hi) - phase(smp))))))
    th = th + args.sgn * (phase(pos) - sum(jumps[a] * pos[:, a] / side for a in range(3)))
if args.wave > 0:
    th = th + 2 * np.pi * args.wave / NC * np.cos(2 * np.pi * args.kw * x[:, 2] / side)
n = np.rint(th * NC / (2 * np.pi)).astype(np.int64) % NC
nm = n.copy()


def defects(n):
    G = np.stack([wrapc(n[nb[:, ka]] - n) + wrapc(n[nb[nb[:, ka], ks]] - n[nb[:, ka]]) + wrapc(n - n[nb[:, kb]]) for ka, kb, ks in tri], 1)
    w = G // NC if False else np.rint(G / NC).astype(int)
    m = w != 0
    cen = np.stack([x + (D[ka] + D[kb]) / 3 for ka, kb, _ in tri], 1)[m]
    cen = wrapx(cen)
    rho = np.hypot(cen[:, 1], cen[:, 2])
    return int(m.sum()), rho.mean() if m.any() else 0.0, np.angle(np.mean(np.exp(2j * np.pi * cen[:, 0] / side))) * side / (2 * np.pi) if m.any() else 0.0


def energy(n, nm):
    r = wrapc(n - nm).astype(float)
    o = wrapc(n[nb] - n[:, None]).astype(float)
    om = wrapc(nm[nb] - nm[:, None]).astype(float)
    return 0.5 * np.sum(r**2) + 0.25 / q * np.sum(WI[None, :] * o * om)


zi = int(np.argmin(np.sum((x - [0.0, 0.0, 0.0])**2, 1)))
print(f"  {'tick':>6s} box_winding(mean,min,max) defects rho_ring x_ring E n[centre]", flush=True)
t0 = time.time()
for s in range(args.ticks + 1):
    if s % args.every == 0:
        b = bw(n)
        nd, rho, xr = defects(n) if args.R > 0 or args.k else (0, 0.0, 0.0)
        el = time.time() - t0
        print(f"  {s:6d} {np.mean(b):+.4f} {min(b):+.0f} {max(b):+.0f} {nd:6d} {rho:7.3f} {xr:+7.3f} {energy(n, nm):.6e} {wrapc(n[zi])}"
              f"  ({el / max(s, 1) * 1e3:.0f} ms/tick, ETA {el / max(s, 1) * (args.ticks - s):.0f} s)", flush=True)
    if s == args.ticks:
        break
    Q = np.rint(imb(n) / q).astype(np.int64)
    n, nm = (2 * n - nm + Q) % NC, n
