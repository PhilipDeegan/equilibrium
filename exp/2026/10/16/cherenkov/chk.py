"""10-16 cherenkov: hop / triangle DEC Maxwell (E edge integrals on the 7 canonical hops per cell, B on the 12 Delaunay
triangles, stars dec r 2), leapfrog. Medium: eps on every hop (wave speed 1 / sqrt(eps)). Source: magnetisation mu on
triangles (flux of a gaussian M through each triangle), moving along +z at v: E-equation term -*1^-1 C^T mu, divergence-
free exactly (d0^T C^T = 0). --rot: M rotates in the xy plane at omega_r (rotating neutral core)."""
import argparse
import time

import numpy as np
import scipy.sparse as sp

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=96)
p.add_argument("--eps", type=float, default=2.25)
p.add_argument("--v", type=float, default=0.8)
p.add_argument("--sig", type=float, default=2.0)
p.add_argument("--T", type=float, default=50.0)
p.add_argument("--dt", type=float, default=0.25)
p.add_argument("--rot", type=float, default=0.0, help="omega of M rotating in xy (0: M along z, static)")
p.add_argument("--save", default="")
args = p.parse_args()
D, MIR = C.D_HAT, C.K_MIRROR
CAN = [4, 5, 6, 7, 9, 11, 13]
r, s2 = C.dec_star("dec")
ac = 2 / np.sqrt(3)
star1 = np.array([r * ac / 8 if k < 8 else ac / 8 for k in CAN])
star2 = s2 / ac
side = args.P / np.sqrt(3)
ac_ = C.AnchorCluster(cpp_box(args.P, n_threads=8))
pos, nb, N = ac_.pos, ac_.nb, ac_.N
wrapx = lambda x: (x + side / 2) % side - side / 2
rows, cols, vals = [], [], []
tri_area, tri_cen = [], []
for t, (s, k, m) in enumerate(C._DEC_TRIS):
    ids = np.arange(N)
    for (cellv, kk, sg) in (("0", k, 1.0), ("k", m, 1.0), ("0", s, -1.0)):
        base = ids if cellv == "0" else nb[:, k]
        if kk in CAN:
            cell, c, se = base, CAN.index(kk), 1.0
        else:
            cell, c, se = nb[base, kk], CAN.index(MIR[kk]), -1.0
        rows.append(ids * 12 + t); cols.append(cell * 7 + c); vals.append(np.full(N, sg * se))
    tri_area.append(0.5 * np.cross(D[k], D[s]))
    tri_cen.append((D[k] + D[s]) / 3)
Cm = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(12 * N, 7 * N))
CT = Cm.T.tocsr()
S1 = np.tile(star1, N)
tri_area, tri_cen = np.array(tri_area), np.array(tri_cen)
tc = (pos[:, None, :] + tri_cen[None]).reshape(-1, 3)
ta = np.tile(tri_area, (N, 1))
a = np.zeros(7 * N)
b = np.zeros(12 * N)
z0 = -side / 2 + 6


def mu(t):
    zs = z0 + args.v * t
    X = wrapx(tc - np.array([0.13, 0.07, zs]))
    g = np.exp(-np.sum(X**2, 1) / (2 * args.sig**2))
    if args.rot:
        Mv = np.tile([np.cos(args.rot * t), np.sin(args.rot * t), 0.0], (len(X), 1))
    else:
        Mv = np.tile([0, 0, 1.0], (len(X), 1))
    return g * np.sum(Mv * ta, 1), zs


en = lambda: 0.5 * args.eps * np.sum(S1 * a**2) + 0.5 * star2 * np.sum(b**2)
nst = int(round(args.T / args.dt))
print(f"P {args.P} N {N} eps {args.eps} c_w {1 / np.sqrt(args.eps):.4f} v {args.v} rot {args.rot} dt {args.dt} steps {nst}", flush=True)
t0 = time.time()
for st in range(nst):
    t = st * args.dt
    b -= 0.5 * args.dt * (Cm @ a)
    m_, zs = mu(t + 0.5 * args.dt)
    a += args.dt * (CT @ (star2 * b - m_)) / (args.eps * S1)
    b -= 0.5 * args.dt * (Cm @ a)
    if st % max(1, nst // 10) == 0:
        print(f"  t {t:6.2f} zs {zs:+7.2f} E {en():.6e}  ({time.time() - t0:.0f} s)", flush=True)
m_, zs = mu(args.T)
Bt = (b * star2 - 0 * m_).reshape(N, 12)
eb = np.sum(Bt**2, 1)
X = wrapx(pos - np.array([0.13, 0.07, zs]))
rho = np.hypot(X[:, 0], X[:, 1])
dz = X[:, 2]
print(f"final zs {zs:+.2f}; Mach front: for each rho bin, dz (behind source) of max B^2 (dz < -1)")
fr = []
for r0 in np.arange(6, min(22, side / 2 - 2), 2.0):
    m = (np.abs(rho - r0) < 1.0) & (dz < -1) & (dz > -side / 2 + 1)
    hist, ed = np.histogram(-dz[m], bins=np.arange(0, side / 2, 1.0), weights=eb[m])
    cnt, _ = np.histogram(-dz[m], bins=ed)
    hm = hist / np.maximum(cnt, 1)
    fr.append((r0, ed[np.argmax(hm)] + 0.5, hm.max()))
    print(f"  rho {r0:5.1f}: behind {ed[np.argmax(hm)] + 0.5:6.1f}  <B^2> peak {hm.max():.3e}  ahead mean {eb[(np.abs(rho - r0) < 1) & (dz > 1)].mean():.3e}")
fr = np.array(fr)
sl = np.polyfit(fr[:, 0], fr[:, 1], 1)[0]
cw = 1 / np.sqrt(args.eps)
print(f"slope d(behind)/d rho {sl:.3f}; Mach cot mu = sqrt(v^2/c_w^2 - 1) = {np.sqrt(max(args.v**2 / cw**2 - 1, 0)):.3f}")
if args.save:
    np.savez(args.save, b=b, a=a, pos=pos, zs=zs)
