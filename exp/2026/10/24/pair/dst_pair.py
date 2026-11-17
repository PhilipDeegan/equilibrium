"""10-24 neutron / antineutron analogue in DST: spheromak + mirror (toroidal winding flipped) on the z axis, step() + Cartan,
no hand-off. Native scorecard: face content per half, centroid gap dz, contact-slab content |z| < 1.5, |dphi| / |phi0|,
unmatched loop share (I - M) w. Compare exp/2026/10/24/ref pair runs (MHD). (Header of the v1 hand-off follows.)
10-24 hand-off of unmatched loop content (numpy reference over the C++ step() + Cartan).
Loops = Delaunay triangles (12 templates per cell), loop sum w = D1 phi (edge integrals). Context <w> = same template
averaged over the cell + its 14 face neighbours (M). Unmatched w_f = (I - M) w. A loop-carried partner e takes and
gives it back:  e_t = g (I - M) D1 phi,   phi_t = -g fold(D1^T (I - M)^T e)
so 1/4 sum_copies phi^2 + 1/2 sum e^2 is exact in continuous time (symplectic Euler: e from old phi, then phi);
gradients have w = 0 and are never touched; matched loop content has no hand-off and is moved by Cartan only.
g calibrated so that with M = 0 a smooth transverse wave sees c^2 k^2 (lattice Maxwell limit).
--hold 0 also makes Cartan transport with <w> only (contextual hold, kappa 0).
Scorecard as 10-22 sph.py (face content in / shell / out, |dphi| / |phi0|, cos) + e content share."""
import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.special import spherical_jn

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

C.spacing_of = lambda lat: 1.0
p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=64)
p.add_argument("--a", type=float, default=6.0)
p.add_argument("--vmax", type=float, default=0.3)
p.add_argument("--T", type=float, default=150.0)
p.add_argument("--every", type=float, default=5.0)
p.add_argument("--smooth", type=float, default=0.444, help="(a / 9)^2 as in 10-22")
p.add_argument("--handoff", type=float, default=1.0, help="g / g_c (0 = baseline step() + Cartan)")
p.add_argument("--hold", type=float, default=1.0, help="Cartan hold kappa (1 = plain Cartan)")
p.add_argument("--order", type=int, default=1, help="high-pass power p: (I - M)^p")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--out", default="")
p.add_argument("--pair", choices=["co", "counter"], required=True)
p.add_argument("--sep", type=float, default=0.0)
args = p.parse_args()
side = args.P / np.sqrt(3)
W = C.W
OFF = np.array([0.13, 0.07, 0.11])
wrapx = lambda x: (x + side / 2) % side - side / 2
a_, lam = args.a, 4.4934094579 / args.a


def taylor(X, zc=0.0, hel=1):
    x = wrapx(X - OFF - np.array([0, 0, zc]))
    r = np.linalg.norm(x, axis=1)
    rs = np.maximum(r, 1e-9)
    ct, st = x[:, 2] / rs, np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) / rs
    rho = np.maximum(np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2), 1e-12)
    cp, sp_ = x[:, 0] / rho, x[:, 1] / rho
    k = lam * rs
    j1, j1p = spherical_jn(1, k), spherical_jn(1, k, derivative=True)
    Br, Bt, Bp = 2 * j1 / k * ct, -(j1 / k + j1p) * st, hel * j1 * st
    B = np.stack([Br * st * cp + Bt * ct * cp - Bp * sp_, Br * st * sp_ + Bt * ct * sp_ + Bp * cp, Br * ct - Bt * st], 1)
    return np.where((r < a_)[:, None], B, 0.0)


alat = cpp_box(args.P, n_threads=args.threads)
ac = C.AnchorCluster(alat)
pos, nb, N = ac.pos, ac.nb, ac.N
x = wrapx(pos - OFF)
sep = args.sep or 2 * a_
hel2 = 1 if args.pair == "co" else -1
f = lambda X: taylor(X, sep / 2, 1) + taylor(X, -sep / 2, hel2)
phi = C.sample_A(ac, f)
for _ in range(int(round(args.smooth / 0.02))):
    phi = phi - 0.02 * ac.curl_curl_dec(phi)
phi = C.project_div_free(ac, phi)
phi *= args.vmax / np.abs(C.reconstruct_A(ac, phi)).max()
ac.u, ac.phi = np.zeros(N), phi
phi0 = phi.copy()
n0 = np.linalg.norm(phi0)
up = x[:, 2] > 0
slab = np.abs(x[:, 2]) < 1.5
D1, D1T = C._dec_ops(ac)
dt = 0.4 / float(np.sum(W**2))
nst, ev = int(round(args.T / dt)), max(1, int(round(args.every / dt)))
print(f"P {args.P} a {a_} N {N} pair {args.pair} sep {sep}  ticks {nst}", flush=True)
tw = time.time()
for s_ in range(nst + 1):
    if s_ % ev == 0:
        ph = ac.phi
        ec = 0.25 * np.sum(ph**2, 1) + 0.5 * ac.u**2
        E = ec.sum()
        zu = np.sum(x[up, 2] * ec[up]) / ec[up].sum()
        zl = np.sum(x[~up, 2] * ec[~up]) / ec[~up].sum()
        w = D1 @ ph.ravel()
        sm = w.copy()
        print(f"  t {s_ * dt:6.1f}  E {E:.6e}  upper {ec[up].sum() / E:.4f}  dz {zu - zl:.3f}  slab {ec[slab].sum() / E:.4f}"
              f"  |dphi| {np.linalg.norm(ph - phi0) / n0:.4f}  |w|^2 {np.sum(w**2):.4e}"
              f"  (ETA {(time.time() - tw) / max(s_, 1) * (nst - s_) / 60:.1f} min)", flush=True)
    if s_ == nst:
        break
    alat.step_cartan(0, dt, 1.0, 2)
