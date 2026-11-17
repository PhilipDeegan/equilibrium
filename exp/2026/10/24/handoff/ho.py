"""10-24 hand-off of unmatched loop content (numpy reference over the C++ step() + Cartan).
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
p.add_argument("--photon", type=float, default=0.0, help="travelling sector: psi on faces <-> e on loops at gamma = photon * g_c (lattice Maxwell)")
p.add_argument("--sponge", type=float, default=0.0, help="photon-sector absorption rate at the box edge (r > 0.38 side), removed energy tracked")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--out", default="")
args = p.parse_args()
side = args.P / np.sqrt(3)
W = C.W
OFF = np.array([0.13, 0.07, 0.11])
wrapx = lambda x: (x + side / 2) % side - side / 2
a_, lam = args.a, 4.4934094579 / args.a


def taylor(X):
    x = wrapx(X - OFF)
    r = np.linalg.norm(x, axis=1)
    rs = np.maximum(r, 1e-9)
    ct, st = x[:, 2] / rs, np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) / rs
    rho = np.maximum(np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2), 1e-12)
    cp, sp_ = x[:, 0] / rho, x[:, 1] / rho
    k = lam * rs
    j1, j1p = spherical_jn(1, k), spherical_jn(1, k, derivative=True)
    Br, Bt, Bp = 2 * j1 / k * ct, -(j1 / k + j1p) * st, j1 * st
    B = np.stack([Br * st * cp + Bt * ct * cp - Bp * sp_, Br * st * sp_ + Bt * ct * sp_ + Bp * cp, Br * ct - Bt * st], 1)
    return np.where((r < a_)[:, None], B, 0.0)


alat = cpp_box(args.P, n_threads=args.threads)
ac = C.AnchorCluster(alat)
pos, nb, N = ac.pos, ac.nb, ac.N
x = wrapx(pos - OFF)
r = np.linalg.norm(x, axis=1)

D1, D1T = C._dec_ops(ac)
tri_of = {}
t0 = 0
for s, k, m in C._DEC_TRIS:
    i = np.nonzero((nb[:, k] >= 0) & (nb[:, s] >= 0))[0]
    j = nb[i, k]
    i = i[nb[j, m] == nb[i, s]]
    arr = np.full(N, -1)
    arr[i] = t0 + np.arange(len(i))
    tri_of[(s, k, m)] = (arr, i)
    t0 += len(i)
NT = t0
r_tri = np.zeros(NT)
for key, (arr, i) in tri_of.items():
    r_tri[arr[i]] = r[i]
rows, cols = [], []
for key, (arr, i) in tri_of.items():
    t = arr[i]
    rows.append(t), cols.append(t)
    for q in range(14):
        n = nb[i, q]
        tn = np.where(n >= 0, arr[np.where(n >= 0, n, 0)], -1)
        ok = tn >= 0
        rows.append(t[ok]), cols.append(tn[ok])
rows, cols = np.concatenate(rows), np.concatenate(cols)
cnt = np.bincount(rows, minlength=NT)
M = sp.csr_matrix((1.0 / cnt[rows], (rows, cols)), shape=(NT, NT))
IM = sp.identity(NT, format="csr") - M
IMT = IM.T.tocsr()
del rows, cols


def fold(y):
    return C._antisym(ac, y.reshape(N, 14))


def L(phi):
    w = D1 @ np.where(nb >= 0, phi, 0.0).ravel()
    for _ in range(args.order):
        w = IM @ w
    return w


def LT(e):
    for _ in range(args.order):
        e = IMT @ e
    return fold(D1T @ e)


def L0(phi):
    return D1 @ np.where(nb >= 0, phi, 0.0).ravel()


def L0T(e):
    return fold(D1T @ e)


# calibration: smooth transverse wave A = y sin(k x), Rayleigh quotient of L0^T L0 against c^2 k^2 (c^2 = C0SQ)
kc = 2 * np.pi / side * 2
ph_w = C.sample_A(ac, lambda X: np.stack([0 * X[:, 0], np.sin(kc * X[:, 0]), 0 * X[:, 0]], 1))
rq = np.sum(ph_w * L0T(L0(ph_w))) / np.sum(ph_w * ph_w)
g_c = np.sqrt(C.C0SQ * kc**2 / rq)
g = args.handoff * g_c
print(f"P {args.P} a {a_} N {N} triangles {NT}  g_c {g_c:.4f}  g {g:.4f}  hold {args.hold}  order {args.order}", flush=True)

phi = C.sample_A(ac, taylor)
eps = 0.02
for _ in range(int(round(args.smooth / eps))):
    phi = phi - eps * ac.curl_curl_dec(phi)
phi = C.project_div_free(ac, phi)
phi *= args.vmax / np.abs(C.reconstruct_A(ac, phi)).max()
ac.u, ac.phi = np.zeros(N), phi
if args.hold != 1.0:
    ac.set_cartan_hold(args.hold)
phi0 = phi.copy()
n0 = np.linalg.norm(phi0)
e = np.zeros(NT)
psi = np.zeros((N, 14))
E_abs = 0.0
gam = args.photon * g_c
r_sp = 0.38 * side
sig_c = args.sponge * np.clip((r - r_sp) / (0.5 * side - r_sp), 0, 1) ** 2
sig_t = args.sponge * np.clip((r_tri - r_sp) / (0.5 * side - r_sp), 0, 1) ** 2
SH = np.array([1.0, 1.5]) * a_
dt = 0.4 / float(np.sum(W**2))
nst, ev = int(round(args.T / dt)), max(1, int(round(args.every / dt)))


def fine_share(ph):
    w = L0(ph)
    return np.sum(L(ph) ** 2) / max(np.sum(w**2), 1e-300)


def score(s, ph):
    u = ac.u
    ec = 0.25 * np.sum(ph**2, 1) + 0.5 * u**2
    Ephi = ec.sum()
    Ee = 0.5 * np.sum(e**2) + 0.25 * np.sum(psi**2)
    sh = np.bincount(np.searchsorted(SH, r), weights=ec, minlength=3) / Ephi
    A = C.reconstruct_A(ac, ph)
    B = C.curl_A(ac, ph)
    m = r < a_
    cos = np.sum(A[m] * B[m]) / np.sqrt(np.sum(A[m] ** 2) * np.sum(B[m] ** 2))
    she = (np.bincount(np.searchsorted(SH, r_tri), weights=0.5 * e**2, minlength=3)
           + np.bincount(np.searchsorted(SH, r), weights=0.25 * np.sum(psi**2, 1), minlength=3)) / max(Ee, 1e-300)
    return dict(t=s * dt, Ephi=Ephi, Ee=Ee, Eabs=E_abs, sh=sh, she=she, d=np.linalg.norm(ph - phi0) / n0, cos=cos, fine=fine_share(ph))


hist = []
tw = time.time()
for s in range(nst + 1):
    if s % ev == 0:
        ph = ac.phi
        row = score(s, ph)
        hist.append(row)
        E0 = hist[0]["Ephi"] + hist[0]["Ee"]
        el = time.time() - tw
        print(f"  t {row['t']:6.1f}  (E+Ee+abs)/E0 {(row['Ephi'] + row['Ee'] + row['Eabs']) / E0:.6f}  abs/E0 {row['Eabs'] / E0:.4f}  Ephi/E0 {row['Ephi'] / E0:.4f}  Ee/E0 {row['Ee'] / E0:.4f}"
              f"  |dphi| {row['d']:.4f}  cos {row['cos']:.4f}  fine {row['fine']:.4f}"
              f"  in/shell/out {row['sh'][0]:.4f} {row['sh'][1]:.4f} {row['sh'][2]:.4f}  e in/shell/out {row['she'][0]:.3f} {row['she'][1]:.3f} {row['she'][2]:.3f}"
              f"  (ETA {el / max(s, 1) * (nst - s) / 60:.1f} min)", flush=True)
    if s == nst:
        break
    alat.step_cartan(0, dt, 1.0, 2)
    if g > 0:
        ph = ac.phi
        e += dt * g * L(ph)
        if gam > 0:
            e += dt * gam * L0(psi)
            psi -= dt * gam * L0T(e)
            if args.sponge > 0:
                before = 0.5 * np.sum(e**2) + 0.25 * np.sum(psi**2)
                e *= np.exp(-sig_t * dt)
                psi *= np.exp(-sig_c * dt)[:, None]
                E_abs += before - (0.5 * np.sum(e**2) + 0.25 * np.sum(psi**2))
        ac.phi = ph - dt * g * LT(e)
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, phi=ac.phi, e=e, hist=np.array([[h["t"], h["Ephi"], h["Ee"], h["d"], h["cos"], h["fine"], *h["sh"]] for h in hist]))
h0, h1 = hist[0], hist[-1]
print(f"RESULT handoff {args.handoff} order {args.order} photon {args.photon} sponge {args.sponge}  abs {h1['Eabs'] / h0['Ephi']:.4f}  Ephi {h1['Ephi'] / h0['Ephi']:.4f}  Ee {h1['Ee'] / h0['Ephi']:.4f}  |dphi| {h1['d']:.4f}"
      f"  cos {h0['cos']:.4f}->{h1['cos']:.4f}  in {h0['sh'][0]:.4f}->{h1['sh'][0]:.4f}  fine {h0['fine']:.4f}->{h1['fine']:.4f}  wall {time.time() - tw:.0f} s")
