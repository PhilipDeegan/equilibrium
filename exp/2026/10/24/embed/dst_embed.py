"""10-24 spheromak embedded in a box-scale ambient disturbance, DST step() + Cartan (Philip: every neutron sits inside
larger-scale disturbances). Ambient phi = B0 (0, sin kx, cos kx) profile (k = 2 pi / side; steady under Cartan,
uniform magnitude, along +z at the body), body = Taylor seed as 10-22 (smoothed, div-free) with orientation
aligned (body axis +z) / opposed (-z) / tilt45 (axis in y-z at 45 deg). B0 = frac * body peak.
Native scorecard on the deviation d = phi - phi_amb0: body content |d|^2 inside r < a (moving centre = content
centroid), overlap <d, body0>, ambient exchange c = <phi, amb0> / |amb0|^2 - 1, loop content |D1 d|^2 vs initial,
centroid drift."""
import argparse
import time
from pathlib import Path

import numpy as np
from scipy.special import spherical_jn

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

C.spacing_of = lambda lat: 1.0
p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=96)
p.add_argument("--a", type=float, default=6.0)
p.add_argument("--vmax", type=float, default=0.3)
p.add_argument("--smooth", type=float, default=0.444)
p.add_argument("--frac", type=float, default=0.0)
p.add_argument("--orient", choices=["aligned", "opposed", "tilt45"], default="aligned")
p.add_argument("--T", type=float, default=150.0)
p.add_argument("--every", type=float, default=5.0)
p.add_argument("--threads", type=int, default=2)
p.add_argument("--save", default="")
args = p.parse_args()
side = args.P / np.sqrt(3)
W = C.W
OFF = np.array([0.13, 0.07, 0.11])
wrapx = lambda x: (x + side / 2) % side - side / 2
a_, lam = args.a, 4.4934094579 / args.a
th = {"aligned": 0.0, "opposed": np.pi, "tilt45": np.pi / 4}[args.orient]
c_, s_ = np.cos(th), np.sin(th)
R = np.array([[1.0, 0, 0], [0, c_, -s_], [0, s_, c_]])


def taylor(X):
    xb = wrapx(X - OFF)
    r = np.linalg.norm(xb, axis=1)
    x = xb @ R
    rs = np.maximum(r, 1e-9)
    ct, st = x[:, 2] / rs, np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) / rs
    rho = np.maximum(np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2), 1e-12)
    cp, sp = x[:, 0] / rho, x[:, 1] / rho
    k = lam * rs
    j1, j1p = spherical_jn(1, k), spherical_jn(1, k, derivative=True)
    Br, Bt, Bp = 2 * j1 / k * ct, -(j1 / k + j1p) * st, j1 * st
    B = np.stack([Br * st * cp + Bt * ct * cp - Bp * sp, Br * st * sp + Bt * ct * sp + Bp * cp, Br * ct - Bt * st], 1)
    return np.where((r < a_)[:, None], B @ R.T, 0.0)


alat = cpp_box(args.P, n_threads=args.threads)
ac = C.AnchorCluster(alat)
pos, nb, N = ac.pos, ac.nb, ac.N
x = wrapx(pos - OFF)
body = C.sample_A(ac, taylor)
for _ in range(int(round(args.smooth / 0.02))):
    body = body - 0.02 * ac.curl_curl_dec(body)
body = C.project_div_free(ac, body)
body *= args.vmax / np.abs(C.reconstruct_A(ac, body)).max()
kk = 2 * np.pi / side
amb = C.sample_A(ac, lambda X: np.stack([0 * X[:, 0], np.sin(kk * X[:, 0]), np.cos(kk * X[:, 0])], 1))
amb = C.project_div_free(ac, amb)
amb *= args.frac * args.vmax / max(np.abs(C.reconstruct_A(ac, amb)).max(), 1e-300) if args.frac > 0 else 0.0
ac.u, ac.phi = np.zeros(N), body + amb
D1, _ = C._dec_ops(ac)
nb0 = np.linalg.norm(body)
na2 = max(np.sum(amb * amb), 1e-300)
w0 = np.sum((D1 @ body.ravel()) ** 2)
dt = 0.4 / float(np.sum(W**2))
nst, ev = int(round(args.T / dt)), max(1, int(round(args.every / dt)))
print(f"P {args.P} a {a_} N {N} orient {args.orient} frac {args.frac}  ticks {nst}", flush=True)
tw, hist = time.time(), []
for s in range(nst + 1):
    if s % ev == 0:
        ph = ac.phi
        d = ph - amb
        ec = 0.25 * np.sum(d**2, 1)
        Ed = ec.sum()
        cen = np.angle(np.sum(ec[:, None] * np.exp(2j * np.pi * x / side), 0)) * side / (2 * np.pi)
        rb = np.linalg.norm(wrapx(x - cen), axis=1)
        inside = ec[rb < a_].sum() / Ed
        ov = np.sum(d * body) / (np.linalg.norm(d) * nb0)
        cx = np.sum(ph * amb) / na2 - 1 if args.frac > 0 else 0.0
        wl = np.sum((D1 @ d.ravel()) ** 2) / w0
        Et = 0.25 * np.sum(ph**2)
        hist.append((s * dt, Et, Ed, inside, ov, cx, wl, *cen))
        print(f"  t {s * dt:6.1f}  Etot {Et:.6e}  Ebody/Ebody0 {Ed / hist[0][2]:.4f}  inside {inside:.4f}  overlap {ov:.4f}"
              f"  amb_exch {cx:+.4e}  loops/loops0 {wl:.3f}  cen {cen[0]:+.2f} {cen[1]:+.2f} {cen[2]:+.2f}"
              f"  (ETA {(time.time() - tw) / max(s, 1) * (nst - s) / 60:.1f} min)", flush=True)
    if s == nst:
        break
    alat.step_cartan(0, dt, 1.0, 2)
h = np.array(hist)
print(f"RESULT {args.orient} frac {args.frac}  Etot drift {h[-1, 1] / h[0, 1] - 1:+.1e}  Ebody {h[-1, 2] / h[0, 2]:.4f}  inside {h[0, 3]:.3f}->{h[-1, 3]:.3f}"
      f"  overlap {h[-1, 4]:.4f}  amb_exch {h[-1, 5]:+.3e}  loops {h[-1, 6]:.2f}  cen {h[-1, 7]:+.2f} {h[-1, 8]:+.2f} {h[-1, 9]:+.2f}")
if args.save:
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.save, phi=ac.phi, amb=amb, body=body, hist=h)
