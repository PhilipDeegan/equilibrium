"""10-24 embedded spheromak, MHD reference (lowest rung: imposed eta, no subsystem interaction), same geometry as
dst_embed.py: box side P / sqrt3, body a, ambient frac * amp (0, sin kx, cos kx), orientations aligned / opposed /
tilt45. Scorecard on the deviation d = B - B_amb0: Ebody, share inside r < a about the content centroid, overlap with
the seed body, ambient exchange <B, amb0> / |amb0|^2 - 1, fine share, centroid, EV."""
import argparse

import numpy as np

from equilibrium.reference import spectral as S

p = argparse.ArgumentParser()
p.add_argument("--P", type=float, default=96)
p.add_argument("--n", type=int, default=64)
p.add_argument("--a", type=float, default=6.0)
p.add_argument("--smooth", type=float, default=0.444)
p.add_argument("--amp", type=float, default=0.3)
p.add_argument("--frac", type=float, default=0.0)
p.add_argument("--orient", choices=["aligned", "opposed", "tilt45"], default="aligned")
p.add_argument("--eta", type=float, default=0.002)
p.add_argument("--T", type=float, default=150.0)
p.add_argument("--every", type=float, default=5.0)
args = p.parse_args()
box = S.Box(args.n, args.P / np.sqrt(3))
th = {"aligned": 0.0, "opposed": np.pi, "tilt45": np.pi / 4}[args.orient]
body = S.taylor_spheromak(box, args.a, args.smooth, args.amp, R=S.rot_x(th))
amb = S.beltrami_ambient(box, args.frac * args.amp)
na2 = max(np.sum(amb**2), 1e-300)
m = S.MHD(box, np.zeros_like(body), body + amb, nu=args.eta, eta=args.eta)
kfine = box.k2 > (np.pi / box.dx / 2) ** 2
wgt = np.where(box.k[2] > 0, 2.0, 1.0)
print(f"MHD n {args.n} a {args.a} orient {args.orient} frac {args.frac} eta {args.eta}", flush=True)
hist = []
times = np.arange(0, args.T + 1e-9, args.every)
for tn in times:
    while m.t < tn - 1e-9:
        m.step(min(m.cfl_dt(), tn - m.t))
    B = m.B
    d = B - amb
    ed = 0.5 * np.sum(d**2, 0)
    Ed = ed.sum() * box.dV
    ph = np.exp(2j * np.pi * box.x / box.side)
    cen = np.angle(np.sum(ed * ph, axis=(1, 2, 3))) * box.side / (2 * np.pi)
    xc = (box.x - cen[:, None, None, None] + box.side / 2) % box.side - box.side / 2
    inside = np.sum(ed[np.sqrt(np.sum(xc**2, 0)) < args.a]) * box.dV / Ed
    ov = np.sum(d * body) / (np.linalg.norm(d) * np.linalg.norm(body))
    cx = np.sum(B * amb) / na2 - 1 if args.frac > 0 else 0.0
    dh = np.sum(np.abs(box.fft(d)) ** 2, 0)
    fine = np.sum(wgt * kfine * dh) / np.sum(wgt * dh)
    EV = S.energy(box, m.v)
    hist.append((m.t, Ed, inside, ov, cx, fine, EV, *cen))
    print(f"  t {m.t:6.1f}  Ebody/Ebody0 {Ed / hist[0][1]:.4f}  inside {inside:.4f}  overlap {ov:.4f}  amb_exch {cx:+.4e}"
          f"  fine {fine:.2e}  EV/Ebody0 {EV / hist[0][1]:.3e}  cen {cen[0]:+.2f} {cen[1]:+.2f} {cen[2]:+.2f}", flush=True)
h = hist[-1]
print(f"RESULT mhd {args.orient} frac {args.frac}  Ebody {h[1] / hist[0][1]:.4f}  inside {hist[0][2]:.3f}->{h[2]:.3f}  overlap {h[3]:.4f}"
      f"  amb_exch {h[4]:+.3e}  fine {h[5]:.1e}  cen {h[7]:+.2f} {h[8]:+.2f} {h[9]:+.2f}")
