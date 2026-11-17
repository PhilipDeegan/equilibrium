"""10-16 bend: Huygens eject with per-cell throughput sigma(x) (axiom: relay; rate from local cell conditions). Each cell
fills an accumulator by sigma per tick and ejects (b = V - a on what it holds) when full; until then arriving amplitude
is held. Energy exact (scatter orthogonal, holding identity). Directed pulse along +z; sigma = 1 - g (x + half) / side:
slower cells at +x. Track the centroid x vs z travel."""
import argparse

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=96)
p.add_argument("--g", type=float, default=0.0)
p.add_argument("--ticks", type=int, default=60)
p.add_argument("--every", type=int, default=10)
p.add_argument("--y0", type=float, default=0.0)
p.add_argument("--sphere", action="store_true")
p.add_argument("--wx", type=float, default=4.0)
p.add_argument("--wz", type=float, default=3.0)
args = p.parse_args()
side = args.P / np.sqrt(3)
D, MIR = C.D_HAT, C.K_MIRROR
ac = C.AnchorCluster(cpp_box(args.P, n_threads=8))
pos, nb, N = ac.pos, ac.nb, ac.N
w = np.asarray(C.LAP_W, float)
w = w / w.sum()
wrapx = lambda x: (x + side / 2) % side - side / 2
x = wrapx(pos)
z0 = -side / 4
env = np.exp(-(x[:, 0]**2 + x[:, 1]**2) / (2 * args.wx**2) - (x[:, 2] - z0)**2 / (2 * args.wz**2))
a = np.where(D[None, :, 2] < 0, env[:, None], 0.0)
if args.sphere:
    z0 = 0.0
    a = np.repeat(np.exp(-np.sum(x**2, 1) / (2 * args.wz**2))[:, None], 14, 1)
y = args.y0 + args.g * (x[:, 0] + side / 2) / side
st = np.zeros(N)
held = a.copy()
En = lambda: np.sum(w * held**2) + np.sum(y * st**2)
E0 = En()
print(f"P {args.P} N {N} stub y0 {args.y0} g {args.g}: y at x=0 {args.y0 + args.g / 2:.3f}, dy/dx {args.g / side:.5f}", flush=True)
print("  tick  dE/E0  z_c  x_c  spread_x  E_fwd", flush=True)
for t in range(args.ticks + 1):
    if t % args.every == 0:
        e = np.sum(w * held**2, 1)
        zc = np.sum(e * x[:, 2]) / e.sum()
        fw = x[:, 2] > z0 + 3
        xc = np.sum(e[fw] * x[fw, 0]) / e[fw].sum()
        sx = np.sqrt(np.sum(e * (x[:, 0] - xc)**2) / e.sum())
        if args.sphere:
            r = np.linalg.norm(x, axis=1)
            rd = [np.sum(e[m] * r[m]) / e[m].sum() for m in ((x @ dd) / np.maximum(r, 1e-9) > np.cos(np.radians(12)) for dd in (np.array([1, 0, 0.]), np.ones(3) / np.sqrt(3)))]
            hh, ed = np.histogram(r, bins=np.arange(0, side / 2, 0.5), weights=e / np.maximum(r, 0.5)**0); hh[:4] = 0
            print(f"  {t:4d}  {En() / E0 - 1:+.1e}  r_peak {ed[np.argmax(hh)] + 0.25:6.2f}  E_in(r<5) {e[r < 5].sum() / e.sum():.3f}  E_stub {np.sum(y * st**2) / E0:.3f}  r_mean {np.sum(e * r) / e.sum():6.3f}  r_sq {rd[0]:6.3f}  r_hex {rd[1]:6.3f}", flush=True)
            continue
        print(f"  {t:4d}  {En() / E0 - 1:+.1e}  {zc:+7.3f}  {xc:+7.4f}  {sx:6.3f}  {e[x[:, 2] > z0].sum() / e.sum():.3f}", flush=True)
    if t == args.ticks:
        break
    V = 2 * (np.sum(w * held, 1) + y * st) / (1 + y)
    out = V[:, None] - held
    st = V - st
    held = np.empty_like(held)
    held[nb, MIR[None, :]] = out
