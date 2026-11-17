"""10-16 eject: Philip -- cells prefer no local perturbation, accept and eject immediately. Per-face port amplitudes a[i, k]
(arriving at cell i through face k). Each tick every cell ejects everything: b = S a, b[i, k] leaves through face k and
arrives at nb[i, k] on face mirror(k). Nothing is stored. Void context (uniform).
straight: b[k_opp] = a[k] (pass through). huygens: junction V = 2 sum w a / sum w, b = V - a (TLM shunt node, energy
sum w a^2 conserved exactly)."""
import argparse

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=96)
p.add_argument("--rule", default="huygens")
p.add_argument("--ticks", type=int, default=40)
p.add_argument("--every", type=int, default=10)
p.add_argument("--width", type=float, default=0.0, help=">0: gaussian pulse in every port")
p.add_argument("--w", default="lap", help="lap | eq: face weights")
args = p.parse_args()
side = args.P / np.sqrt(3)
D, MIR = C.D_HAT, C.K_MIRROR
ac = C.AnchorCluster(cpp_box(args.P, n_threads=8))
pos, nb, N = ac.pos, ac.nb, ac.N
w = np.asarray(C.LAP_W, float) if args.w == "lap" else np.ones(14)
w = w / w.sum()
wrapx = lambda x: (x + side / 2) % side - side / 2
c0 = int(np.argmin(np.sum(wrapx(pos)**2, 1)))
x = wrapx(pos - pos[c0])
r = np.linalg.norm(x, axis=1)
a = np.zeros((N, 14))
a[c0] = 1.0
if args.width > 0:
    a = np.repeat(np.exp(-np.sum(x**2, 1) / (2 * args.width**2))[:, None], 14, 1)
E0 = np.sum(w * a**2)
dirs = {"sq": np.array([1, 0, 0.]), "hex": np.ones(3) / np.sqrt(3), "edge": np.array([1, 1, 0.]) / np.sqrt(2)}
r = np.linalg.norm(x, axis=1)
print(f"P {args.P} N {N} rule {args.rule} weights {np.round(w * 14, 3)} (x 1/14)", flush=True)
print("  tick  sum|E-E0|  r_peak  r_mean  spread  " + "  ".join(f"r_{k}" for k in dirs) + "  E_back(r<t/3)", flush=True)
for t in range(args.ticks + 1):
    if t % args.every == 0 and t:
        e = np.sum(w * a**2, 1)
        rm = np.sum(e * r) / e.sum()
        sp = np.sqrt(np.sum(e * (r - rm)**2) / e.sum())
        hist, edges = np.histogram(r, bins=np.arange(0, side / 2, 0.5), weights=e)
        cnt, _ = np.histogram(r, bins=edges)
        rpk = edges[np.argmax(hist)] + 0.25
        dr = []
        for d in dirs.values():
            cosang = (x @ d) / np.maximum(r, 1e-9)
            m = cosang > np.cos(np.radians(12))
            dr.append(np.sum(e[m] * r[m]) / max(e[m].sum(), 1e-300))
        print(f"  {t:4d}  {abs(np.sum(w * a**2) - E0):.1e}  {rpk:6.2f}  {rm:6.2f}  {sp:6.2f}  " + "  ".join(f"{v:6.2f}" for v in dr)
              + f"  {e[r < t / 3].sum() / e.sum():.3f}", flush=True)
    if t == args.ticks:
        break
    if args.rule == "straight":
        b = a[:, MIR]
    else:
        V = 2 * np.sum(w * a, 1, keepdims=True)
        b = V - a
    an = np.empty_like(a)
    an[nb, MIR[None, :]] = b
    a = an
