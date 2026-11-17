"""10-08: the periodic flow of Gaussian-cored vortex lines as an exact Fourier sum (no grid) evaluated
on the primary strand, u(x) = sum_{k != 0} i k x w_k / k^2 e^{ik.x} + U0 z, w_k = (2 pi q / V) sum_seg dl e^{-ik.x_seg}
e^{-k^2 sigma^2 / 2} (|k| < kmax_sigma / sigma), U0 = 2 pi q pi r0^2 / V per coil (U0 derivation: lines outside every ring wind 0, lines through a hole wind 2 pi q). Lone box = primary
alone (self + images, the self part identical at every P up to the k sum -> differences are the image effect); pair
= + partner sublattice at (s/2, s/2, dz s). Advection v = n / (3 m) <u_z> on the strand."""
import argparse

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--P", type=int, nargs="+", default=[180, 220, 240, 300, 340, 360])
p.add_argument("--N", type=int, default=2)
p.add_argument("--r0", type=float, default=31.0)
p.add_argument("--a", type=float, default=10.0)
p.add_argument("--m", type=float, default=0.8)
p.add_argument("--sigma", type=float, default=3.9)
p.add_argument("--kmax_sigma", type=float, default=6.0)
p.add_argument("--seg", type=int, default=200)
p.add_argument("--npt", type=int, default=48)
p.add_argument("--dz", type=float, default=0.5)
p.add_argument("--qp", type=int, default=1)
args = p.parse_args()
kadv = np.sqrt(1 - args.m**2) / (3 * args.m)


def strand(off=(0, 0, 0), seg=None):
    s = np.linspace(0, 2 * np.pi, args.N * (seg or args.seg), endpoint=False)
    tp = args.N * s
    rr = args.r0 + args.a * np.cos(tp)
    return np.stack([rr * np.cos(s), rr * np.sin(s), args.a * np.sin(tp)], 1) + np.asarray(off)


def uz_on(side, S, X, q=1):
    """z flow (k != 0 part) at points X from the periodic array of strand S (closed polyline)."""
    dk = 2 * np.pi / side
    nmax = int(args.kmax_sigma / args.sigma / dk) + 1
    i = np.arange(-nmax, nmax + 1)
    K = np.stack(np.meshgrid(i, i, i, indexing="ij"), -1).reshape(-1, 3) * dk
    k2 = np.sum(K**2, 1)
    K, k2 = K[(k2 > 0) & (k2 < (args.kmax_sigma / args.sigma)**2)], k2[(k2 > 0) & (k2 < (args.kmax_sigma / args.sigma)**2)]
    dl = np.roll(S, -1, 0) - S
    mid = S + dl / 2
    out = np.zeros(len(X))
    for c0 in range(0, len(K), 20000):
        Kc, kc2 = K[c0:c0 + 20000], k2[c0:c0 + 20000]
        E = np.exp(-1j * mid @ Kc.T)  # (seg, k)
        w = (2 * np.pi * q / side**3) * (dl.T @ E) * np.exp(-0.5 * kc2 * args.sigma**2)  # (3, k)
        uzk = 1j * (Kc[:, 0] * w[1] - Kc[:, 1] * w[0]) / kc2
        out += np.real(np.exp(1j * X @ Kc.T) @ uzk)
    return out


prim = strand()
X = strand(seg=max(1, args.npt // args.N))
area = np.pi * args.r0**2
lone = {}
print(f"r0 {args.r0} a {args.a} N {args.N} sigma {args.sigma}; advection n/(3m) = {kadv:.4f}")
for P in args.P:
    s = P / np.sqrt(3)
    U0 = 2 * np.pi * area / s**3
    own = uz_on(s, prim, X).mean() + U0
    par = uz_on(s, strand((s / 2, s / 2, args.dz * s)), X, args.qp).mean() + args.qp * U0
    lone[P] = own
    d = s * np.sqrt(0.5 + args.dz**2)
    print(f"P {P:4d} s {s:6.1f} d {d:6.1f}: own (self + images + U0) <u_z> {own:+.5e}; partner {par:+.4e} -> dv {kadv * par:+.5f}", flush=True)
ref = args.P[-1]
print("lone-box drift relative to P", ref, ": " + "  ".join(f"P {P} {kadv * (u - lone[ref]):+.5f}" for P, u in lone.items()))
