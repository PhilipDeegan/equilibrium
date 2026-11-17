"""Hopf-texture seed on the C++ walk (m 0.8): which of the two degenerate k = 0 rest states each cell is in.

The rest space at k = 0 is 2-fold degenerate: W v with W = diag(U, U), U in SU(2), is an exact rest state for every U
(residual 1e-16). Seed L, R = a0 U(x) v_L, a0 U(x) v_R with the degree-1 map
    alpha = cos f + i sin f z / r,  beta = sin f (x + i y) / r,  f(r) = pi / (1 + (r / a)^4),
generalised to (alpha^q, beta^p) / norm (fibres on a torus wind q times the short way per p the long way).
U = [[alpha, -beta*], [beta, alpha*]] -> U = 1 at infinity (background), density uniform at t = 0 (no cores).
Read-out: n from the L half, z = (L0, L1) complex: n_z = (|z0|^2 - |z1|^2) / |z|^2 (+1 = background);
the n_z = -1 set (alpha = 0) is the ring r = a, z = 0 for p = q = 1.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium.substrate.walk import CppWalk, cpp_sphere, healing_g, sponge_factor, to_complex, to_real

here = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--m", type=float, default=0.8)
p.add_argument("--xi", type=float, default=3.9)
p.add_argument("--a", type=float, default=15.0, help="texture scale: f(a) = pi / 2")
p.add_argument("--p", type=int, default=1)
p.add_argument("--q", type=int, default=1)
p.add_argument("--R", type=float, default=80.0)
p.add_argument("--T", type=int, default=100)
p.add_argument("--every", type=int, default=10)
p.add_argument("--n_plot", type=int, default=2)
p.add_argument("--sponge", type=float, default=0.25)
p.add_argument("--damp", type=float, default=0.05)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--tag", default="")
args = p.parse_args()

g = healing_g(args.m, args.xi, 1.0, walk=True)
name = f"hopf_p{args.p}_q{args.q}_a{args.a:g}_xi{args.xi:g}_m{args.m:g}_R{args.R:g}{args.tag}"

t0 = time.time()
alat = cpp_sphere(args.R, n_threads=args.threads)
pos = np.array(alat[0].leaf_pos)
x, y, z = pos.T
r = np.linalg.norm(pos, axis=1)
f = np.pi / (1 + (r / args.a) ** 4)
rs = np.where(r > 0, r, 1.0)
al = np.cos(f) + 1j * np.sin(f) * z / rs
be = np.sin(f) * (x + 1j * y) / rs
al, be = al**args.q, be**args.p
nrm = np.sqrt(np.abs(al) ** 2 + np.abs(be) ** 2)
al, be = al / nrm, be / nrm

f_sp, r_inner = sponge_factor(pos, sponge=args.sponge, damp=args.damp)
w = CppWalk(alat, 0, m=args.m, g=g, rho0=1.0, f=f_sp)
v = w.v
vL, vR = v[:2], v[2:]


def rot(u2):
    return np.stack([al * u2[0] - np.conj(be) * u2[1], be * u2[0] + np.conj(al) * u2[1]], 1)


w.set_state(to_real(rot(vL)), to_real(rot(vR)), t=0)
inner = r < r_inner
print(f"{name}: N_cells={len(pos)} build+seed {time.time() - t0:.0f}s; g {g:.4f}; f(r_inner {r_inner:.0f}) = "
      f"{np.pi / (1 + (r_inner / args.a) ** 4):.3g}", flush=True)


def nfield():
    L, _ = w.state()
    zc = to_complex(L)
    s = np.sum(np.abs(zc) ** 2, 1)
    return (np.abs(zc[:, 0]) ** 2 - np.abs(zc[:, 1]) ** 2) / s, 2 * np.conj(zc[:, 0]) * zc[:, 1] / s


def stats():
    rho = w.density()
    nz, _ = nfield()
    flip = (nz < -0.5) & inner
    rf = np.sqrt(x[flip] ** 2 + y[flip] ** 2).mean() if flip.any() else np.nan
    zf = z[flip].mean() if flip.any() else np.nan
    return (rho[inner].min(), rho[inner].max(), rho[inner].std(), nz[inner].min(), int((nz[inner] < 0).sum()),
            int(flip.sum()), rf, zf)


def plot(tag):
    from scipy.interpolate import griddata

    rho = w.density()
    nz, nxy = nfield()
    lim = 3 * args.a
    gg = np.linspace(-lim, lim, 301)
    G1, G2 = np.meshgrid(gg, gg)
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    for row, (sel, X1, X2, lab) in enumerate([(np.abs(z - z[np.argmin(np.abs(z))]) < 0.05, x, y, "xy, z = 0"),
                                              (np.abs(y - y[np.argmin(np.abs(y))]) < 0.05, x, z, "xz, y = 0")]):
        m_ = sel & (np.abs(X1) < lim + 2) & (np.abs(X2) < lim + 2)
        for col, (val, ttl, cm, vr) in enumerate([(rho, "density", "viridis", None), (nz, "n_z", "RdBu", (-1, 1)),
                                                  (np.angle(nxy), "arg(n_x + i n_y)", "twilight", (-np.pi, np.pi))]):
            im = ax[row, col].imshow(griddata((X1[m_], X2[m_]), val[m_], (G1, G2), method="linear"), origin="lower",
                                     extent=(-lim, lim, -lim, lim), cmap=cm,
                                     vmin=None if vr is None else vr[0], vmax=None if vr is None else vr[1])
            ax[row, col].set_title(f"{ttl}, {lab}")
            fig.colorbar(im, ax=ax[row, col], shrink=0.8)
    fig.suptitle(f"{name}  t = {w.t}")
    fig.tight_layout()
    fig.savefig(here / f"{name}_{tag}.png", dpi=80)
    plt.close(fig)


plot("t0000")
plot_every = max(args.every, (args.T // args.n_plot) // args.every * args.every)
rec = []
t0 = time.time()
for t in range(0, args.T + 1, args.every):
    st = stats()
    rec.append((t,) + st)
    print(f"  t {t}: rho min {st[0]:.4f} max {st[1]:.4f} std {st[2]:.2e}; n_z min {st[3]:+.3f}, cells n_z<0 {st[4]}, "
          f"n_z<-0.5 {st[5]} at rho_cyl {st[6]:.1f} z {st[7]:+.1f}", flush=True)
    if t and t % plot_every < args.every:
        plot(f"t{t:05d}")
    if t < args.T:
        w.step(args.every)
print(f"  {args.T} ticks in {time.time() - t0:.0f}s", flush=True)
np.savez(here / f"{name}_rec.npz", rec=np.array(rec, float), args=str(vars(args)))
