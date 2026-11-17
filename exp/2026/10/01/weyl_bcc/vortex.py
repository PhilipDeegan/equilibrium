"""Vortex ring in the real-form Dirac rule (linear): the pair angle (what the complex phase was) winds q times around
the tube centre line of a torus (r0, w), amplitude (d/w)^|q| Gaussian (zero on the centre line, smooth).

Seed: psi(x) = chi(x) v, v = the rule's positive-frequency eigenvector at k = 0 (at rest), chi = (d/w)^|q| e^{-d^2/2w^2}
e^{i q psi}; for m = 0 the k = 0 eigenvectors are degenerate and v is just one of them.
Scores: winding number of the v-projected angle around poloidal loops at d = w (8 toroidal angles), norm in the tube
and in r < r0 + 4w, core radius (density-weighted d), plots of density and pair angle on the xz plane.
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium.substrate.lattice import DampedLattice
from walk import DiracWalk, symbol_dirac, to_real

here = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--m", type=float, default=0.5)
p.add_argument("--q", type=int, default=1)
p.add_argument("--R", type=float, default=40.0)
p.add_argument("--r0", type=float, default=12.0)
p.add_argument("--w", type=float, default=8.0 / 3.0)
p.add_argument("--T", type=int, default=600, help="ticks")
p.add_argument("--every", type=int, default=10)
p.add_argument("--tag", default="")
args = p.parse_args()
name = f"vortex_m{args.m:g}_q{args.q}_R{args.R:g}{args.tag}"

lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.25)
pos = lat.pos
x, y, z = pos.T
rho = np.hypot(x, y)
psi_ang = np.arctan2(z, rho - args.r0)
d = np.hypot(rho - args.r0, z)
r_all = np.linalg.norm(pos, axis=1)
tube = d < 3 * args.w
struct = r_all < args.r0 + 4 * args.w

lam, V = np.linalg.eig(symbol_dirac(np.zeros(3), args.m))
om = -np.angle(lam)
v = V[:, int(np.argmax(om))]
chi = (d / args.w) ** abs(args.q) * np.exp(-(d**2) / (2 * args.w**2)) * np.exp(1j * args.q * psi_ang)
w = DiracWalk(lat, m=args.m)
psi0 = chi[:, None] * v[None, :]
w.L, w.R = to_real(psi0[:, :2]), to_real(psi0[:, 2:])
N0 = w.density().sum()
print(f"{name}: N={lat.N}, omega(k=0) {om.max():.4f}, Compton length c/omega {1 / np.sqrt(3) / max(om.max(), 1e-9):.2f}, "
      f"tube width w {args.w:.2f}", flush=True)

# loop sample points: d = w circle in the meridional plane at 8 toroidal angles, 64 points each -> nearest cells
from scipy.spatial import cKDTree

tree = cKDTree(pos)
loops = []
for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
    ps = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    R_ = args.r0 + args.w * np.cos(ps)
    P = np.stack([R_ * np.cos(th), R_ * np.sin(th), args.w * np.sin(ps)], -1)
    loops.append(tree.query(P)[1])


def amp(wk):
    """v-projected complex amplitude per cell (diagnostic only; the rule itself is real)."""
    Lc = wk.L[:, 0::2] + 1j * wk.L[:, 1::2]
    Rc = wk.R[:, 0::2] + 1j * wk.R[:, 1::2]
    return np.concatenate([Lc, Rc], 1) @ v.conj()


def windings(a):
    return [np.round(np.sum(np.angle(a[l][np.r_[1:64, 0]] / a[l])) / (2 * np.pi), 3) for l in loops]


def plot(tag):
    from scipy.interpolate import griddata

    rho_ = w.density()
    a = amp(w)
    lim = args.r0 + 4 * args.w
    g = np.linspace(-lim, lim, 301)
    G1, G2 = np.meshgrid(g, g)
    m_ = (np.abs(y) < 0.3) & (np.abs(x) < lim + 2) & (np.abs(z) < lim + 2)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for i, (val, lab, cmap) in enumerate([(rho_, "density sum psi^2", "viridis"), (np.angle(a), "pair angle (v-projected)", "twilight"),
                                          (np.abs(a) ** 2 / np.maximum(rho_, 1e-30), "fraction in seeded branch", "magma")]):
        V_ = griddata((x[m_], z[m_]), val[m_], (G1, G2), method="nearest" if i == 1 else "linear")
        im = ax[i].imshow(V_, origin="lower", extent=(-lim, lim, -lim, lim), cmap=cmap)
        ax[i].set_title(f"{lab} (xz)")
        fig.colorbar(im, ax=ax[i], shrink=0.8)
    fig.suptitle(f"{name} {tag}")
    fig.tight_layout()
    fig.savefig(here / f"{name}_{tag}.png", dpi=80)
    plt.close(fig)


plot("t0000")
rec = []
t0 = time.time()
for t in range(args.T + 1):
    if t % args.every == 0:
        rho_ = w.density()
        a = amp(w)
        wd = windings(a)
        dc = np.sum(rho_[tube] * d[tube]) / rho_[tube].sum()
        rec.append([t, rho_.sum() / N0, rho_[tube].sum() / N0, rho_[struct].sum() / N0, dc,
                    np.sum(np.abs(a) ** 2) / rho_.sum(), np.mean(wd), np.min(wd), np.max(wd)])
        if t % (args.T // 4) == 0:
            print(f"  t {t}: norm {rec[-1][1]:.4f} tube {rec[-1][2]:.4f} struct {rec[-1][3]:.4f} d_mean {dc:.2f} "
                  f"branch {rec[-1][5]:.3f} windings {wd}", flush=True)
            if t:
                plot(f"t{t:04d}")
    if t < args.T:
        w.step()
rec = np.array(rec)
print(f"  {args.T} ticks in {time.time() - t0:.0f}s", flush=True)
np.savez(here / f"{name}_rec.npz", rec=rec, cols="t norm tube struct d_mean branch wind_mean wind_min wind_max",
         args=str(vars(args)))
