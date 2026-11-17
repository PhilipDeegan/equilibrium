"""Item C: piece counting in the y = 0 cut without the rho < 0.25 connectivity.

Per saved state: each component a_c = psi . v_c* interpolated to a unit grid on the slab |y| < 0.6; a zero of a_c is a
plaquette with phase winding +-1 (topological: bands and touching cores cannot fake or merge it). Total-density local
minima (minimum filter, window --win) below --rho_cut listed alongside. A ring crossing the cut gives two zeros,
opposite sign, at x ~ +-r.
usage: python3 -u cut_zeros.py <run dir name> [...]   (states from --snap_root/<name>); Cut is used in-run by pair_box.py
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import minimum_filter

from equilibrium.substrate.walk import symbol_dirac, to_complex

here = Path(__file__).parent


def winding(a):
    ph = np.angle(a)

    def d(u, v):
        return np.angle(np.exp(1j * (v - u)))

    s = d(ph[:-1, :-1], ph[:-1, 1:]) + d(ph[:-1, 1:], ph[1:, 1:]) + d(ph[1:, 1:], ph[1:, :-1]) + d(ph[1:, :-1], ph[:-1, :-1])
    return np.rint(np.nan_to_num(s) / (2 * np.pi)).astype(int)


class Cut:
    """y = 0 slab of a box (pos), unit grid; zeros(a) -> [(x, z, sign)] of a complex cell field."""

    def __init__(self, pos):
        self.m = np.abs(pos[:, 1]) < 0.6
        self.xs, self.zs = pos[self.m, 0], pos[self.m, 2]
        self.lim = np.floor(min(np.abs(self.xs).max(), np.abs(self.zs).max()))
        self.gg = np.arange(-self.lim, self.lim + 1.0)
        self.G = np.meshgrid(self.gg, self.gg)

    def grid(self, f):
        return griddata((self.xs, self.zs), f, tuple(self.G))

    def zeros(self, a):
        A = self.grid(a.real) + 1j * self.grid(a.imag)
        W = winding(A)
        iz, ix = np.nonzero(W)
        return [(self.gg[ix[k]] + 0.5, self.gg[iz[k]] + 0.5, int(W[iz[k], ix[k]])) for k in range(len(ix))], A


def fmt_zeros(zc):
    return f"{len(zc)} [" + " ".join(f"({x_:+.0f},{z_:+.0f}){'+' if s_ > 0 else '-'}" for x_, z_, s_ in zc) + "]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+")
    p.add_argument("--m", type=float, default=0.8)
    p.add_argument("--snap_root", default="/bk/dst/2026/10/04/pair")
    p.add_argument("--win", type=int, default=9)
    p.add_argument("--rho_cut", type=float, default=0.85)
    args = p.parse_args()
    lam, V = np.linalg.eig(symbol_dirac(np.zeros(3), args.m))
    pf = np.flatnonzero(-np.angle(lam) > 0)
    vc = {1: V[:, pf[0]], 2: V[:, pf[1]]}
    for name in args.names:
        snap = Path(args.snap_root) / name
        cut = Cut(np.load(snap / "pos.npy"))
        m_, lim, gg = cut.m, cut.lim, cut.gg
        out_dir = here / name
        print(f"== {name}", flush=True)
        for f in sorted(snap.glob("state_t*.npz")):
            st = np.load(f)
            ps = np.concatenate([to_complex(st["L"][m_]), to_complex(st["R"][m_])], 1)
            t = int(st["t"])
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            line = []
            for i, c in enumerate((1, 2)):
                a = ps @ vc[c].conj()
                zc, A = cut.zeros(a)
                line.append(f"v{c} zeros {fmt_zeros(zc)}")
                ax[i].imshow(np.abs(A) ** 2, origin="lower", extent=(-lim, lim, -lim, lim), cmap="viridis")
                for x_, z_, s_ in zc:
                    ax[i].plot(x_, z_, "o" if s_ > 0 else "x", color="r" if s_ > 0 else "w", ms=8, mfc="none", mew=2)
                ax[i].set_title(f"|a{c}|^2, zeros (o +1, x -1): {len(zc)}")
            R = cut.grid((np.abs(ps) ** 2).sum(1))
            Rf = np.nan_to_num(R, nan=9.0)
            iz, ix = np.nonzero((Rf == minimum_filter(Rf, size=args.win, mode="wrap")) & (Rf < args.rho_cut))
            line.append(f"rho minima {len(ix)} [" + " ".join(f"({gg[ix[k]]:+.0f},{gg[iz[k]]:+.0f}) {R[iz[k], ix[k]]:.2f}" for k in range(len(ix))) + "]")
            ax[2].imshow(R, origin="lower", extent=(-lim, lim, -lim, lim), cmap="viridis")
            ax[2].plot(gg[ix], gg[iz], "+", color="r", ms=10, mew=2)
            ax[2].set_title(f"rho, minima < {args.rho_cut}: {len(ix)}")
            fig.suptitle(f"{name}  t = {t}  (y = 0 cut)")
            fig.tight_layout()
            fig.savefig(out_dir / f"zeros_t{t:05d}.png", dpi=80)
            plt.close(fig)
            print(f"  t {t}: " + "; ".join(line), flush=True)


if __name__ == "__main__":
    main()
