"""Plot HDF5 snapshots of run.py: xz (meridional) and xy (equatorial) cuts of u, energy density, sigma,
interpolated onto a regular grid from ONE lattice plane (planes are 1/sqrt3 apart; a thicker slab mixes
phases and makes false speckle). Also prints H and its drift.

usage: plot_snap.py <snapshot dir> [state_XXXXXXX.h5 ...]   (default: every state_*.h5 in the dir)
"""

import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

here = Path(__file__).parent
sd = Path(sys.argv[1])
files = [sd / f for f in sys.argv[2:]] or sorted(sd.glob("state_*.h5"))
with h5py.File(sd / "state_0000000.h5") as f:
    pos, nbi = f["pos"][()], f["nb"][()]
    a0 = dict(f.attrs)
valid = nbi >= 0
x, y, z = pos.T
H0 = None


def energy(u, phi):
    return 0.5 * u**2 + 0.25 * np.sum(np.where(valid, phi, 0.0) ** 2, axis=1)


for fn in files:
    with h5py.File(fn) as f:
        u, phi, lam, at = f["u"][()], f["phi"][()], f["lam"][()], dict(f.attrs)
    e = energy(u, phi)
    H = e.sum()
    H0 = H if H0 is None else H0
    sig = 1.0 / (1.0 + lam)
    print(f"{fn.name}: t = {at['t']:.4f}  H = {H:.10e}  H/H0 - 1 = {H / H0 - 1:+.3e}  max|u| {np.abs(u).max():.4f}  "
          f"lam_max {lam.max():.3f}", flush=True)
    r0 = Path(at["args"] if isinstance(at["args"], str) else "").name
    import json

    args = json.loads(at["args"])
    lim = args["r0"] + 4 * args["w"]
    g = np.linspace(-lim, lim, 321)
    G1, G2 = np.meshgrid(g, g)
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
    for row, (cut, a, b, la, lb) in enumerate([(np.abs(y) < 0.3, x, z, "x", "z"), (np.abs(z) < 0.3, x, y, "x", "y")]):
        m = cut & (np.abs(a) < lim + 2) & (np.abs(b) < lim + 2)
        for col, (val, lab, cmap, sym) in enumerate([(u, "u", "RdBu_r", True), (e, "energy density", "viridis", False),
                                                     (sig, "sigma = 1/(1+lam)", "magma", False)]):
            # interpolate in 3D-to-plane: use slab cells projected, weighted by nearest-plane via linear griddata
            V = griddata((a[m], b[m]), val[m], (G1, G2), method="linear")
            vmax = np.nanmax(np.abs(V)) if sym else None
            im = ax[row, col].imshow(V, origin="lower", extent=(-lim, lim, -lim, lim), cmap=cmap,
                                     vmin=-vmax if sym else None, vmax=vmax, interpolation="nearest")
            ax[row, col].set_xlabel(la)
            ax[row, col].set_ylabel(lb)
            ax[row, col].set_title(f"{lab} ({'xz' if row == 0 else 'xy'} plane, single lattice plane)")
            fig.colorbar(im, ax=ax[row, col], shrink=0.8)
    fig.suptitle(f"{sd.name}  step {at['step']}  t = {at['t']:.3f}  H/H0-1 = {H / H0 - 1:+.2e}")
    fig.tight_layout()
    out = here / f"{sd.name}_step{at['step']:07d}.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"  -> {out}", flush=True)
