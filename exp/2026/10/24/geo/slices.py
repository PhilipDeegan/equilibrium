"""10-24 native look at the 10-22 spheromak states (no reconstruction, no E / B).
Square lattice loop (4 square-face hops, side s = 2 / sqrt3) in a slice plane: circulation of the edge integrals
a = phi / W around it = what threads that loop. Meridional plane (x-z, through the centre): threads toroidally.
Equatorial plane (x-y): threads poloidally. Plus face content sum_k phi_k^2 per cell in the meridional plane.
Rows = states, columns = meridional loops / equatorial loops / meridional content. Colour limits per panel
(99th percentile), value in the title."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

C.spacing_of = lambda lat: 1.0
p = argparse.ArgumentParser()
p.add_argument("--P", type=int, required=True)
p.add_argument("--a", type=float, required=True)
p.add_argument("--zoom", type=float, default=2.0, help="half-width / a")
p.add_argument("--out", required=True)
p.add_argument("states", nargs="+", help="label=path.npz")
args = p.parse_args()
side = args.P / np.sqrt(3)
OFF = np.array([0.13, 0.07, 0.11])
s = 2 / np.sqrt(3)
W = C.W

alat = cpp_box(args.P, n_threads=4)
ac = C.AnchorCluster(alat)
nb = ac.nb
x = (ac.pos - OFF + side / 2) % side - side / 2
dirs = C.D_HAT / np.linalg.norm(C.D_HAT, axis=1)[:, None]
face = lambda v: int(np.argmax(dirs @ np.asarray(v, float)))
F = {n: face(v) for n, v in {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0],
                             "+z": [0, 0, 1], "-z": [0, 0, -1]}.items()}
assert all(abs(dirs[k] @ v) > 0.999 for k, v in zip(F.values(), np.eye(3)[[0, 0, 1, 1, 2, 2]]))


def plane(ax_n):
    """cells of one square sublattice plane closest to the centre, normal axis ax_n."""
    y = x[:, ax_n]
    y0 = y[np.argmin(np.abs(y))]
    return np.nonzero(np.abs(y - y0) < 1e-6)[0]


def loops(phi, cells, u, v):
    """circulation around i -> +u -> +v -> -u -> -v, at the loop centre."""
    a = phi / W[None, :]
    i = cells
    j = nb[i, F["+" + u]]
    k = nb[j, F["+" + v]]
    l_ = nb[k, F["-" + u]]
    ok = (j >= 0) & (k >= 0) & (l_ >= 0) & (nb[l_, F["-" + v]] == i)
    i, j, k, l_ = i[ok], j[ok], k[ok], l_[ok]
    g = a[i, F["+" + u]] + a[j, F["+" + v]] + a[k, F["-" + u]] + a[l_, F["-" + v]]
    return x[i], g


def grid(pts, val, cu, cv):
    L = args.zoom * args.a
    m = (np.abs(pts[:, cu]) < L) & (np.abs(pts[:, cv]) < L)
    iu = np.round((pts[m, cu] + L) / s).astype(int)
    iv = np.round((pts[m, cv] + L) / s).astype(int)
    img = np.full((iv.max() + 1, iu.max() + 1), np.nan)
    img[iv, iu] = val[m]
    return img, [-L, L, -L, L]


AX = {"x": 0, "y": 1, "z": 2}
mer, equ = plane(1), plane(2)
n = len(args.states)
fig, axs = plt.subplots(n, 3, figsize=(13, 4.2 * n), squeeze=False, constrained_layout=True)
for r_, st in enumerate(args.states):
    lab, path = st.split("=")
    phi = np.load(path)["phi"]
    for c_, (cells, u, v, kind) in enumerate([(mer, "x", "z", "loop"), (equ, "x", "y", "loop"), (mer, "x", "z", "content")]):
        if kind == "loop":
            pts, val = loops(phi, cells, u, v)
            pts = pts + 0.5 * s * (np.eye(3)[AX[u]] + np.eye(3)[AX[v]])
            img, ext = grid(pts, val, AX[u], AX[v])
            lim = np.nanpercentile(np.abs(img), 99)
            im = axs[r_, c_].imshow(img, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ttl = f"{lab}: loops in {u}-{v} plane  (99% |circ| {lim:.2e})"
        else:
            val = np.sum(phi[cells] ** 2, 1)
            img, ext = grid(x[cells], val, AX[u], AX[v])
            lim = np.nanpercentile(img, 99)
            im = axs[r_, c_].imshow(img, origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=lim)
            ttl = f"{lab}: face content x-z  (99% {lim:.2e})"
        axs[r_, c_].set_title(ttl, fontsize=9)
        axs[r_, c_].set_xlabel(u)
        axs[r_, c_].set_ylabel(v)
        th = np.linspace(0, 2 * np.pi, 200)
        axs[r_, c_].plot(args.a * np.cos(th), args.a * np.sin(th), color="0.4", lw=0.6, ls="--")
        fig.colorbar(im, ax=axs[r_, c_], shrink=0.8)
    del phi
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=110)
print("wrote", args.out)
