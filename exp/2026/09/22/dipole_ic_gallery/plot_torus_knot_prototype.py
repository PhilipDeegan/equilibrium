"""Visualization of the corrugated-ring (p=1,q=1 z-wobble) prototype from
torus_knot_winding_scope.md -- shows the actual discrete tile path on the
real lattice, for m=1 and m=5 hump heights, side by side."""

import sys

sys.path.insert(0, "/home/deegan/git/eq")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import _MIRROR, winding_number

R_DOMAIN = 40.0
N_THREADS = 8


def build_cluster():
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(N_THREADS)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0.0)
    alat.rebuild(0)
    return alat, cl


def walk_helical_ring(cluster, start_j, m):
    face_seq = [11, 8, 8] + [13] * m + [10, 10] + [12] * m + [9, 9, 11]
    nb_g = cluster.leaf_nb
    leaf_g = cluster.leaf_global_indices
    g_to_j = {int(g): j for j, g in enumerate(leaf_g)}
    tiles = [start_j]
    for f in face_seq[:-1]:
        cur_j = tiles[-1]
        g_nb = int(nb_g[cur_j, int(f)])
        if g_nb < 0:
            return None, None
        j_nb = g_to_j.get(g_nb, -1)
        if j_nb < 0 or j_nb in tiles:
            return None, None
        tiles.append(j_nb)
    last_j = tiles[-1]
    g_close = int(nb_g[last_j, int(face_seq[-1])])
    if g_to_j.get(g_close, -1) != start_j:
        return None, None
    return tiles, [int(f) for f in face_seq]


def find_and_get(cl, pos, m):
    target = np.array([1.0, 0.0, 0.0])
    dists = np.linalg.norm(pos - target, axis=1)
    candidates = np.argsort(dists)[:200]
    for start_j in candidates:
        tiles, faces = walk_helical_ring(cl, int(start_j), m)
        if tiles is not None:
            return tiles
    return None


def main():
    alat, cl = build_cluster()
    pos = np.array(cl.leaf_pos)
    print(f"n_leaves={cl.n_leaves}", flush=True)

    fig = plt.figure(figsize=(13, 12))

    for row, m in enumerate((1, 5)):
        tiles = find_and_get(cl, pos, m)
        assert tiles is not None, f"m={m} did not close"
        tp = pos[tiles]
        tp_closed = np.vstack([tp, tp[0]])  # close the loop for plotting
        wn = winding_number(cl, tiles, axis="z")

        ax3d = fig.add_subplot(2, 2, 2 * row + 1, projection="3d")
        ax3d.plot(tp_closed[:, 0], tp_closed[:, 1], tp_closed[:, 2], "-o", color="C0")
        ax3d.scatter(*tp[0], color="red", s=60, label="start/close tile")
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.set_title(f"m={m} hump  (n_tiles={len(tiles)}, winding_number={wn:.4f})\n3D view")
        ax3d.legend(fontsize=8)

        axz = fig.add_subplot(2, 2, 2 * row + 2)
        rho = np.sqrt(tp_closed[:, 0] ** 2 + tp_closed[:, 1] ** 2)
        axz.plot(rho, tp_closed[:, 2], "-o", color="C1")
        axz.scatter(rho[0], tp_closed[0, 2], color="red", s=60, zorder=5)
        axz.set_xlabel("rho (dist from z-axis)")
        axz.set_ylabel("z")
        axz.set_title(f"m={m} hump: meridional (rho,z) projection\n"
                       "(flat ring detouring out of plane, NOT true rho+z coupling)")
        axz.set_aspect("equal")

    fig.suptitle(
        "Corrugated-ring (p=1,q=1 z-wobble) prototype -- torus_knot_winding_scope.md",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = "exp/2026/09/22/dipole_ic_gallery/torus_knot_prototype_path.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
