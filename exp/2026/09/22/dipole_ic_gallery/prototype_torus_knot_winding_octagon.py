"""Corrugated-OCTAGON-ring (z-wobble) feasibility prototype -- redo of
prototype_torus_knot_winding.py on find_toroidal_ring(mode="octagon", ...)
instead of the raw minimal 8-tile SQUARE pattern, per Philip's catch that
the square-ring regime is exactly the BCC dispersion-anisotropy ("diamond
artifact") band this project's own code already flags as unreliable for
real physics (see init_electron_dipole_ic's "Sizing caution" docstring,
and why _octagon_face_seq/mode="octagon" were built in the first place,
2026-09-08).

Same z-excursion splice idea as the square prototype (insert [+z]*m then,
after the diametrically-opposite far leg, [-z]*m, so the path still closes
at z=0) but spliced into the octagon's own face sequence at the leg
diametrically opposite the start/close point (bracketing "leg 3", the far
straight run), not the square's hard-coded 8-tile pattern.
"""

import sys

sys.path.insert(0, "/home/deegan/git/eq")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import (
    _AXIS_DIR_CYCLE,
    _MIRROR,
    find_toroidal_ring,
    phi_ring_circulation,
    winding_number,
)

R_DOMAIN = 60.0
N_THREADS = 8
HOPS = 4  # octagon hops -- 16*HOPS=64 base tiles, well outside the ~8-tile
          # minimal-loop anisotropy band this project's own code warns about


def build_cluster():
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(N_THREADS)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0.0)
    alat.rebuild(0)
    return alat, cl


def octagon_zwobble_face_seq(axis, hops, m):
    """Same construction as _octagon_face_seq, but with [+z]*m inserted
    right before the leg diametrically opposite the start/close point
    ("leg 3"), and [-z]*m inserted right after it -- the whole far leg
    happens at elevated z, exactly mirroring the square prototype's
    placement (up before the far leg, down right after it)."""
    d1, d2, d3, d4 = _AXIS_DIR_CYCLE[axis]

    def chamfer(da, db):
        return [da, db] * hops

    leg1a = [d1] * hops
    chamfer1 = chamfer(d1, d2)
    leg2 = [d2] * (2 * hops)
    chamfer2 = chamfer(d2, d3)
    leg3 = [d3] * (2 * hops)
    chamfer3 = chamfer(d3, d4)
    leg4 = [d4] * (2 * hops)
    chamfer4 = chamfer(d4, d1)
    leg1b = [d1] * hops

    seq = (
        leg1a + chamfer1 + leg2 + chamfer2
        + [13] * m + leg3 + [12] * m
        + chamfer3 + leg4 + chamfer4 + leg1b
    )
    return np.array(seq, dtype=np.int32)


def walk_face_seq(cluster, start_j, face_seq):
    nb_g = cluster.leaf_nb
    leaf_g = cluster.leaf_global_indices
    g_to_j = {int(g): j for j, g in enumerate(leaf_g)}
    tiles = [start_j]
    for f in face_seq[:-1]:
        cur_j = tiles[-1]
        g_nb = int(nb_g[cur_j, int(f)])
        if g_nb < 0:
            return None, "dead end"
        j_nb = g_to_j.get(g_nb, -1)
        if j_nb < 0:
            return None, "neighbour not a leaf"
        if j_nb in tiles:
            return None, "self-intersection"
        tiles.append(j_nb)
    last_j = tiles[-1]
    g_close = int(nb_g[last_j, int(face_seq[-1])])
    if g_to_j.get(g_close, -1) != start_j:
        return None, "does not close"
    return tiles, "closed"


def main():
    alat, cl = build_cluster()
    pos = np.array(cl.leaf_pos)
    print(f"n_leaves={cl.n_leaves}", flush=True)

    # Confirm the plain octagon base ring itself closes on this build/hops
    # (sanity check before trying the z-wobble variant).
    base_tiles, base_faces = find_toroidal_ring(cl, axis="z", mode="octagon", hops=HOPS)
    if base_tiles is None:
        print(f"FAILED: base octagon ring (hops={HOPS}) did not close at all.")
        return
    base_wn = winding_number(cl, base_tiles, axis="z")
    print(f"base octagon ring: n_tiles={len(base_tiles)}  winding_number={base_wn:.4f}", flush=True)
    start_j = base_tiles[0]

    results = {}
    for m in (1, 3, 6):
        face_seq = octagon_zwobble_face_seq("z", HOPS, m)
        tiles, reason = walk_face_seq(cl, start_j, face_seq)
        if tiles is None:
            print(f"m={m}: FAILED to close ({reason})", flush=True)
            continue
        tp = pos[tiles]
        centroid = tp.mean(axis=0)
        wn = winding_number(cl, tiles, axis="z")

        amp = 1.0
        phi = np.array(cl.leaf_phi)
        for idx in range(len(tiles)):
            j_i = tiles[idx]
            j_nxt = tiles[(idx + 1) % len(tiles)]
            k = int(face_seq[idx])
            mk = int(_MIRROR[k])
            phi[k, j_i] = amp
            phi[mk, j_nxt] = -amp
        cl.set_leaf_phi(phi)
        circ = phi_ring_circulation(cl, tiles, centroid)
        cl.set_leaf_phi(np.zeros_like(phi))

        z_range = (float(tp[:, 2].min()), float(tp[:, 2].max()))
        print(
            f"m={m}: CLOSED  n_tiles={len(tiles)}  z_range={z_range}"
            f"  winding_number={wn:.4f}  phi_ring_circulation={circ:.4f}",
            flush=True,
        )
        results[m] = tiles

    if not results:
        print("No m value closed -- nothing to plot.", flush=True)
        return

    ms = sorted(results.keys())[:2] if len(results) >= 2 else sorted(results.keys())
    fig = plt.figure(figsize=(13, 6 * len(ms)))
    for row, m in enumerate(ms):
        tiles = results[m]
        tp = pos[tiles]
        tp_closed = np.vstack([tp, tp[0]])
        wn = winding_number(cl, tiles, axis="z")

        ax3d = fig.add_subplot(len(ms), 2, 2 * row + 1, projection="3d")
        ax3d.plot(tp_closed[:, 0], tp_closed[:, 1], tp_closed[:, 2], "-o", color="C0", markersize=3)
        ax3d.scatter(*tp[0], color="red", s=60, label="start/close tile")
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.set_title(f"octagon, hops={HOPS}, m={m} hump\n(n_tiles={len(tiles)}, winding_number={wn:.4f})\n3D view")
        ax3d.legend(fontsize=8)

        axz = fig.add_subplot(len(ms), 2, 2 * row + 2)
        rho = np.sqrt(tp_closed[:, 0] ** 2 + tp_closed[:, 1] ** 2)
        axz.plot(rho, tp_closed[:, 2], "-o", color="C1", markersize=3)
        axz.scatter(rho[0], tp_closed[0, 2], color="red", s=60, zorder=5)
        axz.set_xlabel("rho (dist from z-axis)")
        axz.set_ylabel("z")
        axz.set_title(f"octagon, m={m}: meridional (rho,z) projection")
        axz.set_aspect("equal")

    fig.suptitle(
        "Corrugated OCTAGON-ring (z-wobble) prototype -- octagon mode, not square",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = "exp/2026/09/22/dipole_ic_gallery/torus_knot_prototype_path_octagon.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
