"""Feasibility prototype for a discrete (p,q) torus-knot helical winding --
scoping exercise, NOT a production IC. See torus_knot_winding_scope.md for
the full writeup; this script is the empirical closure/circulation check
that writeup's recommendation is based on.

Minimal case built: p=1 (one lap around the major/z-axis circle), q=1 (one
"hump" of height m*D_SQ inserted into the loop, up over roughly half the
path and back down over the other half) -- the simplest path that is
genuinely non-planar (has real z excursion, not just an in-plane loop) while
still closing back to its exact start tile using only legal BCC face-hops.
General q (multiple humps) is a straightforward extension once this closes;
NOT attempted here (see scope doc for why coupled rho+z motion, the genuine
torus-knot case, is materially harder and out of scope for this prototype).
"""

import sys

sys.path.insert(0, "/home/deegan/git/eq")

import numpy as np

from tst.substrate.adaptive.adaptive_flux_tube import (
    _FACE_HOP,
    _MIRROR,
    phi_ring_circulation,
    winding_number,
)
from equilibrium import lattice as _lat

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
    """
    Attempt to walk a (p=1,q=1) helical loop from start_j:
      +y, -x, -x, [+z]*m, -y, -y, [-z]*m, +x, +x, +y(close)
    -- the minimal 8-tile square pattern (_TOROIDAL_PATTERNS['z']) with an
    m-tile z-excursion inserted symmetrically at the two corners furthest
    from the start (after the -x,-x leg, before the -y,-y leg; reversed
    before the +x,+x leg), so the path returns to z=0 before the final
    closing hop. Returns (tiles, faces) or (None, None, reason).
    """
    face_seq = [11, 8, 8] + [13] * m + [10, 10] + [12] * m + [9, 9, 11]
    nb_g = cluster.leaf_nb
    leaf_g = cluster.leaf_global_indices
    g_to_j = {int(g): j for j, g in enumerate(leaf_g)}

    tiles = [start_j]
    for f in face_seq[:-1]:
        cur_j = tiles[-1]
        g_nb = int(nb_g[cur_j, int(f)])
        if g_nb < 0:
            return None, None, f"dead end at face {f}, tile idx {len(tiles)}"
        j_nb = g_to_j.get(g_nb, -1)
        if j_nb < 0:
            return None, None, f"neighbour not a leaf at face {f}"
        if j_nb in tiles:
            return None, None, f"self-intersection at face {f}, tile idx {len(tiles)}"
        tiles.append(j_nb)

    last_j = tiles[-1]
    g_close = int(nb_g[last_j, int(face_seq[-1])])
    if g_to_j.get(g_close, -1) != start_j:
        return None, None, "does not close back to start"
    return tiles, [int(f) for f in face_seq], "closed"


def main():
    print("Building cluster...", flush=True)
    alat, cl = build_cluster()
    print(f"  n_leaves={cl.n_leaves}", flush=True)

    pos = np.array(cl.leaf_pos)
    # start near +x axis, same heuristic find_toroidal_ring uses
    target = np.array([1.0, 0.0, 0.0])
    dists = np.linalg.norm(pos - target, axis=1)
    candidates = np.argsort(dists)[:200]

    for m in (1, 2, 3, 5):
        print(f"\n--- m={m} (hump height, in sq-hop tiles) ---", flush=True)
        found = False
        for start_j in candidates:
            tiles, faces, reason = walk_helical_ring(cl, int(start_j), m)
            if tiles is not None:
                found = True
                tp = pos[tiles]
                z_range = (float(tp[:, 2].min()), float(tp[:, 2].max()))
                centroid = tp.mean(axis=0)
                wn = winding_number(cl, tiles, axis="z")
                # seed a simple constant-amplitude circulation along the path
                amp = 1.0
                phi = np.array(cl.leaf_phi)
                nb_g = cl.leaf_nb
                leaf_g = cl.leaf_global_indices
                g_to_j = {int(g): j for j, g in enumerate(leaf_g)}
                for idx in range(len(tiles)):
                    j_i = tiles[idx]
                    j_nxt = tiles[(idx + 1) % len(tiles)]
                    k = faces[idx]
                    mk = int(_MIRROR[k])
                    phi[k, j_i] = amp
                    phi[mk, j_nxt] = -amp
                cl.set_leaf_phi(phi)
                circ = phi_ring_circulation(cl, tiles, centroid)
                print(
                    f"  CLOSED: start_j={start_j}  n_tiles={len(tiles)}"
                    f"  z_range={z_range}  centroid={centroid}"
                    f"  winding_number={wn:.4f}  phi_ring_circulation={circ:.4f}",
                    flush=True,
                )
                # reset phi for next attempt
                cl.set_leaf_phi(np.zeros_like(phi))
                break
            else:
                pass
        if not found:
            print(f"  no closing path found among {len(candidates)} candidates ({reason})", flush=True)


if __name__ == "__main__":
    main()
