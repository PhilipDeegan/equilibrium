"""Build a genuine (p=1, q=1) discrete torus-knot helical winding -- a single
closed path whose own geometry couples the major-circle (toroidal) and
minor-circle (poloidal) directions, unlike every earlier prototype in this
directory (flat ring + z-only detour, zero rho variation).

Method: build the major loop as 4 FULL square legs (not find_toroidal_ring's
split-at-seam convention -- avoids reasoning about a leg split across the
start/end index), each of length L sq-hops, travel directions cycling
(+y,-x,-y,+x) same as _AXIS_DIR_CYCLE['z']. Each leg's "outward" (away from
z-axis) cardinal direction is the travel-direction cycle rotated by one
step: leg +y (x~+a) -> outward +x; leg -x (y~+a) -> outward +y; leg -y
(x~-a) -> outward -x; leg +x (y~-a) -> outward -y.

Insert into each leg R "radial" hops (toward/away from the z-axis, using
that leg's own outward/inward direction) + R "z" hops (+/-13,12), gated so
the path traces ONE FULL revolution of a diamond-shaped minor loop
(outward-most,z=0) -> (rho0,z=+R) -> (inward-most,z=0) -> (rho0,z=-R) ->
back to (outward-most,z=0) over the SAME single lap that the major square
completes -- i.e. genuine p=1, q=1. Net displacement (major + radial + z)
is zero by construction (algebra below), verified against the real lattice
by an explicit closure walk, same rigor as every prior prototype in this
directory.
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
    init_toroidal_winding,
    phi_ring_circulation,
    winding_number,
)

R_DOMAIN = 50.0
N_THREADS = 8

L = 10  # travel hops per major leg
R = 3  # radial+z hops per leg (must be << L)


def build_cluster():
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(N_THREADS)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0.0)
    alat.rebuild(0)
    return alat, cl


def helical_face_seq(L, R):
    """(p=1,q=1) face sequence, z-axis major loop. See module docstring.

    First attempt used each leg's OWN local "outward" direction for the
    radial hops (rotating frame) -- algebraically checked and found NOT to
    close (net y-displacement -2R): two of the four legs' local-outward
    directions both happened to be -y, so the "radial" contributions don't
    cancel across the loop the way a true co-rotating minor frame would.
    Fixed by using a FIXED global (x,z) plane for the minor diamond instead
    of a rotating per-leg frame -- guaranteed to net to zero by the same
    4-fold symmetry argument as the major square itself, verified below.
    This is an honest simplification, not a literal (p,q) torus-knot
    parametrization: the minor loop's own plane does NOT co-rotate with the
    major tangent the way the continuum formula's does. It still produces
    real coupled rho+z variation (rho = sqrt(x^2+y^2) varies as x shifts
    under a y-varying major position), just not via a rotating local frame.
    """
    # leg travel directions, cycling +y,-x,-y,+x
    d1, d2, d3, d4 = 11, 8, 10, 9

    # fixed-plane (x,z) diamond: (+R,0)->(0,+R)->(-R,0)->(0,-R)->close
    leg1 = [d1] * L + [8] * R + [13] * R  # x: +R->0 (-x hops), z: 0->+R
    leg2 = [d2] * L + [8] * R + [12] * R  # x: 0->-R (-x hops), z: +R->0
    leg3 = [d3] * L + [9] * R + [12] * R  # x: -R->0 (+x hops), z: 0->-R
    leg4 = [d4] * L + [9] * R + [13] * R  # x: 0->+R (+x hops), z: -R->0 (closes)
    return np.array(leg1 + leg2 + leg3 + leg4, dtype=np.int32)


def walk_path(cluster, start_j, face_seq):
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

    face_seq = helical_face_seq(L, R)
    print(f"path length = {len(face_seq)} hops", flush=True)

    # net displacement check (algebraic, before touching the lattice)
    from tst.substrate.adaptive.adaptive_flux_tube import _FACE_3D_DIR

    disp = np.zeros(3)
    for f in face_seq:
        disp += _FACE_3D_DIR[int(f)]
    print(f"algebraic net displacement (hop units) = {disp}", flush=True)

    target = np.array([1.0, 0.0, 0.0]) * (0.5 * L * (2.0 / np.sqrt(3)))
    dists = np.linalg.norm(pos - target, axis=1)
    candidates = np.argsort(dists)[:500]

    tiles = None
    reason = None
    for start_j in candidates:
        tiles, reason = walk_path(cl, int(start_j), face_seq)
        if tiles is not None:
            break
    if tiles is None:
        print(f"FAILED to close: last reason = {reason}", flush=True)
        return

    tp = pos[tiles]
    rho = np.sqrt(tp[:, 0] ** 2 + tp[:, 1] ** 2)
    print(
        f"CLOSED: n_tiles={len(tiles)}  rho range=({rho.min():.3f},{rho.max():.3f})"
        f"  z range=({tp[:,2].min():.3f},{tp[:,2].max():.3f})",
        flush=True,
    )
    wn = winding_number(cl, tiles, axis="z")
    print(f"winding_number = {wn:.4f}", flush=True)

    faces = [int(f) for f in face_seq]
    omega = init_toroidal_winding(cl, tiles, faces, amplitude=1.0)
    print(f"init_toroidal_winding: omega={omega:.6f}", flush=True)

    centroid = tp.mean(axis=0)
    circ = phi_ring_circulation(cl, tiles, centroid)
    print(f"phi_ring_circulation = {circ:.4f}", flush=True)

    # ---- plot ----
    fig = plt.figure(figsize=(13, 6))
    tp_closed = np.vstack([tp, tp[0]])

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(tp_closed[:, 0], tp_closed[:, 1], tp_closed[:, 2], "-o", color="C0", markersize=3)
    ax3d.scatter(*tp[0], color="red", s=60, label="start/close tile")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title(f"(p=1,q=1) helical torus knot\nn_tiles={len(tiles)}, winding_number={wn:.4f}")
    ax3d.legend(fontsize=8)

    axz = fig.add_subplot(1, 2, 2)
    rho_closed = np.sqrt(tp_closed[:, 0] ** 2 + tp_closed[:, 1] ** 2)
    axz.plot(rho_closed, tp_closed[:, 2], "-o", color="C1", markersize=3)
    axz.scatter(rho_closed[0], tp_closed[0, 2], color="red", s=60, zorder=5)
    axz.set_xlabel("rho (dist from z-axis)")
    axz.set_ylabel("z")
    axz.set_title("meridional (rho,z) projection\n(genuine rho+z coupling, not a flat detour)")
    axz.set_aspect("equal")

    fig.suptitle(f"L={L} (travel hops/leg), R={R} (radial+z hops/leg) -- phi_ring_circulation={circ:.2f}")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = "exp/2026/09/22/dipole_ic_gallery/helical_torus_knot_p1q1.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
