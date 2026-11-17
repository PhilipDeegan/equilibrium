"""Probe a DST-native, coordinate-free discrete curl/circulation on the
Kelvin/BCC lattice, and check it against the current (curl-free-by-
construction) dipole-mode IC as a sanity baseline.

Geometric fact (verified analytically against _HEX_DIR/_SQ_DIR): each of the
6 square-face directions is the sum of exactly 2 distinct pairs of the 8
hex-face directions (e.g. sq[+x] = hex[4]+hex[7] = hex[5]+hex[6]). This means
every cell sits at the apex of 12 elementary triangle loops (6 sq-dirs x 2
hex-pairs), each closing as: A -[hex h1]-> B -[hex h2]-> C, where C is ALSO
A's direct sq-face-k neighbour -- i.e. two hex hops and one sq hop connect
the SAME pair of cells by construction. This triangle is intrinsic to the
FIXED per-level face-index table (same for every cell), so it needs no
absolute (x,y,z) coordinates at all -- only nb[] adjacency and the
(h1,h2,sqk) index table below. This is the DST-native candidate for a
"discrete curl": circulation around the loop using each cell's own outgoing
phi value for its leg,

    circ = phi_A[h1] + phi_B[h2] + phi_C[mirror(sqk)]

analogous to a lattice-gauge Wilson loop. For any phi built as a gradient
of a scalar (phi = grad(U), the current dipole-mode IC's construction),
Stokes' theorem says circulation around ANY closed loop is exactly zero in
the continuum -- so on a real discrete lattice this diagnostic should come
out near the lattice's own residual noise floor (~1e-9, see
project_odd27_amr_junction_leak's antisymmetry-residual finding), NOT zero
bit-for-bit, but far below the field's own RMS scale.

This script:
  1. Verifies the nb[nb[i,h1],h2] == nb[i,sqk] identity holds on real leaf
     data (away from AMR junctions/domain boundary -- expected to fail
     there, which is itself informative).
  2. Computes circ for all 12 triangle families on the current dipole IC,
     reports RMS(circ) vs RMS(phi) -- the "is it curl-free" check.

Run:
    PYTHONPATH=$PWD python3 exp/.../probe_discrete_curl.py /bk/l1_wider_margin_active_ic220_pp1fix.h5
"""

import argparse
import sys

sys.path.insert(0, ".")

import numpy as np

from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import _HEX_DIR, _SQ_DIR, _MIRROR

# Enumerate (h1, h2, sqk) triangle families: for each sq direction, the 2
# hex-index pairs whose sum is proportional to it. sqk is the GLOBAL face
# index (8-13) used by nb[]/phi/_MIRROR -- _SQ_DIR itself is a standalone
# 0-5-indexed array (its own local convention), so it needs +8 here. (Bug
# found 2026-08-16: using the bare 0-5 _SQ_DIR index against nb_g/phi/_MIRROR
# silently aliased sq faces to the wrong slots -- e.g. sqk=0 (meant to be
# global face 8, -x) was read as global face 0 (a hex face) -- and made
# every triangle identity check fail (0/N), not just circulation.)
TRIANGLES = []
for sqk_local, sq in enumerate(_SQ_DIR):
    sqk = sqk_local + 8
    for i in range(8):
        for j in range(i + 1, 8):
            s = _HEX_DIR[i] + _HEX_DIR[j]
            n = np.linalg.norm(s)
            if n > 0 and np.allclose(s / n, sq, atol=1e-8):
                TRIANGLES.append((i, j, sqk))
assert len(TRIANGLES) == 12, f"expected 12 triangle families, got {len(TRIANGLES)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument(
        "--r-max",
        type=float,
        default=None,
        help="only probe leaves with |pos|<r_max (avoid domain boundary)",
    )
    args = ap.parse_args()

    alat = _lat.AdaptiveLattice.load(args.path)
    cl = alat[0]
    nb_g = np.array(cl.leaf_nb)  # (N, 14) -- GLOBAL cell indices, not leaf rows
    phi = np.array(cl.leaf_phi)  # (14, N)
    pos = np.array(cl.leaf_pos)  # (N, 3)
    gidx = np.array(cl.leaf_global_indices)  # (N,) leaf row -> global index
    N = nb_g.shape[0]
    print(f"loaded {args.path}  n_leaves={N}", flush=True)

    # global cell index -> leaf row (-1 if not a leaf, e.g. an ancestor-
    # substitution proxy at a coarse-fine junction -- treated as invalid here,
    # consistent with this diagnostic only probing same-level leaf-leaf
    # topology, same convention as the docstring's "uniform region" caveat)
    g2row = np.full(int(gidx.max()) + 1, -1, dtype=np.int64)
    g2row[gidx] = np.arange(N)

    def to_row(g):
        out = np.full(g.shape, -1, dtype=np.int64)
        valid = g >= 0
        gv = g[valid]
        in_range = gv <= g2row.shape[0] - 1
        rows = np.full(gv.shape, -1, dtype=np.int64)
        rows[in_range] = g2row[gv[in_range]]
        out[valid] = rows
        return out

    nb = to_row(nb_g.reshape(-1)).reshape(nb_g.shape)  # (N,14) leaf rows, -1 = invalid

    mask = np.ones(N, dtype=bool)
    if args.r_max is not None:
        mask = np.linalg.norm(pos, axis=1) < args.r_max
    idx_all = np.nonzero(mask)[0]

    phi_rms = float(np.sqrt(np.mean(phi**2)))
    print(f"phi_rms (whole field) = {phi_rms:.6e}", flush=True)

    total_valid = 0
    total_checked = 0
    total_identity_ok = 0
    circ_all = []

    for h1, h2, sqk in TRIANGLES:
        b = nb[idx_all, h1]
        valid_b = b >= 0
        b_valid = b[valid_b]
        a_valid = idx_all[valid_b]

        c_via_hex = nb[b_valid, h2]
        valid_c = c_via_hex >= 0
        c_valid = c_via_hex[valid_c]
        a2 = a_valid[valid_c]
        b2 = b_valid[valid_c]

        c_direct = nb[a2, sqk]
        identity_ok = c_direct == c_valid
        total_checked += identity_ok.size
        total_identity_ok += int(identity_ok.sum())

        # only compute circulation where the triangle identity actually holds
        a3, b3, c3 = a2[identity_ok], b2[identity_ok], c_valid[identity_ok]
        circ = phi[h1, a3] + phi[h2, b3] + phi[_MIRROR[sqk], c3]
        circ_all.append(circ)
        total_valid += circ.size

    circ_all = np.concatenate(circ_all) if circ_all else np.array([])
    print(
        f"triangle identity check: {total_identity_ok}/{total_checked} hold "
        f"({100*total_identity_ok/max(total_checked,1):.3f}%)",
        flush=True,
    )
    print(f"circulation samples: {total_valid}", flush=True)
    if circ_all.size:
        circ_rms = float(np.sqrt(np.mean(circ_all**2)))
        circ_max = float(np.max(np.abs(circ_all)))
        print(
            f"circ_rms = {circ_rms:.6e}   circ_max = {circ_max:.6e}   "
            f"circ_rms/phi_rms = {circ_rms/phi_rms:.6e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
