"""Coordinate-free discrete curl/circulation on the Kelvin/BCC lattice.

Geometric fact (verified exactly against `tst.substrate.adaptive.
adaptive_flux_tube`'s `_HEX_DIR`/`_SQ_DIR` face tables, 2026-08-16): each of
the 6 square-face directions is the sum of exactly 2 distinct pairs of the 8
hex-face directions (e.g. sq[+x] == hex[4]+hex[7] == hex[5]+hex[6]). So every
cell sits at the apex of 12 elementary triangle loops (6 sq-directions x 2
hex-pairs): A -[hex h1]-> B -[hex h2]-> C, where C is ALSO A's direct
sq-face neighbour -- two hex hops and one sq hop connect the SAME pair of
cells. This triangle is intrinsic to the fixed per-level face-index table
(0-13, identical for every cell at a given level) -- it needs only `nb[]`
adjacency and the (h1, h2, sqk) index table below, no absolute (x,y,z)
coordinates anywhere. This is the DST-native analogue of a lattice-gauge
Wilson loop.

**Curl-free = discrete-exact.** A discrete field `phi` has EXACTLY zero
circulation around every such triangle iff it is expressible as
`phi[k, i] = u[nb[i, k]] - u[i]` for some scalar field `u` (pure algebraic
telescoping: `(u_B-u_A) + (u_C-u_B) + (u_A-u_C) == 0` identically, for ANY
scalar `u`, exactly, at any resolution -- confirmed numerically to float64
noise floor, ~1e-18, not just "small"). This is a stronger and cleaner
statement than the continuum Stokes' theorem, which only gives exact zero in
the infinitesimal-loop limit -- a `phi` built by SAMPLING a smooth
continuum gradient (e.g. `equilibrium.substrate.dipole_mode_ic.
seed_dipole_ic`'s phi = grad(U) evaluated at face midpoints) is NOT a
discrete derivative in this sense and picks up a real O(hop^2) curvature
artifact (measured ~0.21 relative to phi's own RMS on the production dipole
IC) -- not evidence of real rotational content, just a reminder that
"curl-free in the continuum" and "discrete-exact on this lattice" are
different conditions. See `project_dst_curl_investigation` memory.

Injecting genuine rotational/"magnetic" free energy means constructing a
`phi` that is NOT expressible as any scalar's discrete derivative -- i.e.
deliberately nonzero circulation on at least some triangles, assigned
directly on the graph, with no scalar potential underneath and no Cartesian
formula (e.g. PHARE's `get_taylor_spheromak` cross-product construction,
`run/phare_plasmoses.py`) required. That generator is not yet built here --
this module currently provides only the identity/measurement primitives.
"""

import numpy as np

from tst.substrate.adaptive.adaptive_flux_tube import _HEX_DIR, _SQ_DIR, _MIRROR

# (h1, h2, sqk): sqk is the GLOBAL face index (8-13) used by nb[]/phi/_MIRROR.
# _SQ_DIR itself is a standalone 0-5-indexed array (its own local convention)
# -- indexing nb[]/phi/_MIRROR with the bare 0-5 value (rather than +8) is a
# real bug found and fixed 2026-08-16: it silently aliases sq faces to hex
# faces and makes every triangle "identity" check fail (0/N), so any change
# here should re-run test_discrete_curl.py's triangle-identity test, which
# would catch that class of bug directly.
TRIANGLES = []
for _sqk_local, _sq in enumerate(_SQ_DIR):
    _sqk = _sqk_local + 8
    for _a in range(8):
        for _b in range(_a + 1, 8):
            _s = _HEX_DIR[_a] + _HEX_DIR[_b]
            _n = np.linalg.norm(_s)
            if _n > 0 and np.allclose(_s / _n, _sq, atol=1e-8):
                TRIANGLES.append((_a, _b, _sqk))
assert len(TRIANGLES) == 12, f"expected 12 triangle families, got {len(TRIANGLES)}"


def leaf_nb_rows(cl):
    """cl.leaf_nb holds GLOBAL cell indices (including non-leaf ancestors,
    e.g. AMR-junction proxies); leaf_* arrays are compacted to leaves only.
    Returns (N,14) neighbour indices translated to leaf-array ROWS, with -1
    for "no neighbour" OR "neighbour is not itself a leaf" (e.g. a coarse-
    fine junction) -- this module only reasons about same-level leaf-leaf
    topology, consistent with resolve_stale_external_proxies's own 1-level
    scope.
    """
    nb_g = np.asarray(cl.leaf_nb)
    gidx = np.asarray(cl.leaf_global_indices)
    N = nb_g.shape[0]

    g2row = np.full(int(gidx.max()) + 1, -1, dtype=np.int64)
    g2row[gidx] = np.arange(N)

    flat = nb_g.reshape(-1)
    out = np.full(flat.shape, -1, dtype=np.int64)
    valid = flat >= 0
    gv = flat[valid]
    in_range = gv <= g2row.shape[0] - 1
    rows = np.full(gv.shape, -1, dtype=np.int64)
    rows[in_range] = g2row[gv[in_range]]
    out[valid] = rows
    return out.reshape(nb_g.shape)


def triangle_triples(nb, idx=None):
    """nb: (N,14) leaf-row neighbour array (see leaf_nb_rows). idx: optional
    subset of leaf rows to probe (e.g. excluding domain boundary/junctions).
    Returns list of (family, a, b, c) -- a,b,c are (M,) leaf-row arrays for
    every (cell, triangle-family) pair where the two-hex-hop path and the
    direct sq-face neighbour genuinely agree (i.e. the geometric identity
    holds -- expected to fail near a real domain boundary or an unresolved
    coarse-fine junction, which is itself diagnostic).
    """
    N = nb.shape[0]
    idx_all = np.arange(N) if idx is None else np.asarray(idx)
    triples = []
    for h1, h2, sqk in TRIANGLES:
        b = nb[idx_all, h1]
        vb = b >= 0
        b2, a2 = b[vb], idx_all[vb]
        c = nb[b2, h2]
        vc = c >= 0
        c2, a3, b3 = c[vc], a2[vc], b2[vc]
        d = nb[a3, sqk]
        ok = d == c2
        triples.append(((h1, h2, sqk), a3[ok], b3[ok], c2[ok]))
    return triples


def circulation(phi, nb, idx=None):
    """phi: (14, N) per-face flux array. Returns 1D array of circulation
    values, one per (valid triangle, cell) instance, using each vertex's OWN
    outgoing phi for its leg -- circ = phi_A[h1] + phi_B[h2] + phi_C[mirror(sqk)].
    """
    out = []
    for (h1, h2, sqk), a, b, c in triangle_triples(nb, idx):
        out.append(phi[h1, a] + phi[h2, b] + phi[_MIRROR[sqk], c])
    return np.concatenate(out) if out else np.array([])


def circulation_from_scalar(u, nb, idx=None):
    """u: (N,) scalar field. Builds the discrete derivative phi[k,i] =
    u[nb[i,k]]-u[i] implicitly and returns its circulation -- exactly zero
    (float64 noise floor) for ANY u, by pure telescoping. Useful as a
    baseline/regression check distinct from `circulation`, which takes a
    real phi array that may not be any scalar's discrete derivative.
    """
    out = []
    for (h1, h2, sqk), a, b, c in triangle_triples(nb, idx):
        out.append((u[b] - u[a]) + (u[c] - u[b]) + (u[a] - u[c]))
    return np.concatenate(out) if out else np.array([])
