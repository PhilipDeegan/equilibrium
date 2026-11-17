"""
Discrete curl(phi) = B on the Kelvin-cell sq faces, built independently of
equilibrium.substrate.mhd (that module is a peripheral MHD/instability
extension, not the core substrate -- see doc/dst.md's own definition of
phi as the vector potential and B as its plaquette curl, which is a core
geometric relationship, not an MHD-specific approximation).

Two things deliberately done differently from mhd.py's SQ_PLAQUETTES here:

1. Face-index-to-physical-direction is determined EMPIRICALLY from real
   lattice geometry (lat.pos, lat.nb), never assumed from a hardcoded
   comment -- equilibrium/substrate/lattice.py and mhd.py's own comments
   disagree with each other about which of face 8/9 is +x vs -x, which is
   exactly the kind of mislabeling this module is built to be immune to.

2. Loop orientation for each axis uses the CYCLIC rule the curl formula
   actually requires: (curl A)_i = d(A_v)/d(u) - d(A_u)/d(v), where
   (u, v) = (next(i), prev(i)) in the cyclic order x->y->z->x (so u x v = i
   for a right-handed loop). mhd.py's SQ_PLAQUETTES built the y-loop by
   naive analogy to the x/z loops (same (first,second) order for all three)
   instead of respecting this cyclic pairing -- a classic, easy-to-miss
   sign bug: curl_y genuinely needs the OPPOSITE cyclic order from curl_x
   and curl_z. Confirmed two independent ways: (a) this derivation, (b) a
   direct numerical test against the exact classical straight-wire B(r)
   profile (zero at the axis, linear ramp to a peak at the wire radius,
   1/r falloff outside) -- mhd.py's convention gave a Bx,By whose tangential
   projection washed out to a pure cos(2*theta) artifact (exactly what a
   flipped-handedness loop produces); this module's convention recovers
   the classical profile shape at a clean constant ratio across the whole
   range (see tst/substrate/test_curl.py).
"""

import numpy as np

_AXES = (0, 1, 2)


def _classify_sq_faces(pos, nb):
    """
    Determine, from real geometry, which of face slots 8-13 point in the
    +x/-x/+y/-y/+z/-z directions. Returns dict axis(0/1/2) -> (face_plus,
    face_minus). Uses the first interior cell found with all 6 sq
    neighbours valid; the lattice is homogeneous so one sample suffices.
    """
    for i in range(pos.shape[0]):
        sq = nb[i, 8:14]
        if np.all(sq >= 0):
            d = pos[sq] - pos[i]  # (6,3)
            slots = np.arange(8, 14)
            out = {}
            for axis in _AXES:
                comp = d[:, axis]
                dominant = np.abs(comp) > 0.5 * np.max(np.abs(d), axis=1)
                cand = slots[dominant & (np.abs(comp) > 1e-9)]
                cand_signs = comp[dominant & (np.abs(comp) > 1e-9)]
                plus = int(cand[cand_signs > 0][0])
                minus = int(cand[cand_signs < 0][0])
                out[axis] = (plus, minus)
            return out
    raise ValueError("no interior cell with all 6 sq-face neighbours found")


def _plaquette_curl_component(phi, nb, i, face_seq):
    """Same loop-walk as mhd._plaquette_curl: sum phi around a 4-cycle."""
    cells = [i]
    for k in face_seq[:-1]:
        j = int(nb[cells[-1], k])
        if j < 0:
            return None
        cells.append(j)
    if int(nb[cells[-1], face_seq[-1]]) != i:
        return None
    return sum(float(phi[cells[m], face_seq[m]]) for m in range(4))


def b_vec(lat, i, face_map=None):
    """
    B = curl(phi) at cell i, cyclic-order-correct on all three axes.
    face_map: precomputed _classify_sq_faces(lat.pos, lat.nb) output, to
    avoid recomputing it per call in a loop.
    """
    if face_map is None:
        face_map = _classify_sq_faces(lat.pos, lat.nb)
    b = []
    for axis in _AXES:
        u = (axis + 1) % 3  # next(axis)
        v = (axis + 2) % 3  # prev(axis)
        up, um = face_map[u]
        vp, vm = face_map[v]
        seq = [up, vp, um, vm]  # +u, +v, -u, -v -- u x v = axis, right-handed
        c = _plaquette_curl_component(lat.phi, lat.nb, i, seq)
        b.append(c)
    if any(c is None for c in b):
        return None
    return np.array(b)


def b_vec_vectorized(pos, nb, phi):
    """
    Vectorized B = curl(phi) for every cell at once. Returns (N,3); NaN
    rows where any of the 4 loop faces hits a boundary (nb[...] < 0).
    """
    face_map = _classify_sq_faces(pos, nb)
    N = phi.shape[0]
    B = np.full((N, 3), np.nan)
    idx0 = np.arange(N)
    for axis in _AXES:
        u = (axis + 1) % 3
        v = (axis + 2) % 3
        up, um = face_map[u]
        vp, vm = face_map[v]
        seq = [up, vp, um, vm]
        cells = [idx0]
        valid = np.ones(N, dtype=bool)
        for k in seq[:-1]:
            nxt = nb[cells[-1], k]
            valid &= nxt >= 0
            cells.append(np.where(nxt >= 0, nxt, 0))
        closes = nb[cells[-1], seq[-1]]
        valid &= closes == idx0
        total = sum(phi[cells[m], seq[m]] for m in range(4))
        B[:, axis] = np.where(valid, total, np.nan)
    return B
