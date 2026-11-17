"""Coordinate-free, genuinely curl-full Hopf-linked-pair spheromak IC —
2026-08-16, follow-on from equilibrium.substrate.discrete_curl. Unlike
equilibrium.substrate.dipole_mode_ic (phi = grad(U), curl-free by
construction, see project_dst_discrete_curl memory), this seeds real,
nonzero circulation directly on two Hopf-linked rings — no scalar potential
underneath, no Cartesian cross-product formula, only ring-index/face-index
assignment (same generative pattern as
tst.substrate.adaptive.adaptive_flux_tube.init_toroidal_winding, but with a
CONSTANT phi value around each ring rather than a cos(K*(idx+0.5)) standing
wave — the cos pattern sums to exactly zero over one period, so it is
curl-free in disguise; a constant value's sum is amplitude*N_ring != 0,
which cannot be any scalar's discrete derivative around a closed loop).

Two things this module adds beyond adaptive_flux_tube.py's existing
ring-finding:

1. `find_square_ring_on_leaves` — a level/subset-restricted generalisation
   of find_toroidal_ring's "square" mode (arbitrary `hops`, hence arbitrary
   physical radius), restricted to an explicit leaf subset (e.g. only
   level-1 leaves in an AMR mesh) via an optional boolean mask — needed
   because `find_toroidal_ring_at_level` only supports the minimal 8-tile
   ring, not the scaled "square" mode.
2. `find_linked_loop_anchored` — fixes a real bug found while building this:
   `adaptive_flux_tube.find_linked_loop` searches for Ring B near Ring A's
   MEAN CENTROID, which is fine for the small "minimal" 8-tile ring (whose
   centroid sits close to the tube) but is WRONG for a large "square" ring
   (whose mean centroid is the torus's central axis point, nowhere near the
   actual tube) — confirmed empirically: hops=41 (production-scale ring,
   radius~54) found Ring A but `find_linked_loop` failed to find Ring B at
   all. This version anchors the search at one of Ring A's OWN tile
   positions instead, and found a valid linking-number -1 loop immediately.

Ring B here is still a small (4-tile) test-current-loop scale poloidal
partner (matching the original project_spheromak_decay Ring A/Ring B
convention) — NOT a fully-scaled poloidal envelope wrapping the whole tube.
That is a real simplification, not hidden: a proper spheromak's poloidal
field wraps smoothly around the ENTIRE minor circumference of the torus;
here only one local linking point carries it. Good enough to test whether
step_cross can erode/reconnect real Hopf-linked circulation at production
scale; not a fully realistic spheromak field.
"""

import numpy as np

from tst.substrate.adaptive.adaptive_flux_tube import (
    _TOROIDAL_PATTERNS,
    _AXIS_DIR_CYCLE,
    _REF_DIR,
    _MIRROR,
    _linking_number_from_pos,
)


def _square_face_seq(axis, hops):
    d1, d2, d3, d4 = _AXIS_DIR_CYCLE[axis]
    seq = (
        [d1] * hops
        + [d2] * (2 * hops)
        + [d3] * (2 * hops)
        + [d4] * (2 * hops)
        + [d1] * hops
    )
    return np.array(seq, dtype=np.int32)


def find_square_ring_on_leaves(cl, axis="z", hops=1, leaf_mask=None, n_candidates=200):
    """Generalises adaptive_flux_tube.find_toroidal_ring's "square" mode to
    an explicit leaf subset (leaf_mask: boolean array over cl.leaf_pos rows,
    None = all leaves) -- needed for a ring confined to one AMR level.
    Returns (ring_js, ring_faces) or (None, None).
    """
    pos = np.asarray(cl.leaf_pos)
    nb_g = np.asarray(cl.leaf_nb)
    leaf_g = np.asarray(cl.leaf_global_indices)
    N = pos.shape[0]
    keep = np.ones(N, dtype=bool) if leaf_mask is None else np.asarray(leaf_mask)
    kept_rows = np.nonzero(keep)[0]
    g_to_j = {int(leaf_g[j]): j for j in kept_rows}

    a = None
    for j in kept_rows:
        for k in range(8, 14):
            g_nb = int(nb_g[j, k])
            j_nb = g_to_j.get(g_nb, -1)
            if j_nb >= 0:
                a = float(np.linalg.norm(pos[j] - pos[j_nb]))
                break
        if a is not None:
            break
    if a is None:
        return None, None

    face_seq = _square_face_seq(axis, hops)
    target = _REF_DIR[axis] * a * hops
    dists = np.linalg.norm(pos[kept_rows] - target, axis=1)
    order = kept_rows[np.argsort(dists)][:n_candidates]

    for start_j in order:
        start_j = int(start_j)
        tiles = [start_j]
        ok = True
        for f in face_seq[:-1]:
            g_nb = int(nb_g[tiles[-1], int(f)])
            j_nb = g_to_j.get(g_nb, -1)
            if j_nb < 0 or j_nb in tiles:
                ok = False
                break
            tiles.append(j_nb)
        if not ok:
            continue
        g_close = int(nb_g[tiles[-1], int(face_seq[-1])])
        if g_to_j.get(g_close, -1) == start_j:
            return tiles, [int(f) for f in face_seq]
    return None, None


def find_linked_loop_anchored(cl, ring_A_js, ring_A_faces, anchor_idx=0):
    """Same 4-tile Hopf-linked-plaquette search as
    adaptive_flux_tube.find_linked_loop, but anchored at one of ring_A's OWN
    tile positions (anchor_idx) instead of ring_A's mean centroid -- correct
    for large "square"-mode rings where the mean centroid is nowhere near
    the tube. See module docstring.
    """
    pos = np.asarray(cl.leaf_pos)
    nb_g = np.asarray(cl.leaf_nb)
    leaf_g = np.asarray(cl.leaf_global_indices)
    g_to_j = {int(g): j for j, g in enumerate(leaf_g)}

    A_pos = pos[ring_A_js]
    anchor = A_pos[anchor_idx].copy()
    plane_axis = int(np.argmin(np.var(A_pos, axis=0)))
    a = float(np.linalg.norm(A_pos[1] - A_pos[0]))
    h = a / 2.0

    base = {
        2: [[9, 13, 8, 12], [11, 13, 10, 12]],
        1: [[9, 11, 8, 10], [11, 13, 10, 12]],
        0: [[9, 11, 8, 10], [9, 13, 8, 12]],
    }[plane_axis]
    patterns = [b[i:] + b[:i] for b in base for i in range(len(b))]
    ring_A_set = set(ring_A_js)

    for sign in (-1, +1):
        target = anchor.copy()
        target[plane_axis] += sign * h
        dists = np.linalg.norm(pos - target, axis=1)
        for pattern in patterns:
            for start_j in np.argsort(dists)[:20]:
                start_j = int(start_j)
                tiles = [start_j]
                ok = True
                for f in pattern[:-1]:
                    g_nb = int(nb_g[tiles[-1], int(f)])
                    j_nb = g_to_j.get(g_nb, -1)
                    if j_nb < 0 or j_nb in tiles:
                        ok = False
                        break
                    tiles.append(j_nb)
                if not ok:
                    continue
                g_close = int(nb_g[tiles[-1], int(pattern[-1])])
                if g_to_j.get(g_close, -1) != start_j:
                    continue
                if set(tiles) & ring_A_set:
                    continue
                if abs(abs(_linking_number_from_pos(A_pos, pos[tiles])) - 1.0) < 0.1:
                    return tiles, pattern
    return None, None


def seed_constant_current(cl, ring_js, ring_faces, amplitude=1.0):
    """Genuine curl-full ring seeding: the SAME phi value all the way
    around the ring (sum = amplitude*N_ring != 0 -> not any scalar's
    discrete derivative), rather than init_toroidal_winding's
    amplitude*cos(K*(idx+0.5)) standing wave (sums to exactly zero -- see
    module docstring). Mutates cl in place.
    """
    N_ring = len(ring_js)
    phi = np.array(cl.leaf_phi)
    for idx in range(N_ring):
        j_i = ring_js[idx]
        j_nxt = ring_js[(idx + 1) % N_ring]
        k = ring_faces[idx]
        mk = int(_MIRROR[k])
        phi[k, j_i] = amplitude
        phi[mk, j_nxt] = -amplitude
    cl.set_leaf_phi(phi)


def seed_curl_full_pair(
    cl, *, hops, amplitude_A=1.0, amplitude_B=1.0, axis="z", leaf_mask=None
):
    """Find a Hopf-linked ring pair (Ring A: `hops`-scaled toroidal loop,
    Ring B: small 4-tile linked plaquette) and seed both with genuine,
    nonzero circulation. leaf_mask restricts ring-finding to a subset of
    leaves (e.g. one AMR level); None = all leaves (uniform mesh).

    Returns dict with ring_A_js/faces, ring_B_js/faces, or None if either
    ring could not be found (caller should fall back to a smaller/different
    hops or a different anchor).
    """
    A_js, A_faces = find_square_ring_on_leaves(
        cl, axis=axis, hops=hops, leaf_mask=leaf_mask
    )
    if A_js is None:
        return None
    B_js, B_faces = find_linked_loop_anchored(cl, A_js, A_faces, anchor_idx=0)
    if B_js is None:
        return None

    seed_constant_current(cl, A_js, A_faces, amplitude=amplitude_A)
    seed_constant_current(cl, B_js, B_faces, amplitude=amplitude_B)
    return {
        "ring_A_js": A_js,
        "ring_A_faces": A_faces,
        "ring_B_js": B_js,
        "ring_B_faces": B_faces,
    }
