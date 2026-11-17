"""
Toroidal ring flux IC and knot lam profile for AdaptiveCellCluster (C++ adaptive lattice).

Face convention matches bcc_geometry.hpp (OPPOSITE sign to the Python SubstrateLattice):
  Sq faces: 8=-x  9=+x  10=-y  11=+y  12=-z  13=+z

Use find_toroidal_ring + init_toroidal_winding to place a winding knot.
"""

import numpy as np

# ---------------------------------------------------------------------------
# BCC face geometry — matches bcc_geometry.hpp exactly
# hex_dir[k]: lexicographic order of {-1,+1}^3  (faces 0-7)
# sq_dir[k-8]: -x,+x,-y,+y,-z,+z               (faces 8-13)
# ---------------------------------------------------------------------------
_HEX_DIR = np.array(
    [
        [-1, -1, -1],
        [-1, -1, +1],
        [-1, +1, -1],
        [-1, +1, +1],
        [+1, -1, -1],
        [+1, -1, +1],
        [+1, +1, -1],
        [+1, +1, +1],
    ],
    dtype=np.float64,
)
_SQ_DIR = np.array(
    [
        [-1, 0, 0],
        [+1, 0, 0],
        [0, -1, 0],
        [0, +1, 0],
        [0, 0, -1],
        [0, 0, +1],
    ],
    dtype=np.float64,
)

# Unit normals (14, 3)
_FACE_NORM = np.vstack([_HEX_DIR / np.sqrt(3), _SQ_DIR])

# Hop vectors to face midpoint in DST coords (D1=1):
#   hex: |hop| = D1 = 1  →  hop = norm * 1
#   sq:  |hop| = 2/√3    →  hop = norm * 2/√3
_D_SQ = 2.0 / np.sqrt(3)
_FACE_HOP = np.vstack([_HEX_DIR / np.sqrt(3), _SQ_DIR * _D_SQ])  # (14, 3)

# mirror_face[k] — from adaptive_lattice.hpp
# hex: 0↔7, 1↔6, 2↔5, 3↔4
# sq:  8↔9, 10↔11, 12↔13
_MIRROR = np.array([7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 11, 10, 13, 12], dtype=np.int32)

# 8-tile toroidal ring patterns in C++ face convention:
#   8=-x  9=+x  10=-y  11=+y  12=-z  13=+z
# z-ring (xy-plane, starting near (+a,0,0)):
#   (+a,0,0) →+y→ (+a,+a,0) →-x→ (0,+a,0) →-x→ (-a,+a,0) →-y→
#   (-a,0,0) →-y→ (-a,-a,0) →+x→ (0,-a,0) →+x→ (+a,-a,0) →+y→ (+a,0,0)
_TOROIDAL_PATTERNS = {
    "z": np.array([11, 8, 8, 10, 10, 9, 9, 11], dtype=np.int32),
    "y": np.array([13, 8, 8, 12, 12, 9, 9, 13], dtype=np.int32),
    "x": np.array([13, 10, 10, 12, 12, 11, 11, 13], dtype=np.int32),
}

_REF_DIR = {
    "z": np.array([1.0, 0.0, 0.0]),
    "y": np.array([1.0, 0.0, 0.0]),
    "x": np.array([0.0, 1.0, 0.0]),
}

# Cyclic sq-face direction order used to walk a square loop around each axis,
# generalising _TOROIDAL_PATTERNS to an arbitrary number of hops per leg —
# derived directly from _TOROIDAL_PATTERNS itself (e.g. "z": [11,8,8,10,10,9,
# 9,11] -> cycle (11,8,10,9)), so hops=1 reproduces the existing minimal
# 8-tile loop's exact face sequence. Mirrors equilibrium/substrate/flux_tube.
# py's _AXIS_DIR_CYCLE, but NOT copied from it directly: that module's cycle
# traces the same loop shape in the OPPOSITE rotational sense (confirmed
# empirically 2026-07-14 — sq-face numbering 8=-x..13=+z is actually IDENTICAL
# between the uniform and adaptive backends, despite this module's older
# docstring claiming otherwise; the two _TOROIDAL_PATTERNS dicts just trace
# opposite winding directions for historical reasons).
_AXIS_DIR_CYCLE = {
    "z": (11, 8, 10, 9),  # +y, -x, -y, +x
    "y": (13, 8, 12, 9),  # +z, -x, -z, +x
    "x": (13, 10, 12, 11),  # +z, -y, -z, +y
}


def _square_face_seq(axis: str, hops: int) -> np.ndarray:
    """
    Face sequence for a square loop of `hops` sq-face hops per half-leg, same
    4-fold direction cycle as the fixed 8-tile _TOROIDAL_PATTERNS. `hops=1`
    reproduces the minimal loop exactly (note: that loop is sometimes called
    "the minimal octagon" elsewhere for having 8 TILES -- it is topologically
    a SQUARE, 4 straight sides with sharp 90-degree corners, not an 8-sided/
    8-direction shape; don't confuse with _octagon_face_seq below, which is
    a genuine 8-direction construction). Total tiles = 8*hops.
    """
    d1, d2, d3, d4 = _AXIS_DIR_CYCLE[axis]
    seq = (
        [d1] * hops
        + [d2] * (2 * hops)
        + [d3] * (2 * hops)
        + [d4] * (2 * hops)
        + [d1] * hops
    )
    return np.array(seq, dtype=np.int32)


def _octagon_face_seq(axis: str, hops: int) -> np.ndarray:
    """
    Face sequence for a genuinely 8-DIRECTION loop -- a square with each
    corner chamfered by a diagonal "staircase" leg (alternating hops between
    the two adjacent cardinal sq-face directions), a rounder approximation
    to a circle than `_square_face_seq`'s 4-direction square (see
    doc/dst/failures/... square-ring-near-field-not-axisymmetric -- built
    2026-09-08 after realising `find_toroidal_ring(mode="square")` is a
    literal square, permanently sharp-cornered at any `hops`/resolution,
    which structurally cannot recover a smooth axisymmetric near field no
    matter how well-resolved it is).

    `hops` sets both the straight-leg half-length and the chamfer length
    (roughly regular-octagon proportions, not geometrically exact). Net
    displacement is exactly zero by construction (verified both
    algebraically and against the real lattice in
    tst/substrate/adaptive/test_octagon_ring.py) -- hops=1 is the smallest
    valid size. Total tiles = 16*hops.
    """
    d1, d2, d3, d4 = _AXIS_DIR_CYCLE[axis]

    def chamfer(da, db):
        return [da, db] * hops

    seq = (
        [d1] * hops
        + chamfer(d1, d2)
        + [d2] * (2 * hops)
        + chamfer(d2, d3)
        + [d3] * (2 * hops)
        + chamfer(d3, d4)
        + [d4] * (2 * hops)
        + chamfer(d4, d1)
        + [d1] * hops
    )
    return np.array(seq, dtype=np.int32)


def winding_number(cluster, ring_js: list, axis: str = "z") -> float:
    """
    Discrete winding number of the ring around its own local axis (parallel
    to `axis`, through the ring's own centroid — not the lattice's coordinate
    origin; see project_spheromak_decay's phi_ring_circulation origin-vs-
    centroid note, same pitfall). Returns ±1.0 for a single winding.
    """
    ax = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[axis]
    pos = np.array(cluster.leaf_pos)
    centroid = pos[ring_js].mean(axis=0)[list(ax)]
    n = len(ring_js)
    total = 0.0
    for j in range(n):
        p = pos[ring_js[j], list(ax)] - centroid
        q = pos[ring_js[(j + 1) % n], list(ax)] - centroid
        total += np.arctan2(p[0] * q[1] - p[1] * q[0], p[0] * q[0] + p[1] * q[1])
    return total / (2.0 * np.pi)


def _sq_hop_dist(pos, nb_g):
    """Estimate sq-face hop distance from first valid sq-face neighbour pair."""
    N = pos.shape[0]
    for j in range(min(N, 20)):
        for k in range(8, 14):
            g = int(nb_g[j, k])
            if g >= 0:
                return float(np.linalg.norm(pos[j] - pos[j]))  # placeholder
    return None


_FACE_3D_DIR = {
    8: np.array([-1.0, 0.0, 0.0]),
    9: np.array([1.0, 0.0, 0.0]),
    10: np.array([0.0, -1.0, 0.0]),
    11: np.array([0.0, 1.0, 0.0]),
    12: np.array([0.0, 0.0, -1.0]),
    13: np.array([0.0, 0.0, 1.0]),
}


def _face_seq_centroid_offset(axis: str, face_seq) -> np.ndarray:
    """
    3D offset of a face-sequence loop's own centroid relative to its start
    tile -- found 2026-09-08 after discovering find_toroidal_ring's search
    `target` (meant to land near the loop's centroid, not just anywhere on
    it) only does so BY COINCIDENCE for mode="square" (whose particular
    face_seq happens to start at its own centroid-symmetric point); for
    mode="octagon" the true centroid sits at 2x that offset (confirmed
    exactly: -hops for square, -2*hops for octagon, in sq-hop-distance
    units, across hops=1/2/4/8/12), landing the ring off-origin with a
    real physical consequence -- an off-center ring in a domain built
    around the coordinate origin is asymmetrically close to one side of
    the sponge/boundary, not just a plotting inconvenience.

    Computed once, generically, from the face_seq's own direction sequence
    (2D d1..d4 basis matching _AXIS_DIR_CYCLE[axis], no lattice needed) --
    works for any current or future mode without per-mode special-casing.
    """
    d1, d2, d3, d4 = _AXIS_DIR_CYCLE[axis]
    unit2d = {
        d1: np.array([0.0, 1.0]),
        d2: np.array([-1.0, 0.0]),
        d3: np.array([0.0, -1.0]),
        d4: np.array([1.0, 0.0]),
    }
    pts = [np.array([0.0, 0.0])]
    for f in face_seq[:-1]:
        pts.append(pts[-1] + unit2d[int(f)])
    offset_2d = np.array(pts).mean(axis=0)  # (x=d2/d4-axis component, y=d1/d3-axis component)
    return offset_2d[1] * _FACE_3D_DIR[int(d1)] + offset_2d[0] * _FACE_3D_DIR[int(d4)]


def find_toroidal_ring(cluster, axis: str = "z", mode: str = "minimal", **kwargs):
    """
    Find a closed loop (leaf-j indices) winding once around `axis`.

    mode="minimal" (default, backward-compatible): the fixed 8-tile loop
        (_TOROIDAL_PATTERNS), one sq-face hop per leg — ~2.8 cells across,
        inside the BCC dispersion anisotropy band
        (project_bcc_dispersion_anisotropy). Prefer mode="square" with
        hops>=3-4 for anything measuring real physics, not just geometry.
    mode="square": a scaled-up loop, `hops` (kwarg, int >= 1) sq-face hops
        per half-leg — total tiles = 8*hops, physical radius grows
        ~linearly with hops. hops=1 reproduces the minimal loop's exact
        face sequence. Optional kwarg `n_candidates` (default 20 for
        minimal, 100 for square) bounds how many start-tile candidates are
        tried before giving up.

    Returns (ring_js, ring_faces) or (None, None).
    The returned ring_js are indices into the leaf arrays (leaf_u, leaf_pos, etc.).
    """
    pos = cluster.leaf_pos  # (N, 3)
    nb_g = cluster.leaf_nb  # (N, 14) global g indices
    leaf_g = cluster.leaf_global_indices  # (N,) global g
    g_to_j = {int(g): j for j, g in enumerate(leaf_g)}

    # Estimate sq hop distance from first available sq-face pair.
    a = None
    N = pos.shape[0]
    for j in range(N):
        for k in range(8, 14):
            nb = int(nb_g[j, k])
            if nb >= 0 and nb in g_to_j:
                j_nb = g_to_j[nb]
                a = float(np.linalg.norm(pos[j] - pos[j_nb]))
                break
        if a is not None:
            break
    if a is None:
        return None, None

    if mode == "minimal":
        face_seq = _TOROIDAL_PATTERNS[axis]
        scale = 1
    elif mode == "square":
        hops = int(kwargs.get("hops", 1))
        if hops < 1:
            raise ValueError("hops must be >= 1")
        face_seq = _square_face_seq(axis, hops)
        scale = hops
    elif mode == "octagon":
        hops = int(kwargs.get("hops", 1))
        if hops < 1:
            raise ValueError("hops must be >= 1")
        face_seq = _octagon_face_seq(axis, hops)
        scale = hops
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # target = -a * centroid_offset so the resulting ring's own centroid
    # lands at the coordinate origin (not just its start tile) -- for
    # mode="minimal" (no face_seq-based centroid_offset available/needed,
    # its fixed 8-tile pattern is already known to center correctly) fall
    # back to the historical target formula.
    if mode == "minimal":
        target = _REF_DIR[axis] * a * scale
    else:
        target = -a * _face_seq_centroid_offset(axis, face_seq)
    n_candidates = int(kwargs.get("n_candidates", 20 if mode == "minimal" else 100))
    dists = np.linalg.norm(pos - target, axis=1)

    for start_j in np.argsort(dists)[:n_candidates]:
        start_j = int(start_j)
        tiles = [start_j]
        ok = True
        for f in face_seq[:-1]:
            cur_j = tiles[-1]
            g_nb = int(nb_g[cur_j, int(f)])
            if g_nb < 0:
                ok = False
                break
            j_nb = g_to_j.get(g_nb, -1)
            if j_nb < 0 or j_nb in tiles:
                ok = False
                break
            tiles.append(j_nb)
        if not ok:
            continue
        # Check closure
        last_j = tiles[-1]
        g_close = int(nb_g[last_j, int(face_seq[-1])])
        if g_to_j.get(g_close, -1) == start_j:
            return tiles, [int(f) for f in face_seq]

    return None, None


def find_toroidal_ring_at_level(cluster, level, axis="z"):
    """
    Find an 8-tile ring among cells at exactly `level` (after AMR splitting).

    Same logic as find_toroidal_ring but restricts the neighbour graph to
    cells at the requested refinement level.  Returns (ring_js, ring_faces)
    or (None, None).
    """
    lev_arr = np.array(cluster.leaf_level)  # (N,) uint8
    pos = cluster.leaf_pos  # (N, 3)
    nb_g = cluster.leaf_nb  # (N, 14)
    leaf_g = cluster.leaf_global_indices  # (N,)

    mask = lev_arr == level
    # build g→j dict restricted to target level
    g_to_j_lev = {int(leaf_g[j]): j for j in np.where(mask)[0]}

    # estimate sq-hop distance among level-L cells
    a = None
    for j in np.where(mask)[0]:
        for k in range(8, 14):
            g_nb = int(nb_g[j, k])
            j_nb = g_to_j_lev.get(g_nb, -1)
            if j_nb >= 0:
                a = float(np.linalg.norm(pos[j] - pos[j_nb]))
                break
        if a is not None:
            break
    if a is None:
        return None, None

    face_seq = _TOROIDAL_PATTERNS[axis]
    level_js = np.where(mask)[0]

    # Hex-splits place children at diagonal positions, so the geometric
    # heuristic [a,0,0] rarely lands on a real cell.  Try every level-L
    # cell as a ring start — the ring is sparse so this is fast.
    for start_j in level_js:
        start_j = int(start_j)
        tiles = [start_j]
        ok = True
        for f in face_seq[:-1]:
            g_nb = int(nb_g[tiles[-1], int(f)])
            j_nb = g_to_j_lev.get(g_nb, -1)
            if j_nb < 0 or j_nb in tiles:
                ok = False
                break
            tiles.append(j_nb)
        if not ok:
            continue
        g_close = int(nb_g[tiles[-1], int(face_seq[-1])])
        if g_to_j_lev.get(g_close, -1) == start_j:
            return tiles, [int(f) for f in face_seq]
    return None, None


def set_knot(cluster, origin_j, lam_target, profile="1/r"):
    """
    BFS lam profile centred on leaf origin_j.

    lam(r) = lam_target / r  (profile='1/r',  default — Coulomb/gravity)
    lam(r) = lam_target / r² (profile='1/r2' — steeper, for comparison)

    Origin cell is set to lam_target directly.
    """
    from collections import deque

    from equilibrium.substrate.discrete import LAM_MAX

    nb_g = cluster.leaf_nb  # (N, 14) global indices
    leaf_g = cluster.leaf_global_indices  # (N,) global g
    g_to_j = {int(g): j for j, g in enumerate(leaf_g)}

    lam = np.array(cluster.leaf_lam)
    lam[origin_j] = lam_target

    visited = {origin_j: 0}
    queue = deque([(origin_j, 0)])

    while queue:
        j, r = queue.popleft()
        for k in range(14):
            g_nb = int(nb_g[j, k])
            if g_nb < 0:
                continue
            j_nb = g_to_j.get(g_nb, -1)
            if j_nb < 0 or j_nb in visited:
                continue
            r_nb = r + 1
            visited[j_nb] = r_nb
            denom = r_nb if profile == "1/r" else r_nb * r_nb
            lam[j_nb] = min(lam_target / denom, LAM_MAX)
            queue.append((j_nb, r_nb))

    cluster.set_leaf_lam(lam)


def set_u_background(cluster, origin_j, u_origin=1.0, profile="1/r"):
    """
    Analytically impose steady-state u background via BFS 1/r from origin.

    Matches the equilibrium solution of the discrete diffusion equation with a
    fixed-u source at origin and absorbing BC at the sphere boundary.
    Origin cell is set to u_origin; all others follow u ~ u_origin / r_hop.
    """
    from collections import deque

    nb_g = cluster.leaf_nb
    leaf_g = cluster.leaf_global_indices
    g_to_j = {int(g): j for j, g in enumerate(leaf_g)}

    u = cluster.leaf_u.copy()
    u[origin_j] = u_origin

    visited = {origin_j: 0}
    queue = deque([(origin_j, 0)])
    while queue:
        j, r = queue.popleft()
        for k in range(14):
            g_nb = int(nb_g[j, k])
            if g_nb < 0:
                continue
            j_nb = g_to_j.get(g_nb, -1)
            if j_nb < 0 or j_nb in visited:
                continue
            r_nb = r + 1
            visited[j_nb] = r_nb
            denom = r_nb if profile == "1/r" else r_nb * r_nb
            u[j_nb] = u_origin / denom
            queue.append((j_nb, r_nb))

    cluster.set_leaf_u(u)


def init_dipole_ic(
    cluster, npz_path, *, scale=3.0, lam_target=9.0, u_max=1.0, cx=0.0, cy=0.0, cz=0.0
):
    """
    Set u and lam from a PHARE relaxed-dipole B-magnitude field (ic_b.npz).

    The npz contains the resistively-relaxed state of a spheromak run — the
    toroidal winding has dissipated, leaving a predominantly axial (dipole)
    field.  Only |B| is used; no phi winding is set.

    scale    : DST units per npz unit.  Dipole radius ≈ 0.5 npz units,
               so scale=3 maps the core to r≈1.5 DST units.
    cx,cy,cz : centre in DST coordinates (default: origin).

    Returns B_max.
    """
    from scipy.interpolate import RegularGridInterpolator

    from equilibrium.substrate.discrete import LAM_MAX

    d = np.load(npz_path)
    b_mag = np.sqrt(d["bx"] ** 2 + d["by"] ** 2 + d["bz"] ** 2)
    b_max = float(b_mag.max())

    interp = RegularGridInterpolator(
        (d["x"], d["y"], d["z"]),
        b_mag,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    pos = np.array(cluster.leaf_pos)
    pts = (pos - np.array([cx, cy, cz])) / scale
    b_vals = interp(pts)

    lam = np.minimum(lam_target * b_vals / b_max, LAM_MAX)
    cluster.set_leaf_lam(lam)
    cluster.set_leaf_u(u_max * b_vals / b_max)

    return b_max


def init_phare_ic(
    cluster,
    npz_path,
    *,
    scale=20.0,
    lam_target=9.0,
    u_max=1.0,
    amplitude=1.0,
    cx=0.0,
    cy=0.0,
    cz=0.0,
):
    """
    Initialise u, lam, and phi from the PHARE relaxed-spheromak B field (ic_b.npz).

    Unlike init_dipole_ic (which only uses |B|), this projects the full vector
    field onto BCC face normals: phi[k] = B(midpoint_k) · n̂_k.

    scale     : DST units per npz unit.  The npz domain spans ±5.05 npz units;
                scale=20 maps that to ±101 DST units, matching R=100.
    amplitude : phi is normalised to [-1,1] then multiplied by this.
    """
    from scipy.interpolate import RegularGridInterpolator

    from equilibrium.substrate.discrete import LAM_MAX

    d = np.load(npz_path)
    centre = np.array([cx, cy, cz])

    def _interp(arr):
        return RegularGridInterpolator(
            (d["x"], d["y"], d["z"]),
            arr,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    ib = [_interp(d["bx"]), _interp(d["by"]), _interp(d["bz"])]

    pos = np.array(cluster.leaf_pos)  # (N, 3)

    # u and lam from |B| at cell centres
    pts_c = (pos - centre) / scale
    B_c = np.stack([ib[i](pts_c) for i in range(3)], axis=1)  # (N, 3)
    b_mag = np.linalg.norm(B_c, axis=1)
    b_max = float(b_mag.max()) or 1.0
    cluster.set_leaf_lam(np.minimum(lam_target * b_mag / b_max, LAM_MAX))
    cluster.set_leaf_u(u_max * b_mag / b_max)

    # phi starts at zero — let the DST update rule generate it naturally from
    # the u/lam field via pass 2.  Projecting the continuous PHARE B field
    # onto discrete face normals introduces large divergence (ratio ~3.7) that
    # cannot be cleaned cleanly post-hoc; zero-phi avoids that entirely.

    return b_max


def init_spheromak_ic(
    cluster,
    *,
    R_sph=1.0,
    scale=1.5,
    B0=1.0,
    lam_target=0.1,
    u_max=1.0,
    amplitude=0.0,
    cx=0.0,
    cy=0.0,
    cz=0.0,
    sigma=None,
):
    """
    Set u, lam, and phi from the analytical Taylor spheromak (force-free, ∇×B = kB).

    The full B vector is mapped onto the lattice:
      |B| at cell centre  → lam and u
      B·n̂_k at midpoint  → phi_k  (directed face flux — carries the winding)

    The winding comes from B_phi ≠ 0 in the force-free solution: each closed
    loop enclosing the knot axis has non-zero phi circulation.  phi is not
    imposed freely — it is derived from the same field that sets lam/u.

    amplitude=0 (default): phi left at zero; circulation builds from pressure
    dynamics.  Set amplitude>0 to pre-load phi from the B-field projection.

    R_sph     : spheromak radius in spheromak coordinates (default 1.0)
    scale     : DST units per spheromak unit; knot boundary at scale*R_sph DST units
    B0        : field amplitude
    amplitude : phi normalisation — max |phi| set to this value (0 = no pre-load)
    sigma     : if given, replace hard r<=R_sph cutoff with Gaussian envelope
                exp(-0.5*(r/sigma)^2) in spheromak units; avoids sharp boundary
                discontinuities that create large discrete div(phi)
    cx,cy,cz  : centre in DST coordinates (default: origin)

    Returns B_max.
    """
    from scipy.special import spherical_jn

    from equilibrium.substrate.discrete import LAM_MAX

    centre = np.array([cx, cy, cz])

    def _B(pos):
        p = (pos - centre) / scale
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        B = np.zeros_like(p)
        mask = (r <= R_sph) & (r > 0)  # hard outer cut at Taylor-sphere boundary
        k_ = 4.493 / R_sph
        rm = r[mask]
        cf = B0 * spherical_jn(1, k_ * rm) / rm
        if sigma is not None:
            cf *= np.exp(-0.5 * (rm / sigma) ** 2)
        B[mask, 0] = cf * (z[mask] * x[mask] / rm - y[mask])
        B[mask, 1] = cf * (z[mask] * y[mask] / rm + x[mask])
        B[mask, 2] = cf * (-(x[mask] ** 2 + y[mask] ** 2) / rm)
        return B

    pos = np.array(cluster.leaf_pos)  # (N, 3)
    N = pos.shape[0]

    # u and lam from |B| at cell centres
    B_c = _B(pos)
    b_mag = np.linalg.norm(B_c, axis=1)
    b_max = float(b_mag.max()) or 1.0
    lam = np.minimum(lam_target * b_mag / b_max, LAM_MAX)
    cluster.set_leaf_lam(lam)
    cluster.set_leaf_u(u_max * b_mag / b_max)

    r_cell = np.linalg.norm((pos - centre) / scale, axis=1)
    n_inner = int((r_cell <= R_sph).sum())
    print(
        f"  init_spheromak_ic: scale={scale:.1f}  R_sph={R_sph}  sigma={sigma}"
        f"  inner_cells={n_inner}/{N}  ({100*n_inner/N:.1f}%)",
        flush=True,
    )

    if amplitude > 0.0:
        # phi from B·n̂_k at face midpoints — encodes the toroidal winding
        midpoints = pos[:, None, :] + 0.5 * _FACE_HOP[None, :, :]  # (N, 14, 3)
        B_mid = _B(midpoints.reshape(-1, 3)).reshape(N, 14, 3)
        phi = np.einsum("jkd,kd->kj", B_mid, _FACE_NORM)  # (14, N)
        phi_max = float(np.abs(phi).max()) or 1.0
        phi *= amplitude / phi_max
        # set_leaf_phi enforces face antisymmetry: phi[k,j]=val causes the
        # C++ to set phi[mirror_k, neighbor]=-val.  To prevent propagation
        # outside R_sph every boundary-crossing face must be zeroed on BOTH
        # sides — inner outward face AND outer inward face — so the pair
        # presents (0, 0), which is already antisymmetric and needs no fix-up.
        nbr_r = np.linalg.norm(
            (pos[:, None, :] + _FACE_HOP[None, :, :] - centre) / scale, axis=2
        )  # (N, 14) neighbour radii in spheromak units
        phi[(nbr_r > R_sph).T] = 0.0  # zero inner cells' outward faces
        phi[:, r_cell > R_sph] = 0.0  # zero all outer cell faces (both sides clean)

        # Pre-store sanity: verify zeroing worked in Python
        n_py_outer = int(
            np.count_nonzero(np.abs(phi[:, r_cell > R_sph]).max(axis=0) > 1e-15)
        )
        n_phi = int(np.count_nonzero(np.abs(phi).max(axis=0) > 1e-15))
        print(
            f"  [pre-store] phi_nonzero={n_phi}  outer_nonzero_python={n_py_outer}"
            f"  inner={n_inner}  N={N}",
            flush=True,
        )

        cluster.set_leaf_phi(np.ascontiguousarray(phi))

        # Post-store readback
        phi_rb = np.array(cluster.leaf_phi)
        n_outer = int(
            np.count_nonzero(np.abs(phi_rb[:, r_cell > R_sph]).max(axis=0) > 1e-15)
        )
        r_phi_max = (
            float(r_cell[np.abs(phi_rb).max(axis=0) > 1e-15].max())
            if n_phi > 0
            else 0.0
        )
        print(
            f"  [post-store] outer_nonzero={n_outer}  max_r_with_phi={r_phi_max:.3f} sph-u",
            flush=True,
        )

    return b_max


def init_electron_dipole_ic(
    cluster,
    *,
    R_sph=1.0,
    scale=4.0,
    B0=1.0,
    lam_target=0.0,
    u_bg=0.0,
    amplitude=1.0,
    cx=0.0,
    cy=0.0,
    cz=0.0,
    sigma=None,
    div_clean=True,
    clean_rtol=1e-8,
):
    """
    Set u, lam, and phi from the poloidal component of the Taylor state —
    the double-ring dipole (candidate electron topology).

    The toroidal (azimuthal) term of the Taylor solution is dropped; what
    remains is the axisymmetric poloidal winding: nested tori of phi
    circulation around a centreline circle at rho ≈ 0.63*R_sph in the
    midplane.  A meridional (XZ) slice shows two counter-circulating
    lobes — the "double ring".  This is the same structure the spheromak
    decays into once its toroidal winding unwinds (the dipole window), so
    the electron IC is the neutron's surviving component at smaller scale.

    Zero-net-force construction: u is set UNIFORM (u_bg — only Δu drives
    phi, so any constant background works), and phi is Hodge-projected
    against the step's exact W-weighted discrete divergence
    (divergence_clean).  The result is an exact fixed point of the
    reversible dynamics — no shed transient, no relaxation phase needed.
    Any subsequent evolution is purely the irreversible operators
    (step_cross etc.).  The cleaning also restores the external return-flux
    tail that the hard cutoff at R_sph removes, so the seam needs no
    dual-zero handling.

    lam_target defaults to 0.0 — whether the electron carries lam at all
    is an open question; pure phi winding is the null hypothesis.  A
    lam ∝ |B| profile does not break stationarity (σ multiplies terms
    that are already zero).

    Sizing caution: scale below ~3 cells approaches the 8-tile regime where
    the BCC k·L dispersion anisotropy (diamond artifact) dominates.

    KNOWN ISSUES (found 2026-09-16, not yet fixed here):
    (1) This phi is a B.n_hat (flux-through-face) sample, not an A.dl
    (circulation) sample -- equilibrium.substrate.curl.b_vec_vectorized's
    curl(phi)=B machinery expects the latter, so any magnetic-moment
    measurement via curl(phi) on THIS phi reads back a meaningless
    (measured: exactly zero) result. See init_electron_dipole_ic_vecpot
    below for an A.dl-seeded alternative, and
    project_dipole_moment_phi_convention_gap memory.
    (2) k_=4.493/R_sph (j1's first zero) is the self-confined force-free
    (Chandrasekhar-Kendall/Taylor-state) eigenvalue -- inherited from the
    neutron/spheromak lineage, not independently justified for a charged,
    field-radiating electron (doc/dst.md: proton is an OPEN knot, neutron
    is CLOSED). This IC structurally cannot source an external field at
    this k regardless of (1). See project_dipole_moment_phi_convention_gap
    memory's 2026-09-16 update.

    Returns (B_max, r_axis) — r_axis is the centreline circle radius in
    DST units.
    """
    from scipy.special import spherical_jn

    from equilibrium.substrate.discrete import LAM_MAX

    centre = np.array([cx, cy, cz])
    k_ = 4.493 / R_sph

    def _B(pos):
        p = (pos - centre) / scale
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        B = np.zeros_like(p)
        mask = (r <= R_sph) & (r > 0)
        rm = r[mask]
        cf = B0 * spherical_jn(1, k_ * rm) / rm
        if sigma is not None:
            cf *= np.exp(-0.5 * (rm / sigma) ** 2)
        # poloidal part only — the toroidal (-y, +x) term is dropped
        B[mask, 0] = cf * (z[mask] * x[mask] / rm)
        B[mask, 1] = cf * (z[mask] * y[mask] / rm)
        B[mask, 2] = cf * (-(x[mask] ** 2 + y[mask] ** 2) / rm)
        return B

    # centreline circle: maximum of the flux function psi(rho) = rho*j1(k*rho)
    rho_g = np.linspace(1e-4, R_sph, 2048)
    psi = rho_g * spherical_jn(1, k_ * rho_g)
    r_axis = float(rho_g[np.argmax(np.abs(psi))]) * scale

    pos = np.array(cluster.leaf_pos)  # (N, 3)
    N = pos.shape[0]

    # lam from |B| at cell centres; u strictly uniform (zero net force)
    B_c = _B(pos)
    b_mag = np.linalg.norm(B_c, axis=1)
    b_max = float(b_mag.max()) or 1.0
    cluster.set_leaf_lam(np.minimum(lam_target * b_mag / b_max, LAM_MAX))
    cluster.set_leaf_u(np.full(N, u_bg, dtype=np.float64))

    r_cell = np.linalg.norm((pos - centre) / scale, axis=1)
    n_inner = int((r_cell <= R_sph).sum())
    print(
        f"  init_electron_dipole_ic: scale={scale:.1f}  R_sph={R_sph}"
        f"  sigma={sigma}  r_axis={r_axis:.2f}  u_bg={u_bg}"
        f"  inner_cells={n_inner}/{N}  ({100*n_inner/N:.1f}%)",
        flush=True,
    )

    if amplitude > 0.0:
        midpoints = pos[:, None, :] + 0.5 * _FACE_HOP[None, :, :]  # (N, 14, 3)
        B_mid = _B(midpoints.reshape(-1, 3)).reshape(N, 14, 3)
        phi = np.einsum("jkd,kd->kj", B_mid, _FACE_NORM)  # (14, N)
        phi_max = float(np.abs(phi).max()) or 1.0
        phi *= amplitude / phi_max
        if div_clean:
            phi, div0, div1 = divergence_clean(cluster, phi, rtol=clean_rtol)
            print(
                f"  divergence_clean: div_rms {div0:.4e} → {div1:.4e}"
                f"  (×{div1 / div0 if div0 else 0:.2e})",
                flush=True,
            )
        else:
            # dual-zero boundary handling — same requirement as
            # init_spheromak_ic: every boundary-crossing face pair must
            # present (0, 0) on both sides.
            nbr_r = np.linalg.norm(
                (pos[:, None, :] + _FACE_HOP[None, :, :] - centre) / scale, axis=2
            )  # (N, 14)
            phi[(nbr_r > R_sph).T] = 0.0
            phi[:, r_cell > R_sph] = 0.0
        cluster.set_leaf_phi(np.ascontiguousarray(phi))

    return b_max, r_axis


def init_electron_dipole_ic_vecpot(
    cluster,
    *,
    R_sph=1.0,
    scale=4.0,
    B0=1.0,
    lam_target=0.0,
    u_bg=0.0,
    amplitude=1.0,
    cx=0.0,
    cy=0.0,
    cz=0.0,
    div_clean=True,
    clean_rtol=1e-8,
):
    """
    Same double-ring/poloidal-Taylor electron topology as
    init_electron_dipole_ic, but seeds phi from the vector potential A
    (dotted along each face's actual discrete edge displacement, a
    Wilson-line-style circulation sample -- A.dl) rather than from B
    dotted with the face NORMAL (a flux sample, B.n_hat).

    Why this exists (see project_dipole_moment_phi_convention_gap memory):
    equilibrium.substrate.curl.b_vec_vectorized's curl(phi)=B reconstruction
    was validated (probe_curl_loop_dipole.py, test_curl.py) assuming phi IS
    an A.dl circulation sample -- init_electron_dipole_ic's B.n_hat phi is a
    different physical quantity and reads back a meaningless (measured:
    exactly zero) result under that machinery. Also: init_electron_dipole_
    ic's raw B formula turns out to have an exact structural defect
    independent of that convention question -- B.r_hat (the radial
    component relative to the object's own centre) is IDENTICALLY zero
    there, for every position, not just on-axis (verified algebraically:
    every Cartesian component is proportional to x, y, or x^2+y^2, and the
    r_hat-projection of that cancels exactly) -- a genuinely degenerate,
    non-dipole-shaped field, not merely a labeling mismatch.

    A_phi(r, theta) = B0 * j1(k*r) * sin(theta) is the standard
    axisymmetric vector-potential ansatz for this Chandrasekhar-Kendall/
    Bessel poloidal mode (consistent with the existing centreline-circle
    convention: rho*A_phi(rho, pi/2) = rho*j1(k*rho), matching this
    module's own psi(rho)=rho*j1(k*rho) flux function). In Cartesian,
    A_phi*phi_hat reduces to the clean form A = cf(r)*(-y, x, 0), cf(r) =
    B0*j1(k*r)/r -- the SAME radial profile init_electron_dipole_ic already
    uses, just applied to the (-y,x,0) direction (a vector POTENTIAL role)
    instead of to the poloidal Cartesian pattern (a vector FIELD role) --
    consistent with apply_toroidal_rotation's B_tor formula elsewhere in
    this module using that same (-y,x,0) direction, suggesting this
    potential/field role mixup is a recurring pattern worth checking
    elsewhere too.

    No continuum derivative of A is taken anywhere (per feedback: only a
    discrete curl on discretely-sampled phi counts as real rotational
    content here, see project_dst_discrete_curl). A itself is sampled
    (algebraic, no derivatives) at face midpoints and dotted with the
    lattice's own discrete edge displacement (_FACE_HOP) to build phi --
    the same Wilson-line-style construction already validated in
    probe_curl_loop_dipole.py. If lam_target>0, the B used to shape lam is
    obtained from the DISCRETE curl of the just-built (and cleaned) phi
    (equilibrium.substrate.curl.b_vec_vectorized), never from an analytic
    formula -- lam_target=0 (the default, matching every current caller)
    skips this entirely since it's multiplied by zero regardless.

    Returns (b_max, r_axis) -- b_max is 0.0 when lam_target<=0 (not
    computed, since unused); r_axis is the centreline circle radius,
    identical convention/value to init_electron_dipole_ic.
    """
    from scipy.special import spherical_jn

    from equilibrium.substrate.discrete import LAM_MAX

    centre = np.array([cx, cy, cz])
    k_ = 4.493 / R_sph

    def _A(pos):
        p = (pos - centre) / scale
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        A = np.zeros_like(p)
        mask = (r <= R_sph) & (r > 0)
        rm = r[mask]
        cf = B0 * spherical_jn(1, k_ * rm) / rm
        A[mask, 0] = -cf * y[mask]
        A[mask, 1] = cf * x[mask]
        return A

    # centreline circle: maximum of the flux function psi(rho) = rho*j1(k*rho)
    # (unchanged from init_electron_dipole_ic -- same psi, same convention:
    # rho*A_phi(rho, pi/2) = rho*j1(k*rho) exactly)
    rho_g = np.linspace(1e-4, R_sph, 2048)
    psi = rho_g * spherical_jn(1, k_ * rho_g)
    r_axis = float(rho_g[np.argmax(np.abs(psi))]) * scale

    pos = np.array(cluster.leaf_pos)  # (N, 3)
    N = pos.shape[0]

    cluster.set_leaf_lam(np.zeros(N))  # replaced below if lam_target > 0
    cluster.set_leaf_u(np.full(N, u_bg, dtype=np.float64))

    r_cell = np.linalg.norm((pos - centre) / scale, axis=1)
    n_inner = int((r_cell <= R_sph).sum())
    print(
        f"  init_electron_dipole_ic_vecpot: scale={scale:.1f}  R_sph={R_sph}"
        f"  r_axis={r_axis:.2f}  u_bg={u_bg}"
        f"  inner_cells={n_inner}/{N}  ({100*n_inner/N:.1f}%)",
        flush=True,
    )

    b_max = 0.0
    if amplitude > 0.0:
        midpoints = pos[:, None, :] + 0.5 * _FACE_HOP[None, :, :]  # (N, 14, 3)
        A_mid = _A(midpoints.reshape(-1, 3)).reshape(N, 14, 3)
        phi = np.einsum("jkd,kd->kj", A_mid, _FACE_HOP)  # (14, N) -- A.dl
        phi_max = float(np.abs(phi).max()) or 1.0
        phi *= amplitude / phi_max
        if div_clean:
            phi, div0, div1 = divergence_clean(cluster, phi, rtol=clean_rtol)
            print(
                f"  divergence_clean: div_rms {div0:.4e} → {div1:.4e}"
                f"  (×{div1 / div0 if div0 else 0:.2e})",
                flush=True,
            )
        else:
            nbr_r = np.linalg.norm(
                (pos[:, None, :] + _FACE_HOP[None, :, :] - centre) / scale, axis=2
            )  # (N, 14)
            phi[(nbr_r > R_sph).T] = 0.0
            phi[:, r_cell > R_sph] = 0.0
        cluster.set_leaf_phi(np.ascontiguousarray(phi))

        if lam_target > 0.0:
            from equilibrium.substrate.curl import b_vec_vectorized
            from equilibrium.substrate.discrete_curl import leaf_nb_rows

            nb_rows = leaf_nb_rows(cluster)
            B_c = b_vec_vectorized(pos, nb_rows, np.array(cluster.leaf_phi).T)
            B_c = np.nan_to_num(B_c, nan=0.0)
            b_mag = np.linalg.norm(B_c, axis=1)
            b_max = float(b_mag.max()) or 1.0
            cluster.set_leaf_lam(np.minimum(lam_target * b_mag / b_max, LAM_MAX))

    return b_max, r_axis


def apply_toroidal_rotation(cluster, centre, R_core, amplitude, spin_axis="z"):
    """
    Add a volumetric toroidal phi field (rotation) to a spherical knot.

    For a knot centred at `centre` spinning around `spin_axis`, the toroidal
    field direction at displacement d from the centre is:
        B_tor = (spin_axis_vec × d_perp) / |d_perp|
    where d_perp is the component of d perpendicular to the spin axis.

    Weighted by a linear taper window = max(0, 1 - |d| / R_core), so the
    winding amplitude peaks at the centre and vanishes at R_core.

    This creates a closed phi circulation (no net radial flux) that provides
    the energy return path missing from a purely static fixed-u source.

    spin_axis : "x", "y", or "z"
    amplitude : max |phi| added (scales with lam_central × R_core²)
    """
    pos = np.array(cluster.leaf_pos) - np.array(centre)  # (N, 3) relative

    B_tor = np.zeros_like(pos)
    if spin_axis == "z":
        rho = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
        mask = rho > 1e-9
        B_tor[mask, 0] = -pos[mask, 1] / rho[mask]
        B_tor[mask, 1] = pos[mask, 0] / rho[mask]
    elif spin_axis == "y":
        rho = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)
        mask = rho > 1e-9
        B_tor[mask, 0] = -pos[mask, 2] / rho[mask]
        B_tor[mask, 2] = pos[mask, 0] / rho[mask]
    elif spin_axis == "x":
        rho = np.sqrt(pos[:, 1] ** 2 + pos[:, 2] ** 2)
        mask = rho > 1e-9
        B_tor[mask, 1] = -pos[mask, 2] / rho[mask]
        B_tor[mask, 2] = pos[mask, 1] / rho[mask]
    else:
        raise ValueError(f"spin_axis must be 'x', 'y', or 'z', got {spin_axis!r}")

    r = np.linalg.norm(pos, axis=1)
    window = np.maximum(0.0, 1.0 - r / R_core)

    phi = np.array(cluster.leaf_phi)
    for k in range(14):
        phi[k] += amplitude * (B_tor @ _FACE_NORM[k]) * window
    cluster.set_leaf_phi(phi)


def init_toroidal_winding(
    cluster, ring_js, ring_faces, amplitude=1.0, frozen=False, phi_only=False
):
    """
    Initialise a winding (phi circulation + optional u pattern) on the ring.

    ring_js    — leaf-j indices returned by find_toroidal_ring
    ring_faces — face sequence returned by find_toroidal_ring
    frozen     — if True, freeze the face phi values (fixed_phi bitmask set)
    phi_only   — if True, skip u initialisation (safe for coupled rings)
    amplitude  — initial amplitude

    Returns resonant angular frequency omega (0.0 if frozen).

    Uses each tile's REAL geometric angle around the ring (not its raw
    path-index) to seed cos(theta) -- for a uniformly-spaced ring (e.g.
    mode="square"/"minimal") these are equivalent up to an overall constant,
    but for a non-uniformly-spaced ring (mode="octagon": straight legs and
    diagonal chamfer legs cover different amounts of real angle per tile,
    confirmed 2026-09-08 -- dtheta/tile ranged 1.03-2.39 degrees on a
    192-tile octagon) a plain cos(K*idx) is NOT the same as cos(theta) and
    produces a real, measured left-right asymmetry in the resulting near
    field (a seeding-construction bug, not a physical DST or lattice
    effect). The ring's own plane is found from its geometry (smallest-
    variance axis of its own positions, same technique as
    `_linking_number_from_pos` below) rather than assumed from an `axis`
    argument -- no preferred axis baked in, works for any ring orientation.
    """
    N_ring = len(ring_js)
    pos = np.asarray(cluster.leaf_pos)
    ring_pos = pos[ring_js]
    centre = ring_pos.mean(axis=0)
    rel = ring_pos - centre
    plane_axis = int(np.argmin(np.var(rel, axis=0)))
    perp = [i for i in range(3) if i != plane_axis]
    theta = np.unwrap(np.arctan2(rel[:, perp[1]], rel[:, perp[0]]))
    theta_wrapped_end = theta[0] + 2.0 * np.pi  # theta "at idx=N_ring" (closing the loop)

    phi = cluster.leaf_phi.copy()  # (14, N_leaves)
    fixed = cluster.leaf_fixed_phi.copy()  # (N_leaves,) uint16

    if not frozen and not phi_only:
        u = cluster.leaf_u.copy()
        for idx, j in enumerate(ring_js):
            u[j] = amplitude * np.cos(theta[idx])
        cluster.set_leaf_u(u)

    for idx in range(N_ring):
        j_i = ring_js[idx]
        j_nxt = ring_js[(idx + 1) % N_ring]
        k = ring_faces[idx]
        mk = int(_MIRROR[k])
        theta_next = theta[idx + 1] if idx + 1 < N_ring else theta_wrapped_end
        theta_mid = 0.5 * (theta[idx] + theta_next)
        phi_val = amplitude * np.cos(theta_mid)
        phi[k, j_i] = phi_val
        phi[mk, j_nxt] = -phi_val
        if frozen:
            fixed[j_i] |= np.uint16(1 << k)
            fixed[j_nxt] |= np.uint16(1 << mk)

    cluster.set_leaf_phi(phi)
    if frozen:
        cluster.set_leaf_fixed_phi(fixed)
        return 0.0

    # resonant freq: omega = 2 * W_SQ * |sin(pi/N)|; W_SQ = 1 in BCC units.
    # Derived for a uniformly-spaced ring -- an approximation for a
    # non-uniformly-spaced one (e.g. mode="octagon"), not re-derived here.
    return 2.0 * abs(np.sin(np.pi / N_ring))


# ---------------------------------------------------------------------------
# Hopf-linked spheromak pair
# ---------------------------------------------------------------------------


def _linking_number_from_pos(pos_A, pos_B):
    """Discrete linking number of ring_B through ring_A using position arrays."""
    plane_axis = int(np.argmin(np.var(pos_A, axis=0)))
    perp = [i for i in range(3) if i != plane_axis]
    z_A = float(pos_A[:, plane_axis].mean())
    poly = [(float(p[perp[0]]), float(p[perp[1]])) for p in pos_A]

    def _inside(pt):
        x, y = pt
        inside = False
        j = len(poly) - 1
        for i, (xi, yi) in enumerate(poly):
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
        return inside

    total = 0.0
    n = len(pos_B)
    for k in range(n):
        p, q = pos_B[k], pos_B[(k + 1) % n]
        zp, zq = float(p[plane_axis]), float(q[plane_axis])
        if (zp - z_A) * (zq - z_A) >= 0:
            continue
        t = (z_A - zp) / (zq - zp)
        cross = (
            float(p[perp[0]] + t * (q[perp[0]] - p[perp[0]])),
            float(p[perp[1]] + t * (q[perp[1]] - p[perp[1]])),
        )
        if _inside(cross):
            total += 1.0 if zq > zp else -1.0
    return total


def find_linked_loop(cluster, ring_A_js, ring_A_faces):
    """
    Find a 4-tile plaquette Hopf-linked (linking number ±1) with ring_A.

    Uses the adaptive face convention (8=-x, 9=+x, 10=-y, 11=+y, 12=-z, 13=+z).
    Returns (ring_B_js, ring_B_faces) or (None, None).
    """
    pos = np.array(cluster.leaf_pos)
    nb_g = np.array(cluster.leaf_nb)
    leaf_g = cluster.leaf_global_indices
    g_to_j = {int(g): j for j, g in enumerate(leaf_g)}

    A_pos = pos[ring_A_js]
    centre = A_pos.mean(axis=0)
    plane_axis = int(np.argmin(np.var(A_pos, axis=0)))

    # sq-hop distance from consecutive ring A tiles
    a = float(np.linalg.norm(A_pos[1] - A_pos[0]))
    h = a / 2.0

    # 4-tile plaquette patterns whose plane threads through ring_A's plane.
    # Adaptive face convention: 8=-x 9=+x 10=-y 11=+y 12=-z 13=+z
    base = {
        2: [
            [9, 13, 8, 12],
            [11, 13, 10, 12],
        ],  # xz or yz plaquettes thread xy-plane ring
        1: [
            [9, 11, 8, 10],
            [11, 13, 10, 12],
        ],  # xy or yz plaquettes thread xz-plane ring
        0: [[9, 11, 8, 10], [9, 13, 8, 12]],  # xy or xz plaquettes thread yz-plane ring
    }[plane_axis]
    patterns = [b[i:] + b[:i] for b in base for i in range(len(b))]

    ring_A_set = set(ring_A_js)

    for sign in (-1, +1):
        target = centre.copy()
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


def find_toroidal_spheromak(cluster, axis="z"):
    """
    Find the Hopf-linked ring pair: Ring A (8-tile toroidal) + Ring B (4-tile plaquette).

    Returns (A_js, A_faces, B_js, B_faces) or (None, None, None, None).
    """
    A_js, A_faces = find_toroidal_ring(cluster, axis=axis)
    if A_js is None:
        return None, None, None, None
    B_js, B_faces = find_linked_loop(cluster, A_js, A_faces)
    if B_js is None:
        return None, None, None, None
    return A_js, A_faces, B_js, B_faces


# ---------------------------------------------------------------------------
# Circulation diagnostics  (phi layout: (14, N_leaves), face-major)
# ---------------------------------------------------------------------------


def phi_ring_circulation(cluster, ring_js, centre):
    """Azimuthal phi circulation summed over all faces of the ring tile set."""
    if not ring_js:
        return 0.0
    phi = np.array(cluster.leaf_phi)  # (14, N)
    pos = np.array(cluster.leaf_pos)  # (N, 3)
    nb_g = np.array(cluster.leaf_nb)  # (N, 14)
    c = np.asarray(centre, dtype=float)
    leaf_g = cluster.leaf_global_indices
    g_to_j = {int(g): j for j, g in enumerate(leaf_g)}

    total = 0.0
    for i_j in ring_js:
        xi = pos[i_j, 0] - c[0]
        yi = pos[i_j, 1] - c[1]
        rho = np.sqrt(xi**2 + yi**2)
        if rho < 1e-10:
            continue
        phi_x = -yi / rho
        phi_y = xi / rho
        for k in range(14):
            g_nb = int(nb_g[i_j, k])
            if g_nb < 0:
                continue
            j_nb = g_to_j.get(g_nb, -1)
            if j_nb < 0:
                continue
            diff = pos[j_nb] - pos[i_j]
            d_len = float(np.linalg.norm(diff))
            if d_len < 1e-12:
                continue
            n_hat = diff / d_len
            proj = n_hat[0] * phi_x + n_hat[1] * phi_y
            total += float(phi[k, i_j]) * proj
    return total


def phi_toroidal_circulation(cluster, centre, r_shell, dr=0.8):
    """Azimuthal phi circulation over faces whose midpoint lies on r_shell ± dr."""
    phi = np.array(cluster.leaf_phi)  # (14, N)
    pos = np.array(cluster.leaf_pos)  # (N, 3)
    nb_g = np.array(cluster.leaf_nb)  # (N, 14)
    c = np.asarray(centre, dtype=float)
    leaf_g = cluster.leaf_global_indices
    g_to_j = {int(g): j for j, g in enumerate(leaf_g)}

    total = 0.0
    N = pos.shape[0]
    for i_j in range(N):
        for k in range(14):
            g_nb = int(nb_g[i_j, k])
            if g_nb < 0:
                continue
            j_nb = g_to_j.get(g_nb, -1)
            if j_nb < 0 or j_nb <= i_j:  # upper triangle — count each face once
                continue
            mid = 0.5 * (pos[i_j] + pos[j_nb]) - c
            r_mid = float(np.linalg.norm(mid))
            if not (r_shell - dr <= r_mid <= r_shell + dr):
                continue
            diff = pos[j_nb] - pos[i_j]
            d_len = float(np.linalg.norm(diff))
            if d_len < 1e-12:
                continue
            n_hat = diff / d_len
            mx, my = mid[0], mid[1]
            rho = np.sqrt(mx**2 + my**2)
            if rho < 1e-10:
                continue
            phi_x = -my / rho
            phi_y = mx / rho
            proj = n_hat[0] * phi_x + n_hat[1] * phi_y
            total += float(phi[k, i_j]) * proj
    return total


# ---------------------------------------------------------------------------
# Vectorized circulation scans — precompute face geometry once, evaluate many
# shells cheaply.  Replaces the per-shell N×14 Python loops for large N.
# ---------------------------------------------------------------------------

# Face weights of the substrate step (matches _W in equilibrium/substrate/
# lattice.py and W2_SUM = 8*9 + 6*1 = 78): hex faces 3, sq faces 1.
_W_FACE = np.array([3.0] * 8 + [1.0] * 6, dtype=np.float64)


def leaf_nb_j(cluster):
    """Leaf-local neighbour index array (N, 14); -1 where no leaf neighbour."""
    nb_g = np.array(cluster.leaf_nb)  # (N, 14)
    leaf_g = np.array(cluster.leaf_global_indices)
    N = nb_g.shape[0]
    size = int(max(leaf_g.max(), nb_g.max())) + 1
    g_to_j = np.full(size, -1, dtype=np.int64)
    g_to_j[leaf_g] = np.arange(N)
    return np.where(nb_g >= 0, g_to_j[np.clip(nb_g, 0, size - 1)], -1)


def divergence_clean(cluster, phi, *, rtol=1e-8, maxiter=10000):
    """
    Hodge-project phi onto the kernel of the substrate step's discrete
    divergence: div_i = Σ_k W_k φ_ik with W = _W_FACE.

    Solves L ψ = div φ on the W-weighted face graph (matrix-free CG), then
    applies the antisymmetric correction φ_ik += ψ_j − ψ_i.  Combined with a
    uniform u field the result is an exact fixed point of the reversible
    step (du ∝ −Σ W φ = 0 per cell, dφ ∝ Δu = 0 per face) — no radiated
    transient, regardless of lam.

    phi : (14, N).  Returns (phi_clean (14, N), div_rms_before, div_rms_after).
    """
    from scipy.sparse.linalg import LinearOperator, cg

    nb_j = leaf_nb_j(cluster)  # (N, 14)
    N = nb_j.shape[0]
    valid = nb_j >= 0
    safe = np.where(valid, nb_j, 0)
    Wv = np.where(valid, _W_FACE[None, :], 0.0)  # (N, 14)
    row_sum = Wv.sum(axis=1)

    phi_T = phi.T  # (N, 14)
    div = (Wv * np.where(valid, phi_T, 0.0)).sum(axis=1)
    div -= div.mean()  # project out the Laplacian nullspace (constants)
    div_rms0 = float(np.sqrt(np.mean(div**2)))

    def matvec(psi):
        psi_nb = np.where(valid, psi[safe], 0.0)
        return row_sum * psi - (Wv * psi_nb).sum(axis=1)

    L = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    try:
        psi, info = cg(L, div, rtol=rtol, maxiter=maxiter)
    except TypeError:  # scipy < 1.12
        psi, info = cg(L, div, tol=rtol, maxiter=maxiter)

    chi = np.where(valid, psi[safe] - psi[:, None], 0.0)  # (N, 14)
    phi_clean = phi_T + chi
    div_after = (Wv * np.where(valid, phi_clean, 0.0)).sum(axis=1)
    div_rms1 = float(np.sqrt(np.mean(div_after**2)))
    if info != 0:
        print(f"  divergence_clean: CG did not fully converge (info={info})")
    return np.ascontiguousarray(phi_clean.T), div_rms0, div_rms1


def face_flux_geometry(cluster, centre):
    """
    Unique-face geometry for circulation scans (each leaf-leaf face once).

    Returns (mid, n_hat, phi_f):
      mid   : (M, 3) face midpoints relative to centre
      n_hat : (M, 3) unit vectors from cell i to neighbour j
      phi_f : (M,)   directed phi on the face, from side i
    """
    phi = np.array(cluster.leaf_phi)  # (14, N)
    pos = np.array(cluster.leaf_pos)  # (N, 3)
    nb_g = np.array(cluster.leaf_nb)  # (N, 14)
    leaf_g = np.array(cluster.leaf_global_indices)
    N = pos.shape[0]

    size = int(max(leaf_g.max(), nb_g.max())) + 1
    g_to_j = np.full(size, -1, dtype=np.int64)
    g_to_j[leaf_g] = np.arange(N)
    nb_j = np.where(nb_g >= 0, g_to_j[np.clip(nb_g, 0, size - 1)], -1)

    ii, kk = np.nonzero(nb_j >= 0)
    jj = nb_j[ii, kk]
    keep = jj > ii  # upper triangle — count each face once
    ii, kk, jj = ii[keep], kk[keep], jj[keep]

    diff = pos[jj] - pos[ii]
    d = np.linalg.norm(diff, axis=1)
    ok = d > 1e-12
    ii, kk, jj, diff, d = ii[ok], kk[ok], jj[ok], diff[ok], d[ok]

    mid = 0.5 * (pos[ii] + pos[jj]) - np.asarray(centre, dtype=float)
    n_hat = diff / d[:, None]
    phi_f = phi[kk, ii]
    return mid, n_hat, phi_f


def phi_toroidal_scan(mid, n_hat, phi_f, r_shells, dr=0.8):
    """Azimuthal phi circulation on spherical shells r_shells ± dr."""
    r_mid = np.linalg.norm(mid, axis=1)
    rho = np.hypot(mid[:, 0], mid[:, 1])
    rho_s = np.where(rho > 1e-10, rho, 1.0)
    proj = (
        n_hat[:, 0] * (-mid[:, 1] / rho_s) + n_hat[:, 1] * (mid[:, 0] / rho_s)
    ) * phi_f
    proj = np.where(rho > 1e-10, proj, 0.0)
    return np.array([proj[np.abs(r_mid - r) <= dr].sum() for r in r_shells])


def phi_poloidal_scan(mid, n_hat, phi_f, ring_r, s_shells, ds=0.8):
    """
    Poloidal phi circulation on tube shells around the centreline circle
    (rho = ring_r, z = 0): s = sqrt((rho - ring_r)^2 + z^2), shells s ± ds.

    Projects each face onto the meridional tangent direction — the same
    reference as _poloidal_phi in probe_decay_compare.  This is the winding
    observable for the double-ring dipole (electron) topology.
    """
    rho = np.hypot(mid[:, 0], mid[:, 1])
    ok = rho > 1e-10
    rho_s = np.where(ok, rho, 1.0)
    rhat_x = mid[:, 0] / rho_s
    rhat_y = mid[:, 1] / rho_s
    s_mid = np.hypot(rho - ring_r, mid[:, 2])
    theta = np.arctan2(mid[:, 2], rho - ring_r)
    s_t, c_t = np.sin(theta), np.cos(theta)
    proj = (
        n_hat[:, 0] * (-s_t * rhat_x)
        + n_hat[:, 1] * (-s_t * rhat_y)
        + n_hat[:, 2] * c_t
    ) * phi_f
    proj = np.where(ok, proj, 0.0)
    return np.array([proj[np.abs(s_mid - s) <= ds].sum() for s in s_shells])


def dipole_moment_fit(
    cluster,
    centre,
    loop_radius,
    axis=2,
    r_min_factor=1.5,
    r_max_factor=4.0,
    theta_min=0.35,
):
    """
    Effective magnetic dipole moment of a circulating phi structure, fit
    from the general far-field point-dipole angular pattern.

    **OPEN ISSUE, verified 2026-09-16 (see project_electron_ic_falsification_
    review memory): this reads back exactly zero (machine noise) on
    init_electron_dipole_ic's actual phi, at every radius/scale tested --
    NOT because that object lacks a moment. Root cause traced by feeding a
    KNOWN classical current loop's field into curl(phi)=B two different
    ways: seeded as A.dl (circulation/vector-potential line-integral, the
    convention equilibrium.substrate.curl.b_vec_vectorized was validated
    against in probe_curl_loop_dipole.py/test_curl.py) reconstructs a real
    (if noisy, not yet fully debugged) signal; seeded as B.n_hat (flux-
    through-face-normal, the convention init_electron_dipole_ic ACTUALLY
    uses) reconstructs to EXACT zero on axis, same degenerate pattern as the
    real electron IC. So init_electron_dipole_ic's phi is a flux-style
    sample, but this function's B=curl(phi) machinery requires a
    circulation-style sample to mean anything -- a real, sharp instance of
    the unresolved phi-convention crisis (project_dst_em_field_convention_
    crisis), not a bug in this function or in today's diagnostic. DO NOT
    trust a near-zero result from this function on the current electron/
    positron IC as "no magnetic moment" until that convention mismatch is
    resolved (either rebuild the IC's phi as A.dl, or derive a moment
    formula native to the flux convention instead of reusing curl(phi)=B).**

    NOTE: an on-axis-only probe would ALSO silently return zero for this
    construction even if the above were fixed -- B=curl(phi) is exactly
    zero on the geometric symmetry axis (rho=0) at every radius for the
    poloidal-Taylor double-ring IC, an exact algebraic consequence of
    init_electron_dipole_ic's Cartesian
    field formula (every component is proportional to x, y, or x^2+y^2,
    all of which vanish at x=y=0), not a near/far-field transition effect.
    This function therefore samples general polar angles instead.

    Reconstructs B = curl(phi) (equilibrium.substrate.curl.b_vec_vectorized,
    validated against classical wire/loop field shapes) and least-squares
    fits it, over cells with r in [r_min_factor, r_max_factor]*loop_radius
    and polar angle theta (from `axis` through `centre`) outside a
    theta_min cone around each pole, against the point-dipole far-field
    pattern (mu0=1 convention):
        B_r     = 2*mu*cos(theta) / (4*pi*r^3)
        B_theta =   mu*sin(theta) / (4*pi*r^3)

    Returns (mu, consistency): consistency = RMS residual / RMS predicted
    signal over the fit -- small means the field genuinely looks
    dipole-shaped at this geometry/resolution; large (or mu~0 with sizable
    residual) means it isn't -- e.g. dominated by a higher multipole or a
    toroidal/anapole-like moment with no simple far-field dipole term. That
    is itself the physical answer, not a probe failure.

    Returns (0.0, nan) if no samples survive the r/theta window.
    """
    from equilibrium.substrate.curl import b_vec_vectorized
    from equilibrium.substrate.discrete_curl import leaf_nb_rows

    pos = np.array(cluster.leaf_pos)
    phi = np.array(cluster.leaf_phi).T  # (N, 14)
    nb = leaf_nb_rows(cluster)

    B = b_vec_vectorized(pos, nb, phi)  # (N, 3)
    have = np.all(np.isfinite(B), axis=1)

    rel = pos - np.asarray(centre, dtype=float)
    other = [k for k in range(3) if k != axis]
    rho = np.hypot(rel[:, other[0]], rel[:, other[1]])
    z = rel[:, axis]
    r = np.hypot(rho, z)

    r_lo, r_hi = r_min_factor * loop_radius, r_max_factor * loop_radius
    ok = have & (r >= r_lo) & (r < r_hi) & (rho > 1e-9)
    theta = np.arctan2(rho, z)  # [0, pi], 0 = +axis pole
    ok &= (theta > theta_min) & (theta < np.pi - theta_min)
    if not np.any(ok):
        return 0.0, float("nan")

    r_ok, theta_ok = r[ok], theta[ok]
    rho_ok = rho[ok]
    rhat0, rhat1 = rel[ok, other[0]] / rho_ok, rel[ok, other[1]] / rho_ok

    sin_t, cos_t = np.sin(theta_ok), np.cos(theta_ok)
    B_ok = B[ok]
    B_rho = B_ok[:, other[0]] * rhat0 + B_ok[:, other[1]] * rhat1
    B_z = B_ok[:, axis]
    B_r = sin_t * B_rho + cos_t * B_z
    B_theta = cos_t * B_rho - sin_t * B_z

    inv_4pi_r3 = 1.0 / (4.0 * np.pi * r_ok**3)
    c = 2.0 * cos_t * inv_4pi_r3
    s = sin_t * inv_4pi_r3

    denom = float(np.sum(c**2 + s**2))
    if denom < 1e-300:
        return 0.0, float("nan")
    mu = float(np.sum(B_r * c + B_theta * s)) / denom

    resid = np.concatenate([B_r - mu * c, B_theta - mu * s])
    signal = np.concatenate([mu * c, mu * s])
    rms_resid = float(np.sqrt(np.mean(resid**2)))
    rms_signal = float(np.sqrt(np.mean(signal**2)))
    consistency = rms_resid / rms_signal if rms_signal > 1e-300 else float("nan")
    return mu, consistency
