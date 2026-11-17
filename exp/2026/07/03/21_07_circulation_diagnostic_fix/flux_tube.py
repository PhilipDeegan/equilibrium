"""
Flux-loop utilities for SubstrateLattice.

Option 1 — fixed plaquette: locate a minimal 4-cycle using sq faces,
set phi antisymmetrically, freeze those faces.  The circulation is held
permanently by the fixed_phi mask.

Option 2 — free resonant loop: same geometry, but phi and u are both
initialised to the K=2π/N traveling-wave normal mode of the loop.  No
faces are frozen; the wave circulates under the free lattice dynamics.
Energy slowly leaks into surrounding tiles, but the first few oscillations
are intact and the initial state is exactly self-consistent with the
linearised wave equation on the isolated loop.
"""

import numpy as np

from equilibrium.substrate.lattice import SubstrateLattice

_MIRROR = np.array([7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 11, 10, 13, 12], dtype=np.int32)

# sq face pairs that form a 4-cycle plaquette: (f0,f1,f2,f3) traces
#   A -f0-> B -f1-> C -f2-> D -f3-> A
_SQ_PLAQUETTES = [
    (8, 10, 9, 11),  # xy: +x,+y,−x,−y
    (8, 12, 9, 13),  # xz: +x,+z,−x,−z
    (10, 12, 11, 13),  # yz: +y,+z,−y,−z
]


def find_shortest_loop(
    lat: SubstrateLattice, start: int = 0, face_type: str = "sq"
) -> tuple:
    """
    Find a minimal closed flux loop starting from tile `start`.

    For face_type='sq': tries each of the three coordinate-plane 4-cycles
    at `start` and returns the first valid one.

    Returns (loop_tiles, loop_faces) where each is a list of length 4, or
    (None, None) if no valid loop exists at `start`.
    """
    if face_type != "sq":
        raise NotImplementedError("Only sq face loops are implemented")

    nb = lat.nb
    for f0, f1, f2, f3 in _SQ_PLAQUETTES:
        B = int(nb[start, f0])
        if B < 0:
            continue
        C = int(nb[B, f1])
        if C < 0:
            continue
        D = int(nb[C, f2])
        if D < 0:
            continue
        if int(nb[D, f3]) == start:
            return [start, B, C, D], [f0, f1, f2, f3]

    return None, None


def set_flux_loop(
    lat: SubstrateLattice,
    loop_tiles: list,
    loop_faces: list,
    flux: float = 1.0,
):
    """
    Imprint a circulation on the loop and freeze those faces.

    For each directed edge i → j via face k:
        phi[i, k]        =  flux
        phi[j, mirror(k)] = -flux
        fixed_phi on both = True
    """
    n = len(loop_tiles)
    for step in range(n):
        i = loop_tiles[step]
        j = loop_tiles[(step + 1) % n]
        k = loop_faces[step]
        mk = int(_MIRROR[k])
        lat.phi[i, k] = flux
        lat.phi[j, mk] = -flux
        lat.fixed_phi[i, k] = True
        lat.fixed_phi[j, mk] = True


def clear_flux_loop(
    lat: SubstrateLattice,
    loop_tiles: list,
    loop_faces: list,
):
    """Zero the phi values on the loop edges and unfreeze those faces."""
    n = len(loop_tiles)
    for step in range(n):
        i = loop_tiles[step]
        j = loop_tiles[(step + 1) % n]
        k = loop_faces[step]
        mk = int(_MIRROR[k])
        lat.phi[i, k] = 0.0
        lat.phi[j, mk] = 0.0
        lat.fixed_phi[i, k] = False
        lat.fixed_phi[j, mk] = False


# ---------------------------------------------------------------------------
# Option 2 — free resonant loop
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Option 3 — toroidal ring
# ---------------------------------------------------------------------------

# Face sequences that trace an 8-tile square ring winding once around each axis.
# Each entry: (face_sequence, reference_start_direction).
# The ring uses only sq faces (8-13) and stays in the plane perpendicular to axis.
#
# Verified positions (with sq-face hop distance a):
#   z: (a,0,0)→(a,a,0)→(0,a,0)→(-a,a,0)→(-a,0,0)→(-a,-a,0)→(0,-a,0)→(a,-a,0)→(a,0,0)
#   y: (a,0,0)→(a,0,a)→(0,0,a)→(-a,0,a)→(-a,0,0)→(-a,0,-a)→(0,0,-a)→(a,0,-a)→(a,0,0)
#   x: (0,a,0)→(0,a,a)→(0,0,a)→(0,-a,a)→(0,-a,0)→(0,-a,-a)→(0,0,-a)→(0,a,-a)→(0,a,0)
_TOROIDAL_PATTERNS = {
    "z": (
        np.array([10, 9, 9, 11, 11, 8, 8, 10], dtype=np.int32),
        np.array([1.0, 0.0, 0.0]),
    ),
    "y": (
        np.array([12, 9, 9, 13, 13, 8, 8, 12], dtype=np.int32),
        np.array([1.0, 0.0, 0.0]),
    ),
    "x": (
        np.array([12, 11, 11, 13, 13, 10, 10, 12], dtype=np.int32),
        np.array([0.0, 1.0, 0.0]),
    ),
}

TOROIDAL_RING_SIZE: int = 8


def _sq_hop_dist(lat: SubstrateLattice) -> float:
    """Infer the sq-face hop distance a from the lattice positions."""
    for i in range(min(10, lat.N)):
        valid = lat.nb[i, 8:14]
        valid = valid[valid >= 0]
        if len(valid):
            return float(np.min(np.linalg.norm(lat.pos[valid] - lat.pos[i], axis=1)))
    return 2.0 / np.sqrt(3)  # fallback: unit lattice spacing


def _trace_loop(lat: SubstrateLattice, start: int, face_seq) -> tuple:
    """Trace a loop from start following face_seq; return (tiles, faces) if it closes."""
    nb = lat.nb
    tiles = [start]
    for f in face_seq[:-1]:
        nxt = int(nb[tiles[-1], int(f)])
        if nxt < 0 or nxt in tiles:
            return None, None
        tiles.append(nxt)
    if int(nb[tiles[-1], int(face_seq[-1])]) == start:
        return tiles, [int(f) for f in face_seq]
    return None, None


def find_toroidal_ring(lat: SubstrateLattice, axis: str = "z") -> tuple:
    """
    Find an 8-tile closed loop that winds once around the given axis ('x', 'y', 'z').

    The ring lies in the plane perpendicular to the axis and uses only sq faces.
    Start tile candidates are sorted by distance to the canonical reference position;
    the first valid one is returned.

    Returns (loop_tiles, loop_faces) — each a list of 8 — or (None, None).
    """
    face_seq, ref_dir = _TOROIDAL_PATTERNS[axis]
    a = _sq_hop_dist(lat)
    target = ref_dir * a
    dists = np.linalg.norm(lat.pos - target, axis=1)
    for start in np.argsort(dists)[:20]:
        tiles, faces = _trace_loop(lat, int(start), face_seq)
        if tiles is not None:
            return tiles, faces
    return None, None


def init_toroidal_winding(
    lat: SubstrateLattice,
    ring_tiles: list,
    ring_faces: list,
    amplitude: float = 1.0,
    frozen: bool = False,
) -> float:
    """
    Initialise a winding on a toroidal ring.

    frozen=False (default): free resonant mode — same travelling-wave
        initialisation as init_resonant_loop, no faces frozen.
    frozen=True: static frozen circulation — same as set_flux_loop.

    Returns ω (resonant angular frequency), or 0.0 if frozen.
    """
    if frozen:
        set_flux_loop(lat, ring_tiles, ring_faces, flux=amplitude)
        return 0.0
    return init_resonant_loop(lat, ring_tiles, ring_faces, amplitude=amplitude)


def winding_number(lat: SubstrateLattice, ring_tiles: list, axis: str = "z") -> float:
    """
    Discrete winding number of the ring around the given axis.

    Computed as the total signed angle traversed in the plane perpendicular
    to the axis, divided by 2π.  Returns ±1.0 for a single winding.
    """
    ax = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[axis]
    n = len(ring_tiles)
    total = 0.0
    for j in range(n):
        p = lat.pos[ring_tiles[j], list(ax)]
        q = lat.pos[ring_tiles[(j + 1) % n], list(ax)]
        total += np.arctan2(p[0] * q[1] - p[1] * q[0], p[0] * q[0] + p[1] * q[1])
    return total / (2.0 * np.pi)


# ---------------------------------------------------------------------------
# Hopf link — spheromak (linked toroidal + poloidal)
# ---------------------------------------------------------------------------


def linking_number(lat: SubstrateLattice, ring_A: list, ring_B: list) -> float:
    """
    Discrete linking number of ring_B through ring_A.

    Finds the plane of ring_A (axis of smallest positional variance), then
    counts signed crossings of ring_B's edges through ring_A's convex polygon:
        +1 for each crossing in the +normal direction
        -1 for each crossing in the -normal direction
    Returns ±1 for a Hopf link, 0 for unlinked loops.
    """
    A_pos = lat.pos[np.array(ring_A)]
    plane_axis = int(np.argmin(np.var(A_pos, axis=0)))
    perp = [i for i in range(3) if i != plane_axis]
    z_A = float(A_pos[:, plane_axis].mean())
    poly = [(float(p[perp[0]]), float(p[perp[1]])) for p in A_pos]

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
    n = len(ring_B)
    for k in range(n):
        p = lat.pos[ring_B[k]]
        q = lat.pos[ring_B[(k + 1) % n]]
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


def find_linked_loop(
    lat: SubstrateLattice, ring_A_tiles: list, ring_A_faces: list
) -> tuple:
    """
    Find a 4-tile plaquette that is Hopf-linked (linking number ±1) with ring_A.

    Searches body-centre and corner tiles near ring_A's centre for a plaquette
    whose plane is perpendicular to ring_A's plane.  The two rings share no tiles.

    Returns (ring_B_tiles, ring_B_faces) or (None, None).
    """
    A_pos = lat.pos[np.array(ring_A_tiles)]
    centre = A_pos.mean(axis=0)
    plane_axis = int(np.argmin(np.var(A_pos, axis=0)))

    a = _sq_hop_dist(lat)
    h = a / 2.0

    # Cross-patterns: plaquettes whose plane is NOT the ring_A plane, so they
    # span both sides of the ring_A plane and can thread through it.
    # plane_axis=2 (xy plane A) → need xz or yz patterns (contain ±z faces 12,13)
    # plane_axis=1 (xz plane A) → need xy or yz patterns (contain ±y faces 10,11)
    # plane_axis=0 (yz plane A) → need xy or xz patterns (contain ±x faces 8,9)
    base = {
        2: [[8, 12, 9, 13], [10, 12, 11, 13]],
        1: [[8, 10, 9, 11], [10, 12, 11, 13]],
        0: [[8, 10, 9, 11], [8, 12, 9, 13]],
    }[plane_axis]
    patterns = [b[i:] + b[:i] for b in base for i in range(len(b))]

    for sign in (-1, +1):
        target = centre.copy()
        target[plane_axis] += sign * h
        dists = np.linalg.norm(lat.pos - target, axis=1)
        for pattern in patterns:
            for start in np.argsort(dists)[:20]:
                tiles, faces = _trace_loop(lat, int(start), pattern)
                if tiles is None:
                    continue
                if set(tiles) & set(ring_A_tiles):
                    continue
                if abs(abs(linking_number(lat, ring_A_tiles, tiles)) - 1.0) < 0.1:
                    return tiles, faces
    return None, None


def find_toroidal_spheromak(lat: SubstrateLattice, axis: str = "z") -> tuple:
    """
    Find the geometrically correct spheromak pair:
        Ring A — 8-tile toroidal ring (winding_number=±1 around axis)
        Ring B — 4-tile plaquette threading through Ring A (linking_number=±1)

    Ring A and Ring B are on different sublattices (corner vs body-centre),
    so they share no tiles.  Their resonant frequencies differ:
        ω_A = 2·W_SQ·|sin(π/8)| ≈ 0.77
        ω_B = 2·W_SQ·|sin(π/4)| = √2 ≈ 1.41
    which eliminates the parametric resonance present when both rings are N=4.

    Returns (A_tiles, A_faces, B_tiles, B_faces) or (None, None, None, None).
    """
    A_tiles, A_faces = find_toroidal_ring(lat, axis=axis)
    if A_tiles is None:
        return None, None, None, None
    B_tiles, B_faces = find_linked_loop(lat, A_tiles, A_faces)
    if B_tiles is None:
        return None, None, None, None
    return A_tiles, A_faces, B_tiles, B_faces


def init_spheromak(
    lat: SubstrateLattice,
    ring_A_tiles: list,
    ring_A_faces: list,
    ring_B_tiles: list,
    ring_B_faces: list,
    amplitude: float = 1.0,
    frozen: bool = False,
    phi_only: bool = True,
) -> tuple:
    """
    Initialise a spheromak: two Hopf-linked flux loops.

    frozen=False (default): resonant mode on both rings (no faces frozen).
        phi_only=True (default): only phi circulation is set; u is left at
        zero.  This avoids the parametric instability that arises when both
        rings are independently initialised with u≠0: the two rings are
        connected by W_HEX=3 hex faces, and independent resonant u patterns
        can place u=+A and u=−A on hex-adjacent tiles, driving exponential
        energy growth.  With u=0 initially the hex coupling is zero at t=0
        and grows only as O(dt·A) per step.
        phi_only=False: full resonant init (u and phi); only safe when the
        caller has verified that u-patterns on the two rings are in phase at
        all hex-contact tile pairs.
    frozen=True: static frozen circulation on both rings; phi_only ignored.

    Returns (omega_A, omega_B) — resonant frequencies (0.0 if frozen).
    """
    if frozen:
        set_flux_loop(lat, ring_A_tiles, ring_A_faces, flux=amplitude)
        set_flux_loop(lat, ring_B_tiles, ring_B_faces, flux=amplitude)
        return 0.0, 0.0
    omega_A = init_resonant_loop(
        lat, ring_A_tiles, ring_A_faces, amplitude=amplitude, phi_only=phi_only
    )
    omega_B = init_resonant_loop(
        lat, ring_B_tiles, ring_B_faces, amplitude=amplitude, phi_only=phi_only
    )
    return omega_A, omega_B


def resonant_freq(loop_size: int, face_type: str = "sq") -> float:
    """
    Angular frequency of the fundamental traveling-wave mode on a closed loop.

    Dispersion relation for a discrete 1-D ring of N tiles with face weight W:
        ω = 2·W·|sin(π/N)|
    """
    from equilibrium.substrate.lattice import W_HEX, W_SQ

    W = W_SQ if face_type == "sq" else W_HEX
    return 2.0 * W * abs(np.sin(np.pi / loop_size))


def init_resonant_loop(
    lat: SubstrateLattice,
    loop_tiles: list,
    loop_faces: list,
    amplitude: float = 1.0,
    phi_only: bool = False,
) -> float:
    """
    Initialise a free (unfrozen) traveling-wave mode on a closed loop.

    Sets phi (and u if phi_only=False) to the K = 2π/N fundamental mode:
        u_j   = A · cos(K·j)          (skipped when phi_only=True)
        phi_j = A · cos(K·(j + ½))    on directed edge j → j+1

    phi_only=True sets only the circulation, leaving u=0.  Useful for
    coupled rings (e.g. spheromak) where setting u independently on
    hex-adjacent loops creates large u-gradients that drive instability.

    No faces are frozen.  Returns the resonant angular frequency ω.
    """
    N = len(loop_tiles)
    K = 2.0 * np.pi / N

    if not phi_only:
        for j, tile_idx in enumerate(loop_tiles):
            lat.u[tile_idx] = amplitude * np.cos(K * j)

    for j in range(N):
        i = loop_tiles[j]
        nxt = loop_tiles[(j + 1) % N]
        k = loop_faces[j]
        mk = int(_MIRROR[k])
        phi_val = amplitude * np.cos(K * (j + 0.5))
        lat.phi[i, k] = phi_val
        lat.phi[nxt, mk] = -phi_val

    return resonant_freq(N)


# ---------------------------------------------------------------------------
# Phi diagnostics
# ---------------------------------------------------------------------------


def phi_toroidal_circulation(lat, centre, r_shell, dr=0.8):
    """
    Sum the azimuthal (z-axis toroidal) phi component over faces whose
    midpoint lies on the shell at r_shell ± dr.  Upper-triangle only
    (i < nb[i,k]) to avoid double-counting.
    """
    c = np.asarray(centre, dtype=float)
    nb_idx = lat.nb
    valid = lat._valid
    pos = lat.pos

    nb_safe = np.where(nb_idx >= 0, nb_idx, 0)
    pos_j = pos[nb_safe]
    pos_i = pos[:, np.newaxis, :]

    mid = 0.5 * (pos_i + pos_j) - c
    r_mid = np.linalg.norm(mid, axis=-1)

    on_shell = valid & (r_mid >= r_shell - dr) & (r_mid <= r_shell + dr)
    upper = on_shell & (np.arange(len(pos))[:, None] < nb_safe)

    mx, my = mid[..., 0], mid[..., 1]
    rho = np.sqrt(mx**2 + my**2)
    rho_s = np.where(rho > 1e-10, rho, 1.0)
    phi_x = -my / rho_s
    phi_y = mx / rho_s

    diff = pos_j - pos_i
    d_len = np.linalg.norm(diff, axis=-1, keepdims=True)
    n_hat = diff / np.where(d_len > 0, d_len, 1.0)

    proj = n_hat[..., 0] * phi_x + n_hat[..., 1] * phi_y
    return float(np.sum(lat.phi[upper] * proj[upper]))


def phi_ring_circulation(lat, tile_idx, centre, axis=None):
    """
    Azimuthal phi sum over all valid faces incident on tiles in tile_idx.
    Each face counted once from the tile's perspective (no upper-triangle
    restriction — we want total flux out of the ring neighbourhood).

    axis: winding axis ('x', 'y', 'z') — the azimuthal direction is computed
        in the plane perpendicular to this axis. If None, inferred from the
        ring's own tile positions as the axis of lowest positional variance
        (the ring lies in the plane perpendicular to its winding axis).
        Previously hardcoded to 'z' (x,y projection only), which silently
        measured the wrong quantity for rings wound around x or y.
    """
    if not tile_idx:
        return 0.0
    c = np.asarray(centre, dtype=float)
    idx = np.array(sorted(tile_idx), dtype=int)
    nb_raw = lat.nb[idx]
    nb_safe = np.where(nb_raw >= 0, nb_raw, 0)

    pos_i = lat.pos[idx]
    pos_j = lat.pos[nb_safe]

    if axis is None:
        plane_axis = int(np.argmin(np.var(pos_i, axis=0)))
    else:
        plane_axis = {"x": 0, "y": 1, "z": 2}[axis]
    a0, a1 = [k for k in range(3) if k != plane_axis]

    diff = pos_j - pos_i[:, np.newaxis, :]
    d_len = np.linalg.norm(diff, axis=-1, keepdims=True)
    n_hat = diff / np.where(d_len > 0, d_len, 1.0)

    xi = (pos_i[:, a0] - c[a0])[:, np.newaxis]
    yi = (pos_i[:, a1] - c[a1])[:, np.newaxis]
    rho = np.sqrt(xi**2 + yi**2)
    rho_s = np.where(rho > 1e-10, rho, 1.0)
    phi_x = -yi / rho_s
    phi_y = xi / rho_s

    proj = n_hat[..., a0] * phi_x + n_hat[..., a1] * phi_y
    valid = lat._valid[idx]
    return float(np.sum(lat.phi[idx] * proj * valid))
