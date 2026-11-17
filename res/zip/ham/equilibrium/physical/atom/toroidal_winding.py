"""
Inverse solver: recover the signed DST u-field from the PHARE toroidal
dipole B-field (the stable endpoint of the spheromak decay at t=3).

Background
──────────
The PHARE simulation of a Taylor spheromak (plasmoses.py) decays after
60 000 timesteps (t=3) into a stable toroidal dipole — the minimum-energy
configuration for the given winding number.  In DST this endpoint IS the
neutron ground state: a closed toroidal winding of u-flux.

The B-field of that endpoint is stored in ic_b.npz:
    x, y, z : 203-point grids over [-5.05, 5.05] PHARE units
    bx,by,bz: (203,203,203) field components
    center  : domain centre in PHARE coordinates

The inner ~70% of the domain (|r| < 3.5 PHARE units) is clean; outside
that the boundary artefacts dominate.  The field decays to noise by
r ≈ 2.5 units.

Inverse problem
───────────────
In DST the face flux for face (i,j) is:
    (u_j - u_i) / d_ij

and should equal the normal component of B at the face:
    B(r_face) · n̂_ij

So for each face:
    u_j - u_i  =  B(r_face) · (pos_j - pos_i)         ... (*)

Stacking all face equations gives an overdetermined linear system
    G u = b
where G is the signed incidence matrix (−1 at row i, +1 at row j),
and b[face] = B(r_face) · Δr_ij (in Kelvin-lattice units).

Since B has non-zero curl (it is NOT a potential field globally), the
system has a non-zero residual.  The least-squares solution minimises the
total squared error over all faces — the best DST scalar approximation
to the vector B-field.

The toroidal dipole B-field has both positive and negative components,
so the recovered u has both signs — this is the key difference from the
scalar magnitude (all-positive) profile, and the mechanism that gives
zero net outward flux from the neutron winding.

Usage
─────
    from equilibrium.physical.atom.toroidal_winding import (
        load_phare_field, solve_u_from_b, winding_summary
    )
    interps, lim = load_phare_field("path/to/ic_b.npz")
    lat, u = solve_u_from_b(interps, phare_scale=0.18, radius_cells=10)
    winding_summary(u)
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


def load_phare_field(npz_path: str, inner_frac: float = 0.70):
    """
    Load the PHARE toroidal dipole B-field from ic_b.npz.

    Parameters
    ----------
    npz_path   : path to ic_b.npz
    inner_frac : fraction of the half-domain to keep (drop boundary artefacts)

    Returns
    -------
    interps : (interp_bx, interp_by, interp_bz) — RegularGridInterpolators
    lim     : half-width of the retained inner region in PHARE units
    """
    d = np.load(npz_path)
    x, y, z = d["x"], d["y"], d["z"]
    lim = float(inner_frac * abs(x[-1]))
    x0 = np.searchsorted(x, -lim)
    x1 = np.searchsorted(x, lim, side="right")
    xi, yi, zi = x[x0:x1], y[x0:x1], z[x0:x1]
    sl = (slice(x0, x1),) * 3

    kw = dict(method="linear", bounds_error=False, fill_value=0.0)
    interp_bx = RegularGridInterpolator((xi, yi, zi), d["bx"][sl], **kw)
    interp_by = RegularGridInterpolator((xi, yi, zi), d["by"][sl], **kw)
    interp_bz = RegularGridInterpolator((xi, yi, zi), d["bz"][sl], **kw)

    return (interp_bx, interp_by, interp_bz), lim


# ---------------------------------------------------------------------------
# Inverse solver
# ---------------------------------------------------------------------------


def solve_u_from_b(
    interps,
    phare_scale: float = 0.18,
    radius_cells: float = 10.0,
    iter_lim: int = 50_000,
):
    """
    Recover the signed DST u-field from the PHARE toroidal dipole B-field.

    Builds a Kelvin lattice and solves the least-squares system:
        u_j - u_i  =  B(r_face) · (pos_j - pos_i)   for every face

    Parameters
    ----------
    interps      : (interp_bx, interp_by, interp_bz) from load_phare_field
    phare_scale  : PHARE units per Kelvin lattice cell-unit
                   (controls how much of the PHARE field fits in the lattice)
    radius_cells : sphere radius in Kelvin cell units
    iter_lim     : max iterations for the LSQR solver

    Returns
    -------
    pts : (N, 3) Kelvin lattice cell positions (lattice units)
    u   : (N,)   signed u-field  — mean-subtracted, unit-normalised
    nb  : (N, 14) neighbour index array (−1 = boundary)
    residual : normalised RMS residual over all faces
    """
    from equilibrium.cpp_dst import DSTLattice  # pylint: disable=no-name-in-module

    interp_bx, interp_by, interp_bz = interps

    lat = DSTLattice(cell_size=phare_scale, radius_cells=radius_cells)
    fps = lat.face_pairs()  # (F, 2) int32
    pos_phare = lat.positions()  # (N, 3) in PHARE units

    # Vectorised face geometry (all in PHARE units)
    pi = pos_phare[fps[:, 0]]
    pj = pos_phare[fps[:, 1]]
    r_face = 0.5 * (pi + pj)
    dr = pj - pi

    # Batch B-field interpolation
    B = np.stack(
        [interp_bx(r_face), interp_by(r_face), interp_bz(r_face)], axis=1
    )  # (F, 3)

    b_face = np.einsum("fi,fi->f", B, dr).astype(np.float64)  # (F,)

    u = lat.solve_lsqr(b_face, iter_lim=iter_lim)  # normalised in C++

    # Residual: ||G u - b|| / (sqrt(F) * mean|b|)
    Gu = u[fps[:, 1]] - u[fps[:, 0]]
    r1norm = float(np.linalg.norm(Gu - b_face))
    residual = r1norm / (np.sqrt(len(b_face)) * (np.abs(b_face).mean() + 1e-30))

    # Return pts in cell units (lattice_spacing=1) for backward compatibility
    pts = pos_phare / phare_scale
    nb = lat.neighbours()  # (N, 14) int32

    return pts, u, nb, residual


# ---------------------------------------------------------------------------
# Perimeter-closure solver
# ---------------------------------------------------------------------------


def solve_with_perimeter(
    interps,
    phare_scale: float = 0.18,
    radius_total: float = 14.0,
    radius_inner: float = 10.0,
    iter_lim: int = 50_000,
):
    """
    Solve u from B for the full domain, then split into free inner cells
    and fixed outer-shell perimeter cells.

    The outer shell (radius_inner < r ≤ radius_total) replicates the role
    of the quiet far-field region in the PHARE simulation: the field there
    barely moves during the decay, and the outer cells act as a topological
    closure — the return path that makes the inner winding self-consistent.

    When the outer shell is fixed on a DSTLattice and the inner cells are
    free to evolve, the winding is no longer being pulled toward zero by
    open vacuum connections.  Instead the far-field boundary drives the
    inner dynamics toward the self-consistent closed-winding solution.

    Parameters
    ----------
    interps      : (interp_bx, interp_by, interp_bz) from load_phare_field
    phare_scale  : PHARE units per Kelvin lattice cell
    radius_total : total sphere radius (cells)
    radius_inner : inner-free / outer-fixed boundary (cells)
    iter_lim     : LSQR iteration limit

    Returns
    -------
    pts          : (N, 3) all cell positions (lattice units)
    u            : (N,)   signed u — mean-subtracted, unit-normalised
    nb           : (N, 14) neighbour array
    inner_mask   : (N,) bool — True for free inner cells
    outer_mask   : (N,) bool — True for fixed perimeter cells
    residual     : normalised RMS residual
    """
    pts, u, nb, residual = solve_u_from_b(
        interps,
        phare_scale=phare_scale,
        radius_cells=radius_total,
        iter_lim=iter_lim,
    )
    r = np.linalg.norm(pts, axis=1)
    inner_mask = r <= radius_inner
    outer_mask = r > radius_inner
    return pts, u, nb, inner_mask, outer_mask, residual


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def save_winding(
    path: str,
    u: np.ndarray,
    residual: float,
    phare_scale: float,
    radius_cells: float,
):
    """
    Save solved winding to an npz file.

    Only the irreducible data is stored: the u solution and the lattice
    parameters needed to reconstruct pts and nb on load.
    """
    np.savez_compressed(
        path,
        u=u,
        residual=np.array([residual]),
        phare_scale=np.array([phare_scale]),
        radius_cells=np.array([radius_cells]),
    )


def load_winding(path: str):
    """
    Load a pre-solved winding.  pts and nb are reconstructed from the
    lattice parameters rather than stored.

    Returns
    -------
    pts         : (N, 3) cell positions (lattice units)
    u           : (N,)   signed u field
    nb          : (N, 14) neighbour indices
    residual    : float
    phare_scale : float
    """
    from equilibrium.cpp_dst import DSTLattice  # pylint: disable=no-name-in-module

    d = np.load(path)
    phare_scale = float(d["phare_scale"][0])
    radius_cells = float(d["radius_cells"][0])
    lat = DSTLattice(cell_size=phare_scale, radius_cells=radius_cells)
    pts = np.array(lat.positions()) / phare_scale
    nb = np.array(lat.neighbours())
    return pts, d["u"], nb, float(d["residual"][0]), phare_scale


def save_perimeter_winding(
    path: str,
    u: np.ndarray,
    residual: float,
    phare_scale: float,
    radius_total: float,
    radius_inner: float,
):
    """
    Save a perimeter-closure winding.  Only u and scalar parameters are
    stored; pts, nb, and masks are reconstructed on load.
    """
    np.savez_compressed(
        path,
        u=u,
        residual=np.array([residual]),
        phare_scale=np.array([phare_scale]),
        radius_total=np.array([radius_total]),
        radius_inner=np.array([radius_inner]),
    )


def load_perimeter_winding(path: str):
    """
    Load a perimeter-closure winding solution.  pts, nb, and masks are
    reconstructed from the stored lattice parameters.

    Returns
    -------
    pts          : (N, 3) all cell positions (lattice units)
    u            : (N,)   signed u field
    inner_mask   : (N,) bool — free inner cells
    outer_mask   : (N,) bool — fixed perimeter cells
    phare_scale  : float
    """
    from equilibrium.cpp_dst import DSTLattice  # pylint: disable=no-name-in-module

    d = np.load(path)
    phare_scale = float(d["phare_scale"][0])
    radius_total = float(d["radius_total"][0])
    radius_inner = float(d["radius_inner"][0])

    lat = DSTLattice(cell_size=phare_scale, radius_cells=radius_total / phare_scale)
    pts = np.array(lat.positions()) / phare_scale
    r = np.linalg.norm(pts, axis=1)
    return (
        pts,
        d["u"],
        r <= radius_inner,
        r > radius_inner,
        phare_scale,
    )


def build_neutron_lattice(
    perimeter_path: str,
    centre: np.ndarray = None,
    amplitude: float = 1.0,
):
    """
    Build a DSTLattice with the neutron perimeter-closure winding.

    The outer shell is fixed (the topological closure / ambient perimeter).
    The inner cells are initialised with the solved winding u-values but left
    free to evolve — they will relax to the self-consistent mode compatible
    with the fixed perimeter.

    This is the DST neutron: a closed toroidal winding that is dynamically
    stable because the perimeter provides the return path for the field.
    The inner winding carries zero net outward flux (charge-neutral by
    topology).  The z-antisymmetry of the converged field is the winding
    chirality — the observable the conventional model calls magnetic moment.

    Parameters
    ----------
    perimeter_path : path to neutron_perimeter.npz
    centre         : (3,) neutron centre in lattice units — snapped to
                     nearest host lattice point.  Defaults to origin.
    amplitude      : scale of u values relative to proton u

    Returns
    -------
    lat  : DSTLattice ready to run.  Call lat.run(n_steps) to evolve.
    """
    from equilibrium.physical.dst.lattice import DSTLattice

    pts, u_w, inner_mask, outer_mask, phare_scale = load_perimeter_winding(
        perimeter_path
    )

    d = np.load(perimeter_path)
    radius_total = float(d["radius_total"][0])
    cs = phare_scale  # 1 winding cell = phare_scale Planck lengths

    lat = DSTLattice(cell_size=cs, radius_cells=radius_total)

    if centre is None:
        centre = np.zeros(3)
    # Snap centre to nearest host lattice point
    snap_dists = np.linalg.norm(lat.pos - centre, axis=1)
    centre = lat.pos[int(np.argmin(snap_dists))]

    # Map winding cells onto host lattice
    cell_u = {}
    cell_fix = {}
    for p, u, is_inner in zip(pts, u_w, inner_mask):
        pos_t = centre + p * cs
        dists = np.linalg.norm(lat.pos - pos_t, axis=1)
        i = int(np.argmin(dists))
        if dists[i] <= cs * 1.5:
            cell_u.setdefault(i, []).append(float(u))
            cell_fix[i] = not bool(is_inner)  # outer → fixed

    host_idx = np.array(list(cell_u.keys()), dtype=np.int32)
    host_vals = np.array([np.mean(v) for v in cell_u.values()])
    host_vals -= host_vals.mean()

    lat.u[host_idx] = amplitude * host_vals
    for i, fix in cell_fix.items():
        if fix:
            lat.fixed[i] = True

    return lat


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def winding_summary(u: np.ndarray):
    """Print key properties of the recovered winding u-field."""
    print(f"  N cells      : {len(u)}")
    print(f"  u range      : [{u.min():.4f}, {u.max():.4f}]")
    print(f"  mean(u)      : {u.mean():.2e}  (≈ 0 → balanced winding)")
    print(f"  sum(u>0)     : {(u > 0).sum()}  sum(u<0): {(u < 0).sum()}")
    print(f"  net outward  : {u.sum():.4f}  (= 0 → closed winding)")
    pos_mass = u[u > 0].sum()
    neg_mass = u[u < 0].sum()
    print(f"  +/- balance  : {pos_mass:.4f} / {neg_mass:.4f}")


def net_flux_fraction(u: np.ndarray) -> float:
    """
    |sum(u)| / sum(|u|) — fraction of u that escapes as net outward flux.
    Should be ≈ 0 for a perfect closed winding, 1 for an open knot.
    """
    total = float(np.abs(u).sum())
    return float(abs(u.sum())) / total if total > 0 else float("nan")
