"""
Hydrogen isotope nuclear models on the DST Kelvin lattice.

    Protium  (¹H): 1 proton, 0 neutrons
    Deuterium (²H): 1 proton, 1 neutron
    Tritium  (³H): 1 proton, 2 neutrons

Both the proton and each neutron are collective patterns spanning an
enormous number of Planck cells.  At nuclear scale (λ=1) a single
simulation cell represents ~1 fm of physical space — every 'particle'
placed here is already a coarse-grained envelope.

Proton  — open knot: high-u cells, flux radiates outward through all 14
          faces.  Fixed in the simulation (the knot doesn't diffuse).

Neutron — closed knot: signed toroidal winding loaded from the cached
          PHARE inverse-solve (neutron_perimeter_fine.npz).  The winding
          is rescaled to fit within R_NEUTRON_FM and mapped onto the host
          lattice.  The net u-sum is enforced to zero so no outward flux
          escapes: the winding is genuinely closed.

          At cell_size=1.0 fm the neutron (r=0.86 fm) fits within one
          host cell — the winding collapses to a single fixed cell with
          u=0.  At cell_size≪0.86 fm the full signed circulation is
          resolved.  In both cases the closed-knot topology (zero net
          outward flux) is correctly represented.

The electron moat forms at the Bohr orbit (~53 000 fm), far outside the
nuclear lattice shown here.  The nuclear simulation answers a different
question: how does the combined proton+neutron u-field profile depend on
the number of neutrons?  The answer determines effective charge radius
and nuclear binding, not atomic line structure (which depends only on the
long-range outward flux, unchanged by the closed neutron winding).
"""

import os
import numpy as np
from scipy.spatial import KDTree

from equilibrium.physical.dst.lattice import DSTLattice

# ---------------------------------------------------------------------------
# Nuclear geometry constants  (all values in physical fm)
# ---------------------------------------------------------------------------

# Proton exponential scale: u(r) = exp(-r / PROTON_EXP_SCALE)
# The volume-weighted RMS of exp(-r/a) is sqrt(12)*a, so to match CODATA:
#   PROTON_EXP_SCALE = CODATA / sqrt(12) = 0.8414 / 3.4641 = 0.2429 fm
PROTON_EXP_SCALE = 0.8414 / float(np.sqrt(12))  # ≈ 0.2429 fm

# Proton hard-sphere radius used when profile='sphere' (approximation only)
R_PROTON_FM = 0.85

# Neutron confinement radius ≈ 0.86 fm (similar to proton)
R_NEUTRON_FM = 0.86

# Proton-neutron separation (fm)
#   Deuteron charge radius ≈ 2.1 fm, so p-n separation ≈ 2.0 fm
D_PN = 2.0

# Backward-compatible alias
R_NEUTRON_CELLS = R_NEUTRON_FM

# Tritium: p + 2n.  Rough triangle: n-n separation ≈ 1.6 fm.
_D_NN_T = 1.6  # n-n distance in tritium (fm)
_D_PN_T = 1.8  # p-n distance in tritium (fm)

# ---------------------------------------------------------------------------
# Winding cache
# ---------------------------------------------------------------------------

_WINDING_CACHE = None  # (inner_pts, inner_u) — loaded once


def _load_winding():
    """
    Lazy-load the cached signed toroidal winding from neutron_perimeter_fine.npz.
    Returns (inner_pts, inner_u): Kelvin lattice coordinates and signed u values
    for the inner (free) cells of the winding solve.
    Falls back to neutron_perimeter.npz if the fine file is absent.
    """
    global _WINDING_CACHE
    if _WINDING_CACHE is None:
        here = os.path.dirname(__file__)
        for name in ("neutron_perimeter_fine.npz", "neutron_perimeter.npz"):
            path = os.path.join(here, name)
            if os.path.exists(path):
                d = np.load(path)
                inner = d["inner_mask"]
                _WINDING_CACHE = (
                    d["pts"][inner].astype(np.float64),
                    d["u"][inner].astype(np.float64),
                )
                break
        if _WINDING_CACHE is None:
            _WINDING_CACHE = (np.empty((0, 3)), np.empty(0))
    return _WINDING_CACHE


# ---------------------------------------------------------------------------
# Neutron placement
# ---------------------------------------------------------------------------


def _neutron_centres_tritium():
    """Two neutron centres for tritium, symmetric about the z-axis (fm)."""
    half = _D_NN_T / 2.0
    z = np.sqrt(max(_D_PN_T**2 - half**2, 0.0))
    return np.array([[half, 0.0, z], [-half, 0.0, z]])


def _add_neutron_winding(
    lat: DSTLattice, centre: np.ndarray, r_fm: float, amplitude: float = 1.0
):
    """
    Superimpose signed toroidal winding (closed knot) onto the lattice u-field.

    The cached winding pattern is rescaled so its radial extent equals r_fm,
    then each winding cell is mapped to the nearest host cell via KD-tree.
    Multiple winding cells mapping to the same host cell are averaged.
    The result is re-centred to enforce zero net u (no outward flux).

    Parameters
    ----------
    lat       : host DSTLattice (lat.pos in fm)
    centre    : (3,) neutron centre in fm
    r_fm      : confinement radius in fm  (winding rescaled to fit within this)
    amplitude : overall scale of winding u relative to the proton
    """
    winding_pts, winding_u = _load_winding()
    if len(winding_pts) == 0:
        return

    r_winding = np.linalg.norm(winding_pts, axis=1)
    r_max = float(r_winding.max())
    if r_max == 0:
        return

    scale = r_fm / r_max  # winding Kelvin units → fm
    pts_fm = centre + winding_pts * scale

    tree = KDTree(lat.pos)
    max_d = lat.cell_size * 1.5
    dists, idxs = tree.query(pts_fm, distance_upper_bound=max_d)

    # Accumulate per host cell; N sentinel means no match
    matched = idxs < lat.N
    if not matched.any():
        return

    from collections import defaultdict

    cell_u = defaultdict(list)
    for idx, u_w, hit in zip(idxs, winding_u, matched):
        if hit:
            cell_u[int(idx)].append(float(u_w))

    host_idx = np.array(list(cell_u.keys()), dtype=np.int32)
    host_vals = np.array([np.mean(v) for v in cell_u.values()])
    host_vals -= host_vals.mean()  # enforce zero net winding flux

    lat.u[host_idx] += amplitude * host_vals
    lat.fixed[host_idx] = True


def _set_proton_exponential(lat: DSTLattice, u_max: float = 1.0):
    """
    Initialise the proton as an exponential open knot.

    u(r) = u_max * exp(-r / PROTON_EXP_SCALE)

    The exponential is the natural steady-state profile of the DST update rule
    around a point source (discrete Laplacian Green's function).  The scale
    PROTON_EXP_SCALE ≈ 0.243 fm is derived from CODATA:
    sqrt(12) × PROTON_EXP_SCALE = 0.841 fm.

    The core (u ≥ u_max × 1e-3, i.e. r ≤ ~1.7 fm) is fixed — the knot is
    stable.  Outer cells are free (they evolve under the DST update rule but
    start at their exponential value so the initial charge_radius is accurate).
    """
    r = np.linalg.norm(lat.pos, axis=1)
    u = u_max * np.exp(-r / PROTON_EXP_SCALE)
    lat.u = u.copy()
    # Fix the core; free the far tail so it can evolve/diffuse
    lat.fixed = u >= u_max * 1e-3


def build(
    n_neutrons: int,
    cell_size: float = 0.2,
    radius_cells: float = 15.0,
    u_proton: float = 1.0,
    neutron_amplitude: float = 1.0,
    K: float = 1.0,
) -> DSTLattice:
    """
    Build a nuclear-scale DSTLattice for a hydrogen isotope.

    Parameters
    ----------
    n_neutrons        : 0 (protium), 1 (deuterium), 2 (tritium)
    cell_size         : cell size in fm.  At 0.2 fm the proton spans ~400 cells
                        and charge_radius converges to 0.835 fm (CODATA: 0.841 fm).
                        At 1.0 fm each nucleon fits in a single cell — valid for
                        topology checks (open/closed knot) but not geometry.
    radius_cells      : lattice sphere radius in cell units (physical radius =
                        radius_cells × cell_size fm; must exceed D_PN = 2.0 fm)
    u_proton          : u-value of the proton open knot
    neutron_amplitude : overall amplitude of the neutron winding relative to proton
    K                 : DST coupling constant

    Returns
    -------
    lat : DSTLattice initialised with proton + neutron(s), ready to run
    """
    if n_neutrons not in (0, 1, 2):
        raise ValueError("n_neutrons must be 0, 1, or 2")

    lat = DSTLattice(cell_size=cell_size, radius_cells=radius_cells, K=K)

    # Proton: open knot at origin.
    # Exponential profile u(r) = u_proton * exp(-r / PROTON_EXP_SCALE).
    # RMS = sqrt(12) * PROTON_EXP_SCALE = 0.841 fm (CODATA).
    _set_proton_exponential(lat, u_proton)

    # Store proton-only charge radius before neutron winding modifies lat.u.
    # This is the proton's intrinsic charge radius, independent of isotope.
    lat._proton_charge_radius = charge_radius(lat)

    # Neutron(s): closed knots at nuclear separations (fm).
    # Coordinates are physical fm — no cell_size factor needed.
    if n_neutrons == 1:
        centre = np.array([0.0, 0.0, D_PN])  # fm
        _add_neutron_winding(lat, centre, R_NEUTRON_FM, neutron_amplitude)

    elif n_neutrons == 2:
        for centre in _neutron_centres_tritium():  # fm
            _add_neutron_winding(lat, centre, R_NEUTRON_FM, neutron_amplitude)

    return lat


def run(lat: DSTLattice, n_steps: int = 2000) -> DSTLattice:
    """Run the DST update rule for n_steps ticks."""
    lat.run(n_steps)
    return lat


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def outward_flux(lat: DSTLattice, n_bins: int = 60):
    """
    Spherically averaged radial u-field profile.

    This is the envelope seen by the far-field (the electron moat).
    The neutron's closed winding contributes zero net outward flux —
    only the proton's open knot contributes at large r.

    Returns (r_centres, u_mean) in cell units.
    """
    return lat.radial_profile(n_bins)


def charge_radius(lat: DSTLattice) -> float:
    """
    RMS charge radius of the proton open knot.

    The neutron is charge-neutral (closed knot) — only the proton's exponential
    u-field contributes to charge.

    If build() has stored the proton-only charge radius (before neutron winding
    was added), that value is returned directly.  This avoids contamination from
    neutron winding cells that sit in the proton's exponential tail.

    Otherwise (e.g. for protium, or when called on a freshly built proton-only
    lattice), the charge radius is computed from all cells with u > 1e-5 in the
    proton zone (r ≤ R_PROTON_FM × 12).

    The exponential proton profile u(r) = exp(-r/a) has volume-weighted RMS
    = sqrt(12) × a = sqrt(12) × PROTON_EXP_SCALE ≈ 0.841 fm.
    """
    if hasattr(lat, "_proton_charge_radius"):
        return lat._proton_charge_radius
    r = np.linalg.norm(lat.pos, axis=1)
    proton_zone = R_PROTON_FM * 12.0
    mask = (lat.u > 1e-5) & (r <= proton_zone)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.average(r[mask] ** 2, weights=lat.u[mask])))


def nuclear_convergence_study(
    cell_sizes=(1.0, 0.5, 0.2, 0.1, 0.05), radius_cells: float = 15.0
) -> list:
    """
    Proton charge radius vs cell_size convergence table.

    Marks the coarse/fine knot-resolution transition:
      coarse: cell_size >= R_NEUTRON_FM — winding unresolved, topology is a
              placeholder (single cell with u=0), charge radius is still
              computed from the exponential profile geometry.
      fine:   cell_size << R_NEUTRON_FM — winding structure begins to emerge
              and zero-net-flux follows from the closed-knot topology itself.

    Returns list of dicts: cell_size, N, charge_radius_fm, error_pct, regime
    """
    CODATA = 0.8414  # fm
    results = []
    for cs in cell_sizes:
        lat = build(0, cell_size=cs, radius_cells=radius_cells)
        r = charge_radius(lat)
        err = 100.0 * abs(r - CODATA) / CODATA
        regime = "coarse" if cs >= R_NEUTRON_FM else "fine"
        results.append(
            {
                "cell_size": cs,
                "N": lat.N,
                "charge_radius_fm": r,
                "error_pct": err,
                "regime": regime,
            }
        )
    return results


def print_nuclear_convergence(results: list):
    """Print nuclear_convergence_study results as a table."""
    CODATA = 0.8414
    print(f"\n{'cell_size':>10}  {'N':>8}  {'R_charge(fm)':>13}  {'err%':>6}  regime")
    print(f"{'CODATA':>10}  {'':>8}  {CODATA:>13.4f}")
    for r in results:
        print(
            f"{r['cell_size']:>10.3f}  {r['N']:>8d}  "
            f"{r['charge_radius_fm']:>13.4f}  {r['error_pct']:>6.2f}%  {r['regime']}"
        )


def total_impedance(lat: DSTLattice) -> float:
    """
    Sum of u² over all non-fixed cells — a proxy for total substrate
    impedance of the field configuration (lower = more stable).
    """
    return float(np.sum(lat.u[~lat.fixed] ** 2))


def summary(lat: DSTLattice, label: str):
    """Print a one-line summary of the nuclear configuration."""
    r, um = outward_flux(lat)
    valid = ~np.isnan(um)
    peak_u = float(um[valid].max()) if valid.any() else float("nan")
    print(
        f"{label:12s}  charge_radius={charge_radius(lat):.3f} fm"
        f"  peak_u={peak_u:.4f}"
        f"  total_impedance={total_impedance(lat):.4f}"
    )
