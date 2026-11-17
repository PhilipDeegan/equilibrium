"""
DST simulation runner for the hydrogen Bohr model.

The proton is a fixed high-u knot at the origin.
The electron moat is an initial deficit shell that evolves to equilibrium.
The Bohr radius emerges as the equilibrium radius of that moat.

Usage:
    from equilibrium.physical.dst.sim import bohr

    r_moat, lat = bohr(cell_size=0.05, radius_cells=40, n_steps=2000)
    print(f"Moat radius: {r_moat:.3f}  (Bohr = {A_0:.3f} Planck lengths)")
"""

import numpy as np
from .lattice import DSTLattice

# Planck-unit physical constants
M_E = 4.18542e-23  # electron mass / Planck mass
ALPHA = 7.29735e-3  # fine-structure constant
A_0 = 1.0 / (M_E * ALPHA)  # Bohr radius in Planck lengths ≈ 3.275e24

# H2 bond length in a_0 units
H2_BOND_A0 = 1.4  # ~1.4 a_0


def bohr(
    cell_size: float = 0.1,
    radius_cells: float = 60.0,
    proton_u: float = 1.0,
    n_steps: int = 5000,
    K: float = 1.0,
    alpha: float = 0.5,
):
    """
    Simulate the hydrogen atom on the Kelvin cell lattice.

    The Bohr radius emerges from L co-evolution with u: high-u cells compress
    (smaller L, slower σ), creating a field-dependent diffusion medium where
    the moat deficit settles at the equilibrium radius without eigenfunctions.

    Parameters
    ----------
    cell_size    : L in units of a_0  (fraction of Bohr radius per cell)
    radius_cells : sphere size in cell units — must be > moat_radius_cells + buffer
    proton_u     : initial essence of the proton cell
    n_steps      : number of update ticks
    K            : substrate coupling
    alpha        : L-u coupling strength

    Returns
    -------
    r_moat : equilibrium moat radius in units of a_0
    lat    : the DSTLattice object after simulation
    """
    lat = DSTLattice(cell_size=cell_size, radius_cells=radius_cells, K=K, alpha=alpha)
    lat.set_proton(u_val=proton_u)

    # Place the initial moat at 2 × Bohr radius (in cells: 2.0/cell_size)
    # Lattice must extend beyond this — radius_cells default ensures it
    moat_r_cells = 2.0 / cell_size
    lat.set_moat(u_val=-proton_u, radius_cells=moat_r_cells, width_cells=2.0)

    lat.run(n_steps)

    r_moat = lat.moat_radius() / cell_size  # in units of a_0
    return r_moat, lat


def h2_scan(
    separations=None,
    cell_size: float = 0.1,
    radius_cells: float = 60.0,
    proton_u: float = 1.0,
    n_steps: int = 3000,
    K: float = 1.0,
    alpha: float = 0.5,
):
    """
    Scan proton-proton separations and measure field tension at equilibrium.

    Two protons are placed at ±d/2 along the x-axis with moats around each.
    The gradient_variance() at equilibrium is a proxy for total field tension.
    The minimum over separations should correspond to the H2 bond length (~1.4 a_0).

    Returns list of (separation_a0, grad_var) tuples.
    """
    if separations is None:
        separations = np.arange(0.5, 3.25, 0.25)

    results = []
    print(f"\n{'sep (a0)':>10}  {'grad_var':>14}  {'N_cells':>8}")
    for d in separations:
        lat = DSTLattice(cell_size=cell_size, radius_cells=radius_cells, K=K, alpha=alpha)
        c1 = np.array([-d / 2.0, 0.0, 0.0])
        c2 = np.array([+d / 2.0, 0.0, 0.0])
        lat.set_proton_at(c1, proton_u)
        lat.set_proton_at(c2, proton_u)
        moat_r = 2.0 / cell_size
        lat.set_moat_at(c1, -proton_u, radius_cells=moat_r, width_cells=2.0)
        lat.set_moat_at(c2, -proton_u, radius_cells=moat_r, width_cells=2.0)
        lat.run(n_steps)
        gv = lat.gradient_variance()
        results.append((float(d), gv))
        print(f"{d:>10.2f}  {gv:>14.4f}  {lat.N:>8d}")

    best = min(results, key=lambda x: x[1])
    print(f"\nMinimum tension at d={best[0]:.2f} a0  (H2 bond target: {H2_BOND_A0} a0)")
    return results


def convergence_table(cell_sizes=(0.15, 0.10, 0.07), radius_cells=60.0, n_steps=5000):
    """
    Print the moat equilibrium radius for decreasing cell sizes.
    Expected: moat_radius → 1.0 a_0 as cell_size → 0.
    """
    print(f"\n{'cell_size':>12}  {'N_cells':>8}  {'r_moat (a0)':>14}  {'err vs 1.0':>12}")
    for cs in cell_sizes:
        r, lat = bohr(cell_size=cs, radius_cells=radius_cells, n_steps=n_steps)
        err = abs(r - 1.0)
        print(f"{cs:>12.3f}  {lat.N:>8d}  {r:>14.4f}  {err:>12.4f}")
