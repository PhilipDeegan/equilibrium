"""
Aharonov-Bohm experiment on the Kelvin cell substrate.

Setup
-----
A solenoid runs along z through the origin — a tube of cells with fixed
integer circulation. Outside the tube B=0 but the accumulated phi along
any path encircling the tube is non-zero (= flux_quanta).

A Gaussian pulse in u is injected on one side (-x). It splits around the
solenoid and recombines on the other side (+x). The interference pattern
at the detector line depends on the enclosed flux.

With flux_quanta=0: symmetric interference (reference).
With flux_quanta=n: shifted interference — the shift is the AB phase.

No physical constants. The flux quantum is the integer count of lattice
circulation units enclosed. Comparison with zero-flux run reveals the
phase shift directly as a lattice-integer quantity.
"""

import numpy as np

from .lattice import SubstrateLattice


def _solenoid_cells(lat: SubstrateLattice, radius: float) -> np.ndarray:
    """Indices of cells within radius of the z-axis (the solenoid tube)."""
    r_xy = np.sqrt(lat.pos[:, 0] ** 2 + lat.pos[:, 1] ** 2)
    return np.where(r_xy <= radius)[0]


def _set_solenoid(lat: SubstrateLattice, sol_idx: np.ndarray, flux_quanta: int):
    """
    Initialise the A-field (phi) for a solenoid along z through the origin.

    For each directed face (i→j), phi[i,k] = flux_quanta * Δθ / (2π)
    where Δθ is the change in azimuthal angle around the solenoid axis.

    This guarantees:
      - sum of phi around any loop enclosing the solenoid = flux_quanta
      - sum of phi around any loop not enclosing       = 0

    Solenoid cells themselves are fixed (B ≠ 0 inside, enforced by holding
    their phi values constant throughout the run).
    """
    from equilibrium.math.geo.kelvin import KelvinLattice

    geo = KelvinLattice(lattice_spacing=1.0)
    offsets = geo.all_offsets  # (14, 3)

    for i in range(lat.N):
        x, y = lat.pos[i, 0], lat.pos[i, 1]
        theta_i = np.arctan2(y, x)
        for k, off in enumerate(offsets):
            j = lat.nb[i, k]
            if j < 0:
                continue
            xj, yj = lat.pos[j, 0], lat.pos[j, 1]
            theta_j = np.arctan2(yj, xj)
            dtheta = theta_j - theta_i
            # wrap to [-pi, pi]
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
            lat.phi[i, k] = flux_quanta * dtheta / (2 * np.pi)

    # freeze the solenoid cells
    for i in sol_idx:
        lat.fixed_phi[i, :] = True


def _inject_pulse(
    lat: SubstrateLattice, x0: float, sigma: float, amplitude: float = 1.0
):
    """Gaussian pulse in u centred at x=x0, spread sigma."""
    dx = lat.pos[:, 0] - x0
    lat.u += amplitude * np.exp(-(dx**2) / (2.0 * sigma**2))


def run(
    radius_cells: float = 20.0,
    spacing: float = 1.0,
    solenoid_radius: float = 1.5,
    flux_quanta: int = 1,
    pulse_x: float = None,
    pulse_sigma: float = 3.0,
    n_steps: int = 200,
):
    """
    Run one AB experiment with the given flux_quanta through the solenoid.

    The solenoid is always present. Compare two calls — flux_quanta=0 vs
    flux_quanta=n — to isolate the AB phase shift.

    Returns the SubstrateLattice after n_steps.
    """
    lat = SubstrateLattice(radius_cells=radius_cells, spacing=spacing)
    sol = _solenoid_cells(lat, solenoid_radius * spacing)
    _set_solenoid(lat, sol, flux_quanta)

    x_max = lat.pos[:, 0].max()
    px = pulse_x if pulse_x is not None else -x_max * 0.6
    _inject_pulse(lat, px, pulse_sigma)
    lat.run(n_steps)
    return lat


def detector_slice(
    lat: SubstrateLattice, x_detect: float, thickness: float = 2.0
) -> tuple:
    """
    u values at cells near x=x_detect, returned as (y, u) sorted by y.
    Use to compare the interference pattern between ref and ab runs.
    """
    mask = np.abs(lat.pos[:, 0] - x_detect) <= thickness
    y = lat.pos[mask, 1]
    u = lat.u[mask]
    order = np.argsort(y)
    return y[order], u[order]
