"""
Truly discrete Kelvin-cell lattice.

Primary state variable: lam (winding occupation per cell).
Throughput:            sigma = 1 / (1 + lam)
Time dilation:         1 + lam  relative to void

Void cell:  lam = 0     →  sigma = 1        (full throughput)
BH core:    lam = 2879  →  sigma = 1/2880   (Sgr A* / S2 periapsis anchor)

NOT stored: chi, ell.  These are descriptions, not variables.
  chi ≈ lam / (1 + lam)  — fraction of throughput consumed by winding
  ell = 1.0 always       — cell geometry is uniform; bending comes from sigma

Every cell ticks once per global step.  No sub-cycling.
dt is the global Planck tick, sized for CFL stability of the wave equation.
The time dilation between any two cells is simply sigma_a / sigma_b = (1+lam_b)/(1+lam_a).
"""

from collections import deque

import numpy as np

from equilibrium.math.geo.kelvin import KelvinLattice

from equilibrium.substrate.lattice import W_HEX as _W_HEX, W_SQ as _W_SQ, _W  # noqa: E402  (sqrt3 : 1)

N_SAGA: int = 2880  # Sgr A* tick ratio — S2 periapsis anchor
LAM_MAX: float = N_SAGA - 1  # = 2879 — BH core; sigma = 1/2880


class DiscreteLattice:
    """
    Kelvin-cell lattice where winding occupation (lam) is the primary variable.

    Fields
    ------
    pos   (N, 3)    cell centre positions
    lam   (N,)      winding occupation — the only state that encodes mass/gravity
    u     (N,)      essence (scalar wave amplitude)
    phi   (N, 14)   directed face fluxes — 8 hex then 6 square faces
    """

    def __init__(self, radius_cells: float = None):
        self.pos = np.empty((0, 3), dtype=np.float64)
        self.lam = np.empty(0, dtype=np.float64)
        self.u = np.empty(0, dtype=np.float64)
        self.phi = np.empty((0, 14), dtype=np.float64)
        self._nb = np.empty((0, 14), dtype=np.int32)

        if radius_cells is not None:
            self._build_sphere(radius_cells)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_sphere(self, radius_cells: float):
        geo = KelvinLattice()
        pts = geo.generate_sphere(radius_cells)
        idx_map, key_fn = geo.point_index_map(pts)
        N = len(pts)

        nb = np.full((N, 14), -1, dtype=np.int32)
        for i, p in enumerate(pts):
            for k, off in enumerate(geo.all_offsets):
                key = key_fn(p + off)
                if key in idx_map:
                    nb[i, k] = idx_map[key]

        self.pos = pts
        self.lam = np.zeros(N, dtype=np.float64)  # void everywhere initially
        self.u = np.zeros(N, dtype=np.float64)
        self.phi = np.zeros((N, 14), dtype=np.float64)
        self._nb = nb

    # ------------------------------------------------------------------
    # Winding initialisation
    # ------------------------------------------------------------------

    def set_radial_lam(self, centre_idx: int, lam_func):
        """
        Set winding occupation by BFS hop distance from centre.

        lam_func(hop) -> lam value.
        hop=0 is the centre (BH/knot); lam should decrease to 0 in void.
        """
        visited = {centre_idx: 0}
        queue = deque([(centre_idx, 0)])
        while queue:
            i, hop = queue.popleft()
            self.lam[i] = float(np.clip(lam_func(hop), 0.0, LAM_MAX))
            for j in self._nb[i]:
                if j >= 0 and j not in visited:
                    visited[j] = hop + 1
                    queue.append((j, hop + 1))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def N(self) -> int:
        return len(self.pos)

    @property
    def sigma(self) -> np.ndarray:
        return 1.0 / (1.0 + self.lam)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def stable_dt(self) -> float:
        """CFL bound: max sigma is 1.0 (void cells), same as uniform lattice."""
        return 0.4 / float((_W**2).sum())

    def step(self, dt: float):
        """
        Symplectic Euler — one global tick, all cells.

        Throughput sigma = 1/(1+lam) per cell.
        Face weights W_k are the bare Kelvin topology integers.
        No ell variation, no chi dynamics, no sub-cycling.
        """
        nb = self._nb
        valid = nb >= 0
        safe_nb = np.where(valid, nb, 0)
        face_w = np.where(valid, _W[None, :], 0.0)

        sigma = self.sigma

        # Step 1: u from current phi
        du = -dt * sigma * (face_w * self.phi).sum(axis=1)
        u_new = self.u + du

        # Step 2: phi from new u — sigma_face is average of both sides
        u_nb = np.where(valid, u_new[safe_nb], u_new[:, None])
        sigma_nb = np.where(valid, sigma[safe_nb], sigma[:, None])
        sigma_face = (sigma[:, None] + sigma_nb) * 0.5
        self.phi += np.where(
            valid,
            -dt * sigma_face * face_w * (u_nb - u_new[:, None]),
            0.0,
        )
        self.u = u_new


# ------------------------------------------------------------------
# Photon helpers
# ------------------------------------------------------------------


def init_photon_packet(
    lat,
    origin: np.ndarray,
    direction: np.ndarray,
    wavelength: float,
    amplitude: float = 1.0,
) -> None:
    """
    Gaussian u-wave packet centred at `origin`, carrier along `direction`.
    phi = 0 — packet splits into two halves; track forward component with a mask.
    wavelength : Gaussian sigma AND carrier wavelength in cell units (keep ≥ 2).
    """
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    r = lat.pos - np.asarray(origin, dtype=float)
    env = np.exp(-np.sum(r**2, axis=1) / (2.0 * wavelength**2))
    carrier = np.cos(2.0 * np.pi / wavelength * (r @ d))
    lat.u[:] = amplitude * env * carrier
    lat.phi[:] = 0.0


def photon_energy_centroid(lat, mask: np.ndarray = None) -> np.ndarray:
    """Energy-weighted position centroid of the u-field."""
    e = lat.u**2
    if mask is not None:
        e = e * mask
    s = e.sum()
    if s == 0.0:
        return lat.pos.mean(axis=0)
    return (lat.pos * e[:, None]).sum(axis=0) / s
