"""
Kelvin cell (truncated octahedron) lattice — the Wigner–Seitz cell of the BCC lattice.

Geometry (all distances in Planck lengths, l_P = 1):
  Each cell has 14 face-sharing neighbors:
    8 through hexagonal faces  — nearest,      d1 = lattice_spacing (= 1 in Planck units)
    6 through square faces     — next-nearest,  d2 = (2/√3) * lattice_spacing ≈ 1.155

  BCC conventional cubic cell side: a = d1 * 2/√3

Discrete Laplacian (isotropic on BCC, exact to 2nd order):
  ∇²f ≈ (1/a²) [w1 Σ_{8 hex} (f_j - f_i) + w2 Σ_{6 sq} (f_j - f_i)]
  with w1 = 2/3, w2 = 1/3  (minimises 4th-order anisotropy on BCC)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class KelvinLattice:
    """
    BCC lattice with Kelvin (truncated octahedron) Wigner–Seitz cells.

    lattice_spacing: nearest cell-centre distance d1 (Planck lengths).
    """

    lattice_spacing: float = 1.0

    def __post_init__(self):
        self.d1 = self.lattice_spacing
        self.a = self.d1 * 2.0 / np.sqrt(3)  # BCC conventional cell side
        self.d2 = self.a  # square-face neighbour distance
        self._hex_off = None
        self._sq_off = None

    # ------------------------------------------------------------------
    # Neighbour offsets
    # ------------------------------------------------------------------

    @property
    def hex_offsets(self) -> np.ndarray:
        """(8, 3) offsets to hexagonal-face neighbours (distance d1)."""
        if self._hex_off is None:
            h = self.a / 2
            self._hex_off = np.array(
                [
                    [sx * h, sy * h, sz * h]
                    for sx in (-1, 1)
                    for sy in (-1, 1)
                    for sz in (-1, 1)
                ],
                dtype=float,
            )
        return self._hex_off

    @property
    def sq_offsets(self) -> np.ndarray:
        """(6, 3) offsets to square-face neighbours (distance d2 = a)."""
        if self._sq_off is None:
            a = self.a
            self._sq_off = np.array(
                [[-a, 0, 0], [a, 0, 0], [0, -a, 0], [0, a, 0], [0, 0, -a], [0, 0, a]],
                dtype=float,
            )
        return self._sq_off

    @property
    def all_offsets(self) -> np.ndarray:
        """(14, 3) all neighbour offsets: 8 hex then 6 sq."""
        return np.vstack([self.hex_offsets, self.sq_offsets])

    @property
    def laplacian_weights(self) -> np.ndarray:
        """(14,) weights for isotropic BCC discrete Laplacian (w1=2/3, w2=1/3)."""
        return np.array([2 / 3] * 8 + [1 / 3] * 6, dtype=float)

    # ------------------------------------------------------------------
    # Lattice generation
    # ------------------------------------------------------------------

    def generate_sphere(self, radius: float) -> np.ndarray:
        """Return (N, 3) array of all BCC lattice points with |r| ≤ radius."""
        a = self.a
        n = int(np.ceil(radius / a)) + 1
        rng = np.arange(-n, n + 1, dtype=float)
        Ix, Jx, Kx = np.meshgrid(rng, rng, rng, indexing="ij")
        grid = np.stack([Ix.ravel(), Jx.ravel(), Kx.ravel()], axis=1)  # (M, 3)

        r2 = radius**2

        # Corner atoms: integer multiples of a
        pts_corner = grid * a
        mask = np.sum(pts_corner**2, axis=1) <= r2

        # Body-centre atoms: half-integer multiples of a
        pts_bc = (grid + 0.5) * a
        mask_bc = np.sum(pts_bc**2, axis=1) <= r2

        parts = []
        if mask.any():
            parts.append(pts_corner[mask])
        if mask_bc.any():
            parts.append(pts_bc[mask_bc])
        return np.vstack(parts)

    def generate_cylinder(
        self, radius: float, half_length: float, axis: int = 2
    ) -> np.ndarray:
        """Return (N, 3) array of all BCC lattice points inside a cylinder:
        transverse distance from `axis` <= radius AND |coord along axis| <= half_length.

        Same candidate-point generation as generate_sphere, just a different mask --
        for a long, thin domain (e.g. a photon/canal propagating along one axis over
        a large separation) this is O(radius^2 * half_length) cells instead of
        O(radius_sphere^3), since cost no longer scales with the cube of the
        longest dimension.
        """
        a = self.a
        n_perp = int(np.ceil(radius / a)) + 1
        n_ax = int(np.ceil(half_length / a)) + 1
        rng_perp = np.arange(-n_perp, n_perp + 1, dtype=float)
        rng_ax = np.arange(-n_ax, n_ax + 1, dtype=float)
        axes = [rng_perp, rng_perp, rng_perp]
        axes[axis] = rng_ax
        Ix, Jx, Kx = np.meshgrid(*axes, indexing="ij")
        grid = np.stack([Ix.ravel(), Jx.ravel(), Kx.ravel()], axis=1)  # (M, 3)

        perp_axes = [i for i in range(3) if i != axis]
        r2 = radius**2

        def _mask(pts):
            perp2 = pts[:, perp_axes[0]] ** 2 + pts[:, perp_axes[1]] ** 2
            return (perp2 <= r2) & (np.abs(pts[:, axis]) <= half_length)

        pts_corner = grid * a
        mask = _mask(pts_corner)

        pts_bc = (grid + 0.5) * a
        mask_bc = _mask(pts_bc)

        parts = []
        if mask.any():
            parts.append(pts_corner[mask])
        if mask_bc.any():
            parts.append(pts_bc[mask_bc])
        return np.vstack(parts)

    def point_index_map(self, points: np.ndarray) -> dict:
        """Build {rounded-key → index} dict for O(1) neighbour lookup."""
        scale = 2.0 / self.a  # maps a/2 steps → integers

        def key(p):
            return tuple(np.round(p * scale).astype(np.int64))

        return {key(p): i for i, p in enumerate(points)}, key
