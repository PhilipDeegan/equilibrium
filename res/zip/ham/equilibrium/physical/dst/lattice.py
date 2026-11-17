"""
Discrete Substrate Theory — Kelvin cell lattice.

All lengths in Planck units (l_P = 1).

Each cell has a size L (Planck lengths between adjacent cell centres).
Changing L rescales all geometric accounting uniformly:
    d_hex = L               A_hex = L^2 * sqrt(3)/4
    d_sq  = L * 2/sqrt(3)   A_sq  = L^2 / 6

The face geometry is exact for the BCC Wigner-Seitz (truncated octahedron):
    edge length s = L / sqrt(6)
    hex face: regular hexagon, area = (3*sqrt(3)/2) * s^2 = sqrt(3)/4 * L^2
    sq  face: square,          area =              s^2 = L^2/6

Fractal / multi-resolution structure
─────────────────────────────────────
The same 14-neighbour topology applies at every scale.  What changes is the
accounting (L, λ) per cell.  The resolution index λ encodes how many times a
cell has been subdivided:

    λ = 0          vacuum — one Planck cell, full throughput
    λ = 1          proton/electron knot — first subdivision level
    λ = λ_max      sintered macro-cell  — black-hole limit

Every stable structure at every scale produces a knot/moat pair: a high-u
region (small L, high λ) surrounded by a compensating low-u moat (larger L,
lower λ).  This is a product of the update rule — it repeats from the nuclear
scale (proton knot + electron moat) up to the galactic scale and beyond.

Throughput function σ(λ)
────────────────────────
    σ(λ) = L / (1 + λ)

At λ=0: σ = L (vacuum rate).  At each subdivision level the local clock slows
by a further factor of (1+λ).  Crucially, σ(λ_max) = L/(1+λ_max) > 0 always —
**there is no zero**.  A black hole (λ_max) is not "frozen"; it is ticking at
the minimum substrate rate, which only appears frozen to an outside observer
because of the extreme time-dilation ratio.
"""

import numpy as np
from equilibrium.math.geo.kelvin import KelvinLattice

# Unit face geometry (L = 1 Planck length = d1)
_A_HEX = np.sqrt(3) / 4  # ≈ 0.4330
_A_SQ = 1.0 / 6.0  # ≈ 0.1667
_D_HEX = 1.0
_D_SQ = 2.0 / np.sqrt(3)  # ≈ 1.1547

_A_UNITS = np.array([_A_HEX] * 8 + [_A_SQ] * 6, dtype=np.float64)
_D_UNITS = np.array([_D_HEX] * 8 + [_D_SQ] * 6, dtype=np.float64)


class DSTLattice:
    """
    Kelvin cell lattice carrying the DST state variables.

    Arrays of length N (one entry per cell):
        pos   (N, 3)   cell-centre positions in Planck lengths
        u     (N,)     local potential / essence
        L     (N,)     cell size in Planck lengths
        lam   (N,)     resolution index λ  (0=vacuum, 1=knot, …)
        fixed (N,)     bool — cells that do not update (e.g. proton core)

    Neighbour arrays (N × 14):
        nb  (N, 14)  neighbour index, or -1 at sphere boundary
        A   (N, 14)  face area (Planck lengths^2)
        d   (N, 14)  centre-to-centre distance (Planck lengths)
    """

    def __init__(self, cell_size: float, radius_cells: float, K: float = 1.0, alpha: float = 0.0):
        """
        cell_size    : L  — Planck lengths per cell
        radius_cells : sphere radius in cell units
        K            : substrate coupling constant  K = l_P / (t_P * Z_0)
        alpha        : L-u coupling — cells compress where u is high, expand where low
        """
        self.cell_size = float(cell_size)
        self.K = float(K)
        self._alpha = float(alpha)
        self._build(radius_cells)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self, radius_cells: float):
        L = self.cell_size
        geo = KelvinLattice(lattice_spacing=1.0)
        pts = geo.generate_sphere(radius_cells)  # (N, 3) in cell units
        N = len(pts)
        idx_map, key_fn = geo.point_index_map(pts)

        nb = np.full((N, 14), -1, dtype=np.int32)

        offsets = geo.all_offsets  # (14, 3)
        for i, p in enumerate(pts):
            for k, off in enumerate(offsets):
                key = key_fn(p + off)
                if key in idx_map:
                    nb[i, k] = idx_map[key]

        # Unit face geometry (L-independent).
        # sigma(cell) = L  is the throughput: at L=1 (Planck) rate = 1 tick/Planck-time.
        # Larger cells have larger sigma → faster updates.
        # Full coupling per face = sigma * A_unit / D_unit  (scales as L).
        self._face_unit = _A_UNITS / _D_UNITS  # (14,)  — geometry factor, dimensionless

        self.pos = pts * L
        self.u = np.zeros(N, dtype=np.float64)
        self.L = np.full(N, L, dtype=np.float64)  # sigma = L per cell
        self.lam = np.zeros(N, dtype=np.int32)
        self.fixed = np.zeros(N, dtype=bool)
        self.nb = nb
        self.N = N

        # safe neighbour indices: replace -1 with own index → zero flux at boundary
        self._safe_nb = nb.copy()
        for i in range(N):
            self._safe_nb[i, nb[i] < 0] = i

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def set_proton(self, u_val: float, radius_cells: float = 0.0):
        """
        Fix cells within radius_cells of origin as the proton knot.
        At minimum the single origin cell is fixed.
        The proton is stable — it does not diffuse.
        """
        r = np.linalg.norm(self.pos, axis=1)
        mask = r <= radius_cells * self.cell_size
        if not mask.any():
            mask[np.argmin(r)] = True
        self.u[mask] = u_val
        self.fixed[mask] = True

    def set_moat(self, u_val: float, radius_cells: float, width_cells: float = 1.0):
        """Place an electron moat (deficit, u_val < 0) at a given shell radius."""
        r = np.linalg.norm(self.pos, axis=1)
        L = self.cell_size
        mask = (r >= (radius_cells - width_cells * 0.5) * L) & (
            r <= (radius_cells + width_cells * 0.5) * L
        )
        self.u[mask] = u_val

    def set_proton_at(self, centre: np.ndarray, u_val: float, radius_cells: float = 0.0):
        """Fix cells near centre as a proton knot."""
        r = np.linalg.norm(self.pos - np.asarray(centre, dtype=float), axis=1)
        mask = r <= radius_cells * self.cell_size
        if not mask.any():
            mask[np.argmin(r)] = True
        self.u[mask] = u_val
        self.fixed[mask] = True

    def set_proton_winding(
        self,
        R: float,
        B0: float = 1.0,
        charge: float = 1.0,
        lam_knot: int = 500,
        centre: np.ndarray = None,
        chirality: int = 1,
    ):
        """
        Initialise the proton as an open winding — the through-flux redirect.

        The proton is a Taylor spheromak winding (toroidal, self-confining,
        topologically stable) plus a monopole charge component (the net outward
        u-excess from the convergent pressure passing through and back out).

        Neutron = closed winding (sum(u) ≈ 0).
        Proton  = open winding  (sum(u) > 0 for chirality=+1).
        Antiproton = chirality=-1 (sum(u) < 0, opposite through-flux direction).

        Cells are made high-λ (kinetically slow, not fixed) so total u is
        conserved and the proton persists through kinetic suppression rather
        than by fiat.

        Parameters
        ----------
        R         : winding confinement radius (same units as pos)
        B0        : spheromak winding amplitude
        charge    : monopole (outward flux) amplitude — net u-excess = charge
        lam_knot  : λ for proton cells — sets kinetic suppression rate
        centre    : (3,) proton centre; defaults to origin
        chirality : +1 for proton, -1 for antiproton
        """
        from equilibrium.physical.atom.spheromak import taylor_u_profile

        if centre is None:
            centre = np.zeros(3)
        centre = np.asarray(centre, dtype=float)

        r = np.linalg.norm(self.pos - centre, axis=1)
        mask = r <= R

        # Winding component: Taylor spheromak (zero net flux, self-confining)
        u_wind = taylor_u_profile(r, R, B0) * chirality

        # Charge component: Gaussian monopole (net outward flux = charge)
        # sigma ~ R/2 so it sits inside the confinement radius
        sigma = R / 2.0
        u_charge = charge * chirality * np.exp(-(r**2) / (2.0 * sigma**2))

        self.u[mask] += u_wind[mask] + u_charge[mask]
        self.lam[mask] = lam_knot

    def set_moat_at(self, centre: np.ndarray, u_val: float, radius_cells: float, width_cells: float = 1.0):
        """Place a moat shell at radius_cells from centre."""
        r = np.linalg.norm(self.pos - np.asarray(centre, dtype=float), axis=1)
        L = self.cell_size
        inner = (radius_cells - width_cells * 0.5) * L
        outer = (radius_cells + width_cells * 0.5) * L
        mask = (r >= inner) & (r <= outer)
        self.u[mask] = u_val

    def set_neutron_winding(
        self,
        winding_pts: np.ndarray,
        winding_u: np.ndarray,
        centre: np.ndarray = None,
        phare_scale: float = 0.18,
        amplitude: float = 1.0,
    ):
        """
        Superimpose the signed toroidal winding (recovered from the PHARE
        toroidal dipole B-field) onto the lattice u-field.

        Unlike set_spheromak(), which uses a positive-only scalar profile,
        this uses the signed u recovered by the inverse solver — positive on
        one half of the winding, negative on the other.  The net sum is zero,
        so no outward flux escapes: the winding is genuinely closed.

        The neutron centre is snapped to the nearest host lattice point so
        that the winding and host lattices are commensurate (1-to-1 cell
        mapping).  When multiple winding cells map to the same host cell
        their u contributions are averaged, then the result is re-centred
        to guarantee zero net winding sum.

        The winding cells are fixed in place.  In DST the proton's surrounding
        u-gradient is what physically holds the winding shut; fixing the cells
        models that constraint in this first approximation.

        Parameters
        ----------
        winding_pts   : (M, 3) Kelvin lattice points from solve_u_from_b
        winding_u     : (M,)   signed u values from solve_u_from_b
        centre        : (3,) desired neutron centre in Planck lengths.
                        Snapped to nearest host lattice point.  Defaults to origin.
        phare_scale   : PHARE units per winding lattice cell — must match the
                        value used when solve_u_from_b was called
        amplitude     : overall scale of the winding u relative to the proton
        """
        if centre is None:
            centre = np.zeros(3)

        # Snap to nearest host lattice point for commensurate mapping
        snap_dists = np.linalg.norm(self.pos - centre, axis=1)
        snap_idx = int(np.argmin(snap_dists))
        centre = self.pos[snap_idx]

        # Scale: 1 winding cell = phare_scale Planck lengths (both are Kelvin lattices)
        scale = self.cell_size / phare_scale

        # Accumulate winding contributions per host cell
        cell_u = {}  # host_cell_idx -> list of winding u values
        for p_w, u_w in zip(winding_pts, winding_u):
            pos_target = centre + p_w * scale
            dists = np.linalg.norm(self.pos - pos_target, axis=1)
            i = int(np.argmin(dists))
            if dists[i] <= self.cell_size * 1.5:
                cell_u.setdefault(i, []).append(float(u_w))

        if not cell_u:
            return

        # Average multi-contributions, then re-centre to enforce sum=0
        host_idx = np.array(list(cell_u.keys()), dtype=np.int32)
        host_vals = np.array([np.mean(v) for v in cell_u.values()])
        host_vals -= host_vals.mean()

        self.u[host_idx] += amplitude * host_vals
        self.fixed[host_idx] = True

    def set_spheromak(self, R: float, B0: float = 1.0, centre: np.ndarray = None):
        """
        Initialise the u-field with the Taylor spheromak profile (neutron mode).

        This sets u(r) = B₀ j₁(kr) / r with k = 4.493 / R, which is the
        scalar magnitude approximation of the Woltjer-Taylor force-free
        winding.  The profile is self-confining: u = 0 at r = R and at r = 0.

        The neutron is a collective pattern across an enormous number of
        Planck cells.  Each simulation cell here is a coarse-grained
        representative of that collective; this profile describes the
        impedance envelope of the winding, not individual lattice hops.

        Parameters
        ----------
        R      : confinement radius in the same Planck-length units as pos
        B0     : amplitude (substrate impedance scale)
        centre : (3,) centre of the spheromak; defaults to origin
        """
        from equilibrium.physical.atom.spheromak import taylor_u_profile

        if centre is None:
            centre = np.zeros(3)
        r = np.linalg.norm(self.pos - centre, axis=1)
        self.u = taylor_u_profile(r, R, B0)

    # ------------------------------------------------------------------
    # Time evolution
    # ------------------------------------------------------------------

    def sigma(self) -> np.ndarray:
        """
        Per-cell throughput: σ(λ) = L / (1 + λ).

        Encodes local rate of time.  Vacuum cells (λ=0) tick at rate L.
        Each subdivision level (λ=1,2,…) slows the local clock proportionally.
        Always positive — the substrate never stops.
        """
        return self.L / (1.0 + self.lam)

    def stable_dt(self) -> float:
        """
        Conservative stable timestep for explicit integration.
        Uses max σ over all cells; higher-λ cells have reduced σ so the
        vacuum cells (λ=0) typically set the limit.
        """
        max_sigma = self.sigma().max()
        total_geom = self._face_unit.sum()
        return 0.4 / (self.K * max_sigma * total_geom)

    def step(self, dt: float):
        """
        One tick of the DST update rule:

            du_i = dt * K * σ(λ_i) * Σ_j [ face_unit_k * (u_j - u_i) ]

        σ(λ) = L/(1+λ): vacuum cells tick at full rate; knots and sintered
        cells tick proportionally slower.  Boundary faces and fixed cells
        contribute zero change.

        If alpha > 0, cell sizes co-evolve with u after each tick:
            L_i = L0 * (1 - alpha * u_i)
        High-u cells compress (smaller L, slower σ); low-u cells expand.
        This is regression to the local mean in a field-dependent medium —
        the mechanism by which the Bohr radius emerges without eigenfunctions.
        """
        u = self.u
        valid = self.nb >= 0  # (N, 14)
        u_nb = u[self._safe_nb]  # (N, 14)
        du_face = self._face_unit * (u_nb - u[:, None])  # (N, 14)
        du_face[~valid] = 0.0
        du = dt * self.K * self.sigma() * du_face.sum(axis=1)
        self.u = np.where(self.fixed, u, u + du)
        if self._alpha > 0.0:
            new_L = self.cell_size * (1.0 - self._alpha * self.u)
            self.L = np.where(
                self.fixed,
                self.L,
                np.maximum(self.cell_size * 0.01, new_L),
            )

    def run(self, n_steps: int, dt: float = None):
        """Run n_steps ticks. Uses stable_dt() if dt not given."""
        if dt is None:
            dt = self.stable_dt()
        for _ in range(n_steps):
            self.step(dt)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def radial_profile(self, n_bins: int = 80):
        """
        Spherically averaged radial profile of u.
        Returns (r_centres, u_mean) in Planck lengths.
        """
        r = np.linalg.norm(self.pos, axis=1)
        edges = np.linspace(0.0, r.max(), n_bins + 1)
        rc = 0.5 * (edges[:-1] + edges[1:])
        um = np.full(n_bins, np.nan)
        for i in range(n_bins):
            mask = (r >= edges[i]) & (r < edges[i + 1])
            if mask.any():
                um[i] = self.u[mask].mean()
        return rc, um

    def gradient_variance(self) -> float:
        """Total squared u-gradient across all faces — proxy for field tension."""
        u = self.u
        u_nb = u[self._safe_nb]
        diffs = (u_nb - u[:, None]) ** 2
        diffs[self.nb < 0] = 0.0
        return float((self._face_unit * diffs).sum())

    def moat_radius(self) -> float:
        """
        Radius (Planck lengths) of the minimum-u shell — the electron moat.
        Returns the r at which u is most negative.
        """
        r, um = self.radial_profile()
        valid = ~np.isnan(um)
        return float(r[valid][np.argmin(um[valid])])

    # ------------------------------------------------------------------
    # Gravity
    # ------------------------------------------------------------------

    def set_gravity(self, rs_planck: float, r_clamp: float = None):
        """
        Impose a Schwarzschild weak-field gravity profile on the lattice.

        u(r) = rs / (2r)       [= GM/(rc²) in Planck units where G=c=1]
        L(r) = cell_size × (1 − u(r))

        Parameters
        ----------
        rs_planck : Schwarzschild radius of the central mass in Planck lengths
                    (= 2GM/c² = 2M_planck in Planck units).
        r_clamp   : minimum radius for potential regularisation.
                    Defaults to cell_size.
        """
        if r_clamp is None:
            r_clamp = self.cell_size
        r = np.linalg.norm(self.pos, axis=1)
        r = np.maximum(r, r_clamp)
        u = rs_planck / (2.0 * r)
        self.u = u
        self.L = self.cell_size * (1.0 - u)

    def clock_rate_field(self) -> np.ndarray:
        """
        Per-cell local clock rate = L(r) / L_vacuum.
        Dimensionless; values < 1 near mass (time runs slow there).
        """
        return self.L / self.cell_size
