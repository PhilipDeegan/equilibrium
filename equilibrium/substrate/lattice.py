"""
Kelvin cell substrate — adaptive tile graph.

Each SubstrateTile is a Kelvin cell node. It owns its 14 neighbour pointers,
allowing the topology to be reconfigured at any resolution level.  All field
arrays (u, phi, lam, …) live on SubstrateLattice so they remain contiguous
and vectorisable.

Constitutive relation
---------------------
    σ  =  1 / (1 + λ)    local throughput  (1 in void, 1/2880 at BH)
    χ  =  λ / (1 + λ)    winding occupation (emergent — NOT a stored field)

λ is the primary state variable encoding winding content.
χ and σ are derived from λ; they are never stored or updated independently.

T_void (λ=0, σ=1) is the master tick rate.  Every cell ticks once per global
step.  Time dilation at a cell = 1/σ = 1 + λ relative to void.
Light bends toward mass because it follows the path of least resistance
through the throughput gradient ∇σ.

Face coupling between adjacent tiles i and j
--------------------------------------------
    w_k^{ij} = W_k    (bare Kelvin topology weights)

Wave update rule (one tick)
---------------------------
    Δu_i    = −dt · σ_i · Σ_k  w_k · φ_{ik}
    Δφ_{ij} = −dt · σ̄_{ij} · w_k · (u_j − u_i)

where σ̄_{ij} = (σ_i + σ_j)/2 preserves flux antisymmetry across σ gradients.

Note: ell is kept as a unit array for C++ extension compatibility.
"""

from collections import deque

import numpy as np

# Per-leg face weights. The Kelvin geometry fixes the Hodge star |dual face| /
# |link| = 3:1 (W_STAR: complete operators, e.g. a Laplacian); step() applies W
# on both legs, so W = sqrt(W_STAR). Before 2026-09-28: W_HEX = 3 (star 9:1).
# exp/2026/09/28/hodge_star; matches C++ bcc::face_weight / face_star.
W_HEX: float = 3.0**0.5
W_SQ: float = 1.0
_W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
W_STAR = np.array([3.0] * 8 + [1.0] * 6)  # = _W**2

# Physical centre-to-centre distance per face type (in lattice units)
# hex faces: d₁ = 1;  sq faces: d₂ = 2/√3 ≈ 1.1547
_D_FACE = np.array([1.0] * 8 + [2.0 / 3.0**0.5] * 6)

# Reverse-face map: face k_rev is the face pointing opposite to face k.
# Hex faces 0-7 in lex order (sx,sy,sz)∈{-1,+1}³: reverse flips all signs.
#   0(-,-,-)↔7(+,+,+)  1(-,-,+)↔6(+,+,-)  2(-,+,-)↔5(+,-,+)  3(-,+,+)↔4(+,-,-)
# Sq faces 8-13 (±x,±y,±z): 8(-x)↔9(+x)  10(-y)↔11(+y)  12(-z)↔13(+z)
_K_REV = np.array([7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 11, 10, 13, 12], dtype=np.intp)


def wave_speed(spacing: float = 1.0) -> float:
    """step()'s long-wavelength speed, per lattice time unit (dt as passed to
    step()): c^2 = 1/2 sum_k W_k^2 d_kx^2 with hops d_k in units of the hex
    distance, times spacing. 3.651 at the pre-2026-09-28 W = 3:1, 2.309 now."""
    from equilibrium.math.geo.kelvin import KelvinLattice

    d = KelvinLattice(lattice_spacing=1.0).all_offsets
    return float(np.sqrt(0.5 * np.sum(_W**2 * d[:, 0] ** 2))) * spacing

# Face adjacency — faces that share an edge on the Kelvin cell (truncated octahedron).
# Hex face (sx,sy,sz) is adjacent to:
#   3 sq faces: one per sign component (sx→face8/9, sy→face10/11, sz→face12/13)
#   3 hex faces: obtained by flipping any two of the three signs
# Sq face is adjacent to the 4 hex faces whose direction contains the same sign component.
# Padded to width 6; sq faces use only columns 0-3 (columns 4-5 are -1).
# Straight-through (opposite face) is NOT in the adjacency list — it is the longest path.
_FACE_ADJ = np.full((14, 6), -1, dtype=np.int32)
_FACE_ADJ[0] = [3, 5, 6, 8, 10, 12]  # (-,-,-): hex {3,5,6}  sq {8,10,12}
_FACE_ADJ[1] = [2, 4, 7, 8, 10, 13]  # (-,-,+): hex {2,4,7}  sq {8,10,13}
_FACE_ADJ[2] = [1, 4, 7, 8, 11, 12]  # (-,+,-): hex {1,4,7}  sq {8,11,12}
_FACE_ADJ[3] = [0, 5, 6, 8, 11, 13]  # (-,+,+): hex {0,5,6}  sq {8,11,13}
_FACE_ADJ[4] = [1, 2, 7, 9, 10, 12]  # (+,-,-): hex {1,2,7}  sq {9,10,12}
_FACE_ADJ[5] = [0, 3, 6, 9, 10, 13]  # (+,-,+): hex {0,3,6}  sq {9,10,13}
_FACE_ADJ[6] = [0, 3, 5, 9, 11, 12]  # (+,+,-): hex {0,3,5}  sq {9,11,12}
_FACE_ADJ[7] = [1, 2, 4, 9, 11, 13]  # (+,+,+): hex {1,2,4}  sq {9,11,13}
_FACE_ADJ[8, :4] = [0, 1, 2, 3]  # -x sq: hex with sx=-1
_FACE_ADJ[9, :4] = [4, 5, 6, 7]  # +x sq: hex with sx=+1
_FACE_ADJ[10, :4] = [0, 1, 4, 5]  # -y sq: hex with sy=-1
_FACE_ADJ[11, :4] = [2, 3, 6, 7]  # +y sq: hex with sy=+1
_FACE_ADJ[12, :4] = [0, 2, 4, 6]  # -z sq: hex with sz=-1
_FACE_ADJ[13, :4] = [1, 3, 5, 7]  # +z sq: hex with sz=+1
_K_IDX = np.arange(14, dtype=np.intp)

N_SAGA: int = 2880  # Sgr A* tick ratio — S2 periapsis anchor
L_MIN: float = 1.0 / N_SAGA  # sigma_min at BH core (≈3.47e-4)
L_MAX: float = 1.0  # sigma in void
LAM_MAX: float = N_SAGA - 1  # = 2879 — maximum winding; sigma = 1/N_SAGA at BH

# Kelvin cell (truncated octahedron) geometry at unit lattice spacing (d1=1).
# Edge length e = 1/√3.  Used as the volumetric conversion factor in step_lam_sink:
# creating winding (lam) from face flux costs N_SAGA ticks × the cell volume.
_KELVIN_EDGE = 1.0 / 3.0**0.5  # e ≈ 0.5774
_KELVIN_VOLUME = 8.0 * 2.0**0.5 * _KELVIN_EDGE**3  # V ≈ 2.177
_KELVIN_SURF = (8 * 3.0 * 3.0**0.5 / 2.0 + 6 * 1.0) * _KELVIN_EDGE**2  # S ≈ 8.928
# Isotropic sink conversion cost: N_SAGA ticks × volume of one cell
_LAM_SINK_COST = N_SAGA * _KELVIN_VOLUME  # ≈ 6271


class SubstrateTile:
    """
    One Kelvin cell node in the substrate graph.

    Owns its 14 neighbour pointers and subdivision state.  Field values
    (u, phi, ell, lam) live in the parent SubstrateLattice arrays; this
    object provides a convenient per-tile view of them.

    Subdivision (9-tree / 15-tree) is a future implementation; children
    and parent are reserved for that use.
    """

    __slots__ = ("_lat", "children", "idx", "neighbours", "parent")

    def __init__(self, idx: int, lattice: "SubstrateLattice"):
        self.idx = idx
        self._lat = lattice
        self.neighbours: list = [None] * 14
        self.parent: SubstrateTile | None = None
        self.children: list | None = None

    @property
    def pos(self) -> np.ndarray:
        return self._lat.pos[self.idx]

    @property
    def ell(self) -> float:
        return 1.0  # uniform cell size; stored array stays 1.0 for C++ compat

    @property
    def lam(self) -> float:
        return float(self._lat.lam[self.idx])

    @lam.setter
    def lam(self, v: float):
        self._lat.lam[self.idx] = float(v)

    @property
    def chi(self) -> float:
        lam = float(self._lat.lam[self.idx])
        return lam / (1.0 + lam)

    @property
    def sigma(self) -> float:
        return 1.0 / (1.0 + float(self._lat.lam[self.idx]))

    @property
    def u(self) -> float:
        return float(self._lat.u[self.idx])

    @u.setter
    def u(self, v: float):
        self._lat.u[self.idx] = float(v)

    def subdivide(self):
        # placeholder — 9-tree or 15-tree split based on χ / ∇χ thresholds
        ...


class SubstrateLattice:
    """
    Global Kelvin substrate graph.

    All field arrays are stored here for contiguous, vectorisable access.
    SubstrateTile objects hold the neighbour pointers and form the graph
    topology; that topology can be reconfigured at each junction without
    rebuilding the arrays.

    Array layout
    ------------
    pos       (N, 3)       cell centre positions
    ell       (N,)         always 1.0 — kept for C++ extension compatibility
    lam       (N,) float64 winding occupation λ  (primary state)
    u         (N,)         scalar essence
    phi       (N, 14)      directed face flux
    fixed_u   (N,) bool    frozen u mask
    fixed_phi (N, 14) bool frozen phi mask

    Derived (computed, not stored)
    --------------------------------
    chi   = lam / (1 + lam)
    sigma = 1 / (1 + lam)
    nb    = (N, 14) int32 neighbour index array, rebuilt when topology changes
    """

    def __init__(
        self,
        radius_cells: float = None,
        spacing: float = 1.0,
    ):
        self.tiles: list[SubstrateTile] = []
        self.pos = np.empty((0, 3), dtype=np.float64)
        self.ell = np.empty(0, dtype=np.float64)
        self.lam = np.empty(0, dtype=np.float64)
        self.u = np.empty(0, dtype=np.float64)
        self.phi = np.empty((0, 14), dtype=np.float64)
        self.fixed_u = np.empty(0, dtype=bool)
        self.fixed_phi = np.empty((0, 14), dtype=bool)
        self._nb_idx: np.ndarray = np.empty((0, 14), dtype=np.int32)
        self._dirty = True

        if radius_cells is not None:
            self._build_sphere(radius_cells, spacing)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_sphere(self, radius_cells: float, spacing: float):
        from equilibrium.math.geo.kelvin import KelvinLattice

        geo = KelvinLattice(lattice_spacing=spacing)
        pts = geo.generate_sphere(radius_cells)
        self._build_from_points(geo, pts)

    def _build_cylinder(
        self,
        radius_cells: float,
        half_length_cells: float,
        spacing: float,
        axis: int = 2,
    ):
        """Cylindrical domain -- O(radius^2 * half_length) cells instead of
        O(radius_sphere^3); use for long, thin domains (e.g. two points separated
        along one axis by a large distance, with only modest transverse extent
        needed to resolve the propagating structure)."""
        from equilibrium.math.geo.kelvin import KelvinLattice

        geo = KelvinLattice(lattice_spacing=spacing)
        pts = geo.generate_cylinder(radius_cells, half_length_cells, axis=axis)
        self._build_from_points(geo, pts)

    @classmethod
    def cylinder(
        cls,
        radius_cells: float,
        half_length_cells: float,
        spacing: float = 1.0,
        axis: int = 2,
    ):
        """Public constructor for a cylindrical domain (see _build_cylinder)."""
        lat = cls(spacing=spacing)
        lat._build_cylinder(radius_cells, half_length_cells, spacing, axis=axis)
        return lat

    def _build_from_points(self, geo, pts: np.ndarray):
        idx_map, key_fn = geo.point_index_map(pts)
        N = len(pts)

        self.pos = pts
        self.ell = np.ones(N, dtype=np.float64)
        self.lam = np.zeros(N, dtype=np.float64)
        self.u = np.zeros(N, dtype=np.float64)
        self.phi = np.zeros((N, 14), dtype=np.float64)
        self.fixed_u = np.zeros(N, dtype=bool)
        self.fixed_phi = np.zeros((N, 14), dtype=bool)

        self.tiles = [SubstrateTile(i, self) for i in range(N)]
        for i, tile in enumerate(self.tiles):
            for k, off in enumerate(geo.all_offsets):
                key = key_fn(pts[i] + off)
                if key in idx_map:
                    tile.neighbours[k] = self.tiles[idx_map[key]]

        self._dirty = True

    # ------------------------------------------------------------------
    # Topology cache — rebuilt whenever neighbour pointers change
    # ------------------------------------------------------------------

    def _rebuild_cache(self):
        N = len(self.tiles)
        nb_idx = np.full((N, 14), -1, dtype=np.int32)
        for i, tile in enumerate(self.tiles):
            for k, nb in enumerate(tile.neighbours):
                if nb is not None:
                    nb_idx[i, k] = nb.idx
        self._nb_idx = nb_idx
        self._dirty = False

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def N(self) -> int:
        return len(self.tiles)

    @property
    def chi(self) -> np.ndarray:
        lam = self.lam.astype(np.float64)
        return lam / (1.0 + lam)

    @property
    def sigma(self) -> np.ndarray:
        return 1.0 / (1.0 + self.lam.astype(np.float64))

    @property
    def nb(self) -> np.ndarray:
        if self._dirty:
            self._rebuild_cache()
        return self._nb_idx

    @property
    def _valid(self) -> np.ndarray:
        return self.nb >= 0

    # ------------------------------------------------------------------
    # Knot initialisation
    # ------------------------------------------------------------------

    def set_winding(
        self,
        centre: SubstrateTile,
        chi_0: float,
        profile: str = "1/r",
    ):
        """
        Initialise winding occupation (lam) around a knot centre.

        lam is set so that sigma(r) = 1 - chi_0/r, matching the DST 1/r
        gravitational well.  From sigma = 1/(1+lam), this gives:
            lam(r) = chi_0 / (r - chi_0)

        profile='1/r' (default): chi_0 / r well.
        profile='1/r2': chi_0 / r² (steeper, non-Newtonian — for comparison).

        Centre tile (r=0): lam set to a large sentinel (near-BH).
        """
        self.lam[centre.idx] = LAM_MAX
        visited = {centre.idx: 0}
        queue = deque([(centre, 0)])
        while queue:
            tile, r = queue.popleft()
            for nb in tile.neighbours:
                if nb is None or nb.idx in visited:
                    continue
                r_nb = r + 1
                visited[nb.idx] = r_nb
                denom = r_nb if profile == "1/r" else r_nb**2
                chi_r = float(np.clip(chi_0 / denom, 0.0, 1.0 - L_MIN))
                lam_r = chi_r / (1.0 - chi_r)
                self.lam[nb.idx] = float(np.clip(lam_r, 0.0, LAM_MAX))
                queue.append((nb, r_nb))

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def stable_dt(self) -> float:
        """CFL bound for σ=1/(1+λ). Max σ is 1.0 (void cells), so dt is fixed."""
        return 0.4 / float((_W**2).sum())

    def step(self, dt: float):
        """Symplectic Euler step."""
        if self._dirty:
            self._rebuild_cache()

        sigma = 1.0 / (1.0 + self.lam)
        nb_idx = self._nb_idx
        valid = nb_idx >= 0
        safe_nb = np.where(valid, nb_idx, 0)
        face_w = np.where(valid, _W[None, :], 0.0)

        # sigma at the face (mean of the two cells) on both legs: sum u and H conserved for any lam
        sigma_nb = np.where(valid, sigma[safe_nb], sigma[:, None])
        sigma_face = (sigma[:, None] + sigma_nb) / 2.0

        # Step 1 — update u from current phi
        phi_eff = np.where(valid, self.phi, 0.0)
        du = -dt * (sigma_face * face_w * phi_eff).sum(axis=1)
        u_new = np.where(self.fixed_u, self.u, self.u + du)

        # Step 2 — update phi from new u
        u_nb = np.where(valid, u_new[safe_nb], u_new[:, None])
        drive = u_nb - u_new[:, None]
        dphi = np.where(
            valid & ~self.fixed_phi,
            -dt * sigma_face * face_w * drive,
            0.0,
        )

        self.u = u_new
        self.phi = np.where(self.fixed_phi, self.phi, self.phi + dphi)

    def step_cpp(self, dt: float, n_threads: int = 1):
        """
        C++ symplectic Euler step (requires equilibrium.cpp to be built).

        Identical semantics to step() but runs the inner loop in C++ with
        optional std::thread parallelism.  n_threads=1 uses the C++ single-
        threaded path (still ~10-50× faster than NumPy for large N).

        Falls back to the Python step() if the extension is not available.
        """
        try:
            from equilibrium import cpp as _cpp
        except ImportError:
            SubstrateLattice.step(self, dt)
            return

        if not hasattr(_cpp, "substrate_step"):
            SubstrateLattice.step(self, dt)
            return

        if self._dirty:
            self._rebuild_cache()

        _cpp.substrate_step(
            self.u,
            self.phi,
            self.ell,
            self.lam,
            self.fixed_u.view(np.uint8),
            self.fixed_phi.view(np.uint8),
            self._nb_idx,
            dt,
            n_threads,
        )

    def step_lam(self, dt: float, scale: float = 1.0, n_threads: int = 1):
        """
        Lam transport step: Δlam_i/Δt = -σ_i · lam_i · Σ_k W_k · φ_{ik}

        Winding is carried by the same phi field that drives u — the net phi
        flux at a cell reflects whether the substrate is being compressed or
        expanded there.  A stationary omnibalanced winding has Σ W_k·φ_{ik}=0
        and therefore Δlam=0: no translation without imbalance.  When phi is
        asymmetric (a proton being driven by a u gradient), lam redistributes
        in the direction of compression, and continues doing so as long as the
        imbalance persists.  Once the excess is offloaded into the substrate,
        the winding settles at a new balanced state — Newton's first law.

        σ_i = 1/(1+lam_i) self-buffers high-lam cores: the proton centre is
        nearly frozen in lam, while the vacuum/orbit region updates freely.

        scale: coupling strength (start with 1e-3; increase if lam moves too
               slowly).  fixed_u cells are also frozen in lam.
        Lam is clamped to [0, LAM_MAX] each step.
        """
        if self._dirty:
            self._rebuild_cache()

        try:
            from equilibrium import cpp as _cpp

            if hasattr(_cpp, "substrate_step_lam"):
                _cpp.substrate_step_lam(
                    self.lam,
                    self.phi,
                    self.fixed_u.view(np.uint8),
                    self._nb_idx,
                    dt,
                    scale,
                    n_threads,
                )
                return
        except ImportError:
            pass

        nb_idx = self._nb_idx
        valid = nb_idx >= 0
        face_w = np.where(valid, _W[None, :], 0.0)

        sigma = 1.0 / (1.0 + self.lam)
        phi_eff = np.where(valid, self.phi, 0.0)
        phi_sum = (face_w * phi_eff).sum(axis=1)

        dlam = -dt * scale * sigma * self.lam * phi_sum
        self.lam = np.clip(
            np.where(self.fixed_u, self.lam, self.lam + dlam),
            0.0,
            LAM_MAX,
        )

    def step_lam_force(self, dt: float, scale: float = 1.0):
        """
        Force-driven lam advection along ∇u (toward higher-u neighbors).

        For each face k, lam at cell i flows toward neighbor k wherever
        u_k > u_i, weighted by sigma_i and face_w.  The flux is conservative:
        lam leaving i is deposited at the receiving cell.

        This gives the 1/r² gravitational attraction:
          - B's 1/r u-field produces a gradient ∝ 1/r² at A, pulling A's lam
            toward B.
          - A's own symmetric u-field produces zero net COM drift (all
            directions cancel).

        sigma_i = 1/(1+lam_i) buffers high-lam cores so the knot interior is
        nearly frozen while the halo and vacuum regions respond freely.
        fixed_u cells are also frozen in lam.
        """
        if self._dirty:
            self._rebuild_cache()

        nb_idx = self._nb_idx
        valid = nb_idx >= 0
        safe_nb = np.where(valid, nb_idx, 0)
        sigma = 1.0 / (1.0 + self.lam)
        face_w = np.where(valid, W_STAR[None, :], 0.0)  # scalar -> scalar operator: the star
        u_nb = np.where(valid, self.u[safe_nb], self.u[:, None])

        # Pull toward higher-u neighbors (gravity toward mass)
        pull = np.maximum(u_nb - self.u[:, None], 0.0)

        # Lam flux from i toward k
        flux = dt * scale * sigma[:, None] * face_w * pull * self.lam[:, None]

        dlam = np.zeros_like(self.lam)
        dlam -= flux.sum(axis=1)

        for k in range(14):
            v = valid[:, k]
            np.add.at(dlam, nb_idx[v, k], flux[v, k])

        self.lam = np.clip(
            np.where(self.fixed_u, self.lam, self.lam + dlam),
            0.0,
            LAM_MAX,
        )

    def stable_lam_dt(self) -> float:
        """
        Stability bound for phi-driven lam transport.

        The RHS magnitude is bounded by max_lam * max(|Σ W_k phi_ik|).
        Returns inf if phi is zero everywhere.
        """
        if self._dirty:
            self._rebuild_cache()
        nb_idx = self._nb_idx
        valid = nb_idx >= 0
        face_w = np.where(valid, _W[None, :], 0.0)
        phi_eff = np.where(valid, self.phi, 0.0)
        max_phi_sum = float(np.max(np.abs((face_w * phi_eff).sum(axis=1))))
        max_lam = float(np.max(self.lam)) if np.any(self.lam > 0) else 1.0
        if max_phi_sum * max_lam == 0:
            return float("inf")
        return 0.4 / (max_lam * max_phi_sum)

    def step_cross(
        self,
        dt: float,
        scale: float = 1.0,
        n_threads: int = 1,
        gate: str = "v1",
        m0: float = None,
    ):
        """
        Face-adjacent short-circuit crossing step.

        Flux through face k (in either direction) drives flux through the faces
        edge-adjacent to k on the Kelvin cell — the short-circuit path through
        the cell interior.  Symmetric in sign: inward phi redistributes outward
        to neighbours; outward phi redistributes inward to neighbours.

        This couples sq-face z-forcing (face 12/13) to the hex-face circulation
        of a ring winding (faces 0-7), enabling the collective gate condition
        for winding translation.  No stored field — instantaneous coupling from
        the Kelvin cell face-adjacency graph.

        The coupling weight is uniform over adjacent faces.  scale sets the
        overall crossing resistance: smaller scale = higher resistance = larger
        inertial mass.

        gate="v1" (default): continuous, always-fires crossing — the original
            behaviour, unchanged. C++-accelerated when the extension is built.
        gate="barrier": stateless, instantaneous, all-or-nothing per-cell gate
            (doc/dst.md: "the crossing either happens or it doesn't: a gate
            condition on the instantaneous phi state"). Pooled per-cell drive
            Σ_k W_k·max(-φ_{ik},0) is compared against `m0` (required — no
            default; see doc/dst_next.md §8 for how to calibrate it per knot).
            C++-accelerated when the extension is built
            (`_cpp.substrate_step_cross_barrier`, inc/eq/substrate.hpp),
            falls back to the Python implementation below otherwise. Ported
            from `run/substrate/cross_v2_uniform.py`'s `CrossV2(gate="barrier")`,
            where it was verified: sub-threshold drive never crosses over any
            number of ticks; over-threshold drive fires and matches this same
            kernel bit-for-bit on firing cells. The C++ kernel is verified
            bit-for-bit against this Python path.
        """
        if gate not in ("v1", "barrier"):
            raise ValueError(f"unknown gate {gate!r}")
        if gate == "barrier" and m0 is None:
            raise ValueError(
                "gate='barrier' requires m0 (no default -- see doc/dst_next.md sec 8 "
                "for per-knot calibration)"
            )

        if self._dirty:
            self._rebuild_cache()

        if gate == "v1":
            try:
                from equilibrium import cpp as _cpp

                if hasattr(_cpp, "substrate_step_cross"):
                    _cpp.substrate_step_cross(
                        self.phi,
                        self.fixed_phi.view(np.uint8),
                        self._nb_idx,
                        self.lam,
                        dt,
                        scale,
                        n_threads,
                    )
                    return
            except (ImportError, TypeError):
                pass

            nb_idx = self._nb_idx  # (N, 14), int32, -1 = boundary
            dphi = np.zeros_like(self.phi)

            for k in range(14):
                delta = (
                    -dt * scale * self.phi[:, k]
                )  # sign tracks phi: inward→+, outward→-
                adj = [j for j in _FACE_ADJ[k] if j >= 0]
                n_adj = len(adj)
                mk = int(_K_REV[k])

                valid_k = nb_idx[:, k] >= 0
                nb_k = np.where(valid_k, nb_idx[:, k], 0)

                # Drain source face k
                dphi[:, k] += n_adj * delta
                # Mirror of drain at neighbour nb_k
                np.add.at(dphi, (nb_k[valid_k], mk), -(n_adj * delta[valid_k]))

                for j in adj:
                    mj = int(_K_REV[j])
                    valid_j = nb_idx[:, j] >= 0
                    nb_j = np.where(valid_j, nb_idx[:, j], 0)
                    # Redistribute to adjacent face j
                    dphi[:, j] += delta
                    # Mirror at neighbour nb_j
                    np.add.at(dphi, (nb_j[valid_j], mj), -delta[valid_j])

            self.phi = np.where(self.fixed_phi, self.phi, self.phi + dphi)
            return

        # gate == "barrier"
        try:
            from equilibrium import cpp as _cpp

            if hasattr(_cpp, "substrate_step_cross_barrier"):
                _cpp.substrate_step_cross_barrier(
                    self.phi,
                    self.fixed_phi.view(np.uint8),
                    self._nb_idx,
                    self.lam,
                    dt,
                    scale,
                    m0,
                    n_threads,
                )
                return
        except (ImportError, TypeError):
            pass

        nb_idx = self._nb_idx
        valid = nb_idx >= 0
        safe_nb = np.where(valid, nb_idx, 0)
        phiv = np.where(valid, self.phi, 0.0)
        sig2 = (1.0 / (1.0 + self.lam)) ** 2

        drive = (_W[None, :] * np.maximum(-phiv, 0.0)).sum(
            axis=1
        )  # (N,) pooled per cell
        fire = (drive > m0).astype(np.float64)[:, None]  # (N, 1), broadcasts over faces

        inward = np.maximum(-phiv, 0.0)
        delta = (dt * scale) * sig2[:, None] * fire * inward  # (N, 14), >= 0

        L = np.zeros_like(self.phi)
        for k in range(14):
            d = delta[:, k]
            adj = [j for j in _FACE_ADJ[k] if j >= 0]
            n_adj = len(adj)
            L[:, k] += n_adj * d
            for j in adj:
                L[:, j] += d

        mirror = np.where(valid, L[safe_nb, _K_REV[None, :]], 0.0)
        new_phi = self.phi + L - mirror
        self.phi = np.where(self.fixed_phi, self.phi, new_phi)

    def step_lam_sink(self, dt: float, scale: float = 1.0, amp: float = 1.0):
        """
        φ → λ (sink) → φ → u  absorption and cross re-emission.

        Inward phi on each face is absorbed into lam growth and re-emitted
        through cross-adjacent faces weighted by lam_i.

          Δλ_i    = -scale · σ_i · λ_i · Σ_k min(φ_{ik}, 0) / _LAM_SINK_COST
          Δφ_{adj} = -scale · λ_i · min(φ_{ik}, 0)

        Gradient pull (chi-based convergent phi drive) was removed from step().

        Apparent non-conservation is not patched.
        """
        if self._dirty:
            self._rebuild_cache()

        sigma = 1.0 / (1.0 + self.lam)

        dphi = np.zeros_like(self.phi)
        dlam = np.zeros_like(self.lam)

        # Conservative absorption + cross re-emission.
        #
        # For each face k carrying inward phi:
        #   (1) Absorbed phi is removed from face k (phi source closed).
        #   (2) Lam grows from the absorbed phi / _LAM_SINK_COST.
        #   (3) Lam releases u locally at the same rate (lam → u, closes cycle).
        #   (4) Remaining absorbed phi is re-emitted outward on cross-adjacent faces.
        #
        # Net: u → phi → lam → u (stable cycle); phi is routed not created.
        du = np.zeros_like(self.u)
        for k in range(14):
            inward_k = np.minimum(self.phi[:, k], 0.0)  # ≤ 0
            absorbed = dt * scale * sigma * self.lam * (-inward_k)  # ≥ 0

            # (1) remove absorbed phi from face k
            dphi[:, k] += absorbed

            # (2) lam grows
            dlam += absorbed / _LAM_SINK_COST

            # (3) lam releases u (closes the loop)
            du += absorbed / _LAM_SINK_COST

            # (4) re-emit absorbed phi across cross-adjacent faces.
            # amp=1.0: chi amplification (×1 vacuum, ×2 knot core — feedback loop)
            # amp=0.0: flat re-emission, no feedback
            adj = [j for j in _FACE_ADJ[k] if j >= 0]
            n_adj = len(adj)
            if n_adj:
                if amp:
                    chi = self.lam / (1.0 + self.lam)
                    factor = 1.0 + amp * chi
                else:
                    factor = 1.0
                for j in adj:
                    dphi[:, j] += absorbed * factor / n_adj

        self.lam = np.clip(
            np.where(self.fixed_u, self.lam, self.lam + dlam),
            0.0,
            LAM_MAX,
        )
        self.phi = np.where(self.fixed_phi, self.phi, self.phi + dphi)
        self.u = np.where(self.fixed_u, self.u, self.u + du)

    def step_lam_relax(self, dt: float, rate: float, alpha: float = 1.0):
        """
        Local lam relaxation toward alpha * |phi|.

        lam_target_i = clip(alpha * |phi_i|, 0, LAM_MAX)
        dlam_i/dt    = rate * (lam_target_i - lam_i)

        No transport between cells — pure local coupling to field amplitude.
        Where phi is strong, lam grows; where phi collapses, lam drains.
        """
        phi_mag = np.sqrt((self.phi**2).sum(axis=1))
        lam_target = np.clip(alpha * phi_mag, 0.0, LAM_MAX)
        new_lam = np.clip(self.lam + dt * rate * (lam_target - self.lam), 0.0, LAM_MAX)
        self.lam = np.where(self.fixed_u, self.lam, new_lam)

    def run(self, n_steps: int, dt: float = None):
        if dt is None:
            dt = self.stable_dt()
        for _ in range(n_steps):
            self.step(dt)

    # ------------------------------------------------------------------
    # Physical radius — cell-size-weighted BFS distance
    # ------------------------------------------------------------------

    def physical_radius(self, centre: "SubstrateTile") -> np.ndarray:
        """BFS hop distance from centre, weighted by _D_FACE per face type.

        ell=1 always, so each hop contributes d_face[k] directly.
        Returns an array of length N (NaN for unreachable tiles).
        """
        r_phys = np.full(len(self.tiles), np.nan)
        r_phys[centre.idx] = 0.0
        queue = deque([(centre, 0.0)])
        visited = {centre.idx}
        while queue:
            tile, r = queue.popleft()
            for k, nb in enumerate(tile.neighbours):
                if nb is None or nb.idx in visited:
                    continue
                visited.add(nb.idx)
                r_new = r + _D_FACE[k]
                r_phys[nb.idx] = r_new
                queue.append((nb, r_new))
        return r_phys

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str, step_dt: float = None):
        """Dump full lattice state to a compressed .npz file.

        step_dt: if provided, the timestep is saved and restored by load().
        """
        if self._dirty:
            self._rebuild_cache()
        meta = np.array([step_dt if step_dt is not None else np.nan])
        np.savez_compressed(
            path,
            pos=self.pos,
            u=self.u,
            phi=self.phi,
            ell=self.ell,
            lam=self.lam,
            fixed_u=self.fixed_u.view(np.uint8),
            fixed_phi=self.fixed_phi.view(np.uint8),
            nb_idx=self._nb_idx,
            meta=meta,
        )

    @classmethod
    def load(cls, path: str) -> "SubstrateLattice":
        """Restore a lattice previously saved with save().

        Tile neighbour pointers are wired from the saved nb_idx, so no
        KelvinLattice geometry rebuild is needed — load time is O(N).
        """
        data = np.load(path)
        lat = cls()
        lat.pos = data["pos"]
        lat.u = data["u"]
        lat.phi = data["phi"]
        lat.ell = data["ell"]
        lat.lam = data["lam"].astype(np.float64)
        lat.fixed_u = data["fixed_u"].view(bool).copy()
        lat.fixed_phi = data["fixed_phi"].view(bool).copy()
        lat._nb_idx = data["nb_idx"]
        # restore saved step_dt if present
        if "meta" in data:
            saved_dt = float(data["meta"][0])
            lat.saved_step_dt = None if np.isnan(saved_dt) else saved_dt
        else:
            lat.saved_step_dt = None
        lat.tiles = [SubstrateTile(i, lat) for i in range(len(lat.pos))]
        nb = lat._nb_idx
        for i, tile in enumerate(lat.tiles):
            for k in range(14):
                j = int(nb[i, k])
                if j >= 0:
                    tile.neighbours[k] = lat.tiles[j]
        lat._dirty = False
        return lat


# ---------------------------------------------------------------------------
# Absorbing boundary conditions
# ---------------------------------------------------------------------------


class DampedLattice(SubstrateLattice):
    """
    SubstrateLattice with a volumetric sponge (absorbing layer) at r > r_inner.

    Rate g(r) = gamma_max * t^power, t = 0 at the inner edge -> 1 at the boundary. After each
    step, u relaxes toward u_bg by exp(-g dt) per cell and each face (phi, toward phi_bg) by
    exp(-g_face dt), g_face = mean of the two cells' rates, applied to both stored copies so
    phi_ki = -phi_ik survives (exp/2026/09/29/sponge_assess). dt-consistent.

    Strength: gamma_max, or damp_max = the legacy per-step fraction, converted at stable_dt()
    (same damping per step at the stable dt as before). Default: gamma_max = 10 c / width,
    the reflection minimum of exp/2026/09/29/bc_design.

    A sponge that damps phi cannot hold a static potential: a steady u gradient needs phi to
    keep growing (exp/2026/09/29/bc_design). It is a wave absorber.
    """

    def __init__(
        self,
        radius_cells: float,
        spacing: float = 1.0,
        sponge_fraction: float = 0.2,
        damp_max: float | None = None,
        u_bg: float = 0.0,
        phi_bg: np.ndarray = None,
        gamma_max: float | None = None,
        power: float = 2.0,
    ):
        super().__init__(radius_cells=radius_cells, spacing=spacing)
        r = np.linalg.norm(self.pos, axis=1)
        self._init_damping(r, self._sponge_depth(r, sponge_fraction), damp_max, gamma_max, power, u_bg, phi_bg)

    @staticmethod
    def _sponge_depth(x: np.ndarray, sponge_fraction: float) -> np.ndarray:
        """Normalised sponge depth: 0 inside the inner edge, ->1 at the boundary."""
        x_max = float(x.max())
        x_inner = x_max * (1.0 - sponge_fraction)
        return np.where(x > x_inner, (x - x_inner) / (x_max - x_inner), 0.0)

    def _init_damping(self, x, t, damp_max, gamma_max, power, u_bg, phi_bg):
        if damp_max is not None and gamma_max is not None:
            raise ValueError("DampedLattice: give at most one of damp_max, gamma_max")
        if gamma_max is None:
            if damp_max is not None:
                gamma_max = -np.log1p(-damp_max) / self.stable_dt()
            else:
                i, k = np.nonzero(self.nb[:, :8] >= 0)
                d1 = float(np.linalg.norm(self.pos[self.nb[i[0], k[0]]] - self.pos[i[0]]))
                width = float(x[t > 0].max() - x[t > 0].min()) if np.any(t > 0) else d1
                gamma_max = 10.0 * wave_speed(d1) / max(width, d1)
        self._gamma = (gamma_max * t**power).astype(np.float64)
        self._u_bg = u_bg
        self.phi_bg = np.zeros_like(self.phi) if phi_bg is None else np.array(phi_bg, dtype=np.float64)
        self._last_dt = self.stable_dt()

    @property
    def _gamma(self):
        return self._gamma_cell

    @_gamma.setter
    def _gamma(self, g):
        self._gamma_cell = np.asarray(g, dtype=np.float64)
        valid = self.nb >= 0
        gn = np.where(valid, self._gamma_cell[np.where(valid, self.nb, 0)], self._gamma_cell[:, None])
        self._gamma_face = 0.5 * (self._gamma_cell[:, None] + gn)

    @property
    def _sponge(self):
        """Per-step damped fraction per cell at stable_dt() (the legacy damp profile)."""
        return -np.expm1(-self._gamma * self.stable_dt())

    @_sponge.setter
    def _sponge(self, frac):
        self._gamma = -np.log1p(-np.asarray(frac, dtype=np.float64)) / self.stable_dt()

    @classmethod
    def cylinder(
        cls,
        radius_cells: float,
        half_length_cells: float,
        spacing: float = 1.0,
        axis: int = 2,
        sponge_fraction: float = 0.2,
        damp_max: float | None = None,
        u_bg: float = 0.0,
        phi_bg: np.ndarray = None,
        gamma_max: float | None = None,
        power: float = 2.0,
    ):
        """Cylindrical domain with a sponge on the lateral wall AND both end
        caps: sponge depth is the deeper of the radial (perpendicular-to-axis)
        and axial penetrations, so outgoing waves are absorbed whichever
        boundary they reach.  Depths are normalised per-extent, so the sponge
        layer is proportionally thin along the (long) axis too."""
        lat = cls.__new__(cls)
        SubstrateLattice.__init__(lat, spacing=spacing)
        lat._build_cylinder(radius_cells, half_length_cells, spacing, axis=axis)
        r_perp = np.linalg.norm(np.delete(lat.pos, axis, axis=1), axis=1)
        z = np.abs(lat.pos[:, axis])
        t_r = cls._sponge_depth(r_perp, sponge_fraction)
        t_z = cls._sponge_depth(z, sponge_fraction)
        x = np.where(t_r >= t_z, r_perp, z)
        lat._init_damping(x, np.maximum(t_r, t_z), damp_max, gamma_max, power, u_bg, phi_bg)
        return lat

    def step(self, dt: float, n_threads: int = 1):
        self.step_cpp(dt, n_threads)
        self.apply_sponge(dt)

    def apply_sponge(self, dt: float | None = None):
        """
        Relax u toward u_bg and phi toward phi_bg over dt (default stable_dt()), independent of
        step_cpp's wave evolution, so callers that evolve by other means can call it alone.
        With u_bg = 0 / phi_bg = 0 (the defaults) this is plain absorption; phi_bg is the
        far-field background used by mhd.step_hall (see capture_ground_state there).
        """
        dt = self.stable_dt() if dt is None else dt
        self._last_dt = dt
        fu = np.exp(-self._gamma * dt)
        ff = np.exp(-self._gamma_face * dt)
        u_new = self._u_bg + fu * (self.u - self._u_bg)
        self.u = np.where(self.fixed_u, self.u, u_new)
        phi_new = self.phi_bg + ff * (self.phi - self.phi_bg)
        self.phi = np.where(self.fixed_phi, self.phi, phi_new)

    def bc_flux(self) -> float:
        """u the sponge would inject over the last dt from the current state (positive =
        injection into the domain)."""
        frac = -np.expm1(-self._gamma * self._last_dt)
        return float(np.sum(np.where(self.fixed_u, 0.0, frac * (self._u_bg - self.u))))

    @classmethod
    def load_snapshot(
        cls,
        path: str,
        sponge_fraction: float = 0.15,
        damp_max: float | None = 0.003,
        u_bg: float = 0.0,
        gamma_max: float | None = None,
        power: float = 2.0,
    ) -> "DampedLattice":
        """Load a SubstrateLattice snapshot and attach a sponge layer."""
        plain = SubstrateLattice.load(path)
        lat = cls.__new__(cls)
        lat.__dict__.update(plain.__dict__)
        phi_bg = getattr(lat, "phi_bg", None)
        r = np.linalg.norm(lat.pos, axis=1)
        if gamma_max is not None:
            damp_max = None
        lat._init_damping(r, cls._sponge_depth(r, sponge_fraction), damp_max, gamma_max, power, u_bg, phi_bg)
        return lat


class RadiationBCLattice(SubstrateLattice):
    """
    SubstrateLattice with a Sommerfeld-style radiation boundary condition.

    After each step the outermost shell of tiles (r > r_inner) is overwritten
    with u[i] = A / r[i] + u_bg, where A and u_bg are inferred from the
    interior.  This preserves the DC Coulomb 1/r tail instead of damping it.

    A sponge layer absorbs AC waves as in DampedLattice.  Call freeze_bc()
    after warmup to lock in the fitted A and u_bg; before that the BC is
    inactive and the lattice behaves like DampedLattice.

    Parameters
    ----------
    bc_fraction : fraction of lattice radius used as the BC shell
    fit_lo/hi   : band [fit_lo, fit_hi] × r_max used to fit A each step
    A_min       : minimum A below which BC is not applied
    """

    def __init__(
        self,
        radius_cells: float,
        spacing: float = 1.0,
        bc_fraction: float = 0.15,
        damp_max: float = 0.003,
        fit_lo: float = 0.25,
        fit_hi: float = 0.60,
        A_min: float = 1e-6,
    ):
        super().__init__(radius_cells=radius_cells, spacing=spacing)
        r_all = np.linalg.norm(self.pos, axis=1)
        r_max = float(r_all.max())
        r_inner = r_max * (1.0 - bc_fraction)
        beyond = r_all > r_inner
        t = np.where(beyond, (r_all - r_inner) / (r_max - r_inner), 0.0)
        self._sponge = (damp_max * t**2).astype(np.float64)
        self._bc_mask = beyond.copy()
        self._fit_mask = (r_all > r_max * fit_lo) & (r_all < r_max * fit_hi)
        self._r = r_all
        self._A_min = A_min
        self._A: float = 0.0
        self._u_bg: float = 0.0
        self._frozen: bool = False
        self.phi_bg = np.zeros_like(self.phi)

    def refresh_masks(self):
        """Exclude fixed_u tiles from the fit band and BC shell. Call after pinning sources."""
        self._bc_mask &= ~self.fixed_u
        self._fit_mask &= ~self.fixed_u

    def freeze_bc(self):
        """Fit A and u_bg from the current interior field and lock the BC.

        Call after warmup (quasi-steady state).  Subsequent step() calls apply
        u[bc] = A/r + u_bg with no further feedback from the interior to A.
        """
        r_fit = self._r[self._fit_mask]
        ur_fit = self.u[self._fit_mask] * r_fit
        coeffs = np.polyfit(r_fit, ur_fit, 1)  # [u_bg, A]
        self._u_bg = float(coeffs[0])
        self._A = float(coeffs[1])
        self._frozen = True

    @classmethod
    def load_snapshot(
        cls,
        path: str,
        bc_fraction: float = 0.15,
        fit_lo: float = 0.25,
        fit_hi: float = 0.60,
        A_min: float = 1e-6,
    ) -> "RadiationBCLattice":
        """Load a saved lattice snapshot and attach radiation BC.

        Re-fits A and u_bg from the loaded u field so the Coulomb tail is
        immediately active.  fixed_u cells are excluded from the fit band
        and BC shell.
        """
        plain = SubstrateLattice.load(path)
        lat = cls.__new__(cls)
        lat.__dict__.update(plain.__dict__)
        r_all = np.linalg.norm(lat.pos, axis=1)
        r_max = float(r_all.max())
        r_inner = r_max * (1.0 - bc_fraction)
        beyond = r_all > r_inner
        t = np.where(beyond, (r_all - r_inner) / (r_max - r_inner), 0.0)
        lat._sponge = (0.003 * t**2).astype(np.float64)
        lat._bc_mask = beyond & ~lat.fixed_u
        lat._fit_mask = (
            (r_all > r_max * fit_lo) & (r_all < r_max * fit_hi) & ~lat.fixed_u
        )
        lat._r = r_all
        lat._A_min = A_min
        lat._A = 0.0
        lat._u_bg = 0.0
        lat._frozen = False
        lat.phi_bg = np.zeros_like(lat.phi)
        lat.freeze_bc()
        return lat

    def step(self, dt: float, n_threads: int = 1):
        self.step_cpp(dt, n_threads)
        # phi relaxes toward phi_bg (far-field background, see DampedLattice
        # and equilibrium/substrate/mhd.py's capture_ground_state/step_hall)
        # instead of decaying to zero; phi_bg=0 by default reproduces the
        # original absorbing behaviour.
        u_damp = np.where(self.fixed_u, 1.0, 1.0 - self._sponge)
        self.u *= u_damp
        phi_new = self.phi - self._sponge[:, None] * (self.phi - self.phi_bg)
        self.phi = np.where(self.fixed_phi, self.phi, phi_new)
        if self._frozen and self._A > self._A_min:
            self.u[self._bc_mask] = self._A / self._r[self._bc_mask] + self._u_bg
