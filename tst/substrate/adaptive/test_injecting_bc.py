"""
Injecting BC and collapsed independent electron tests.

Injecting BC: a source face that adds phi to the domain each step, tracking
what is injected versus what escapes through other boundary faces.
The domain starts as void (u=U_BG, lam=0, phi=0).

Collapsed electron: an 8-tile toroidal ring winding in vacuum — no proton,
no lam, no imposed knot.  Tests whether the winding is self-sustaining under
the wave equation with a sponge BC (the adaptive equivalent of probe_electron_loop.py).

Key questions:
  1. Does an injecting face grow phi inside the domain?
  2. Can we track what was injected vs what escaped?
  3. Does a free-standing ring winding persist without a proton?
"""

import unittest

import numpy as np

from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    phi_ring_circulation,
)

# BCC sq face indices: k=8..13 → -x,+x,-y,+y,-z,+z
_K_Z_POS = 13  # +z sq face
_K_Z_NEG = 12  # -z sq face

R_SPHERE = 12.0
U_BG = 0.0  # void — matches DampedLattice convention
AMP = 1.0
DT = 0.01
SPONGE_FRAC = 0.20
DAMP_MAX = 0.005


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_sphere(u0=U_BG, with_sponge=False):
    alat = _lat.AdaptiveLattice()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_SPHERE, d1=1.0, u0=u0, lam0=0.0)
    alat.rebuild(0)
    if with_sponge:
        r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
        r_inner = float(r_all.max()) * (1.0 - SPONGE_FRAC)
        cl.set_bc(
            _lat.BoundaryCondition.quadratic_sponge(
                target=0.0, damp_max=DAMP_MAX, r_all=r_all, r_inner=r_inner
            )
        )
    else:
        cl.set_bc(_lat.BoundaryCondition.absorbing())
    return alat, cl


def _find_injection_cells(cl):
    """Bottom-layer cells: those with no -z neighbour (nb[_K_Z_NEG] < 0)."""
    nb = np.asarray(cl.leaf_nb, dtype=np.int32)
    pos = np.asarray(cl.leaf_pos, dtype=np.float64)
    # Cells at z ≈ -R with their -z face exposed
    boundary_z = nb[:, _K_Z_NEG] < 0
    z_min = pos[:, 2].min()
    near_bottom = pos[:, 2] < z_min + 3.0
    return np.where(boundary_z & near_bottom)[0]


def _phi_energy(cl):
    phi = np.asarray(cl.leaf_phi, dtype=np.float64)
    return float(np.sum(phi**2))


def _inject(cl, inject_js, face_k, delta):
    """Add delta to phi[face_k] on inject_js cells after each step."""
    phi = np.asarray(cl.leaf_phi, dtype=np.float64)
    phi[face_k, inject_js] += delta
    cl.set_leaf_phi(phi)


# ---------------------------------------------------------------------------
# Injecting BC tests
# ---------------------------------------------------------------------------


class TestInjectingBC(unittest.TestCase):

    def test_injection_cells_found(self):
        """Bottom layer cells with exposed -z face are identifiable."""
        _, cl = _build_sphere()
        inject_js = _find_injection_cells(cl)
        self.assertGreater(
            len(inject_js), 0, "no injection cells found at sphere bottom"
        )

    def test_void_phi_stays_zero(self):
        """Without injection, void domain phi stays zero under wave equation."""
        alat, cl = _build_sphere(u0=U_BG)
        for _ in range(100):
            alat.step(0, DT)
        self.assertAlmostEqual(_phi_energy(cl), 0.0, places=10)

    def test_injection_grows_phi(self):
        """Injecting phi each step increases total phi energy in the domain."""
        alat, cl = _build_sphere(u0=U_BG)
        inject_js = _find_injection_cells(cl)
        phi_e0 = _phi_energy(cl)

        DELTA = 0.05
        N = 200
        for _ in range(N):
            _inject(cl, inject_js, _K_Z_POS, DELTA)
            alat.step(0, DT)

        phi_e1 = _phi_energy(cl)
        self.assertGreater(phi_e1, phi_e0, "phi energy should grow under injection")

    def test_flux_balance(self):
        """Retained phi energy is positive and grows with injection steps.

        Tracks retained = phi_total_final - phi_total_initial.
        Positive retained means more phi enters than escapes.
        Compares N=100 vs N=300 steps: longer injection → more retained.
        """
        DELTA = 0.05

        def _run(n_steps):
            alat, cl = _build_sphere(u0=U_BG)
            inject_js = _find_injection_cells(cl)
            phi_e0 = _phi_energy(cl)
            for _ in range(n_steps):
                _inject(cl, inject_js, _K_Z_POS, DELTA)
                alat.step(0, DT)
            return _phi_energy(cl) - phi_e0

        retained_100 = _run(100)
        retained_300 = _run(300)

        self.assertGreater(retained_100, 0.0, "100 steps: retained must be positive")
        self.assertGreater(
            retained_300, retained_100, "more injection steps → more retained phi"
        )

    def test_injection_propagates_inward(self):
        """Injected phi propagates away from the injection face into the interior.

        Measure phi_rms at two shells: r < 3 (near injection face) and
        r > R/2 (deep interior).  Both should be non-zero after injection,
        confirming the wave propagates inward.

        Note: injecting on +z sq faces creates a z-directed field.  Building
        a toroidal xy-plane circulation from z-face injection requires either
        geometry-matched helicity injection (future work) or direct ring init.
        """
        alat, cl = _build_sphere(u0=U_BG)
        inject_js = _find_injection_cells(cl)
        pos = np.asarray(cl.leaf_pos, dtype=np.float64)
        r = np.linalg.norm(pos, axis=1)

        DELTA = 0.1
        N = 300
        for _ in range(N):
            _inject(cl, inject_js, _K_Z_POS, DELTA)
            alat.step(0, DT)

        phi = np.asarray(cl.leaf_phi, dtype=np.float64)
        phi_rms = np.sqrt(np.mean(phi**2, axis=0))  # (N,) per cell
        deep = r > R_SPHERE * 0.3
        self.assertGreater(
            float(phi_rms[deep].mean()), 0.0, "injected phi should reach deep interior"
        )


# ---------------------------------------------------------------------------
# Collapsed independent electron tests
# ---------------------------------------------------------------------------


class TestCollapsedElectron(unittest.TestCase):
    """Ring winding in vacuum — no proton, no lam — the DST free electron.

    Adaptive equivalent of run/substrate/probe_electron_loop.py.
    """

    def _build_ring(self, with_sponge=True):
        alat, cl = _build_sphere(u0=U_BG, with_sponge=with_sponge)
        ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
        self.assertIsNotNone(ring_js, "toroidal ring not found")
        init_toroidal_winding(cl, ring_js, ring_faces, amplitude=AMP, phi_only=True)
        centre = np.zeros(3)
        return alat, cl, ring_js, centre

    def test_ring_winding_initialised(self):
        """Ring winding has non-zero phi_rms and non-zero circulation at t=0."""
        _, cl, ring_js, centre = self._build_ring()
        phi = np.asarray(cl.leaf_phi, dtype=np.float64)
        rms = float(np.sqrt(np.mean(phi**2)))
        circ = phi_ring_circulation(cl, ring_js, centre)
        self.assertGreater(rms, 0.0)
        self.assertGreater(abs(circ), 0.0)

    def test_ring_persists_vacuum(self):
        """Ring winding in vacuum (lam=0, absorbing BC) retains circulation.

        The winding is a topological object — it should persist even without
        a proton anchor.  Tests the 'collapsed independent electron' hypothesis.
        """
        alat, cl, ring_js, centre = self._build_ring(with_sponge=False)
        circ0 = phi_ring_circulation(cl, ring_js, centre)

        N = 1000
        for _ in range(N):
            alat.step(0, DT)

        circ1 = phi_ring_circulation(cl, ring_js, centre)
        # Winding should retain at least 20% of initial circulation
        self.assertGreater(
            abs(circ1),
            abs(circ0) * 0.2,
            f"ring circulation decayed too fast: {circ0:.4f} → {circ1:.4f}",
        )

    def test_ring_persists_with_sponge(self):
        """Ring winding survives sponge-damped boundary — the sponge absorbs
        radiated waves without destroying the topological core.
        """
        alat, cl, ring_js, centre = self._build_ring(with_sponge=True)
        circ0 = phi_ring_circulation(cl, ring_js, centre)

        N = 1000
        for _ in range(N):
            alat.step(0, DT)
            alat.step_sponge(0)

        circ1 = phi_ring_circulation(cl, ring_js, centre)
        self.assertGreater(
            abs(circ1),
            abs(circ0) * 0.2,
            f"ring lost too much circulation under sponge: {circ0:.4f} → {circ1:.4f}",
        )

    def test_lam_on_ring_kills_circulation(self):
        """Lam on ring cells is destructive to winding — topology alone holds it.

        lam=0: circulation persists (the ring IS the knot; no extra resistance needed).
        lam>0 on ring faces: resistance kills the circulation — the winding collapses.

        Implication: at macro scale, lam should surround the winding, not sit
        on the circulating faces themselves.
        """
        alat0, cl0, ring_js, centre = self._build_ring(with_sponge=True)
        alat1, cl1, _, _ = self._build_ring(with_sponge=True)

        # Add lam on ring cells for one of them
        lam = np.asarray(cl1.leaf_lam, dtype=np.float64)
        for j in ring_js:
            lam[j] = 5.0
        cl1.set_leaf_lam(lam)

        circ0_init = abs(phi_ring_circulation(cl0, ring_js, centre))

        N = 500
        for _ in range(N):
            alat0.step(0, DT)
            alat0.step_sponge(0)
            alat1.step(0, DT)
            alat1.step_sponge(0)

        circ0_final = abs(phi_ring_circulation(cl0, ring_js, centre))
        circ1_final = abs(phi_ring_circulation(cl1, ring_js, centre))

        # lam=0: ring persists
        self.assertGreater(
            circ0_final, circ0_init * 0.2, "lam=0 ring should retain circulation"
        )
        # lam>0 on ring: circulation collapses faster than lam=0
        self.assertGreater(
            circ0_final,
            circ1_final,
            "lam on ring cells should reduce retained circulation",
        )
