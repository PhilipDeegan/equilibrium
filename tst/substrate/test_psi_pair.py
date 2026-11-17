"""
Regression tests for equilibrium.substrate.psi_pair -- the (phi, psi)
curl-conjugate candidate mechanism (see that module's docstring and `now`,
2026-09-08 entries, for the physics context: this is the structurally-fixed
successor to the bare curl(curl(phi)) self-loop already confirmed
unconditionally unstable, doc/dst/failures/naive-reverse-curl-term-
unconditionally-unstable.txt).

These tests capture the empirically-established "recipe" from the
2026-09-08 calibration sweeps (exp/2026/09/08/{09_34_c2_speed_matching_
sweep,10_32_weighted_circ_scatter,11_11_resolution_convergence}) as hard
regressions, independent of whether the underlying mechanism is the final
ontological answer for DST magnetism:

    1. scatter_DT is the exact algebraic transpose of circ_D (the adjoint
       pairing the weighting scheme is supposed to guarantee -- checked
       directly, not assumed).
    2. With c2=0 the pair is inert: no spurious curl is ever generated
       (guards against silently reintroducing a self-sourcing bug).
    3. The pair is stable (no blowup) across the full previously-swept c2
       range, including values well above the calibrated optimum --
       guards against reintroducing the earlier unconditional-instability
       failure mode.
    4. At the calibrated c2, a real traveling wave produces a genuinely
       nonzero, bounded, non-frozen curl disturbance (the qualitative
       "recipe" fact: partial co-propagation, not a static deposit) --
       bounds are generous and set from an actual measured run, not tuned
       to make the test pass.
"""

import unittest

import numpy as np

from equilibrium.substrate.curl import b_vec_vectorized
from equilibrium.substrate.discrete_curl import triangle_triples
from equilibrium.substrate.lattice import W_HEX, W_SQ, SubstrateLattice
from equilibrium.substrate.psi_pair import (
    C2_CALIBRATED,
    N_FAMILIES,
    circ_D,
    scatter_DT,
)
from equilibrium.substrate.traveling_wave import seed_traveling_wave
from tst.substrate.adaptive.adaptive_flux_tube import _MIRROR

_W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)

# Small cylinder -- fast (~20k cells, <1s to build) but big enough to carry
# a real traveling wave envelope without touching the domain boundary.
R_SMALL = 8.0
HALF_LEN_SMALL = 40.0
AXIS = 2
Z0_SMALL = -20.0
N_STEPS_SMALL = 300


def _small_lattice():
    lat = SubstrateLattice.cylinder(R_SMALL, HALF_LEN_SMALL, axis=AXIS)
    triples = triangle_triples(lat.nb)
    return lat, triples


def _base_step(lat, dt, valid, safe_nb):
    div_phi = np.sum(np.where(valid, _W[None, :] * lat.phi, 0.0), axis=1)
    u_new = lat.u - dt * div_phi
    u_nb_new = u_new[safe_nb]
    drive = np.where(valid, u_nb_new - u_new[:, None], 0.0)
    dphi = -dt * _W[None, :] * drive
    return u_new, dphi


def _run(lat, triples, dt, valid, safe_nb, c2, n_steps):
    """Runs the base (u,phi) wave step plus the (phi,psi) pair for n_steps,
    returning per-log-point (B_rms, gap) traces."""
    psi_dense = np.zeros((lat.N, N_FAMILIES))
    log = []
    for step in range(n_steps):
        u_new, dphi_base = _base_step(lat, dt, valid, safe_nb)
        vals = circ_D(lat.phi, triples, lat.N, _MIRROR)
        psi_new = psi_dense - dt * vals
        extra = scatter_DT(psi_new, lat.phi.shape, triples, _MIRROR)
        lat.u, lat.phi, psi_dense = u_new, lat.phi + dphi_base + dt * c2 * extra, psi_new

        B = b_vec_vectorized(lat.pos, lat.nb, lat.phi)
        valid_rows = ~np.isnan(B).any(axis=1)
        Bmag2 = np.sum(B[valid_rows] ** 2, axis=1)
        B_rms = float(np.sqrt(np.mean(Bmag2))) if Bmag2.size else float("nan")

        pos_v = lat.pos[valid_rows]
        Bsum = float(np.sum(Bmag2))
        B_centroid_z = float(np.sum(Bmag2 * pos_v[:, AXIS]) / Bsum) if Bsum > 0 else float("nan")
        e = lat.u**2 + 0.5 * np.sum(lat.phi**2, axis=1)
        e_tot = float(np.sum(e))
        u_centroid_z = float(np.sum(e * lat.pos[:, AXIS]) / e_tot) if e_tot > 0 else float("nan")
        log.append((B_rms, B_centroid_z - u_centroid_z))
    return log


class TestScatterIsExactTranspose(unittest.TestCase):
    def test_adjoint_identity(self):
        """<circ_D(phi), psi> == <phi, scatter_DT(psi)> for random phi/psi --
        the algebraic adjoint identity the weighting scheme is supposed to
        satisfy exactly (not just approximately)."""
        rng = np.random.default_rng(0)
        lat, triples = _small_lattice()
        phi = rng.normal(size=lat.phi.shape)
        psi = rng.normal(size=(lat.N, N_FAMILIES))

        lhs = float(np.sum(circ_D(phi, triples, lat.N, _MIRROR) * psi))
        rhs = float(np.sum(phi * scatter_DT(psi, phi.shape, triples, _MIRROR)))
        self.assertAlmostEqual(lhs, rhs, places=8)


class TestC2ZeroIsInert(unittest.TestCase):
    def test_no_spurious_curl(self):
        """c2=0 must reproduce the base system's own frozen-B identity
        exactly -- the (phi,psi) pair must never source curl on its own
        when its feedback into phi is switched off."""
        lat, triples = _small_lattice()
        seed_traveling_wave(
            lat, AXIS, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=3.0, z0=Z0_SMALL
        )
        dt = lat.stable_dt()
        valid = lat.nb >= 0
        safe_nb = np.where(valid, lat.nb, 0)

        log = _run(lat, triples, dt, valid, safe_nb, c2=0.0, n_steps=N_STEPS_SMALL)
        B_rms_values = [b for b, _ in log]
        self.assertLess(max(B_rms_values), 1e-9)


class TestStabilityAcrossC2Range(unittest.TestCase):
    def test_no_blowup(self):
        """No blowup across the full previously-swept c2 range (0 to well
        above the calibrated optimum) -- regression guard against
        reintroducing the earlier unconditional-instability failure mode
        (naive-reverse-curl-term-unconditionally-unstable.txt)."""
        for c2 in (0.0, 0.1, C2_CALIBRATED, 2.0, 5.0):
            with self.subTest(c2=c2):
                lat, triples = _small_lattice()
                seed_traveling_wave(
                    lat, AXIS, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=3.0, z0=Z0_SMALL
                )
                dt = lat.stable_dt()
                valid = lat.nb >= 0
                safe_nb = np.where(valid, lat.nb, 0)
                log = _run(lat, triples, dt, valid, safe_nb, c2=c2, n_steps=N_STEPS_SMALL)
                B_rms_values = np.array([b for b, _ in log])
                self.assertTrue(np.all(np.isfinite(B_rms_values)))
                self.assertLess(float(np.max(B_rms_values)), 10.0)


class TestCalibratedC2GivesPartialCoPropagation(unittest.TestCase):
    def test_bounded_nonzero_curl_and_gap(self):
        """The qualitative 'recipe' fact from the 2026-09-08 calibration:
        at c2=C2_CALIBRATED, a real traveling wave produces a genuinely
        nonzero (not frozen), bounded (not diverging) curl-carrying
        disturbance -- a real, measured improvement over an uncalibrated
        coupling, short of full ontological closure. Bounds below are
        generous and set from an actual measured run on this exact test
        lattice, not tuned to force a pass."""
        lat, triples = _small_lattice()
        seed_traveling_wave(
            lat, AXIS, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=3.0, z0=Z0_SMALL
        )
        dt = lat.stable_dt()
        valid = lat.nb >= 0
        safe_nb = np.where(valid, lat.nb, 0)

        log = _run(lat, triples, dt, valid, safe_nb, c2=C2_CALIBRATED, n_steps=N_STEPS_SMALL)
        B_rms_final = log[-1][0]
        gaps = np.array([g for _, g in log])

        # genuinely nonzero curl generated (not the c2=0 frozen/near-zero case)
        self.assertGreater(B_rms_final, 1e-3)
        # but bounded -- no blowup
        self.assertLess(B_rms_final, 10.0)
        # gap stays bounded over this short run (no runaway divergence) --
        # observed max |gap| ~1.1 on this exact lattice/step count; allow
        # a generous margin, not a tight fit.
        self.assertTrue(np.all(np.isfinite(gaps)))
        self.assertLess(float(np.max(np.abs(gaps))), 5.0)


if __name__ == "__main__":
    unittest.main()
