"""
Verify equilibrium.substrate.energy_flux's local energy-flux identity
against the real SubstrateLattice.step() -- promoted from the exploratory
check in exp/ per feedback_save_validations_as_tests: this replaces the
E:=-dphi/dt, B:=curl(phi), S:=ExB convention (three times falsified, see
project_dst_em_field_convention_crisis) as DST's local "Poynting vector".

    E_i = u_i^2/2 + (1/4)*sum_k phi[i,k]^2
    dE_i/dt = -sum_k S_i,k     S_i,k = sigma_face_k * W_k * phi[i,k] * ubar_i,k
(sigma relational, at the face on both legs since 2026-09-29).

This is a continuous-time identity; the real step() is symplectic-Euler
(operator-split), so the per-step residual is expected to be O(dt^2), not
exactly zero. The test checks that scaling: halving dt should quarter the
residual (within a loose tolerance -- this is a numerical-analysis
convergence-order check, not a bit-exact assertion).
"""

import unittest

import numpy as np

from equilibrium.substrate.energy_flux import W_FACE, local_energy_flux_vector
from equilibrium.substrate.lattice import SubstrateLattice


def _local_energy(u, phi, sigma, sigma_face, valid):
    return 0.5 * u**2 + 0.25 * np.where(valid, phi**2, 0.0).sum(axis=1)


def _residual_rms(radius_cells, dt, seed=0):
    rng = np.random.default_rng(seed)
    lat = SubstrateLattice(radius_cells=radius_cells)
    N = lat.N
    lat.u = rng.normal(size=N) * 0.7
    lat.phi = rng.normal(size=(N, 14)) * 0.3
    lat.lam = np.abs(rng.normal(size=N)) * 0.5  # nonuniform sigma -- gravity-relevant case

    sigma = 1.0 / (1.0 + lat.lam)
    nb = lat.nb
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    sigma_face = (sigma[:, None] + np.where(valid, sigma[safe_nb], sigma[:, None])) / 2.0

    u_old, phi_old = lat.u.copy(), lat.phi.copy()
    E_old = _local_energy(u_old, phi_old, sigma, sigma_face, valid)

    lat.step(dt)
    u_new, phi_new = lat.u.copy(), lat.phi.copy()
    E_new = _local_energy(u_new, phi_new, sigma, sigma_face, valid)
    dE = E_new - E_old

    # Pair phi(before)/u(after), matching step()'s own du-from-old-phi,
    # dphi-from-new-u ordering (see energy_flux module docstring).
    S = local_energy_flux_vector(lat.pos, nb, phi_old, u_new, w_face=W_FACE, sigma_face=sigma_face)
    S_scalar_out = np.where(
        valid,
        sigma_face * W_FACE[None, :] * phi_old * (0.5 * (u_new[:, None] + np.where(valid, u_new[safe_nb], u_new[:, None]))),
        0.0,
    ).sum(axis=1)
    pred_dE = -dt * S_scalar_out
    resid = dE - pred_dE

    interior = valid.all(axis=1)
    return float(np.sqrt(np.mean(resid[interior] ** 2))), S


class TestLocalEnergyFlux(unittest.TestCase):
    def test_residual_scales_as_dt_squared(self):
        """Halving dt should quarter the O(dt^2) symplectic-splitting residual."""
        rms_a, _ = _residual_rms(6.0, 1e-3, seed=0)
        rms_b, _ = _residual_rms(6.0, 5e-4, seed=0)
        self.assertGreater(rms_a, 0.0)
        ratio = rms_a / rms_b
        self.assertAlmostEqual(ratio, 4.0, delta=0.05)

    def test_residual_shrinks_with_dt(self):
        """Sanity floor: at dt=1e-5 the identity should hold far tighter than the field scale."""
        rms, S = _residual_rms(6.0, 1e-5, seed=1)
        self.assertLess(rms, 1e-8)
        self.assertTrue(np.all(np.isfinite(S)))

    def test_vector_shape_and_zero_on_uniform_field(self):
        """A uniform u with phi=0 has no flux anywhere (nothing to source S)."""
        lat = SubstrateLattice(radius_cells=4.0)
        lat.u[:] = 1.0
        lat.phi[:] = 0.0
        S = local_energy_flux_vector(lat.pos, lat.nb, lat.phi, lat.u)
        self.assertEqual(S.shape, (lat.N, 3))
        self.assertTrue(np.allclose(S, 0.0))


if __name__ == "__main__":
    unittest.main()
