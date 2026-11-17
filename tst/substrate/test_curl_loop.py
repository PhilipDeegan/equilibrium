"""
Verify equilibrium.substrate.curl's discrete curl(phi)=B against the exact
classical circular-current-loop field -- the PROTON candidate test (open
toroidal ring circulation, poloidal dipole field DERIVED from it via
curl(phi)=B, not independently seeded the way the closed neutron spheromak
IC does -- see `now`, 2026-09-02).

Method: same style as test_curl.py's straight-wire check. Seed phi from the
exact analytic vector potential A_phi(rho,z) of a circular current loop
(elliptic-integral closed form), sample it onto phi via a discrete line
integral per directed edge, then check the recovered B against the known
closed-form B(rho,z) (also elliptic-integral exact) -- on-axis (pure dipole
1/z^3-style falloff, zero near-field complication) and in the equatorial
plane (near-field sign reversal just outside the loop AND far-field falloff,
a genuinely two-region shape check). Pure static geometry test, no
time-stepping. mu0*I=1 (shape/ratio only, not absolute calibration).
"""

import unittest

import numpy as np
from scipy.special import ellipe, ellipk

from equilibrium.substrate.curl import b_vec_vectorized
from equilibrium.substrate.lattice import SubstrateLattice

R = 30.0
LOOP_A = 8.0
# Same hop-distance^2 calibration factor found for the straight wire
# (test_curl.py) -- confirms this is a universal discrete-curl geometric
# constant, not specific to that source's geometry.
_HOP_SQ_FACTOR = (2.0 / np.sqrt(3.0)) ** 2


def _k2(rho, z, a):
    return 4.0 * a * rho / ((a + rho) ** 2 + z**2)


def _a_phi(rho, z, a):
    rho = np.asarray(rho, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    safe_rho = np.where(rho > 1e-9, rho, 1.0)
    k2 = np.clip(_k2(safe_rho, z, a), 0.0, 1.0 - 1e-12)
    K, E = ellipk(k2), ellipe(k2)
    val = (1.0 / np.pi) * np.sqrt(a / safe_rho) / np.sqrt(k2) * ((1.0 - k2 / 2.0) * K - E)
    return np.where(rho > 1e-6, val, 0.0)


def _b_field(rho, z, a):
    rho = np.asarray(rho, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    safe_rho = np.where(rho > 1e-9, rho, 1.0)
    k2 = np.clip(_k2(safe_rho, z, a), 0.0, 1.0 - 1e-12)
    K, E = ellipk(k2), ellipe(k2)
    denom_pm = (a - safe_rho) ** 2 + z**2
    denom_pm = np.where(denom_pm > 1e-9, denom_pm, 1e-9)
    root = np.sqrt((a + safe_rho) ** 2 + z**2)
    prefactor = 1.0 / (2.0 * np.pi)
    b_rho = prefactor * (z / (safe_rho * root)) * (-K + (a**2 + safe_rho**2 + z**2) / denom_pm * E)
    b_z = prefactor * (1.0 / root) * (K + (a**2 - safe_rho**2 - z**2) / denom_pm * E)
    b_rho = np.where(rho > 1e-6, b_rho, 0.0)
    return b_rho, b_z


def _b_z_axis(z, a):
    return a**2 / (2.0 * (a**2 + z**2) ** 1.5)


def _build_phi(lat, a):
    pos, nb = lat.pos, lat.nb
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    mid = 0.5 * (pos[:, None, :] + pos[safe_nb])
    rho_mid = np.linalg.norm(mid[..., :2], axis=2)
    z_mid = mid[..., 2]
    a_phi_mid = _a_phi(rho_mid, z_mid, a)
    safe_rho_mid = np.where(rho_mid > 1e-9, rho_mid, 1.0)
    phihat_x = -mid[..., 1] / safe_rho_mid
    phihat_y = mid[..., 0] / safe_rho_mid
    diff = pos[safe_nb] - pos[:, None, :]
    a_dot_dl = a_phi_mid * (phihat_x * diff[..., 0] + phihat_y * diff[..., 1])
    return np.where(valid, a_dot_dl, 0.0)


def _measure():
    lat = SubstrateLattice(radius_cells=R)
    pos, nb = lat.pos, lat.nb
    lat.phi = _build_phi(lat, LOOP_A)
    B = b_vec_vectorized(pos, nb, lat.phi)
    Bx, By, Bz = B[:, 0], B[:, 1], B[:, 2]
    have = np.isfinite(Bx) & np.isfinite(By) & np.isfinite(Bz)
    rho = np.linalg.norm(pos[:, :2], axis=1)
    z = pos[:, 2]
    theta = np.arctan2(pos[:, 1], pos[:, 0])
    d_loop = np.sqrt((rho - LOOP_A) ** 2 + z**2)
    b_phi_meas = -Bx * np.sin(theta) + By * np.cos(theta)
    return pos, rho, z, Bx, By, Bz, b_phi_meas, have, d_loop


class TestCurlCircularLoop(unittest.TestCase):
    def test_b_phi_vanishes(self):
        """An axisymmetric loop current has NO azimuthal B component."""
        _, rho, z, Bx, By, Bz, b_phi_meas, have, d_loop = _measure()
        mask = have & (d_loop > 2.0) & (rho > 0.5)
        b_phi_rms = float(np.sqrt(np.mean(b_phi_meas[mask] ** 2)))
        signal_rms = float(np.sqrt(np.mean(Bx[mask] ** 2 + By[mask] ** 2 + Bz[mask] ** 2)))
        self.assertLess(b_phi_rms, 0.1 * signal_rms)

    def test_bz_axis_matches_classical_loop_profile(self):
        """
        On-axis Bz(z): pure classical dipole-like falloff (rho=0, no near-
        field complication) -- constant ratio to the exact target, at the
        SAME hop-distance^2 calibration factor as the straight-wire test.
        """
        _, rho, z, Bx, By, Bz, _, have, _ = _measure()
        axis_mask = have & (rho < 1.0)
        zs = np.sort(np.unique(np.round(z[axis_mask])))
        ratios = []
        for zv in zs:
            m = axis_mask & (np.abs(z - zv) < 0.5)
            if m.sum() < 1:
                continue
            bz_t = float(_b_z_axis(zv, LOOP_A))
            if abs(bz_t) < 1e-9:
                continue
            ratios.append(float(np.mean(Bz[m])) / bz_t)
        self.assertGreaterEqual(len(ratios), 10, "not enough populated on-axis bins")
        ratios = np.array(ratios)
        self.assertAlmostEqual(float(ratios.std()), 0.0, delta=0.05)
        self.assertAlmostEqual(float(ratios.mean()), _HOP_SQ_FACTOR, delta=0.05)

    def test_bz_equatorial_matches_classical_loop_profile(self):
        """
        Equatorial-plane Bz(rho): near-field sign reversal just outside the
        loop AND far-field falloff, both at the same constant ratio -- a
        genuinely two-region shape check, not just a monotonic decay.
        """
        pos, rho, z, Bx, By, Bz, _, have, d_loop = _measure()
        eq_mask = have & (np.abs(z) < 1.0) & (d_loop > 2.0)
        edges = np.linspace(0.0, R * 0.85, 20)
        ratios = []
        saw_sign_flip = False
        prev_sign = None
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = eq_mask & (rho >= lo) & (rho < hi)
            if m.sum() < 3:
                continue
            rho_c = float(np.mean(rho[m]))
            bz_m = float(np.mean(Bz[m]))
            _, bz_t_arr = _b_field(np.full(m.sum(), rho_c), np.zeros(m.sum()), LOOP_A)
            bz_t = float(np.mean(bz_t_arr))
            if abs(bz_t) < 1e-9:
                continue
            sign = bz_t > 0
            if prev_sign is not None and sign != prev_sign:
                saw_sign_flip = True
            prev_sign = sign
            ratios.append(bz_m / bz_t)
        self.assertTrue(saw_sign_flip, "expected a near-field sign reversal outside the loop")
        self.assertGreaterEqual(len(ratios), 10, "not enough populated equatorial bins")
        ratios = np.array(ratios)
        self.assertAlmostEqual(float(ratios.std()), 0.0, delta=0.05)
        self.assertAlmostEqual(float(ratios.mean()), _HOP_SQ_FACTOR, delta=0.05)


if __name__ == "__main__":
    unittest.main()
