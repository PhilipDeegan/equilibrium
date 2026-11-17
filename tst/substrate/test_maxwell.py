"""
Maxwell emergence tests on the Kelvin cell substrate.

Verify that the two-field (u, phi) update rule on the Kelvin lattice
reproduces Maxwell's equations in vacuum in the long-wavelength, linear,
source-free limit.

Field identification:
    u        <->  E  (scalar proxy for the electric field component)
    phi[i,k] <->  B  (magnetic face flux through each directed face)

The update rule then reads:
    dphi[i,k]/dt = -w_k (u[j] - u[i])   ->  dB/dt = -curl E   (Faraday)
    du[i]/dt     = -sum_k w_k phi[i,k]   ->  dE/dt = -curl B   (Ampere, vacuum)

Domain of validity (where Maxwell is recovered):
    - Wavelength >> cell spacing  (continuum limit)
    - |delta_u| << u_background   (linear regime, small perturbations)
    - No topological defects      (pure vacuum, no knots)
    - Uniform sigma               (single-clock approximation, small domain)

Tests are independent:
    TestDiscreteFaraday      -- phi update rule in isolation
    TestDiscreteAmpere       -- u update rule in isolation
    TestDispersionRelation   -- integration: both together give omega = c*k
    TestDivergencePreservation -- constraint: div B = 0 maintained by dynamics
"""

import unittest

import numpy as np

from equilibrium.substrate.ab import _set_solenoid, _solenoid_cells
from equilibrium.substrate.lattice import W_HEX, W_SQ, SubstrateLattice
from equilibrium.substrate.mhd import _SQ_AXIS_PAIRS


class TestDiscreteFaraday(unittest.TestCase):
    """
    Verify: dphi[i,k]/dt = -w_k * (u[j] - u[i])

    Discrete Faraday: dB/dt = -curl E.
    Each directed face (i->j, weight w_k) carries magnetic flux phi[i,k].
    Its rate of change is set by the electric potential difference across
    the face, weighted by w_k.

    Hex faces (k < 8, w=W_HEX=3) and sq faces (k >= 8, w=W_SQ=1) are
    tested separately because they have different weights and distances.
    """

    def test_faraday_hex_face(self):
        """
        Set up a lattice with a known u gradient, zero phi.
        Step once with small dt.
        For a hex face (i->j, k < 8), verify:
            (phi[i,k](t+dt) - phi[i,k](t)) / dt == -W_HEX * (u[j] - u[i])
        """
        lat = SubstrateLattice(radius_cells=4.0)
        lat.u = lat.pos[:, 0].copy()  # linear x-gradient

        # central cell with a valid hex neighbour
        i = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
        k = next((kk for kk in range(8) if lat.nb[i, kk] >= 0), None)
        self.assertIsNotNone(k, "no valid hex face found near centre")

        j = lat.nb[i, k]
        u_i, u_j = float(lat.u[i]), float(lat.u[j])
        phi_before = float(lat.phi[i, k])

        dt = 1e-4
        lat.step(dt)

        rate = (float(lat.phi[i, k]) - phi_before) / dt
        self.assertAlmostEqual(rate, -W_HEX * (u_j - u_i), places=12)

    def test_faraday_sq_face(self):
        """
        Same as test_faraday_hex_face but for a sq face (k >= 8).
        Verify:
            (phi[i,k](t+dt) - phi[i,k](t)) / dt == -W_SQ * (u[j] - u[i])
        """
        lat = SubstrateLattice(radius_cells=4.0)
        lat.u = lat.pos[:, 0].copy()

        i = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
        k = next((kk for kk in range(8, 14) if lat.nb[i, kk] >= 0), None)
        self.assertIsNotNone(k, "no valid sq face found near centre")

        j = lat.nb[i, k]
        u_i, u_j = float(lat.u[i]), float(lat.u[j])
        phi_before = float(lat.phi[i, k])

        dt = 1e-4
        lat.step(dt)

        rate = (float(lat.phi[i, k]) - phi_before) / dt
        self.assertAlmostEqual(rate, -W_SQ * (u_j - u_i), places=12)

    def test_faraday_plaquette_circulation(self):
        """
        Integral form of Faraday's law:

            d/dt sum_boundary phi[i,k] = -sum_boundary w_k (u[j] - u[i])

        Three loops are tested:

        Sq 4-cycle  (k=8,10,9,11 — all W_SQ):  RHS telescopes to 0 for any u
            → precision check: |d(circ)/dt| < 1e-12.

        Hex 4-cycle (k=7,4,2,1   — all W_HEX): same telescoping argument
            → same precision check.

        Mixed 3-cycle (k=8 sq, k=3 hex, k=0 hex): weights differ, no
            telescoping. With u=pos[:,0]:
            RHS = (W_HEX - W_SQ) * (u[c1] - u[c0]) = (W_HEX - W_SQ) * lattice_a
            → non-trivial assertion.
        """
        w = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
        dt = 1e-4

        def plaquette_rate_and_rhs(lat, cells, faces):
            n = len(cells)
            u0 = lat.u.copy()
            rhs = sum(
                -w[faces[m]] * (u0[cells[(m + 1) % n]] - u0[cells[m]]) for m in range(n)
            )
            circ_before = sum(float(lat.phi[cells[m], faces[m]]) for m in range(n))
            lat.step(dt)
            circ_after = sum(float(lat.phi[cells[m], faces[m]]) for m in range(n))
            return (circ_after - circ_before) / dt, rhs

        # --- sq 4-cycle ---
        lat = SubstrateLattice(radius_cells=6.0)
        lat.u = lat.pos[:, 0].copy()
        c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
        sq = [c0, int(lat.nb[c0, 8]), -1, -1]
        sq[2] = int(lat.nb[sq[1], 10])
        sq[3] = int(lat.nb[sq[2], 9])
        self.assertTrue(all(c >= 0 for c in sq))
        self.assertEqual(lat.nb[sq[3], 11], c0)
        rate, rhs = plaquette_rate_and_rhs(lat, sq, [8, 10, 9, 11])
        self.assertAlmostEqual(rate, rhs, places=12)

        # --- hex 4-cycle (offsets +++ / +-- / -+- / --+ close the loop) ---
        lat = SubstrateLattice(radius_cells=6.0)
        lat.u = lat.pos[:, 0].copy()
        c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
        hx = [c0, int(lat.nb[c0, 7]), -1, -1]
        hx[2] = int(lat.nb[hx[1], 4])
        hx[3] = int(lat.nb[hx[2], 2])
        self.assertTrue(all(c >= 0 for c in hx))
        self.assertEqual(lat.nb[hx[3], 1], c0)
        rate, rhs = plaquette_rate_and_rhs(lat, hx, [7, 4, 2, 1])
        self.assertAlmostEqual(rate, rhs, places=12)

        # --- mixed 3-cycle: hex k=0, hex k=3, sq k=9 --- RHS = (W_HEX - W_SQ)*a (non-trivial)
        #
        # NOTE: the original loop here (faces [8, 3, 0], i.e. sq(-x) -> hex(-,+,+)
        # -> hex(-,-,-)) did not actually close -- its offsets summed to
        # (-a-2e, 0, 0), not zero (a=2/sqrt(3), e=1/sqrt(3): no single hex face
        # can supply the compensating (+a+e,-e,-e) needed). That made
        # `assertEqual(lat.nb[mx[2], 0], c0)` fail unconditionally: it was
        # asserting a topological fact about the Kelvin lattice that isn't
        # true, not a numerical tolerance issue. Replaced with a loop verified
        # by brute-force search over closing (offset-sum == 0) triples of
        # faces to actually return to c0.
        lat = SubstrateLattice(radius_cells=6.0)
        lat.u = lat.pos[:, 0].copy()
        c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
        mx = [c0, int(lat.nb[c0, 0]), -1]
        mx[2] = int(lat.nb[mx[1], 3])
        self.assertTrue(all(c >= 0 for c in mx))
        self.assertEqual(lat.nb[mx[2], 9], c0)
        rate, rhs = plaquette_rate_and_rhs(lat, mx, [0, 3, 9])
        a = float(
            lat.pos[int(lat.nb[c0, 9]), 0]
        )  # = lattice_a (+x sq offset, unambiguous sign)
        self.assertAlmostEqual(rhs, (W_HEX - W_SQ) * a, places=12)
        self.assertAlmostEqual(rate, rhs, places=12)


class TestDiscreteAmpere(unittest.TestCase):
    """
    Verify: du[i]/dt = -sum_k w_k * phi[i,k]

    Discrete Ampere in vacuum: dE/dt = -curl B.
    The electric field u at cell i changes at a rate given by the weighted
    sum of surrounding face fluxes phi.

    The algebraic form follows directly from the update rule; the physical
    content here is that the discrete weighted sum correctly approximates
    the continuum curl B in the long-wavelength limit.
    """

    def test_ampere_single_cell(self):
        """
        Set up a lattice with a known phi configuration, zero u.
        Step once with small dt.
        For a cell i not on the boundary, verify:
            (u[i](t+dt) - u[i](t)) / dt == -sum_k w_k * phi[i,k](t)

        With phi=1 everywhere and all 14 neighbours valid:
            expected = -(8*W_HEX + 6*W_SQ) = -30
        """
        lat = SubstrateLattice(radius_cells=4.0)
        lat.phi = np.where(lat._valid, 1.0, 0.0)

        i = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
        self.assertTrue(lat._valid[i].all(), "central cell must have all 14 neighbours")

        u_before = float(lat.u[i])
        dt = 1e-4
        lat.step(dt)

        rate = (float(lat.u[i]) - u_before) / dt
        # phi=1 on all 14 valid faces, u=0 so phi unchanged during step
        self.assertAlmostEqual(rate, -(8 * W_HEX + 6 * W_SQ), places=12)

    def test_ampere_weight_ratio(self):
        """
        Verify that a single hex face (w=W_HEX=3) contributes exactly 3x more
        than a single sq face (w=W_SQ=1) to the Ampere update.

        Using all-hex vs all-sq would give (8*W_HEX)/(6*W_SQ)=4, not 3, because
        the face counts differ. One active face each isolates the weight ratio.
        """
        dt = 1e-4
        i = None

        def make_lat():
            lat = SubstrateLattice(radius_cells=4.0)
            nonlocal i
            i = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
            return lat

        # lattice A: exactly one hex face active (k=0)
        lat_a = make_lat()
        lat_a.phi[i, 0] = 1.0
        u_before = float(lat_a.u[i])
        lat_a.step(dt)
        du_a = float(lat_a.u[i]) - u_before

        # lattice B: exactly one sq face active (k=8)
        lat_b = make_lat()
        lat_b.phi[i, 8] = 1.0
        u_before = float(lat_b.u[i])
        lat_b.step(dt)
        du_b = float(lat_b.u[i]) - u_before

        self.assertAlmostEqual(du_a / du_b, W_HEX / W_SQ, places=12)


class TestDispersionRelation(unittest.TestCase):
    """
    Verify that the two-field update rule produces waves with dispersion
    relation omega = c * k, where c is set by the integer coupling weights.

    Integration test: Faraday and Ampere must both hold together to give
    the correct wave speed. A failure here means at least one of them is
    breaking down (but does not say which).

    The expected c must be derived analytically from the Kelvin cell geometry
    and the integer weights before writing the assertion. The derivation
    belongs in a comment or docstring here.

    Expected: c^2 = (sum_k w_k^2 * d_k^2) / (3 * something from cell volume)
    -- fill in the exact expression before implementing.
    """

    def test_wave_speed_x_direction(self):
        """
        Exact discrete dispersion relation for a wave along x:

            omega^2 = 8*W_HEX^2*(1 - cos(K*h)) + 2*W_SQ^2*(1 - cos(K*a))

        where h = a/2, a = BCC cell side.  Continuum limit K->0 gives
        c^2 = (W_HEX^2 + W_SQ^2)*a^2 = 10*a^2.

        Method: initialise u = sin(K*x), phi=0.  This is a standing-wave
        initial condition (du/dt = 0 at t=0, d^2u/dt^2 != 0).  The cell
        at x = lambda/4 starts at u=+1 and evolves as cos(omega*t).
        Time the first zero-crossing; omega = pi / (2 * t_cross).
        Only ~T/4 / dt ~ 125 steps needed.
        """
        lat = SubstrateLattice(radius_cells=10.0)
        c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
        a = float(lat.pos[int(lat.nb[c0, 8]), 0])
        h = a / 2.0

        n_cells = 8  # cells per wavelength; K*h = pi/8, dispersion error ~0.3%
        K = 2 * np.pi / (n_cells * a)

        omega_exact = np.sqrt(
            8 * W_HEX**2 * (1 - np.cos(K * h)) + 2 * W_SQ**2 * (1 - np.cos(K * a))
        )

        lat.u = np.sin(K * lat.pos[:, 0])

        # cell at x = lambda/4 = n_cells/4 * a, on the x-axis (y=z=0)
        target_x = (n_cells // 4) * a
        i_peak = int(
            np.argmin(
                np.abs(lat.pos[:, 0] - target_x)
                + 8 * np.abs(lat.pos[:, 1])
                + 8 * np.abs(lat.pos[:, 2])
            )
        )
        self.assertAlmostEqual(float(lat.u[i_peak]), 1.0, places=10)

        dt = lat.stable_dt()
        u_prev = float(lat.u[i_peak])
        t_cross = None
        limit = int(np.pi / (2 * omega_exact * dt)) + 80
        for n in range(1, limit):
            lat.step(dt)
            u_curr = float(lat.u[i_peak])
            if u_curr <= 0.0 and u_prev > 0.0:
                t_cross = dt * (n - 1 + u_prev / (u_prev - u_curr))
                break
            u_prev = u_curr

        self.assertIsNotNone(t_cross, "no zero crossing found within expected window")
        omega_meas = np.pi / (2 * t_cross)
        self.assertAlmostEqual(omega_meas / omega_exact, 1.0, delta=0.01)

    def test_dispersion_isotropy(self):
        """
        Verify wave speed is the same along x, y, z, and (1,1,0)/sqrt(2).

        General exact discrete dispersion (any K_vec = (Kx,Ky,Kz)):
            omega^2 = 8*W_HEX^2*(1 - cos(h*Kx)*cos(h*Ky)*cos(h*Kz))
                    + W_SQ^2*(6 - 2*(cos(a*Kx)+cos(a*Ky)+cos(a*Kz)))

        For n_cells=8 at K*a=pi/4: x/y/z give omega~2.463, (1,1,0) gives
        omega~2.452 — 0.46% anisotropy.  All are within 1% of each other.
        The anisotropy vanishes as K->0 (4th-order effect on BCC).
        """
        n_cells = 8

        def omega_exact(K_vec, a):
            h = a / 2
            Kx, Ky, Kz = K_vec
            return np.sqrt(
                W_HEX**2 * 8 * (1 - np.cos(h * Kx) * np.cos(h * Ky) * np.cos(h * Kz))
                + W_SQ**2 * (6 - 2 * (np.cos(a * Kx) + np.cos(a * Ky) + np.cos(a * Kz)))
            )

        def measure_omega(direction):
            lat = SubstrateLattice(radius_cells=10.0)
            c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
            a = float(lat.pos[int(lat.nb[c0, 8]), 0])
            K_vec = (2 * np.pi / (n_cells * a)) * direction
            lat.u = np.sin(lat.pos @ K_vec)

            interior = np.linalg.norm(lat.pos, axis=1) < 5.0
            i_peak = int(np.argmax(lat.u * interior.astype(float)))
            self.assertGreater(float(lat.u[i_peak]), 0.9)

            om_ex = omega_exact(K_vec, a)
            dt = lat.stable_dt()
            u_prev, t_cross = float(lat.u[i_peak]), None
            for n in range(1, int(np.pi / (2 * om_ex * dt)) + 80):
                lat.step(dt)
                u_curr = float(lat.u[i_peak])
                if u_curr <= 0.0 and u_prev > 0.0:
                    t_cross = dt * (n - 1 + u_prev / (u_prev - u_curr))
                    break
                u_prev = u_curr
            self.assertIsNotNone(t_cross, f"no zero crossing in direction {direction}")
            return np.pi / (2 * t_cross)

        directions = {
            "x": np.array([1, 0, 0], dtype=float),
            "y": np.array([0, 1, 0], dtype=float),
            "z": np.array([0, 0, 1], dtype=float),
            "110": np.array([1, 1, 0], dtype=float) / np.sqrt(2),
        }

        omegas = {name: measure_omega(v) for name, v in directions.items()}

        # x, y, z are identical by Oh symmetry; all four within 1% of each other
        self.assertAlmostEqual(omegas["x"], omegas["y"], delta=1e-10)
        self.assertAlmostEqual(omegas["x"], omegas["z"], delta=1e-10)
        ratio = max(omegas.values()) / min(omegas.values())
        self.assertAlmostEqual(ratio, 1.0, delta=0.01, msg=f"anisotropy: {omegas}")


class TestFullPotentialFieldIsZero(unittest.TestCase):
    """
    Fast regression for a 2026-08-19 finding (see project_dst_em_field_
    convention_crisis memory): treating `u` as a scalar potential and `phi`
    as a vector potential, the full field `E := -grad(u) - dphi/dt` is
    IDENTICALLY zero under plain step() (free wave dynamics), to machine
    precision -- a genuine algebraic identity of the update rule, general
    (not specific to any one field configuration), not an approximation.

    This matters because it makes the "u=potential, phi=vector-potential"
    interpretation a dead end for computing radiated energy flux (E is
    trivially zero under any free dynamics in that reading) -- any future
    Poynting-vector/S construction must NOT be built on this identification.
    Each term (-grad(u), -dphi/dt) is individually real and nonzero; they
    exactly cancel. Cheap (~seconds), keep this passing as a guardrail.
    """

    def test_grad_u_plus_dphi_dt_is_exactly_zero(self):
        lat = SubstrateLattice(radius_cells=8.0)
        sigma = 2.0
        lat.u = np.exp(-np.sum(lat.pos**2, axis=1) / sigma**2)
        dt = 1e-4

        nb = lat.nb
        valid = nb >= 0
        safe_nb = np.where(valid, nb, 0)

        def grad_u_faces(u):
            u_nb = np.where(valid, u[safe_nb], u[:, None])
            return u_nb - u[:, None]

        def face_vec(face_arr):
            return np.stack(
                [face_arr[:, fp] - face_arr[:, fm] for fp, fm in _SQ_AXIS_PAIRS], axis=1
            )

        phi_before = lat.phi.copy()
        grad_u_before = grad_u_faces(lat.u.copy())
        lat.step(dt)
        dphi = lat.phi - phi_before

        e_full = -face_vec(grad_u_before) - face_vec(dphi) / dt
        grad_term = face_vec(grad_u_before)

        self.assertGreater(
            float(np.sqrt(np.mean(grad_term**2))),
            1e-3,
            "sanity check: grad(u) term must be genuinely nonzero, or this "
            "test is vacuous",
        )
        self.assertLess(
            float(np.max(np.abs(e_full))),
            1e-10,
            "E_full = -grad(u) - dphi/dt should be zero to machine precision",
        )


class TestDivergencePreservation(unittest.TestCase):
    """
    Characterise div B on the substrate.

    In continuum EM, div B = 0 is an initial condition that must be imposed
    and then shown to be preserved. In DST it is different: the substrate is
    everywhere, with no edges and no boundary. There is nowhere for a flux
    line to terminate. div B = 0 is a topological fact of the substrate --
    it is not a constraint the update rule needs to maintain, it is something
    that was never violated in the first place.

    The interesting question is therefore not "is it preserved?" (it is, by
    construction) but rather: does a specific initialisation -- in particular
    the AB solenoid angle formula -- introduce spurious per-cell divergence,
    and if so, is it localised or does it spread?

    test_div_b_zero_initial_preserved is kept as a numerical sanity check on
    the implementation (floating-point should not accumulate divergence), but
    it is not a physical test of the theory.
    """

    def test_div_b_zero_initial_preserved(self):
        """
        The correct discrete 'div B = 0' invariant is antisymmetry:

            phi[i,k] + phi[nb[i,k], mirror[k]] == 0  for all valid (i,k)

        This is preserved BIT-EXACTLY by the update rule: both phi[i,k] and
        phi[j,mirror[k]] are computed from the same (u[j]-u[i]) expression,
        so their sum cancels algebraically regardless of float rounding.

        NOTE: the per-cell sum sum_k phi[i,k] is NOT zero after t>0 — it
        equals the discrete Laplacian of u (up to dt) and grows to O(1).
        That quantity is NOT the div B = 0 constraint; the antisymmetry is.
        """
        mirror = np.array([7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 11, 10, 13, 12])

        lat = SubstrateLattice(radius_cells=8.0)
        sigma = 2.0
        lat.u = np.exp(-np.sum(lat.pos**2, axis=1) / sigma**2)
        dt = lat.stable_dt()

        for checkpoint in range(4):
            for _ in range(50):
                lat.step(dt)

            max_err = 0.0
            for k in range(14):
                mk = int(mirror[k])
                j = lat.nb[:, k]
                mask = j >= 0
                errs = lat.phi[mask, k] + lat.phi[j[mask], mk]
                if errs.size:
                    max_err = max(max_err, float(np.max(np.abs(errs))))

            self.assertEqual(
                max_err, 0.0, f"antisymmetry broken at checkpoint {checkpoint}"
            )

    def test_div_b_ab_solenoid_structure(self):
        """
        Characterisation of the per-cell divergence introduced by the angle
        formula initialisation of the AB solenoid.

        Findings (flux_quanta=1, solenoid_radius=1.5, radius_cells=8):

          t=0  inside solenoid: max|div| ~ 0.75   (large, localised)
               outside solenoid: max|div| ~ 0.04  (small, ~20x smaller)

          t=50 inside solenoid: max|div| ~ 0.75   (unchanged — phi fixed)
               outside solenoid: max|div| ~ 0.3   (grows as field propagates)

        The angle formula also breaks antisymmetry (phi[i,k]+phi[j,mirror(k)]
        is O(1)), unlike the zero-phi case where antisymmetry is bit-exact.
        This is a known limitation of the angle formula on a discrete lattice:
        it guarantees correct loop integrals but not per-cell flux balance.

        Assertions:
          - Inside divergence is at least 10x outside at t=0 (localisation).
          - Inside divergence is preserved to 1e-12 (phi is frozen there).
          - Outside divergence grows after 50 steps (spreading, not confined).
        """
        _mirror = np.array([7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 11, 10, 13, 12])

        lat = SubstrateLattice(radius_cells=8.0)
        sol_idx = _solenoid_cells(lat, 1.5)
        _set_solenoid(lat, sol_idx, flux_quanta=1)

        sol_mask = np.zeros(lat.N, dtype=bool)
        sol_mask[sol_idx] = True

        def max_abs_div(mask):
            phi_eff = np.where(lat._valid, lat.phi, 0.0)
            div = phi_eff.sum(axis=1)
            return float(np.max(np.abs(div[mask])))

        div_in_t0 = max_abs_div(sol_mask)
        div_out_t0 = max_abs_div(~sol_mask)

        # solenoid divergence is strongly localised at t=0
        self.assertGreater(div_in_t0, 10 * div_out_t0)

        lat.run(50)

        div_in_t50 = max_abs_div(sol_mask)
        div_out_t50 = max_abs_div(~sol_mask)

        # inside stays constant (phi is frozen on solenoid cells)
        self.assertAlmostEqual(div_in_t50, div_in_t0, places=12)

        # outside divergence spreads into the bulk during dynamics
        self.assertGreater(div_out_t50, div_out_t0)


if __name__ == "__main__":
    unittest.main()
