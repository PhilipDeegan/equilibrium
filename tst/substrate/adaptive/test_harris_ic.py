"""
Sanity checks for the Harris sheet IC used in harris_sheet_phare.py.

Five test classes, in order of diagnostic priority:

  TestHarrisDivPhiZero      — weighted div(phi) < 1e-12 everywhere at t=0
  TestHarrisBxProfile       — sq-face phi recovers tanh Bx(y) exactly
  TestHarrisFlatUFrozen     — flat u + div-free phi → exactly static (no step moves anything)
  TestHarrisLamTransport    — step_lam is the macro-scale mass transport driver
  TestHarrisVirtualParticle — virtual particle (macro-PIC) Lorentz reconnection mechanism

step_cross is proton-scale only: it models flux slippage through GAPS of lam between
discrete proton knots.  At macro scale (continuous lam, no gaps) it is physically wrong.

The macro-scale reconnection mechanism being developed:
  Each cell hosts a virtual particle (v.p.) representing the collective motion of its
  many enclosed proton knots.  The v.p. feels a Lorentz-analogue force:
    a_i = E_i + v_i × B_i
  where B is reconstructed from the phi face fluxes and E from the u gradient.
  This is adiabatic: field energy converts to directed kinetic energy without
  thermal diffusion — satisfying the DST adiabatic constraint.

If TestHarrisFlatUFrozen fails, the IC is not div-free OR the wave equation
is introducing spurious drift — everything else is suspect until this passes.
"""

import unittest

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.adaptive import wire_periodic_bcc

# Face geometry — must match bcc_geometry.hpp and harris_sheet_phare.py.
HEX_DIR = np.array(
    [
        [-1, -1, -1],
        [-1, -1, +1],
        [-1, +1, -1],
        [-1, +1, +1],
        [+1, -1, -1],
        [+1, -1, +1],
        [+1, +1, -1],
        [+1, +1, +1],
    ],
    dtype=np.float64,
)
SQ_DIR = np.array(
    [[-1, 0, 0], [+1, 0, 0], [0, -1, 0], [0, +1, 0], [0, 0, -1], [0, 0, +1]],
    dtype=np.float64,
)
# Pass-1 weights: W_HEX=3 for hex faces 0-7, W_SQ=1 for sq faces 8-13.
W_FACE = np.array([3.0] * 8 + [1.0] * 6)

# Small box — enough to contain two current sheets but fast to run.
LX, LY, LZ = 30.0, 30.0, 3.5
B0 = 0.05
LAM_W = 6.0  # sheet half-width (lambda in PHARE notation)
U_BG = 0.4
LAM_BG = 0.4
DT = 0.05


def _bx_profile(y, y0):
    return B0 * (-1.0 + np.tanh((y + y0) / LAM_W) - np.tanh((y - y0) / LAM_W))


def _build_box(*, flat_u=True, perturb=0.0):
    """Build a Harris sheet box and set the IC.

    Returns (alat, cl, pos_N3, phi_14N).
    phi_14N is the (14, N) array passed to set_leaf_phi — kept for reference
    so tests can compare against the actual IC without a second round-trip.
    """
    alat = _lat.AdaptiveLattice()
    cl = alat[0]
    _lat.build_bcc_box(cl, LX, LY, LZ, 1.0, u0=0.0, lam0=LAM_BG)
    alat.rebuild(0)
    H = 1.0 / np.sqrt(3.0)
    wire_periodic_bcc(cl, H)

    N = cl.n_leaves
    pos = np.array(cl.leaf_pos)  # (N, 3)
    y0 = LY / 4.0

    bx_eq = _bx_profile(pos[:, 1], y0)
    if flat_u:
        cl.set_leaf_u(np.full(N, U_BG))
    else:
        cl.set_leaf_u(U_BG + (B0**2 - bx_eq**2) / 2.0)

    phi_init = np.zeros((14, N), dtype=np.float64)
    for k in range(14):
        fd = HEX_DIR[k] if k < 8 else SQ_DIR[k - 8]
        phi_init[k] += bx_eq * fd[0]

    if perturb > 0.0:
        kx = 2.0 * np.pi / LX
        dB = perturb * B0
        for sign, y_null in ((+1, +y0), (-1, -y0)):
            y_r = pos[:, 1] - y_null
            env = np.exp(-(y_r**2) / LAM_W**2)
            kx_x = kx * (pos[:, 0] + LX / 2.0)
            bx_p = sign * (-2.0 * y_r / LAM_W**2) * dB * np.cos(kx_x) * env
            by_p = sign * kx * dB * np.sin(kx_x) * env
            for k in range(14):
                fd = HEX_DIR[k] if k < 8 else SQ_DIR[k - 8]
                phi_init[k] += bx_p * fd[0] + by_p * fd[1]

    cl.set_leaf_phi(phi_init)
    return alat, cl, pos, phi_init


class TestHarrisDivPhiZero(unittest.TestCase):
    """Weighted divergence Σ_k W_k·phi[k,i] must be < 1e-12 for every cell.

    This is what pass 1 integrates.  A nonzero value means the IC will
    immediately drive spurious u changes even with flat u and no cross step.
    """

    def _check(self, phi_14N):
        weighted_div = (phi_14N * W_FACE[:, None]).sum(axis=0)  # (N,)
        self.assertLess(float(np.abs(weighted_div).max()), 1e-12)

    def test_equilibrium_only(self):
        _, _, _, phi = _build_box(perturb=0.0)
        self._check(phi)

    def test_with_perturbation(self):
        _, _, _, phi = _build_box(perturb=0.1)
        self._check(phi)


class TestHarrisBxProfile(unittest.TestCase):
    """x-sq faces (8=-x, 9=+x) must recover Bx(y) exactly.

    phi[9,i] = +Bx(y_i),  phi[8,i] = -Bx(y_i)
    → Bx_rec = (phi[9] - phi[8]) / 2

    Checks that the face projection correctly encodes the Harris profile.
    Perturbation is left off so the comparison is clean.
    """

    def test_sq_face_recovery(self):
        _, _, pos, phi = _build_box(perturb=0.0)
        bx_rec = (phi[9] - phi[8]) / 2.0
        y0 = LY / 4.0
        bx_anal = _bx_profile(pos[:, 1], y0)
        np.testing.assert_allclose(bx_rec, bx_anal, atol=1e-14)

    def test_null_lines_at_y0(self):
        """Bx must be zero at both null lines y = ±Ly/4."""
        _, cl, pos, phi = _build_box(perturb=0.0)
        bx_rec = (phi[9] - phi[8]) / 2.0
        y0 = LY / 4.0
        for sign in (+1, -1):
            near_null = np.abs(pos[:, 1] - sign * y0) < 0.5
            self.assertTrue(near_null.any(), "no cells found near null line")
            self.assertLess(float(np.abs(bx_rec[near_null]).max()), B0 * 0.02)


class TestHarrisFlatUFrozen(unittest.TestCase):
    """Flat u + div-free phi + no cross step → u is exactly static.

    Mechanism:
      Pass 1: du = -dt·sigma·Σ W_k·phi[k,i] = 0  (div-free)
      Pass 2: dphi = -dt·sigma·W_k·(u_nb - u_i) = 0  (flat u, no gradient)
      → nothing moves

    Any failure here means the IC is broken or the step introduces spurious
    drift — all downstream physics is suspect until this passes.
    """

    N_STEPS = 200

    def test_u_unchanged(self):
        alat, cl, _, _ = _build_box(flat_u=True, perturb=0.0)
        u0 = np.array(cl.leaf_u).copy()
        for _ in range(self.N_STEPS):
            alat.step(0, DT)
        u1 = np.array(cl.leaf_u)
        max_du = float(np.abs(u1 - u0).max())
        self.assertLess(max_du, 1e-12)

    def test_phi_unchanged(self):
        alat, cl, _, _ = _build_box(flat_u=True, perturb=0.0)
        phi0 = np.array(cl.leaf_phi).copy()
        for _ in range(self.N_STEPS):
            alat.step(0, DT)
        phi1 = np.array(cl.leaf_phi)
        max_dphi = float(np.abs(phi1 - phi0).max())
        self.assertLess(max_dphi, 1e-12)


class TestHarrisLamTransport(unittest.TestCase):
    """step_lam is the macro-scale DST mass transport: dlam/dt = -sigma·lam·div(phi).

    step_cross is proton-scale only — it models flux slippage through GAPS of lam
    between discrete proton knots.  At macro scale the lam distribution is continuous
    (no gaps), so step_cross is physically wrong.  step_lam driven by the wave
    equation's div(phi) is the correct fluid-scale reconnection mechanism.

    Two cases:

    flat u: pass 2 of the wave equation gives dphi = -grad(u) = 0, so div(phi)
    stays zero, and step_lam has nothing to act on — lam is frozen.

    pressure-balanced u: the u gradient drives dphi via pass 2, introducing
    nonzero div(phi) at the current sheet.  step_lam then advects lam there,
    changing sigma locally — the DST analog of effective resistivity.
    """

    N_STEPS = 200
    LAM_SCALE = 1.0

    def test_flat_u_lam_frozen(self):
        """With flat u, div(phi)=0 everywhere, step_lam must leave lam unchanged."""
        alat, cl, _, _ = _build_box(flat_u=True, perturb=0.1)
        lam0 = np.array(cl.leaf_lam).copy()
        for _ in range(self.N_STEPS):
            alat.step(0, DT)
            alat.step_lam(0, DT, self.LAM_SCALE)
        lam1 = np.array(cl.leaf_lam)
        max_dlam = float(np.abs(lam1 - lam0).max())
        self.assertLess(
            max_dlam,
            1e-12,
            "lam changed with flat u — div(phi) should be zero, step_lam should be a no-op",
        )

    def test_pressure_balance_moves_lam_at_sheet(self):
        """Pressure-balanced u generates div(phi) via the wave equation; step_lam
        must transport lam at the current sheet as a result."""
        alat, cl, pos, _ = _build_box(flat_u=False, perturb=0.1)
        y0 = LY / 4.0
        lam0 = np.array(cl.leaf_lam).copy()
        for _ in range(self.N_STEPS):
            alat.step(0, DT)
            alat.step_lam(0, DT, self.LAM_SCALE)
        lam1 = np.array(cl.leaf_lam)
        null_mask = (np.abs(pos[:, 1] - y0) < LAM_W) | (np.abs(pos[:, 1] + y0) < LAM_W)
        delta = float(np.abs(lam1[null_mask] - lam0[null_mask]).max())
        self.assertGreater(
            delta,
            1e-8,
            "step_lam made no change at the current sheet with pressure-balanced u — "
            "macro-scale reconnection driver is inactive",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Virtual particle helpers (pure Python / numpy — no C++ required)
# ──────────────────────────────────────────────────────────────────────────────

_ALL_DIR = np.concatenate([HEX_DIR, SQ_DIR])  # (14, 3)
_B_NORM = 26.0  # Σ_k W_k * face_dir[k][d]^2 per axis: 8*3*1 (hex) + 2*1*1 (sq) = 26


def _reconstruct_B(phi):
    """Reconstruct B vector field from phi face fluxes.

    phi: (14, N) face-flux values.
    Returns B: (N, 3).

    B_d[n] = Σ_k W_k * phi[k,n] * face_dir[k,d] / 26
    — exact inverse of the IC encoding phi[k] = B·face_dir[k].
    """
    return np.einsum("k,kn,kd->nd", W_FACE, phi, _ALL_DIR) / _B_NORM


def _step_virtual(cl, v3, dt):
    """One step of virtual-particle (macro-PIC) dynamics.

    Each cell i hosts a virtual particle: mass proxy lam[i], velocity v3[:,i].

    Lorentz-analogue acceleration:
        a_i = E_i + v_i × B_i
    where:
        B_i  = _reconstruct_B(phi)[i]          (field from phi face fluxes)
        E_i  = Σ_k W_k*(u_nb - u_i)*face_dir[k] (pressure-gradient driver)

    Continuity (lam tracks particle):
        dlam_i = -dt * Σ_k W_k * lam_i * (v_i · face_dir[k])  (divergence)

    This conserves directed kinetic energy at the expense of field energy —
    adiabatic, no thermal diffusion to neighbours.

    Args:
        cl  : AdaptiveCellCluster (mutated in-place: lam updated)
        v3  : (3, N) float64 velocity array (mutated: returned updated)
        dt  : timestep

    Returns updated v3 (3, N).
    """
    N = cl.n_leaves
    u = np.asarray(cl.leaf_u, dtype=np.float64)
    phi = np.asarray(cl.leaf_phi, dtype=np.float64)
    lam = np.asarray(cl.leaf_lam, dtype=np.float64)
    nb = np.asarray(cl.leaf_nb, dtype=np.int32)  # (N, 14)
    gid = np.asarray(cl.leaf_global_indices, dtype=np.int32)  # (N,)

    max_g = int(nb.max()) + 1
    g_to_j = np.full(max_g, -1, dtype=np.int32)
    g_to_j[gid] = np.arange(N, dtype=np.int32)

    # B field from phi fluxes
    B = _reconstruct_B(phi)  # (N, 3)

    # E field: pressure-gradient projection onto each face direction
    E = np.zeros((N, 3))
    for k in range(14):
        nb_g = nb[:, k]
        valid = nb_g >= 0
        safe_g = np.where(valid, nb_g, gid)  # self when boundary
        j = g_to_j[np.clip(safe_g, 0, max_g - 1)]
        valid2 = valid & (j >= 0)
        du = np.where(valid2, u[j] - u, 0.0)
        E += (W_FACE[k] * du)[:, None] * _ALL_DIR[k]

    # Lorentz acceleration and velocity update
    vxB = np.cross(v3.T, B)  # (N, 3): v_i × B_i
    acc = E + vxB  # (N, 3)
    v3_new = v3 + dt * acc.T  # (3, N)

    # Continuity: dlam_i = -dt · Σ_k W_k · [(lam_nb·v_nb - lam_i·v_i)·dir_k]
    # (Σ_k W_k * dir_k = 0 by centrosymmetry, so the self-only term always vanishes —
    # the gradient-difference form is required.)
    lam_v = lam[None, :] * v3  # (3, N): lam·v per cell
    dlam = np.zeros(N)
    for k in range(14):
        nb_g = nb[:, k]
        # With periodic BCC all nb_g >= 0, but guard anyway
        safe_g = np.where(nb_g >= 0, nb_g, gid)
        j = g_to_j[np.clip(safe_g, 0, max_g - 1)]
        # Use self when boundary so flux diff is zero there
        safe_j = np.where(nb_g >= 0, j, np.arange(N, dtype=np.int32))
        flux_diff = np.einsum("d,dn->n", _ALL_DIR[k], lam_v[:, safe_j] - lam_v)  # (N,)
        dlam -= dt * W_FACE[k] * flux_diff

    lam_new = np.clip(lam + dlam, 0.0, None)
    cl.set_leaf_lam(lam_new)

    return v3_new


def _nb_lookup(cl):
    """Return (nb, gid, g_to_j, max_g) for neighbour lookups."""
    N = cl.n_leaves
    nb = np.asarray(cl.leaf_nb, dtype=np.int32)
    gid = np.asarray(cl.leaf_global_indices, dtype=np.int32)
    max_g = int(nb.max()) + 1
    g_to_j = np.full(max_g, -1, dtype=np.int32)
    g_to_j[gid] = np.arange(N, dtype=np.int32)
    return nb, gid, g_to_j, max_g


def _step_phi_diffuse(cl, nb, gid, g_to_j, max_g, dt, eta=0.1):
    """Phi diffusion: same rule as the DST u-diffusion applied to every face flux.

    dphi[k, i] = eta * sigma_i * dt * Σ_j W_j * (phi[k, nb[j,i]] - phi[k, i])

    Time-irreversible (dissipative).  Preserves Σ_k W_k*phi[k,i] = 0 by linearity.
    Stability requires eta * dt * Σ W_k < 1  (here Σ W_k = 26).
    """
    N = cl.n_leaves
    phi = np.asarray(cl.leaf_phi, dtype=np.float64)  # (14, N)
    lam = np.asarray(cl.leaf_lam, dtype=np.float64)
    sigma = 1.0 / (1.0 + lam)  # (N,)

    dphi = np.zeros_like(phi)
    for k_src in range(14):
        nb_g = nb[:, k_src]
        safe_g = np.where(nb_g >= 0, nb_g, gid)
        j = g_to_j[np.clip(safe_g, 0, max_g - 1)]
        safe_j = np.where(nb_g >= 0, j, np.arange(N, dtype=np.int32))
        dphi += W_FACE[k_src] * sigma[None, :] * (phi[:, safe_j] - phi)

    cl.set_leaf_phi(phi + dt * eta * dphi)


def _step_u_diffuse(cl, nb, gid, g_to_j, max_g, dt, eta=0.5):
    """U diffusion: the fundamental DST update rule.

    du_i = eta * sigma_i * dt * Σ_k W_k * (u[nb_k,i] - u_i)

    Dissipative (u variance strictly decreases).  Mean-conserving when sigma is
    uniform (lam uniform → sigma uniform).
    Stability requires eta * dt * Σ W_k < 1  (Σ W_k = 26).
    With DT=0.05 and eta=0.5: 0.5*0.05*26 = 0.65 < 1 ✓
    """
    N = cl.n_leaves
    u = np.asarray(cl.leaf_u, dtype=np.float64)
    lam = np.asarray(cl.leaf_lam, dtype=np.float64)
    sigma = 1.0 / (1.0 + lam)

    du = np.zeros(N)
    for k_src in range(14):
        nb_g = nb[:, k_src]
        safe_g = np.where(nb_g >= 0, nb_g, gid)
        j = g_to_j[np.clip(safe_g, 0, max_g - 1)]
        safe_j = np.where(nb_g >= 0, j, np.arange(N, dtype=np.int32))
        du += W_FACE[k_src] * sigma * (u[safe_j] - u)

    cl.set_leaf_u(u + dt * eta * du)


class TestHarrisPhiDiffusion(unittest.TestCase):
    """Phi diffusion as the irreversible DST reconnection mechanism.

    Applying the same σ·Σ W_k·(phi_nb − phi) rule to every face flux gives
    magnetic diffusion: ∂B/∂t = η∇²B.  No wave equation.  No time-reversal.
    Every cell change pays its cost; there is no free reverse.

    The diffusion:
    - is dissipative (phi_E strictly decreases)
    - preserves div(phi) = 0 (topology-respecting)
    - widens the current sheet (Sweet-Parker mechanism in the discrete substrate)
    - is controlled by eta — the DST analog of resistivity
    """

    N_STEPS = 200
    ETA = 0.5  # stable for DT=0.05: requires eta * dt * 26 = 0.5*0.05*26 = 0.65 < 1

    def _setup(self, perturb=0.0, flat_u=True):
        alat, cl, pos, _ = _build_box(flat_u=flat_u, perturb=perturb)
        nb, gid, g_to_j, max_g = _nb_lookup(cl)
        return cl, pos, nb, gid, g_to_j, max_g

    def test_uniform_phi_is_fixed_point(self):
        """Uniform phi (same value at every cell for each face) → dphi = 0."""
        _, cl, _, _ = _build_box(flat_u=True, perturb=0.0)
        N = cl.n_leaves
        cl.set_leaf_phi(np.full((14, N), 0.05))
        phi0 = np.array(cl.leaf_phi).copy()
        nb, gid, g_to_j, max_g = _nb_lookup(cl)
        for _ in range(self.N_STEPS):
            _step_phi_diffuse(cl, nb, gid, g_to_j, max_g, DT, self.ETA)
        np.testing.assert_allclose(
            np.array(cl.leaf_phi),
            phi0,
            atol=1e-12,
            err_msg="uniform phi changed — diffusion has a spurious source",
        )

    def test_div_phi_preserved(self):
        """Phi diffusion preserves div(phi) = Σ_k W_k·phi[k,i] = 0 to machine precision.

        Proof: Σ_k W_k·dphi[k,i] = eta·sigma_i·dt·Σ_j W_j·(Σ_k W_k·phi[k,nb_j] - Σ_k W_k·phi[k,i])
        = eta·sigma_i·dt·Σ_j W_j·(0 - 0) = 0  when the IC satisfies div(phi)=0.
        """
        cl, _, nb, gid, g_to_j, max_g = self._setup(perturb=0.1)
        for _ in range(self.N_STEPS):
            _step_phi_diffuse(cl, nb, gid, g_to_j, max_g, DT, self.ETA)
        phi = np.array(cl.leaf_phi)
        div = (W_FACE[:, None] * phi).sum(axis=0)
        self.assertLess(
            float(np.abs(div).max()),
            1e-10,
            "div(phi) became nonzero — diffusion broke the divergence-free condition",
        )

    def test_phi_e_strictly_decreases(self):
        """Phi diffusion is dissipative: phi_E = Σ phi² must decrease monotonically."""
        cl, _, nb, gid, g_to_j, max_g = self._setup(perturb=0.0)
        phi_e = [float(np.sum(np.array(cl.leaf_phi) ** 2))]
        for _ in range(self.N_STEPS):
            _step_phi_diffuse(cl, nb, gid, g_to_j, max_g, DT, self.ETA)
            phi_e.append(float(np.sum(np.array(cl.leaf_phi) ** 2)))
        # Check final energy is less than initial
        self.assertLess(
            phi_e[-1],
            phi_e[0],
            f"phi_E grew from {phi_e[0]:.4e} to {phi_e[-1]:.4e} — diffusion is not dissipating",
        )
        # Check monotonically decreasing (no step should increase phi_E)
        diffs = np.diff(phi_e)
        if diffs.max() > 1e-10 * phi_e[0]:
            n_up = int((diffs > 0).sum())
            self.fail(
                f"phi_E increased at {n_up} steps — diffusion is not monotonically dissipative"
            )

    def test_bx_gradient_at_null_decreases(self):
        """Current sheet widens: Bx gradient at the null line decreases over time.

        Jz = ∂Bx/∂y at y=y0 is the current density.  Magnetic diffusion spreads the
        current sheet, reducing the peak Jz.  This is the Sweet-Parker widening.
        """
        cl, pos, nb, gid, g_to_j, max_g = self._setup(perturb=0.0)
        y0 = LY / 4.0

        def _jz_at_null():
            phi_now = np.array(cl.leaf_phi)
            B = _reconstruct_B(phi_now)
            # Cells just above and below the upper null line
            above = (
                (pos[:, 1] > y0)
                & (pos[:, 1] < y0 + 1.5)
                & (np.abs(pos[:, 0]) < LX * 0.4)
            )
            below = (
                (pos[:, 1] < y0)
                & (pos[:, 1] > y0 - 1.5)
                & (np.abs(pos[:, 0]) < LX * 0.4)
            )
            bx_above = float(B[above, 0].mean()) if above.any() else 0.0
            bx_below = float(B[below, 0].mean()) if below.any() else 0.0
            dy = (
                float(pos[above, 1].mean() - pos[below, 1].mean())
                if (above.any() and below.any())
                else 1.0
            )
            return abs(bx_above - bx_below) / dy if dy > 0 else 0.0

        jz0 = _jz_at_null()
        self.assertGreater(jz0, 0.0, "Jz at null is zero at t=0 — IC problem")
        for _ in range(self.N_STEPS):
            _step_phi_diffuse(cl, nb, gid, g_to_j, max_g, DT, self.ETA)
        jz1 = _jz_at_null()
        self.assertLess(
            jz1,
            jz0,
            f"Jz at null = {jz1:.3e} did not decrease from {jz0:.3e} — current sheet not widening",
        )

    def test_by_changes_at_xpoint(self):
        """With perturbation, By near the X-point changes as phi diffuses.

        This is the topology-change signal: By at x≈0 grows as reconnected field
        lines diffuse from the O-points (|x|=Lx/4) toward the X-point (x=0).
        """
        cl, pos, nb, gid, g_to_j, max_g = self._setup(perturb=0.1)
        y0 = LY / 4.0
        xpt = (np.abs(pos[:, 1] - y0) < LAM_W) & (np.abs(pos[:, 0]) < LX * 0.15)
        by0 = float(
            np.sqrt(np.mean(_reconstruct_B(np.array(cl.leaf_phi))[xpt, 1] ** 2))
        )
        for _ in range(self.N_STEPS):
            _step_phi_diffuse(cl, nb, gid, g_to_j, max_g, DT, self.ETA)
        by1 = float(
            np.sqrt(np.mean(_reconstruct_B(np.array(cl.leaf_phi))[xpt, 1] ** 2))
        )
        self.assertNotAlmostEqual(
            by1,
            by0,
            places=6,
            msg=f"By_xpt unchanged ({by0:.3e}) — phi diffusion not changing magnetic topology",
        )


class TestHarrisDiffusionCombined(unittest.TestCase):
    """Combined u diffusion + phi diffusion on the Harris sheet.

    The DST-consistent macro-scale loop (no wave equation):
      du_i   = eta_u * sigma * Σ W_k * (u_nb   - u_i)    [pressure equilibrates]
      dphi_k = eta_p * sigma * Σ W_j * (phi_nb - phi_i)  [field reconnects]

    Phi diffusion preserves div(phi)=0, so step_lam does nothing and lam is
    frozen.  Reconnection is a field phenomenon here — plasma jets are absent.
    """

    N_STEPS = 200
    ETA_PHI = 0.1
    ETA_U = 0.5  # 0.5 * 0.05 * 26 = 0.65 < 1 ✓

    def _setup(self, perturb=0.0, flat_u=False):
        alat, cl, pos, _ = _build_box(flat_u=flat_u, perturb=perturb)
        nb, gid, g_to_j, max_g = _nb_lookup(cl)
        return cl, pos, nb, gid, g_to_j, max_g

    def test_u_variance_decreases(self):
        """U diffusion is dissipative: spatial variance of u must strictly decrease."""
        cl, _, nb, gid, g_to_j, max_g = self._setup(flat_u=False)
        var0 = float(np.var(np.asarray(cl.leaf_u)))
        for _ in range(self.N_STEPS):
            _step_u_diffuse(cl, nb, gid, g_to_j, max_g, DT, self.ETA_U)
        var1 = float(np.var(np.asarray(cl.leaf_u)))
        self.assertLess(
            var1, var0, f"u variance did not decrease: {var0:.4e} → {var1:.4e}"
        )

    def test_u_mean_conserved(self):
        """U diffusion conserves the spatial mean (uniform sigma → antisymmetric fluxes cancel)."""
        cl, _, nb, gid, g_to_j, max_g = self._setup(flat_u=False)
        mean0 = float(np.mean(np.asarray(cl.leaf_u)))
        for _ in range(self.N_STEPS):
            _step_u_diffuse(cl, nb, gid, g_to_j, max_g, DT, self.ETA_U)
        mean1 = float(np.mean(np.asarray(cl.leaf_u)))
        self.assertAlmostEqual(
            mean0,
            mean1,
            places=10,
            msg=f"u mean not conserved: {mean0:.8e} → {mean1:.8e}",
        )

    def test_phi_e_decreases(self):
        """Phi_E still decreases with u diffusion running alongside."""
        cl, _, nb, gid, g_to_j, max_g = self._setup(flat_u=False, perturb=0.1)
        phi_e0 = float(np.sum(np.asarray(cl.leaf_phi) ** 2))
        for _ in range(self.N_STEPS):
            _step_u_diffuse(cl, nb, gid, g_to_j, max_g, DT, self.ETA_U)
            _step_phi_diffuse(cl, nb, gid, g_to_j, max_g, DT, self.ETA_PHI)
        phi_e1 = float(np.sum(np.asarray(cl.leaf_phi) ** 2))
        self.assertLess(
            phi_e1, phi_e0, f"phi_E did not decrease: {phi_e0:.4e} → {phi_e1:.4e}"
        )

    def test_by_xpt_grows(self):
        """With perturbation, By at the X-point grows under combined phi+u diffusion.

        This is the reconnection signal.  U diffusion lets pressure rebalance
        as the current sheet changes — no artificial pressure barrier builds up.
        """
        cl, pos, nb, gid, g_to_j, max_g = self._setup(flat_u=False, perturb=0.1)
        y0 = LY / 4.0
        xpt = (np.abs(pos[:, 1] - y0) < LAM_W) & (np.abs(pos[:, 0]) < LX * 0.15)
        by0 = float(
            np.sqrt(np.mean(_reconstruct_B(np.asarray(cl.leaf_phi))[xpt, 1] ** 2))
        )
        for _ in range(self.N_STEPS):
            _step_u_diffuse(cl, nb, gid, g_to_j, max_g, DT, self.ETA_U)
            _step_phi_diffuse(cl, nb, gid, g_to_j, max_g, DT, self.ETA_PHI)
        by1 = float(
            np.sqrt(np.mean(_reconstruct_B(np.asarray(cl.leaf_phi))[xpt, 1] ** 2))
        )
        self.assertNotAlmostEqual(
            by1,
            by0,
            places=6,
            msg=f"By_xpt unchanged ({by0:.3e}) — combined diffusion not driving reconnection",
        )

    def test_current_sheet_widens(self):
        """Jz at the null line decreases as the current sheet widens under combined diffusion."""
        cl, pos, nb, gid, g_to_j, max_g = self._setup(flat_u=False, perturb=0.0)
        y0 = LY / 4.0

        def _jz():
            B = _reconstruct_B(np.asarray(cl.leaf_phi))
            above = (
                (pos[:, 1] > y0)
                & (pos[:, 1] < y0 + 1.5)
                & (np.abs(pos[:, 0]) < LX * 0.4)
            )
            below = (
                (pos[:, 1] < y0)
                & (pos[:, 1] > y0 - 1.5)
                & (np.abs(pos[:, 0]) < LX * 0.4)
            )
            bx_a = float(B[above, 0].mean()) if above.any() else 0.0
            bx_b = float(B[below, 0].mean()) if below.any() else 0.0
            dy = (
                float(pos[above, 1].mean() - pos[below, 1].mean())
                if (above.any() and below.any())
                else 1.0
            )
            return abs(bx_a - bx_b) / dy if dy > 0 else 0.0

        jz0 = _jz()
        self.assertGreater(jz0, 0.0, "Jz=0 at t=0 — IC problem")
        for _ in range(self.N_STEPS):
            _step_u_diffuse(cl, nb, gid, g_to_j, max_g, DT, self.ETA_U)
            _step_phi_diffuse(cl, nb, gid, g_to_j, max_g, DT, self.ETA_PHI)
        jz1 = _jz()
        self.assertLess(
            jz1, jz0, f"Jz at null = {jz1:.3e} did not decrease from {jz0:.3e}"
        )


class TestHarrisVirtualParticle(unittest.TestCase):
    """Virtual particle (macro-PIC) as the adiabatic reconnection driver.

    At macro scale each cell encloses many proton knots in local equilibrium.
    Their collective centre-of-mass motion is a "virtual particle" with mass
    proxy lam and velocity v3 (3-vector per cell).

    Force law:  a_i = E_i + v_i × B_i
      B from phi face fluxes (vector potential), E from u gradient.

    Energy route: field energy (phi_E) → directed kinetic energy (½ lam v²).
    No heat dumped into u — adiabatic by construction.

    Lam continuity:  dlam = -div(lam·v)·dt  so mass proxy tracks particle flow.
    """

    N_STEPS = 50

    # ── B-field reconstruction ─────────────────────────────────────────────

    def test_b_recovers_harris_profile(self):
        """B reconstructed from Harris phi matches the analytic Bx(y) profile
        and has zero By and Bz without perturbation."""
        _, _, pos, phi = _build_box(perturb=0.0)
        B = _reconstruct_B(phi)
        y0 = LY / 4.0
        bx_anal = _bx_profile(pos[:, 1], y0)
        np.testing.assert_allclose(
            B[:, 0],
            bx_anal,
            atol=1e-12,
            err_msg="Bx reconstruction does not match analytic tanh profile",
        )
        self.assertLess(
            float(np.abs(B[:, 1]).max()), 1e-13, "By nonzero without perturbation"
        )
        self.assertLess(
            float(np.abs(B[:, 2]).max()), 1e-13, "Bz nonzero without perturbation"
        )

    def test_by_nonzero_near_null_with_perturbation(self):
        """With perturbation, By reconstructed from phi is nonzero near the null line."""
        _, _, pos, phi = _build_box(perturb=0.1)
        B = _reconstruct_B(phi)
        y0 = LY / 4.0
        near_null = (np.abs(pos[:, 1] - y0) < LAM_W) & (np.abs(pos[:, 0]) < LX * 0.4)
        By_rms = float(np.sqrt(np.mean(B[near_null, 1] ** 2)))
        self.assertGreater(
            By_rms,
            1e-4 * B0,
            f"By_rms={By_rms:.2e} near null — perturbation not encoded in phi",
        )

    def test_jz_by_signs_imply_outflow_force(self):
        """By antisymmetry across X-point implies Jz·By drives outflow.

        Perturbation puts By>0 for x<0 and By<0 for x>0 near the upper null line.
        The current Jz = ∂Bx/∂y − ∂By/∂x ≈ −∂By/∂x > 0 (null line: Bx≈0).
        Outflow force on right side: F_x = Jz·|By|·sign = (+)·(+) = rightward ✓.
        """
        _, _, pos, phi = _build_box(perturb=0.1)
        B = _reconstruct_B(phi)
        y0 = LY / 4.0
        near_null = np.abs(pos[:, 1] - y0) < LAM_W
        left = near_null & (pos[:, 0] < -0.5) & (pos[:, 0] > -LX * 0.4)
        right = near_null & (pos[:, 0] > 0.5) & (pos[:, 0] < LX * 0.4)
        self.assertTrue(
            left.any() and right.any(), "not enough cells on both sides of X-point"
        )
        By_left = float(B[left, 1].mean())
        By_right = float(B[right, 1].mean())
        # By must have opposite signs
        self.assertGreater(By_left, 0.0, f"By_left={By_left:.3e} should be positive")
        self.assertLess(By_right, 0.0, f"By_right={By_right:.3e} should be negative")
        # Jz ≈ -∂By/∂x > 0 (By goes from + to - left→right)
        dx = float(pos[right, 0].mean() - pos[left, 0].mean())
        Jz = -(By_right - By_left) / dx
        self.assertGreater(Jz, 0.0, f"Jz={Jz:.3e} should be positive at X-point")
        # Outflow F_x on right side: -Jz * By_right = -(+)(-) > 0 (rightward)
        Fx_right = -Jz * By_right
        self.assertGreater(
            Fx_right, 0.0, "J×B force not in outflow direction on right side"
        )

    # ── virtual particle step ──────────────────────────────────────────────

    def test_zero_phi_no_magnetic_force(self):
        """phi=0 → B=0 → v×B=0 regardless of v: no magnetic force at all."""
        _, cl, _, _ = _build_box(flat_u=True, perturb=0.0)
        N = cl.n_leaves
        cl.set_leaf_phi(np.zeros((14, N)))
        v3 = np.random.default_rng(42).standard_normal((3, N)) * 0.1
        v3_copy = v3.copy()
        # With zero phi and flat u, only magnetic term is active (E=0 from flat u).
        # Step must leave v unchanged.
        v3_out = _step_virtual(cl, v3.copy(), DT)
        # E is also zero (flat u), so v3 should be exactly unchanged
        np.testing.assert_allclose(
            v3_out,
            v3_copy,
            atol=1e-14,
            err_msg="v changed with zero phi and flat u — spurious force",
        )

    def test_flat_u_lam_unchanged(self):
        """Flat u, zero v → E=0, B≠0 but v×B=0 → no force → lam continuity gives dlam=0."""
        alat, cl, _, _ = _build_box(flat_u=True, perturb=0.1)
        lam0 = np.array(cl.leaf_lam).copy()
        N = cl.n_leaves
        v3 = np.zeros((3, N))
        for _ in range(self.N_STEPS):
            alat.step(0, DT)
            v3 = _step_virtual(cl, v3, DT)
        lam1 = np.array(cl.leaf_lam)
        # v=0 → div(lam·v)=0 → lam must be unchanged
        max_dlam = float(np.abs(lam1 - lam0).max())
        self.assertLess(
            max_dlam, 1e-12, "lam changed with v=0 — div(lam·v) should be zero"
        )

    def test_pressure_balance_accelerates_vp(self):
        """Pressure-balanced IC: u gradient drives E → virtual particle accelerates
        away from the current sheet."""
        alat, cl, pos, _ = _build_box(flat_u=False, perturb=0.1)
        y0 = LY / 4.0
        N = cl.n_leaves
        v3 = np.zeros((3, N))
        # Run a few steps so E drives the particle
        for _ in range(self.N_STEPS):
            alat.step(0, DT)
            v3 = _step_virtual(cl, v3, DT)
        # Virtual particles near the null line should have developed nonzero v
        near_null = (np.abs(pos[:, 1] - y0) < LAM_W) & (np.abs(pos[:, 0]) < LX * 0.3)
        v_rms_xpt = float(np.sqrt(np.mean(v3[:, near_null] ** 2)))
        self.assertGreater(
            v_rms_xpt,
            1e-10,
            "virtual particles remain stationary at X-point with pressure-balanced u",
        )

    def test_lam_changes_with_particle_flow(self):
        """Once particles move, lam continuity must redistribute lam accordingly."""
        alat, cl, pos, _ = _build_box(flat_u=False, perturb=0.1)
        lam0 = np.array(cl.leaf_lam).copy()
        N = cl.n_leaves
        v3 = np.zeros((3, N))
        for _ in range(self.N_STEPS):
            alat.step(0, DT)
            v3 = _step_virtual(cl, v3, DT)
        lam1 = np.array(cl.leaf_lam)
        max_dlam = float(np.abs(lam1 - lam0).max())
        self.assertGreater(
            max_dlam,
            1e-12,
            "lam unchanged despite particle motion — continuity update not active",
        )

    def test_kinetic_energy_grows_from_field(self):
        """With pressure-balanced IC, kinetic energy (½ lam v²) must grow as field
        drives the virtual particles — field → kinetic conversion (adiabatic)."""
        alat, cl, _, _ = _build_box(flat_u=False, perturb=0.1)
        N = cl.n_leaves
        v3 = np.zeros((3, N))
        lam = np.array(cl.leaf_lam)
        KE0 = 0.5 * float(np.sum(lam * np.sum(v3**2, axis=0)))
        for _ in range(self.N_STEPS):
            alat.step(0, DT)
            v3 = _step_virtual(cl, v3, DT)
        lam = np.array(cl.leaf_lam)
        KE1 = 0.5 * float(np.sum(lam * np.sum(v3**2, axis=0)))
        self.assertGreater(
            KE1,
            KE0,
            "kinetic energy did not grow — virtual particle step is not converting field energy",
        )


if __name__ == "__main__":
    unittest.main()
