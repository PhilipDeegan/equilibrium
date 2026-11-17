"""
Macro vacuum plasma: does a resistance knot spontaneously produce a u moat?

DST hypothesis: lam encodes knot density (resistance). A high-lam region
should generate a u deficit (moat) in the surrounding area if σ=1/(1+lam)
is the right coupling.

Two routes tested:
  pure u diffusion  — parabolic, no phi
  wave equation     — alat.step(), phi active

If diffusion generates a moat: σ-coupling alone is sufficient.
If only the wave equation does: phi dynamics are necessary for the moat.
If neither does: the mechanism is still missing.
"""

import unittest

import numpy as np

from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
)

_HEX_DIR = np.array(
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
_SQ_DIR = np.array(
    [[-1, 0, 0], [+1, 0, 0], [0, -1, 0], [0, +1, 0], [0, 0, -1], [0, 0, +1]],
    dtype=np.float64,
)
W_FACE = np.array([3.0] * 8 + [1.0] * 6)

L_BOX = 8.0  # half-width — full box 16×16×16
LAM_KNOT = 10.0
U_BG = 1.0
U_KNOT = 2.0  # knot has elevated u (winding = stored energy)
KNOT_R = 1.5  # radius of knot in lattice units
DT = 0.01


def _build(u_knot=U_KNOT, lam_knot=LAM_KNOT):
    alat = _lat.AdaptiveLattice()
    cl = alat[0]
    _lat.build_bcc_box(cl, L_BOX, L_BOX, L_BOX, 1.0, u0=U_BG, lam0=0.0)
    alat.rebuild(0)
    N = cl.n_leaves
    pos = np.asarray(cl.leaf_pos, dtype=np.float64)
    r = np.linalg.norm(pos, axis=1)
    u = np.full(N, U_BG)
    lam = np.zeros(N)
    knot = r <= KNOT_R
    u[knot] = u_knot
    lam[knot] = lam_knot
    cl.set_leaf_u(u)
    cl.set_leaf_lam(lam)
    nb = np.asarray(cl.leaf_nb, dtype=np.int32)
    gid = np.asarray(cl.leaf_global_indices, dtype=np.int32)
    max_g = int(nb.max()) + 1
    g_to_j = np.full(max_g, -1, dtype=np.int32)
    g_to_j[gid] = np.arange(N, dtype=np.int32)
    return alat, cl, pos, r, nb, gid, g_to_j, max_g


def _diffuse_u(cl, nb, gid, g_to_j, max_g, dt):
    N = cl.n_leaves
    u = np.asarray(cl.leaf_u, dtype=np.float64)
    sigma = 1.0 / (1.0 + np.asarray(cl.leaf_lam, dtype=np.float64))
    du = np.zeros(N)
    for k in range(14):
        nb_g = nb[:, k]
        safe_g = np.where(nb_g >= 0, nb_g, gid)
        j = g_to_j[np.clip(safe_g, 0, max_g - 1)]
        safe_j = np.where(nb_g >= 0, j, np.arange(N, dtype=np.int32))
        du += W_FACE[k] * sigma * (u[safe_j] - u)
    cl.set_leaf_u(u + dt * du)


def _shell_mean(pos, u, r_lo, r_hi):
    r = np.linalg.norm(pos, axis=1)
    m = (r >= r_lo) & (r < r_hi)
    return float(u[m].mean()) if m.any() else float("nan")


class TestResistanceEffect(unittest.TestCase):
    """Baseline: σ-heterogeneity slows diffusion as expected."""

    def test_flat_u_unchanged(self):
        """Flat u with any lam: diffusion does nothing (Δu=0 everywhere)."""
        _, cl, _, _, nb, gid, g2j, mg = _build(u_knot=U_BG)  # flat u, high-lam knot
        u0 = np.asarray(cl.leaf_u, dtype=np.float64).copy()
        for _ in range(20):
            _diffuse_u(cl, nb, gid, g2j, mg, DT)
        u1 = np.asarray(cl.leaf_u, dtype=np.float64)
        np.testing.assert_allclose(u1, u0, atol=1e-12)

    def test_knot_retains_u(self):
        """High-lam knot loses u more slowly than lam=0 control."""
        _, cl_hi, _, r, nb, gid, g2j, mg = _build(lam_knot=LAM_KNOT)
        _, cl_lo, _, _, nb, gid, g2j, mg = _build(lam_knot=0.0)
        for _ in range(300):
            _diffuse_u(cl_hi, nb, gid, g2j, mg, DT)
            _diffuse_u(cl_lo, nb, gid, g2j, mg, DT)
        u_hi = np.asarray(cl_hi.leaf_u, dtype=np.float64)[r <= KNOT_R].mean()
        u_lo = np.asarray(cl_lo.leaf_u, dtype=np.float64)[r <= KNOT_R].mean()
        self.assertGreater(
            u_hi, u_lo, "high-lam knot should retain more u than lam=0 control"
        )

    def test_outer_ring_elevated_by_knot(self):
        """Shell just outside knot is pulled up early — knot is a u source."""
        _, cl, pos, _, nb, gid, g2j, mg = _build()
        u0 = _shell_mean(
            pos, np.asarray(cl.leaf_u, dtype=np.float64), KNOT_R + 0.5, KNOT_R + 3.0
        )
        for _ in range(50):
            _diffuse_u(cl, nb, gid, g2j, mg, DT)
        u1 = _shell_mean(
            pos, np.asarray(cl.leaf_u, dtype=np.float64), KNOT_R + 0.5, KNOT_R + 3.0
        )
        self.assertGreater(u1, u0)


class TestMoatEmergence(unittest.TestCase):
    """Does a u moat (deficit) emerge around the knot?"""

    def test_no_moat_from_diffusion(self):
        """Pure diffusion: outer shell stays at or above U_BG.

        The knot is a u source. Diffusion only moves u outward — no
        mechanism exists to pull the surrounding region below background.
        """
        _, cl, pos, r, nb, gid, g2j, mg = _build()
        for _ in range(200):
            _diffuse_u(cl, nb, gid, g2j, mg, DT)
        u = np.asarray(cl.leaf_u, dtype=np.float64)
        # Shell outside knot, well inside box boundary
        mask = (r >= KNOT_R + 1.0) & (r < L_BOX * 0.6)
        u_shell = u[mask]
        self.assertTrue(
            np.all(u_shell >= U_BG * 0.97),
            f"unexpected moat under pure diffusion: min={u_shell.min():.4f}",
        )

    def test_wave_equation_creates_deficit(self):
        """Wave equation radiates u outward; trailing phase dips below U_BG.

        In 3D, a spherical wave from a localised source has a trailing
        negative phase (Huygens). If u dips below U_BG in the outer shell,
        phi dynamics are necessary — and sufficient — for moat formation.
        """
        alat, cl, pos, r, *_ = _build()
        shell_mask = (r >= KNOT_R + 1.0) & (r < L_BOX * 0.6)
        deficit_seen = False
        for step in range(2000):
            alat.step(0, DT)
            if step % 50 == 0:
                u = np.asarray(cl.leaf_u, dtype=np.float64)
                if u[shell_mask].min() < U_BG * 0.99:
                    deficit_seen = True
                    break
        self.assertTrue(
            deficit_seen,
            "wave equation should create transient u deficit (trailing phase)",
        )


class TestSpinningKnot(unittest.TestCase):
    """Lam knot WITH phi ring winding (spinning knot) — does a stable moat emerge?

    A proper DST knot is not just a resistance lump — it has phi circulation
    (the winding IS the spinning).  The ring winding provides a centripetal
    trap that can hold a u deficit at a preferred radius.
    """

    R_SPHERE = 15.0
    AMP = 1.0
    N_STEPS = 2000
    DT = 0.01

    def _build_spinning_knot(self, lam_ring=LAM_KNOT):
        alat = _lat.AdaptiveLattice()
        cl = alat[0]
        _lat.build_bcc_sphere(cl, self.R_SPHERE, d1=1.0, u0=U_BG, lam0=0.0)
        alat.rebuild(0)

        ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
        self.assertIsNotNone(ring_js, "no toroidal ring found in sphere")

        init_toroidal_winding(
            cl, ring_js, ring_faces, amplitude=self.AMP, phi_only=True
        )

        if lam_ring > 0:
            lam = np.asarray(cl.leaf_lam, dtype=np.float64)
            for j in ring_js:
                lam[j] = lam_ring
            cl.set_leaf_lam(lam)

        pos = np.asarray(cl.leaf_pos, dtype=np.float64)
        r = np.linalg.norm(pos, axis=1)
        return alat, cl, pos, r, ring_js

    def test_winding_initialised(self):
        """Ring winding IC has non-zero phi after init."""
        _, cl, _, _, _ = self._build_spinning_knot()
        phi = np.asarray(cl.leaf_phi, dtype=np.float64)
        self.assertGreater(np.sqrt(np.mean(phi**2)), 0.0)

    def test_moat_emerges_from_phi_winding(self):
        """Phi winding alone (lam=0) produces a u deficit outside the ring.

        The wave equation's 3D trailing phase creates a u moat around the
        winding.  This is the DST mechanism: the phi ring is the knot,
        the deficit is the electron moat.
        """
        alat, cl, pos, r, ring_js = self._build_spinning_knot(lam_ring=0.0)
        ring_r = float(r[ring_js].mean())
        shell_mask = (r >= ring_r + 1.0) & (r < self.R_SPHERE * 0.6)

        deficit_seen = False
        for step in range(500):
            alat.step(0, self.DT)
            if step % 50 == 0:
                u = np.asarray(cl.leaf_u, dtype=np.float64)
                if u[shell_mask].min() < U_BG * 0.99:
                    deficit_seen = True
                    break

        self.assertTrue(
            deficit_seen,
            f"no u moat appeared from phi winding alone (ring_r={ring_r:.2f})",
        )

    def test_lam_on_ring_reduces_moat_depth(self):
        """Resistance on the ring source slows the outgoing wave → shallower moat.

        lam on the ring cells reduces σ at the source, attenuating the wave
        that creates the trailing deficit.  The pure winding (lam=0) produces
        a deeper moat than the resistive winding (lam>0).

        Implication: at macro scale, lam should NOT sit on the winding faces
        themselves — it should represent surrounding knot density.
        """
        alat_hi, cl_hi, pos, r, ring_js = self._build_spinning_knot(lam_ring=LAM_KNOT)
        alat_lo, cl_lo, _, _, _ = self._build_spinning_knot(lam_ring=0.0)
        ring_r = float(r[ring_js].mean())
        shell_mask = (r >= ring_r + 1.0) & (r < self.R_SPHERE * 0.6)

        for _ in range(300):
            alat_hi.step(0, self.DT)
            alat_lo.step(0, self.DT)

        u_hi = np.asarray(cl_hi.leaf_u, dtype=np.float64)[shell_mask].min()
        u_lo = np.asarray(cl_lo.leaf_u, dtype=np.float64)[shell_mask].min()
        self.assertLess(
            u_lo,
            u_hi,
            "pure winding (lam=0) should produce deeper deficit than resistive winding",
        )
