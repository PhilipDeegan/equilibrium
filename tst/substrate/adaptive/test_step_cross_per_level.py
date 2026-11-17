"""
Regression test for step_cross's cell-size dependence (found 2026-08-13 --
see `now` / project_cross_scale_calibration memory): production runs
`uniform_r180_active` (flat H/H0 over 570+ steps) vs `l1_wider_margin_active`
/ `pure_l1size_control` (steady ~25%+ decay over the same real time), at the
identical dt and cross_scale=1e-3, differing ONLY in cell size, showed
step_cross's dissipation is NOT level-independent as its own `cell.L` scaling
(adaptive_lattice.hpp:2475) was intended to make it.

Uses the same real dipole IC production runs use
(equilibrium.substrate.dipole_mode_ic), at a much smaller scale
(cells_across=22 vs 220) so this runs in a fraction of a second, with the
same uniform-vs-uniform construction `pure_l1size_control`/
`uniform_r180_active` use directly (build_bcc_sphere at d1=1 vs d1=1/3 --
no splitting, no coarse-fine junction, isolating the per-level RATE
question from any junction confound).

An earlier from-scratch C++ attempt at this same question
(tst/eq/core/data/lattice/test_step_cross_per_level.cpp) used a degenerate
uniform-phi-on-every-face synthetic IC and got a misleading answer (near-zero
dissipation at the finer level, opposite of the real production asymmetry) --
almost certainly a translation-symmetric cancellation artifact of that
synthetic field, not a real result. The real dipole IC used here has no such
symmetry.

RESOLVED 2026-08-15: `_step_cross_local`'s `cell.L^(+1)` factor is only
correct for a FLAT, shared L0-calibrated `dt` reused across a mixed-level AMR
mesh (its original, still-valid motivation). This test's/`pure_l1size_control`'s
`dt` is instead ALREADY matched to each mesh's own uniform cell size
(`_compute_dt_l0(cl, D1=cell_size)`), so it needs `cell.L^0` (no extra
compensation -- `dt` is already the cell's own locally-correct CFL value):
set via the new `AdaptiveLattice.set_step_cross_level_scale_power(0.0)`
diagnostic knob. Also fixed the comparison itself to be TIME-MATCHED (1 step
at L0's dt == 3 steps at L1's dt, since dt0 == 3*dt1) rather than comparing
one call at each resolution directly -- a resistive process's fractional
loss per call legitimately scales with that call's own dt, so single-call
equality was never the right invariant; equal loss over equal REAL elapsed
time is. Power=0 + time-matched together give ratio=0.999 (was 0.111 with
the old power=+1/single-call comparison).
"""

import unittest

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.dipole_mode_ic import seed_dipole_ic
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0
from tst.substrate.adaptive.adaptive_flux_tube import _FACE_HOP, _FACE_NORM

CELLS_ACROSS = 22
R_SPH = 1.0
U_MAX, AMPLITUDE = 1.0, 1.0
LAM_SCALE = (
    1.0  # matches build_l1_wider_margin_active.py / build_uniform_r180_active.py
)
R_DOMAIN = 18.0  # physical units -- matches production's padding/scale ratio at 1/10th the size


def _build(d1):
    scale = (CELLS_ACROSS / 2.0) * d1
    r_cells = (
        R_DOMAIN / d1
    )  # build_bcc_sphere's radius arg is in d1-units, not absolute

    alat = _lat.AdaptiveLattice()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, r_cells, d1=d1, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)

    pos = np.array(cl.leaf_pos)
    cell_size = np.full(cl.n_leaves, d1)
    seed_dipole_ic(
        cl,
        pos,
        cell_size,
        scale,
        _FACE_HOP,
        _FACE_NORM,
        r_sph=R_SPH,
        u_max=U_MAX,
        amplitude=AMPLITUDE,
    )

    # lam = lam_scale * (u^2 + 0.5*sum(phi^2)) -- both "everything active"
    # production builds seed this immediately after the dipole IC; sigma =
    # 1/(1+lam) feeds step_cross's si2 term, so skipping this is a real
    # divergence from the configs this test is meant to characterize.
    u = np.array(cl.leaf_u)
    phi = np.array(cl.leaf_phi)
    h = u**2 + 0.5 * np.sum(phi**2, axis=0)
    cl.set_leaf_lam(LAM_SCALE * h)
    return alat, cl


def _total_H(cl):
    u = np.array(cl.leaf_u)
    phi = np.array(cl.leaf_phi)
    return float(np.sum(u**2) + 0.5 * np.sum(phi**2))


class TestStepCrossPerLevel(unittest.TestCase):
    def tearDown(self):
        # step_cross_level_scale_power is a global static -- always restore
        # the default so other tests in the same process see the original
        # (flat-shared-L0-dt/AMR) behaviour regardless of how this test exits.
        _lat.AdaptiveLattice.set_step_cross_level_scale_power(1.0)

    def test_fractional_loss_level_independent(self):
        cross_scale = 1e-3

        alat0, cl0 = _build(d1=1.0)
        alat1, cl1 = _build(d1=1.0 / 3.0)

        # Matches production EXACTLY: each resolution gets its own CFL dt via
        # _compute_dt_l0(cl, D1=cell_size) -- pure_l1size_control's step
        # script passes D1=CELL_SIZE=1/3, NOT the same flat dt as
        # uniform_r180_active. A fixed dt for both (my first attempt at this
        # test) is not the real comparison basis and gave a misleading ratio.
        dt0 = _compute_dt_l0(cl0, D1=1.0)
        dt1 = _compute_dt_l0(cl1, D1=1.0 / 3.0)
        assert abs(dt0 / dt1 - 3.0) < 1e-9, "test assumes dt0 == 3*dt1"

        # _step_cross_local's cell.L factor (adaptive_lattice.hpp) was
        # designed for a FLAT, shared L0-calibrated dt reused across every
        # level of one AMR mesh (cell.L^(+1) converts that shared dt into an
        # effective per-cell dt). This test's dt is instead ALREADY matched
        # to each mesh's own uniform cell size (D1=cell_size above) --
        # exactly `pure_l1size_control`'s own convention -- so `dt` is
        # already the cell's own locally-correct CFL dt and needs NO further
        # cell.L compensation: power=0 (2026-08-15).
        _lat.AdaptiveLattice.set_step_cross_level_scale_power(0.0)

        H0_before = _total_H(cl0)
        H1_before = _total_H(cl1)
        self.assertGreater(H0_before, 0.0)
        self.assertGreater(H1_before, 0.0)

        # Time-matched comparison (2026-08-15, matching `pure_l1size_control`'s
        # own report.md methodology): dt0 == 3*dt1, so 1 step at L0 and 3
        # steps at L1 cover the SAME real elapsed time. Comparing single
        # calls directly (1 vs 1) is not the right invariant -- a resistive
        # process's fractional loss per call scales with that call's own dt,
        # so a 3x-smaller dt should legitimately produce a smaller per-call
        # fraction; the level-independent quantity is the loss accumulated
        # over EQUAL elapsed time, not per call.
        alat0.step_cross_all(dt0, cross_scale)
        for _ in range(3):
            alat1.step_cross_all(dt1, cross_scale)

        frac0 = (H0_before - _total_H(cl0)) / H0_before
        frac1 = (H1_before - _total_H(cl1)) / H1_before

        print(
            f"frac_L0(1 step)={frac0:.6e}  frac_L1(3 steps)={frac1:.6e}  ratio={frac1 / frac0:.4f}"
        )
        self.assertAlmostEqual(
            frac1 / frac0,
            1.0,
            delta=0.05,
            msg="fractional H loss from step_cross over EQUAL elapsed time (1 step @ dt0 == 3 steps @ dt1) "
            "must be level-independent (within 5%) for a fixed cross_scale",
        )


if __name__ == "__main__":
    unittest.main()
