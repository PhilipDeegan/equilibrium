"""
Quick uniform-lattice check of BC compositions for the cross_scale boundary-
buildup question.

tst/substrate/adaptive/test_reciprocal_bc.py::TestReciprocalWithRingDecay (C++
AdaptiveLattice backend) found that "reciprocal INSTEAD OF sponge" makes the
boundary buildup ~35x WORSE, not better — compensate() is pure global
u-bookkeeping with no local wave absorption, so removing the sponge removes
the only absorbing mechanism and outgoing waves reflect off the true domain
edge instead. See project_cross_scale_calibration memory for the full
writeup.

This checks the follow-up hypothesis — "sponge AND reciprocal, composed" —
on the plain-Python-callable SubstrateLattice/DampedLattice, which is much
faster to iterate on than the C++ AdaptiveLattice backend:

  - sponge alone        DampedLattice.step() — baseline, same onset-seam
                         artifact expected as the adaptive-lattice sweep
  - reciprocal alone     SubstrateLattice.step_cpp(), no sponge — expect to
                         reproduce the "worse" adaptive-lattice result here
  - sponge + reciprocal  DampedLattice.step() + ReciprocalBC.compensate()
                         each step — the untested composed hypothesis

ReciprocalBC.compensate() already restores u_total to its ORIGINAL value each
call regardless of what drained it (see equilibrium/substrate/reciprocal_bc.py) — so
"sponge + reciprocal" needs no new plumbing (no bc_flux() bookkeeping) beyond
calling both each step.

Run:
  PYTHONPATH=$PWD python3 -m unittest tst.substrate.test_sponge_reciprocal_bc -v
"""

import unittest

import numpy as np

from equilibrium.substrate.flux_tube import find_toroidal_spheromak, init_spheromak
from equilibrium.substrate.lattice import DampedLattice, SubstrateLattice, wave_speed
from equilibrium.substrate.reciprocal_bc import ReciprocalBC

R = 15.0
DT = 0.005
CROSS_SCALE = 0.05
LAM_SCALE = 0.05
N_STEPS = 4000
SPONGE_FRAC = 0.15
DAMP_MAX = 0.003


def _init_ring(lat, amplitude=1.0, lam_rings=1.0):
    A_tiles, A_faces, B_tiles, B_faces = find_toroidal_spheromak(lat, axis="z")
    if A_tiles is None:
        raise RuntimeError(f"no toroidal spheromak found at R={R} — too small?")
    init_spheromak(
        lat, A_tiles, A_faces, B_tiles, B_faces, amplitude=amplitude, phi_only=True
    )
    if lam_rings > 0.0:
        lat.lam[list(set(A_tiles) | set(B_tiles))] = lam_rings
    return A_tiles, B_tiles


def _boundary_phi_frac(lat, sponge_frac=SPONGE_FRAC):
    """Fraction of total phi-energy sitting in the outer sponge shell —
    same diagnostic as the adaptive-lattice test (phi is (N,14) here, not
    (14,N) as on the AdaptiveLattice leaf arrays)."""
    r = np.linalg.norm(lat.pos, axis=1)
    r_inner = float(r.max()) * (1.0 - sponge_frac)
    shell = r > r_inner
    return float(np.sum(lat.phi[shell] ** 2) / max(np.sum(lat.phi**2), 1e-300))


class TestBCCompositionForRingDecay(unittest.TestCase):
    """Fast (uniform-lattice) companion to
    tst/substrate/adaptive/test_reciprocal_bc.py::TestReciprocalWithRingDecay — same
    question, quicker to iterate: does composing the sponge with reciprocal
    reinjection (rather than reciprocal replacing the sponge) change the
    boundary buildup, and in which direction?
    """

    def _run(self, mode):
        if mode == "sponge":
            lat = DampedLattice(
                radius_cells=R, sponge_fraction=SPONGE_FRAC, damp_max=DAMP_MAX, u_bg=0.0
            )
            bc = None
        elif mode == "reciprocal":
            lat = SubstrateLattice(radius_cells=R)
            bc = ReciprocalBC(lat, rng=np.random.default_rng(0))
        elif mode == "sponge+reciprocal":
            lat = DampedLattice(
                radius_cells=R, sponge_fraction=SPONGE_FRAC, damp_max=DAMP_MAX, u_bg=0.0
            )
            bc = ReciprocalBC(lat, rng=np.random.default_rng(0))
        else:
            raise ValueError(mode)

        _init_ring(lat)

        for _ in range(N_STEPS):
            if mode == "reciprocal":
                lat.step_cpp(DT)
            else:
                lat.step(DT)
            if bc is not None:
                bc.compensate(lat)
            lat.step_cross(DT, CROSS_SCALE)
            lat.step_lam(DT, LAM_SCALE)

        return _boundary_phi_frac(lat)

    def test_bc_composition_comparison(self):
        bnd_sponge = self._run("sponge")
        bnd_recip = self._run("reciprocal")
        bnd_composed = self._run("sponge+reciprocal")
        print(
            f"\nboundary_phi_frac — sponge={bnd_sponge:.3e}  "
            f"reciprocal={bnd_recip:.3e}  sponge+reciprocal={bnd_composed:.3e}"
        )
        # Reproduces the adaptive-lattice finding: reciprocal alone (no
        # absorption at all) is worse than the sponge.
        self.assertGreater(
            bnd_recip,
            bnd_sponge,
            f"expected reciprocal-alone to reproduce the adaptive-lattice "
            f"result (worse than sponge): sponge={bnd_sponge:.3e} vs "
            f"reciprocal={bnd_recip:.3e}",
        )
        # Measured 2026-07-05: sponge+reciprocal (6.45e-6) ~= sponge alone
        # (6.07e-6), NOT closer to reciprocal-alone's catastrophic 2.52e-4.
        # ReciprocalBC.compensate() reinjects at the single outermost
        # missing-neighbour layer, restoring global u_total — it doesn't
        # touch the sponge's local damping *ramp* (zero at r_inner, rising
        # outward), which is the actual mechanism behind the onset-seam
        # buildup. So composing gets you real conservation (energy doesn't
        # vanish into nothing — the original "more universe out there"
        # motivation) essentially for free, but does NOT by itself fix the
        # onset-seam artifact — that still needs the damping-ramp redesign
        # (non-zero floor at r_inner / steeper or earlier ramp) discussed in
        # project_cross_scale_calibration memory. Generous 3x band (vs. the
        # reciprocal-alone case's ~40x blowup) to allow for RNG/step-count
        # noise while still catching a real regression either direction.
        self.assertLess(
            bnd_composed,
            bnd_sponge * 3.0,
            f"sponge+reciprocal unexpectedly far worse than sponge alone — "
            f"composing may be interacting with the onset seam after all: "
            f"sponge={bnd_sponge:.3e} vs composed={bnd_composed:.3e}",
        )


# ---------------------------------------------------------------------------
# Sponge damping-ramp shape — does killing the zero-at-onset artifact fix the
# local seam, without changing the interior decay physics?
# ---------------------------------------------------------------------------


def _make_damped_lattice(
    profile="quadratic", damp_max=DAMP_MAX, damp_min=0.0, sponge_frac=SPONGE_FRAC
):
    """DampedLattice with a swappable damping-ramp shape — prototyped by
    overwriting the plain-numpy _sponge array DampedLattice.__init__ builds,
    no C++ change needed to test the idea first.

    profile:
      "quadratic" — current behaviour, damp_max*t**2 (zero exactly at
                    r_inner — the mechanism behind the onset-seam buildup)
      "floor"     — damp_min + (damp_max-damp_min)*t**2 (kills the exact-zero
                    onset, keeps the rest of the quadratic shape)
      "linear"    — damp_max*t (rises immediately instead of flattening near
                    onset, no floor needed)
    t = (r-r_inner)/(r_max-r_inner), 0 for r<=r_inner.
    """
    lat = DampedLattice(
        radius_cells=R, sponge_fraction=sponge_frac, damp_max=damp_max, u_bg=0.0
    )
    r_all = np.linalg.norm(lat.pos, axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - sponge_frac)
    beyond = r_all > r_inner
    t = np.where(beyond, (r_all - r_inner) / (r_max - r_inner), 0.0)
    if profile == "quadratic":
        damp = damp_max * t**2
    elif profile == "floor":
        damp = np.where(beyond, damp_min + (damp_max - damp_min) * t**2, 0.0)
    elif profile == "linear":
        damp = damp_max * t
    else:
        raise ValueError(profile)
    lat._sponge = damp.astype(np.float64)
    return lat


def _seam_excess(
    lat, sponge_frac=SPONGE_FRAC, trend_band=(0.6, 0.95), seam_band=(0.97, 1.08)
):
    """Ratio of the actual energy-density right at r_inner to the smooth
    interior trend extrapolated (log-linear fit) from just inside it.

    ~1.0 means the profile passes through r_inner smoothly (no artifact);
    >>1.0 means a local bump (the onset-seam artifact) has formed there.
    Bands are fractions of r_inner, not r_max.
    """
    r = np.linalg.norm(lat.pos, axis=1)
    r_max = float(r.max())
    r_inner = r_max * (1.0 - sponge_frac)
    e_dens = lat.u**2 + 0.5 * np.sum(lat.phi**2, axis=1)

    trend_mask = (r > r_inner * trend_band[0]) & (r < r_inner * trend_band[1])
    seam_mask = (r > r_inner * seam_band[0]) & (r < r_inner * seam_band[1])
    if not np.any(trend_mask) or not np.any(seam_mask):
        return float("nan")

    r_t = r[trend_mask]
    e_t = np.maximum(e_dens[trend_mask], 1e-300)
    slope, intercept = np.polyfit(r_t, np.log(e_t), 1)
    predicted_at_inner = np.exp(intercept + slope * r_inner)

    seam_val = float(np.mean(e_dens[seam_mask]))
    return seam_val / max(predicted_at_inner, 1e-300)


class TestRateSpongeWidth(unittest.TestCase):
    """The production sponge (rate, one factor per face, gamma_max = 10 c / width) reflects less
    as it widens (exp/2026/09/29/bc_design: 5.6e-3 / 2.8e-3 / 1.0e-3 at 2.4 / 4.8 / 9.6 cells).
    Replaces TestSpongeRampShape / TestSpongeWidth, which tuned the old per-call, per-cell
    sponge's onset seam."""

    RI = 9.6

    def _reflection(self, width):
        R_tot = self.RI + width
        lat = DampedLattice(radius_cells=R_tot, sponge_fraction=width / R_tot)
        r = np.linalg.norm(lat.pos, axis=1)
        lat.u[:] = np.exp(-(r**2) / (2 * 1.5**2))
        m = r < self.RI
        e = lambda: 0.5 * np.sum(lat.u[m] ** 2) + 0.25 * np.sum((lat.phi[m] * (lat.nb[m] >= 0)) ** 2)
        e0 = e()
        c = wave_speed()
        t0, t1 = (self.RI + 4.5) / c, (self.RI + 4.5) / c + 2 * self.RI / c
        dt = lat.stable_dt()
        vals = []
        for s in range(int(np.ceil(t1 / dt))):
            lat.step(dt)
            if (s + 1) * dt >= t0 and s % 10 == 0:
                vals.append(e() / e0)
        return float(np.mean(vals))

    def test_wider_reflects_less(self):
        narrow, wide = self._reflection(2.4), self._reflection(9.6)
        self.assertLess(narrow, 1e-2)
        self.assertLess(wide, 0.3 * narrow, f"narrow {narrow:.2e} wide {wide:.2e}")




# ---------------------------------------------------------------------------
# Structured (spatially-correlated) reinjection — a de Broglie-motivated
# alternative to ReciprocalBC's pointwise-random reinjection. No DST-native
# hbar/k_B exists yet to derive a real thermal wavelength (see
# project_cross_scale_calibration memory), so kernel_width is swept as a free
# parameter in ring-radius units rather than computed from first principles.
# ---------------------------------------------------------------------------


class StructuredReciprocalBC(ReciprocalBC):
    """
    Same conservation contract as ReciprocalBC (compensate() restores
    u_total to its original value) but the reinjection weights are a
    spatially-correlated random field (Gaussian-kernel-smoothed noise, with
    correlation length kernel_width) instead of i.i.d. per boundary cell —
    a coherent "packet" landing on the boundary rather than pointwise shot
    noise, as a stand-in for a real de Broglie thermal wavelength once one
    exists for DST.

    kernel_width=0 (or None) reproduces plain ReciprocalBC exactly (smoothing
    with a zero-width kernel is the identity).
    """

    def __init__(self, lat, u_target=None, rng=None, kernel_width=1.5):
        super().__init__(lat, u_target=u_target, rng=rng)
        self._kernel_width = kernel_width
        if kernel_width and self.n_boundary > 0:
            pos_b = lat.pos[self._bc_idx]
            d2 = np.sum((pos_b[:, None, :] - pos_b[None, :, :]) ** 2, axis=-1)
            self._kernel = np.exp(-d2 / (2.0 * kernel_width**2))
        else:
            self._kernel = None

    def compensate(self, lat):
        deficit = self._u_target - float(lat.u.sum())
        if self.n_boundary == 0 or abs(deficit) < 1e-15:
            return
        raw = self._rng.exponential(size=self.n_boundary)
        if self._kernel is not None:
            smoothed = self._kernel @ raw
        else:
            smoothed = raw
        weights = smoothed / smoothed.sum()
        lat.u[self._bc_idx] += deficit * weights


class TestStructuredInjection(unittest.TestCase):
    """Does spatially-correlated reinjection (StructuredReciprocalBC) change
    the boundary-buildup picture vs. plain (uncorrelated) ReciprocalBC, when
    composed with the sponge? kernel_width swept in ring-radius units
    (ring_r≈1.5, so 0.5x/1x/3x roughly probes sub-ring/ring-scale/
    much-larger-than-ring correlation lengths).
    """

    KERNEL_WIDTHS = (0.75, 1.5, 4.5)

    def _run(self, kernel_width):
        lat = _make_damped_lattice(
            "quadratic", damp_max=DAMP_MAX, sponge_frac=SPONGE_FRAC
        )
        if kernel_width is None:
            bc = ReciprocalBC(lat, rng=np.random.default_rng(0))
        else:
            bc = StructuredReciprocalBC(
                lat, rng=np.random.default_rng(0), kernel_width=kernel_width
            )
        _init_ring(lat)
        for _ in range(N_STEPS):
            lat.step(DT)
            bc.compensate(lat)
            lat.step_cross(DT, CROSS_SCALE)
            lat.step_lam(DT, LAM_SCALE)
        seam = _seam_excess(lat)
        bnd = _boundary_phi_frac(lat)
        rms = float(np.sqrt(np.mean(lat.phi**2)))
        return seam, bnd, rms

    def test_structured_vs_unstructured_reinjection(self):
        seam_u, bnd_u, rms_u = self._run(None)
        print(
            f"\nunstructured        — seam_excess={seam_u:.2f}x  bnd={bnd_u:.3e}  phi_rms={rms_u:.3e}"
        )
        results = {}
        for kw in self.KERNEL_WIDTHS:
            seam, bnd, rms = self._run(kw)
            results[kw] = (seam, bnd, rms)
            print(
                f"kernel_width={kw:.2f}  — seam_excess={seam:.2f}x  "
                f"bnd={bnd:.3e}  phi_rms={rms:.3e}"
            )
        # Exploratory — report only, no directional assertion yet (we don't
        # have a prior for which way spatial correlation should push this).
        # Sanity floor: none of the structured variants should blow up the
        # way reciprocal-alone (no sponge at all) did (~40x) — that would
        # indicate a bug in the kernel construction, not a real effect.
        for kw, (seam, bnd, rms) in results.items():
            self.assertLess(
                seam,
                seam_u * 5.0,
                f"kernel_width={kw} seam_excess blew up vs unstructured "
                f"({seam_u:.2f}x) — check kernel construction: {seam:.2f}x",
            )


if __name__ == "__main__":
    unittest.main()
