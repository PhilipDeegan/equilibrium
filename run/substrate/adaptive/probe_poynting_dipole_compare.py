"""
Quantitative calibration check: does the barrier-gated step_cross Poynting
pattern (run/substrate/adaptive/probe_poynting_ring.py) match the classical
oscillating-dipole radiation law in the far zone?

Classical magnetic dipole (moment along the ring axis) time-averaged Poynting
flux in the radiation zone: <S> ~ sin^2(theta) / r^2, theta measured from the
dipole axis (so flux peaks at the equator theta=90, vanishes on-axis
theta=0). This script does NOT time-average (a single late-time snapshot,
well after transients, at a fixed oscillation phase) -- it checks the radial
FALLOFF EXPONENT of |S| in two angular sectors (near-equatorial, near-polar)
against the classical -2 power law, log-log. A real radiative far zone
should show a much shallower (less negative) log-log slope than whatever
locally-confined pattern the barrier gate alone would produce if it's simply
attenuating with distance from the source rather than genuinely radiating.

Usage:
    PYTHONPATH=$PWD python3 run/substrate/adaptive/probe_poynting_dipole_compare.py
    PYTHONPATH=$PWD python3 run/substrate/adaptive/probe_poynting_dipole_compare.py \\
        --R 50 --hops 4 --m0 1e-4 --n-steps 8000

Calibration result so far (R=50, hops=4, m0=1e-4, step=8000; see
doc/mems/project_poynting_dipole_calibration.md): polar p99-envelope slope
-2.44 (near classical -2), equatorial slope -6.19 (far steeper) -- opposite
of the naive magnetic-dipole expectation (radiation peaking at the equator).
Best current explanation is BCC dispersion anisotropy
(project_bcc_dispersion_anisotropy) plus a structural fact about this ring
IC: ring_r/period is a FIXED ~0.166 across hops=1..8 (both scale together
via omega=2*sin(pi/N_ring)), so the ring is never small compared to its own
wavelength (diameter ~0.33*lambda) regardless of hops -- not something
--hops can fix.

Next steps (not yet done):
  - Full angular sweep (many theta bins, not just polar/equatorial) to map
    the falloff exponent vs angle and test whether it correlates with BCC
    lattice-axis alignment the way the anisotropy hypothesis predicts.
  - Confirm at a second (R, hops) pair that polar stays ~-2 and equatorial
    stays much steeper -- rules out a ring-size-specific coincidence.
  - Try refining the central region (r < 2-3*ring_r) to L1 and rebuilding
    the ring there: tests whether the steep equatorial falloff is partly a
    discretization artifact from only 32 cells resolving the ring at
    uniform L=1. Caution: init_toroidal_winding's omega=2*sin(pi/N) is a
    topology-only formula (no L dependence) tuned for uniform L=1 cells --
    it will likely NOT describe the real oscillation frequency correctly on
    refined cells, so the resonant period would need to be re-measured
    empirically rather than assumed.
  - The real test of "is the dipole too small (relative to its own
    wavelength)" needs a source DECOUPLED from its own natural frequency:
    an externally-forced low-omega drive on the ring tiles (continuously
    reset each step, not a free-running normal-mode IC), independent of
    hops/ring size -- init_toroidal_winding can't do this, it locks omega to
    ring topology by construction.
  - R=100 run: queued, but revisit whether it's the right next step given
    the size/wavelength finding above -- larger R gives more absolute cells
    to look at, not more wavelengths, since lambda is fixed by hops alone.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.mhd import _SQ_AXIS_PAIRS, SQ_PLAQUETTES
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    leaf_nb_j,
    winding_number,
)


def _plaquette_vectorized(phi_T, nb_j, faces):
    """Vectorized eq. of mhd._plaquette_curl over ALL cells at once.
    faces = (ka, kb, kc, kd) as in SQ_PLAQUETTES. Returns (N,) curl values,
    0.0 wherever the 4-cycle doesn't close (boundary)."""
    N = phi_T.shape[0]
    c0 = np.arange(N)
    ka, kb, kc, kd = faces
    c1 = nb_j[c0, ka]
    valid = c1 >= 0
    c1s = np.where(valid, c1, 0)
    c2 = nb_j[c1s, kb]
    valid &= c2 >= 0
    c2s = np.where(c2 >= 0, c2, 0)
    c3 = nb_j[c2s, kc]
    valid &= c3 >= 0
    c3s = np.where(c3 >= 0, c3, 0)
    close = nb_j[c3s, kd]
    valid &= close == c0
    term = phi_T[c0, ka] + phi_T[c1s, kb] + phi_T[c2s, kc] + phi_T[c3s, kd]
    return np.where(valid, term, 0.0)


def b_vec_vectorized(phi_T, nb_j):
    """Vectorized eq. of mhd._b_vec over ALL cells at once. Returns (N, 3)."""
    Bx = _plaquette_vectorized(phi_T, nb_j, SQ_PLAQUETTES["x"])
    By = _plaquette_vectorized(phi_T, nb_j, SQ_PLAQUETTES["y"])
    Bz = _plaquette_vectorized(phi_T, nb_j, SQ_PLAQUETTES["z"])
    return np.stack([Bx, By, Bz], axis=1)


def face_vec_vectorized(dphi):
    """Vectorized eq. of face_vec over ALL cells. dphi: (N,14) -> (N,3)."""
    return np.stack([dphi[:, fp] - dphi[:, fm] for fp, fm in _SQ_AXIS_PAIRS], axis=1)


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--R", type=float, default=50.0)
parser.add_argument("--hops", type=int, default=4)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--lam-ring", type=float, default=1.0)
parser.add_argument("--cross-scale", type=float, default=0.01)
parser.add_argument("--m0", type=float, default=1e-4)
parser.add_argument("--axis", default="z")
parser.add_argument("--n-steps", type=int, default=8000)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--threads", type=int, default=10)
parser.add_argument(
    "--sector-half-angle",
    type=float,
    default=20.0,
    help="half-angle (degrees) of the equatorial/polar angular sectors used for the radial fit",
)
parser.add_argument("--r-min-fit", type=float, default=None, help="default: 1.5*ring_r")
parser.add_argument("--r-max-fit", type=float, default=None, help="default: 0.8*R")
parser.add_argument("--plot-out", type=str, default="poynting_dipole_compare.png")


def main():
    args = parser.parse_args()
    ax_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    perp = [i for i in range(3) if i != ax_idx]

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)

    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - args.sponge)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0, damp_max=args.damp_max, r_all=r_all, r_inner=r_inner
        )
    )

    ring_js, ring_faces = find_toroidal_ring(
        cl, axis=args.axis, mode="square", hops=args.hops
    )
    if ring_js is None:
        print("ERROR: no toroidal ring found")
        sys.exit(1)
    w = winding_number(cl, ring_js, axis=args.axis)
    print(f"winding={w:+.4f}")
    omega = init_toroidal_winding(
        cl, ring_js, ring_faces, amplitude=args.amplitude, phi_only=True
    )
    pos = np.array(cl.leaf_pos)
    ring_pos = pos[ring_js]
    ring_centre = ring_pos.mean(axis=0)
    ring_r = float(np.linalg.norm(ring_pos - ring_centre, axis=1).mean())

    lam = np.array(cl.leaf_lam)
    for j in ring_js:
        lam[j] = args.lam_ring
    cl.set_leaf_lam(lam)

    dt = compute_dt(cl)
    period = 2.0 * np.pi / omega if omega > 0 else None
    print(
        f"N={cl.n_leaves}  R={args.R}  hops={args.hops}  ring_r={ring_r:.3f}  "
        f"omega={omega:.4f}  period={period}  dt={dt:.4e}  m0={args.m0:.2e}",
        flush=True,
    )

    nb_j = leaf_nb_j(cl)
    r_from_centre = np.linalg.norm(pos - ring_centre, axis=1)

    r_min_fit = args.r_min_fit if args.r_min_fit is not None else 1.5 * ring_r
    r_max_fit = args.r_max_fit if args.r_max_fit is not None else 0.8 * args.R

    # Angular sectors relative to the ring axis, using ALL cells (not just a
    # thin planar slice) for better radial-bin statistics.
    rel = pos - ring_centre
    ax_comp = rel[:, ax_idx]
    perp_comp = np.linalg.norm(rel[:, perp], axis=1)
    theta = np.degrees(
        np.arctan2(perp_comp, np.abs(ax_comp))
    )  # 0=polar axis, 90=equator
    half = args.sector_half_angle
    equatorial = theta > (90.0 - half)
    polar = theta < half
    in_fit_r = (r_from_centre >= r_min_fit) & (r_from_centre <= r_max_fit)

    print(
        f"fit range r=[{r_min_fit:.2f},{r_max_fit:.2f}]  "
        f"equatorial cells in range={int((equatorial & in_fit_r).sum())}  "
        f"polar cells in range={int((polar & in_fit_r).sum())}",
        flush=True,
    )

    for step in range(args.n_steps):
        alat.step(0, dt)
        alat.step_sponge(0)
        if args.cross_scale > 0.0:
            alat.step_cross(0, dt, args.cross_scale, gate="barrier", m0=args.m0)
        if step % 1000 == 0:
            print(f"  step={step}", flush=True)

    print(
        f"reached step={args.n_steps}, computing E/B/S over full domain...", flush=True
    )

    phi_before = np.array(cl.leaf_phi).T.copy()  # (N, 14)
    alat.step_cross(0, dt, args.cross_scale, gate="barrier", m0=args.m0)
    phi_after = np.array(cl.leaf_phi).T
    dphi_cross = phi_after - phi_before

    E3 = -face_vec_vectorized(dphi_cross) / dt
    B3 = b_vec_vectorized(phi_after, nb_j)
    S3 = np.cross(E3, B3)
    Smag = np.linalg.norm(S3, axis=1)

    # -------------------------------------------------------------------
    # Radial profile fit: the barrier gate produces a sparse, patchy field
    # (most cells sit at/near the noise floor between firing events), so a
    # blanket per-cell log|S| vs log(r) fit is dominated by non-firing cells
    # and says nothing about the radiated envelope. Bin r logarithmically and
    # fit the log-log slope of the PEAK (percentile) |S| per bin instead --
    # the standard way to characterize a noisy wave amplitude trend.
    # -------------------------------------------------------------------
    def _fit_slope(mask, percentile=99.0, n_bins=16, min_per_bin=5):
        m = mask & in_fit_r & (Smag > 0)
        n_total = int(m.sum())
        if n_total < min_per_bin * 3:
            return None, None, n_total, None
        r_m, s_m = r_from_centre[m], Smag[m]
        edges = np.logspace(np.log10(r_min_fit), np.log10(r_max_fit), n_bins + 1)
        centres, envelope = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = (r_m >= lo) & (r_m < hi)
            if in_bin.sum() < min_per_bin:
                continue
            centres.append(np.sqrt(lo * hi))
            envelope.append(np.percentile(s_m[in_bin], percentile))
        if len(centres) < 3:
            return None, None, n_total, None
        centres, envelope = np.array(centres), np.array(envelope)
        lr, ls = np.log(centres), np.log(envelope)
        A = np.vstack([lr, np.ones_like(lr)]).T
        slope, intercept = np.linalg.lstsq(A, ls, rcond=None)[0]
        return float(slope), float(intercept), n_total, (centres, envelope)

    slope_eq, intercept_eq, n_eq, env_eq = _fit_slope(equatorial)
    slope_pol, intercept_pol, n_pol, env_pol = _fit_slope(polar)

    print()
    print("=== Radial |S| falloff (log-log slope; classical far-zone dipole = -2) ===")
    if slope_eq is not None:
        print(f"equatorial (theta>{90-half:.0f} deg, n={n_eq}): slope={slope_eq:+.3f}")
    else:
        print(f"equatorial: insufficient nonzero cells in fit range (n={n_eq})")
    if slope_pol is not None:
        print(f"polar      (theta<{half:.0f} deg, n={n_pol}): slope={slope_pol:+.3f}")
    else:
        print(f"polar: insufficient nonzero cells in fit range (n={n_pol})")

    equator_over_pole = None
    if slope_eq is not None and slope_pol is not None:
        # Compare sector amplitudes at r_min_fit (intercept comparison at a
        # common radius) -- classical dipole predicts equatorial >> polar
        # (ideally polar -> 0).
        s_eq_at_rmin = np.exp(intercept_eq + slope_eq * np.log(r_min_fit))
        s_pol_at_rmin = np.exp(intercept_pol + slope_pol * np.log(r_min_fit))
        equator_over_pole = s_eq_at_rmin / max(s_pol_at_rmin, 1e-300)
        print(
            f"|S|(equator)/|S|(pole) at r={r_min_fit:.2f}: {equator_over_pole:.3e}  "
            f"(classical magnetic-dipole prediction: >> 1, ideally -> inf)"
        )

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[plot] SKIPPED -- matplotlib import failed ({e})", flush=True)
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for mask, label, color in [
        (equatorial, "equatorial", "C0"),
        (polar, "polar", "C1"),
    ]:
        m = mask & (Smag > 0) & (r_from_centre > ring_r)
        ax.scatter(
            r_from_centre[m],
            Smag[m],
            s=3,
            alpha=0.08,
            color=color,
            label=f"_{label} cells",
        )
    for env, label, color in [
        (env_eq, "equatorial p99 envelope", "C0"),
        (env_pol, "polar p99 envelope", "C1"),
    ]:
        if env is not None:
            ax.plot(
                env[0], env[1], "o", color=color, ms=6, mec="k", mew=0.5, label=label
            )
    r_ref = np.array([r_min_fit, r_max_fit])
    if slope_eq is not None:
        ax.plot(
            r_ref,
            np.exp(intercept_eq) * r_ref**slope_eq,
            "C0-",
            lw=2,
            label=f"equatorial fit slope={slope_eq:+.2f}",
        )
    if slope_pol is not None:
        ax.plot(
            r_ref,
            np.exp(intercept_pol) * r_ref**slope_pol,
            "C1-",
            lw=2,
            label=f"polar fit slope={slope_pol:+.2f}",
        )
    # classical -2 reference, anchored at equatorial fit's value at r_min_fit
    if slope_eq is not None:
        s0 = np.exp(intercept_eq) * r_min_fit**slope_eq
        ax.plot(
            r_ref,
            s0 * (r_ref / r_min_fit) ** (-2.0),
            "k--",
            lw=1.5,
            label="classical r^-2",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("r (cells, from ring centre)")
    ax.set_ylabel("|S|")
    ax.set_title(
        f"|S| radial falloff vs classical dipole  R={args.R} hops={args.hops} "
        f"m0={args.m0:.1e}  step={args.n_steps}"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"\n[plot] {args.plot_out}", flush=True)


if __name__ == "__main__":
    main()
