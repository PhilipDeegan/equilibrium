"""
Simplified Poynting-ring far/near-field falloff, computed ENTIRELY from
plain evolver.step() (face-hopping wave dynamics) -- no step_cross, no
gate, no m0 anywhere.

S is the local discrete energy-flux vector derived and numerically
verified against the real step() in equilibrium.substrate.energy_flux:
S_i = sum_k W_k * phi[i,k] * ubar_i,k * n_hat[i,k] (scalar face flux times
face-average u, projected onto real face-direction vectors) -- NOT E x B.
Replaces this script's original E:=-dphi/dt, B:=curl(phi), S:=ExB
convention, which project_dst_em_field_convention_crisis.md later found
failed three independent checks (local Poynting theorem, two Hertzian-
dipole constructions). DST's (u, phi) update is structurally a scalar/
acoustic-style staggered wave scheme, not a full vector-EM one -- there is
no cross product in its local energy law, only a scalar flux per face.
This sidesteps the whole step_cross/m0 diagnosis in
project_poynting_dipole_calibration -- doc/dst.md's own claim is that EM
wave propagation is a face-hopping phenomenon, so this is the natural,
much simpler way to measure it.

New angle (2026-08-19): every existing Poynting-ring run (including the
original -6.19/-2.44 near-field calibration) seeds `lam_ring=1.0` on the
ring tiles by default, against a lam=0 background -- a SHARP, localized
sigma step right at the source. The energy-split thread the same day
derived dCirc/dt = -dt*W_sq*d2^2*(grad(sigma) x grad(u)) as DST's genuine
continuum electric<->magnetic leak mechanism (project_curl_full_dipole_
spheromak memory) -- vanishes for a uniform lam, real for a sharp one. This
script's --lam-ring sweep (0 vs the historical default 1.0) tests directly
whether that leak mechanism is (part of) the real source of the near-field
equatorial/polar asymmetry, rather than BCC dispersion anisotropy or a
settle-time artifact (the two explanations tried so far).

Usage:
    PYTHONPATH=$PWD python3 run/substrate/adaptive/probe_poynting_step_only.py --lam-ring 0
    PYTHONPATH=$PWD python3 run/substrate/adaptive/probe_poynting_step_only.py --lam-ring 1.0
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.energy_flux import local_energy_flux_vector
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    leaf_nb_j,
    winding_number,
)


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--R", type=float, default=50.0)
parser.add_argument("--hops", type=int, default=4)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--lam-ring", type=float, default=1.0)
parser.add_argument("--axis", default="z")
parser.add_argument("--n-steps", type=int, default=1500)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--sector-half-angle", type=float, default=20.0)
parser.add_argument("--r-min-fit", type=float, default=None, help="default: 1.5*ring_r")
parser.add_argument("--r-max-fit", type=float, default=None, help="default: 0.8*R")


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
        f"N={cl.n_leaves}  R={args.R}  hops={args.hops}  lam_ring={args.lam_ring}  "
        f"ring_r={ring_r:.3f}  omega={omega:.4f}  period={period}  dt={dt:.4e}",
        flush=True,
    )

    nb_j = leaf_nb_j(cl)
    r_from_centre = np.linalg.norm(pos - ring_centre, axis=1)

    r_min_fit = args.r_min_fit if args.r_min_fit is not None else 1.5 * ring_r
    r_max_fit = args.r_max_fit if args.r_max_fit is not None else 0.8 * args.R

    rel = pos - ring_centre
    ax_comp = rel[:, ax_idx]
    perp_comp = np.linalg.norm(rel[:, perp], axis=1)
    theta = np.degrees(np.arctan2(perp_comp, np.abs(ax_comp)))
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
        if step % 500 == 0:
            print(f"  step={step}", flush=True)

    print(
        f"reached step={args.n_steps}, computing local energy-flux S from plain step() only...",
        flush=True,
    )

    phi_before = np.array(cl.leaf_phi).T.copy()
    alat.step(0, dt)  # ONE more plain step -- no step_cross anywhere
    u_after = np.array(cl.leaf_u)

    # S_i = sum_k W_k*phi[i,k]*ubar_i,k*n_hat[i,k], paired phi(before)/u(after)
    # to match step()'s own symplectic-Euler ordering (du from OLD phi, dphi
    # from NEW u) -- see equilibrium.substrate.energy_flux module docstring.
    S3 = local_energy_flux_vector(pos, nb_j, phi_before, u_after)
    Smag = np.linalg.norm(S3, axis=1)

    def _fit_slope(mask, percentile=99.0, n_bins=16, min_per_bin=5):
        m = mask & in_fit_r & (Smag > 0)
        n_total = int(m.sum())
        if n_total < min_per_bin * 3:
            return None, None, n_total
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
            return None, None, n_total
        centres, envelope = np.array(centres), np.array(envelope)
        lr, ls = np.log(centres), np.log(envelope)
        A = np.vstack([lr, np.ones_like(lr)]).T
        slope, intercept = np.linalg.lstsq(A, ls, rcond=None)[0]
        return float(slope), float(intercept), n_total

    slope_eq, intercept_eq, n_eq = _fit_slope(equatorial)
    slope_pol, intercept_pol, n_pol = _fit_slope(polar)

    print()
    print(
        "=== Radial |S| falloff (log-log slope; classical far-zone dipole = -2), plain step() only ==="
    )
    if slope_eq is not None:
        print(f"equatorial (theta>{90-half:.0f} deg, n={n_eq}): slope={slope_eq:+.3f}")
    else:
        print(f"equatorial: insufficient nonzero cells (n={n_eq})")
    if slope_pol is not None:
        print(f"polar      (theta<{half:.0f} deg, n={n_pol}): slope={slope_pol:+.3f}")
    else:
        print(f"polar: insufficient nonzero cells (n={n_pol})")

    if slope_eq is not None and slope_pol is not None:
        s_eq_at_rmin = np.exp(intercept_eq + slope_eq * np.log(r_min_fit))
        s_pol_at_rmin = np.exp(intercept_pol + slope_pol * np.log(r_min_fit))
        ratio = s_eq_at_rmin / max(s_pol_at_rmin, 1e-300)
        print(f"|S|(equator)/|S|(pole) at r={r_min_fit:.2f}: {ratio:.3e}")


if __name__ == "__main__":
    main()
