"""
Philip (2026-09-04): quantify how far DST's two-fixed-source result is from
the classical continuous prediction, not just eyeball it.

Reuses the exact methodology already validated in this project for a
single point source (exp/2026/08/27/19_25_gauss_charge_tests/
probe_derived_coulomb.py): fit the actual field against the classical
closed-form solution and report R^2 + residuals, rather than a fresh
from-scratch Maxwell/FDTD build (a much bigger undertaking, not attempted
here).

Classical target for TWO fixed point potentials (a dipole-like pair) in
the electrostatic limit: u_classical(r_a, r_b) = A/r_a + B/r_b, A,B fit by
least squares (expected A~-B by symmetry, matching the sources' opposite
sign). This is meaningful even though the run (probe_vacuum_current.py,
20000 steps) has NOT reached global steady state (H/I_mid still growing,
now.md 2026-09-04 finding 2) -- the classical WAVE equation also predicts
exactly zero field outside a sharp causal horizon (Huygens' principle, no
wake in 3D) for a suddenly-switched-on source, so a still-growing outer
extent is not itself evidence against the near-field shape being right.
Restrict the fit to cells within a radius where the near-source field has
plausibly settled (excludes both the singular source cells themselves and
the not-yet-reached far boundary).
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice

R = 40.0
SEP = 16.0
U0 = 1.0
N_STEPS = 20000
N_THREADS = 8
FIT_R_MIN = 2.0  # exclude cells too close to either source (near-singular)
FIT_R_MAX = 14.0  # exclude cells near the outer boundary/sponge


def build(lat):
    pos = lat.pos
    src_a = np.array([-SEP / 2, 0.0, 0.0])
    src_b = np.array([SEP / 2, 0.0, 0.0])
    ia = int(np.argmin(np.linalg.norm(pos - src_a, axis=1)))
    ib = int(np.argmin(np.linalg.norm(pos - src_b, axis=1)))
    lat.u[ia] = U0
    lat.u[ib] = -U0
    lat.fixed_u[ia] = True
    lat.fixed_u[ib] = True
    return lat.pos[ia].copy(), lat.pos[ib].copy()


def main():
    lat = DampedLattice(radius_cells=R, sponge_fraction=0.15, damp_max=0.003)
    pa, pb = build(lat)
    dt = lat.stable_dt()
    print(f"N={lat.N}  dt={dt:.4e}  src_a={pa}  src_b={pb}", flush=True)

    for step in range(N_STEPS + 1):
        if step % 2000 == 0:
            print(f"  step {step}", flush=True)
        lat.step(dt, n_threads=N_THREADS)

    pos = lat.pos
    r_a = np.linalg.norm(pos - pa, axis=1)
    r_b = np.linalg.norm(pos - pb, axis=1)
    r_a_safe = np.where(r_a > 1e-9, r_a, 1e-9)
    r_b_safe = np.where(r_b > 1e-9, r_b, 1e-9)

    mask = (r_a > FIT_R_MIN) & (r_b > FIT_R_MIN) & (r_a < FIT_R_MAX) & (r_b < FIT_R_MAX)
    print(f"fit cells: {mask.sum()} / {lat.N}")

    X = np.stack([1.0 / r_a_safe[mask], 1.0 / r_b_safe[mask]], axis=1)
    y = lat.u[mask]
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    A, B = coeffs
    y_pred = X @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rel_resid = float(np.sqrt(ss_res / len(y)) / (np.abs(y).mean() + 1e-30))

    print(f"\nfit: u ~= A/r_a + B/r_b,  A={A:.6f}  B={B:.6f}  (expect A ~ -B)")
    print(f"R^2 = {r2:.6f}")
    print(f"RMS residual / mean|u| = {rel_resid:.4%}")

    # per-radius-bin residual breakdown
    print("\nresidual by distance-from-nearest-source bin:")
    r_near = np.minimum(r_a, r_b)[mask]
    resid = np.abs(y - y_pred)
    bins = np.linspace(FIT_R_MIN, FIT_R_MAX, 7)
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = (r_near >= lo) & (r_near < hi)
        if b.sum() == 0:
            continue
        print(
            f"  r in [{lo:.1f},{hi:.1f})  n={b.sum():6d}  "
            f"mean|u|={np.abs(y[b]).mean():.4e}  mean|resid|={resid[b].mean():.4e}  "
            f"rel={resid[b].mean()/(np.abs(y[b]).mean()+1e-30):.2%}"
        )


if __name__ == "__main__":
    main()
