"""
Charge test 2b: "ion trap" style -- impose a KNOWN, exact boundary condition
instead of trying to approximate free space with a sponge.

A point charge at the centre of a grounded conducting sphere of radius R has
an EXACT classical solution (image-charge method):
    u(r) = q * (1/r - 1/R)    for r < R
i.e. plain 1/r, offset by a constant so u is forced to exactly 0 at the
wall. This sidesteps the "how big does the domain need to be to approximate
infinity" ambiguity from probe_derived_coulomb.py's sponge-based version --
here the target is EXACT for any R, so the comparison isn't approximate.

Method: fix u=0 on the outer shell of cells (r > r_max - shell_thickness)
instead of using the sponge's soft damping (damp_max=0, sponge inert),
fix u=u_source at the origin cell, relax under bare step() to steady
state, then fit u_measured(r) = q*(1/r - 1/R_eff) for a single free
parameter q (R_eff = the actual fixed-boundary radius) via least squares
and report the fit quality -- a clean, unambiguous shape-match diagnostic.

Run:
  PYTHONPATH=$PWD python3 -u exp/2026/08/27/19_25_gauss_charge_tests/probe_grounded_sphere.py
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice, _W

_D_FACE = np.array([1.0] * 8 + [2.0 / 3.0**0.5] * 6)


def compute_Q(lat):
    if lat._dirty:
        lat._rebuild_cache()
    nb = lat._nb_idx
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    u = lat.u
    u_nb = np.where(valid, u[safe_nb], u[:, None])
    du = u_nb - u[:, None]
    return np.where(valid, _W[None, :] * du / _D_FACE[None, :], 0.0).sum(axis=1)


def main():
    R = 40.0
    shell_thickness = 2.0
    u_source = 1.0
    n_steps = 60000
    report = 2000

    lat = DampedLattice(radius_cells=R, sponge_fraction=0.15, damp_max=0.003)
    pos = lat.pos
    r = np.linalg.norm(pos, axis=1)
    origin = int(np.argmin(r))
    r_max = float(r.max())
    R_eff = r_max  # the actual grounded-shell radius

    lat.fixed_u = np.zeros(lat.N, dtype=bool)
    outer_mask = r > (r_max - shell_thickness)
    lat.fixed_u[outer_mask] = True
    lat.u[outer_mask] = 0.0
    lat.fixed_u[origin] = True
    lat.u[origin] = u_source

    dt = lat.stable_dt()
    print(
        f"N={lat.N}  R={R}  R_eff={R_eff:.2f}  outer_shell_n={int(outer_mask.sum())}"
        f"  dt={dt:.4e}  u_source={u_source}  n_steps={n_steps}"
    )

    interior = ~outer_mask & (np.arange(lat.N) != origin)

    for s in range(1, n_steps + 1):
        lat.step(dt, n_threads=8)
        if s % report == 0:
            print(
                f"  {s:6d}  max|u|(interior)={np.abs(lat.u[interior]).max():.4e}"
                f"  mean|du/dt proxy|~n/a",
                flush=True,
            )

    Q_i = compute_Q(lat)
    u = lat.u

    far_from_src = r > 2.0
    print("\n--- final state ---")
    print(f"Q_i at origin={Q_i[origin]:.4e}")
    interior_only = interior & far_from_src
    print(
        f"interior (r>2, not boundary) |Q_i|: mean={np.abs(Q_i[interior_only]).mean():.4e}"
        f"  max={np.abs(Q_i[interior_only]).max():.4e}"
    )

    # fit u(r) = q*(1/r - 1/R_eff) for q via least squares, using radial bin means
    r_bins = np.linspace(1.0, r_max - shell_thickness - 1.0, 100)
    r_mid, u_mean = [], []
    for i in range(1, len(r_bins)):
        mask = (r >= r_bins[i - 1]) & (r < r_bins[i])
        if mask.sum() > 0:
            r_mid.append(0.5 * (r_bins[i - 1] + r_bins[i]))
            u_mean.append(float(u[mask].mean()))
    r_mid = np.array(r_mid)
    u_mean = np.array(u_mean)
    basis = 1.0 / r_mid - 1.0 / R_eff
    q_fit = float(np.dot(basis, u_mean) / np.dot(basis, basis))
    pred = q_fit * basis
    resid = u_mean - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((u_mean - u_mean.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"\nfit: u(r) = q*(1/r - 1/{R_eff:.2f}),  q_fit={q_fit:.4e}  R^2={r2:.6f}")
    print(
        f"\n{'r':>8}  {'u_measured':>12}  {'u_fit':>12}  {'residual':>12}  {'residual/u':>10}"
    )
    for i in range(0, len(r_mid), 2):
        rel = resid[i] / u_mean[i] if abs(u_mean[i]) > 1e-12 else float("nan")
        print(
            f"{r_mid[i]:8.2f}  {u_mean[i]:12.4e}  {pred[i]:12.4e}  {resid[i]:12.4e}  {rel:10.4f}"
        )

    np.savez(
        "exp/2026/08/27/19_25_gauss_charge_tests/grounded_sphere_result.npz",
        u=u,
        Q_i=Q_i,
        r=r,
        pos=pos,
        origin=origin,
        u_source=u_source,
        R_eff=R_eff,
        q_fit=q_fit,
        r2=r2,
        r_mid=r_mid,
        u_mean=u_mean,
    )
    print("\nsaved grounded_sphere_result.npz")


if __name__ == "__main__":
    main()
