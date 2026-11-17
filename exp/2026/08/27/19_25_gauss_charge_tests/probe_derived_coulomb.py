"""
Charge test 2 (foundational, before touching the electron IC specifically):
does DST's own claimed derivation actually hold?

doc/dst.md, "Steady-state form -- discrete Poisson equation":
  "At equilibrium every free cell has zero net tension flow across its 14
  faces. At a fixed-source cell (a knot), the boundary condition is
  nonzero. This is the discrete Poisson equation. Its far-field solution
  is u(r) ~ 1/r -- Coulomb's law, derived, not imposed."

Never numerically checked in this codebase (confirmed by search this
session). Both objects tested all session (A2's set_winding proton-knot,
init_electron_dipole_ic's "zero-net-force" electron) have u=0 everywhere
by construction -- neither ever exercised this claimed mechanism at all.

Method: fix u at ONE source cell (the lattice origin) to a nonzero
constant, leave every other cell at u=0, and relax under bare step()
(u's own diffusive/Laplacian dynamics -- no step_cross, no phi seeding)
until d u/dt ~ 0 everywhere except the fixed source. Then check:
  1. Does u(r) actually converge to ~1/r in the far field?
  2. Is Q_i = Sum_k W_k (u_nb-u_i)/d_k ~0 away from the source (doc's
     "free cell: Q_i=0 by equilibrium" claim)?
  3. Is Q_enc(r) = Sum_{r'<r} Q_i constant once the shell encloses the
     source -- the actual non-trivial Gauss's law check?

Run:
  PYTHONPATH=$PWD python3 -u exp/2026/08/27/19_25_gauss_charge_tests/probe_derived_coulomb.py
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice

_D_FACE = np.array([1.0] * 8 + [2.0 / 3.0**0.5] * 6)


def compute_Q(lat):
    u = lat.u
    nb = lat._nb_idx if hasattr(lat, "_nb_idx") else None
    if nb is None:
        if lat._dirty:
            lat._rebuild_cache()
        nb = lat._nb_idx
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    from equilibrium.substrate.lattice import _W

    u_nb = np.where(valid, u[safe_nb], u[:, None])
    du = u_nb - u[:, None]
    Q_i = np.where(valid, _W[None, :] * du / _D_FACE[None, :], 0.0).sum(axis=1)
    return Q_i


def main():
    R = 60.0
    u_source = 1.0
    n_steps = 80000
    report = 4000

    lat = DampedLattice(radius_cells=R, sponge_fraction=0.15, damp_max=0.003)
    pos = lat.pos
    r = np.linalg.norm(pos, axis=1)
    origin = int(np.argmin(r))

    lat.u[origin] = u_source
    lat.fixed_u = np.zeros(lat.N, dtype=bool)
    lat.fixed_u[origin] = True

    dt = lat.stable_dt()
    print(f"N={lat.N}  R={R}  dt={dt:.4e}  u_source={u_source}  n_steps={n_steps}")

    for s in range(1, n_steps + 1):
        lat.step(dt, n_threads=8)
        if s % report == 0:
            Q = compute_Q(lat)
            far = r > 5.0
            print(
                f"  {s:6d}  max|u|(non-src)={np.abs(lat.u[~lat.fixed_u]).max():.4e}"
                f"  Q_far_mean={Q[far].mean():.4e}  Q_far_max_abs={np.abs(Q[far]).max():.4e}",
                flush=True,
            )

    Q_i = compute_Q(lat)
    u = lat.u

    print("\n--- final state ---")
    print(f"Q_i: min={Q_i.min():.4e}  max={Q_i.max():.4e}  at origin={Q_i[origin]:.4e}")
    far_mask = r > 5.0
    print(
        f"far-field (r>5) |Q_i|: mean={np.abs(Q_i[far_mask]).mean():.4e}  "
        f"max={np.abs(Q_i[far_mask]).max():.4e}"
    )

    print(f"\n{'r_bin':>8}  {'u_mean':>10}  {'u*r':>10}  {'Q_enc':>12}  {'n':>8}")
    r_bins = np.linspace(1.0, R * 0.85, 25)
    for i, rb in enumerate(r_bins):
        if i == 0:
            shell = r < rb
        else:
            shell = (r >= r_bins[i - 1]) & (r < rb)
        enc = r < rb
        if shell.sum() > 0:
            u_mean = u[shell].mean()
            print(
                f"{rb:8.2f}  {u_mean:10.4e}  {u_mean*rb:10.4e}  {Q_i[enc].sum():12.4e}  {int(shell.sum()):8d}"
            )

    np.savez(
        "exp/2026/08/27/19_25_gauss_charge_tests/derived_coulomb_result.npz",
        u=u,
        Q_i=Q_i,
        r=r,
        pos=pos,
        origin=origin,
        u_source=u_source,
    )
    print("\nsaved derived_coulomb_result.npz")


if __name__ == "__main__":
    main()
