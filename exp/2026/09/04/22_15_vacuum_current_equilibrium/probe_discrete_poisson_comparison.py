"""
Philip (2026-09-04): "what happens if you consider discrete derivatives
for the maxwellian comparison" -- instead of fitting DST's field against
the smooth CONTINUUM 1/r Coulomb solution (probe_classical_comparison.py,
found a real, growing-with-distance few-percent residual), solve the
DISCRETE Poisson equation directly on the SAME lattice graph (same 14
neighbours, same W_HEX=3/W_SQ=1 face weights lattice.py actually uses --
NOT doc/dst.md's stated w1=2/3/w2=1/3, a real doc/code discrepancy worth
flagging separately) with the same two fixed-u Dirichlet sources, and
compare DST's dynamic step()-evolved field against THAT reference instead.

doc/dst.md's own claim (line 121, "Gauss's Theorem on the Kelvin Cell"):
Delta u_i/Delta t = sum_j A_ij (u_j-u_i)/d_ij -- the discrete Laplacian.
Steady state (line 129) is exactly the discrete Poisson equation this
solves. If DST's dynamic result matches THIS discrete reference much
better than it matched the smooth continuum 1/r fit, the earlier
"growing-with-distance residual" was mostly discretization truncation
error (expected, not a real physics gap) -- doc/dst.md itself says the
weights only make the discrete Laplacian "maximally isotropic", not exact,
so a real leading-order truncation-error term is expected and named in the
text itself.

Solves the reduced (Dirichlet-eliminated) symmetric system via CG.
"""

import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice, W_HEX, W_SQ

R = 40.0
SEP = 16.0
U0 = 1.0
N_STEPS = 20000
N_THREADS = 8
FIT_R_MIN = 2.0
FIT_R_MAX = 14.0


def build_graph_laplacian(lat):
    N = lat.N
    nb = lat.nb
    W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
    rows, cols, vals = [], [], []
    diag = np.zeros(N)
    for k in range(14):
        valid = nb[:, k] >= 0
        i_idx = np.nonzero(valid)[0]
        j_idx = nb[i_idx, k]
        w = W[k]
        rows.append(i_idx)
        cols.append(j_idx)
        vals.append(-w * np.ones(len(i_idx)))
        np.add.at(diag, i_idx, w)
    rows.append(np.arange(N))
    cols.append(np.arange(N))
    vals.append(diag)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    L = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return L


def solve_discrete_poisson(lat, ia, ib, u_a, u_b):
    N = lat.N
    L = build_graph_laplacian(lat)
    fixed = np.zeros(N, dtype=bool)
    fixed[ia] = True
    fixed[ib] = True
    u_fixed = np.zeros(N)
    u_fixed[ia] = u_a
    u_fixed[ib] = u_b

    free = ~fixed
    L_ff = L[free][:, free]
    L_fx = L[free][:, fixed]
    rhs = -(L_fx @ u_fixed[fixed])

    print(f"solving discrete Poisson: {free.sum()} free unknowns", flush=True)
    u_free, info = spla.cg(L_ff, rhs, rtol=1e-10, maxiter=20000)
    print(f"CG info={info}", flush=True)

    u_full = np.zeros(N)
    u_full[free] = u_free
    u_full[fixed] = u_fixed[fixed]
    return u_full


def build_sources(lat):
    pos = lat.pos
    src_a = np.array([-SEP / 2, 0.0, 0.0])
    src_b = np.array([SEP / 2, 0.0, 0.0])
    ia = int(np.argmin(np.linalg.norm(pos - src_a, axis=1)))
    ib = int(np.argmin(np.linalg.norm(pos - src_b, axis=1)))
    return ia, ib


def main():
    lat = DampedLattice(radius_cells=R, sponge_fraction=0.15, damp_max=0.003)
    ia, ib = build_sources(lat)
    pa, pb = lat.pos[ia].copy(), lat.pos[ib].copy()
    lat.u[ia] = U0
    lat.u[ib] = -U0
    lat.fixed_u[ia] = True
    lat.fixed_u[ib] = True
    dt = lat.stable_dt()
    print(f"N={lat.N}  dt={dt:.4e}  src_a={pa}  src_b={pb}", flush=True)

    for step in range(N_STEPS + 1):
        if step % 4000 == 0:
            print(f"  step {step}", flush=True)
        lat.step(dt, n_threads=N_THREADS)
    u_dynamic = lat.u.copy()

    u_discrete = solve_discrete_poisson(lat, ia, ib, U0, -U0)

    pos = lat.pos
    r_a = np.linalg.norm(pos - pa, axis=1)
    r_b = np.linalg.norm(pos - pb, axis=1)
    mask = (r_a > FIT_R_MIN) & (r_b > FIT_R_MIN) & (r_a < FIT_R_MAX) & (r_b < FIT_R_MAX)

    y_dyn = u_dynamic[mask]
    y_disc = u_discrete[mask]
    resid = y_dyn - y_disc
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_dyn - y_dyn.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rel = float(np.sqrt(np.mean(resid**2)) / (np.abs(y_dyn).mean() + 1e-30))

    print(f"\nDST-dynamic vs discrete-Poisson-reference (same mask as continuum comparison):")
    print(f"R^2 = {r2:.6f}  RMS residual/mean|u| = {rel:.4%}")

    r_near = np.minimum(r_a, r_b)[mask]
    bins = np.linspace(FIT_R_MIN, FIT_R_MAX, 7)
    print("\nresidual by distance-from-nearest-source bin (DST dynamic vs discrete Poisson):")
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = (r_near >= lo) & (r_near < hi)
        if b.sum() == 0:
            continue
        print(
            f"  r in [{lo:.1f},{hi:.1f})  n={b.sum():6d}  "
            f"mean|u_dyn|={np.abs(y_dyn[b]).mean():.4e}  mean|resid|={np.abs(resid[b]).mean():.4e}  "
            f"rel={np.abs(resid[b]).mean()/(np.abs(y_dyn[b]).mean()+1e-30):.2%}"
        )

    # for reference: also fit u_discrete itself against the smooth continuum 1/r
    # form, to see how much of the ORIGINAL residual (probe_classical_comparison.py)
    # was already present in the discrete-Poisson reference itself (pure
    # discretization truncation error, nothing to do with DST's dynamics)
    r_a_safe = np.where(r_a > 1e-9, r_a, 1e-9)
    r_b_safe = np.where(r_b > 1e-9, r_b, 1e-9)
    X = np.stack([1.0 / r_a_safe[mask], 1.0 / r_b_safe[mask]], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(X, y_disc, rcond=None)
    y_pred = X @ coeffs
    ss_res2 = float(np.sum((y_disc - y_pred) ** 2))
    ss_tot2 = float(np.sum((y_disc - y_disc.mean()) ** 2))
    r2_disc_vs_continuum = 1.0 - ss_res2 / ss_tot2 if ss_tot2 > 0 else float("nan")
    rel2 = float(np.sqrt(ss_res2 / len(y_disc)) / (np.abs(y_disc).mean() + 1e-30))
    print(f"\n[reference] discrete-Poisson-solution vs smooth continuum 1/r fit:")
    print(f"R^2 = {r2_disc_vs_continuum:.6f}  RMS residual/mean|u| = {rel2:.4%}")


if __name__ == "__main__":
    main()
