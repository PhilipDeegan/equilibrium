"""
Nuclear shell structure from the Kelvin cell lattice at resolution λ=1.

Same discrete Laplacian as bohr_kelvin.py — same 14-neighbour topology,
same isotropic BCC weights (w1=2/3 hex, w2=1/3 sq).  What changes is
the effective potential:

  Atomic scale (λ=0):  V = -Z/r          (Coulomb, long-range)
  Nuclear scale (λ=1): V = ½ r²          (harmonic oscillator, short-range)

Units: scaled so that ħ = m_N = ω = 1  (oscillator units).
The oscillator length a_osc = sqrt(ħ/m_N ω) = 1 in these units.

Harmonic oscillator magic numbers (no spin-orbit):
    N   d_N  cumulative  closed-shell?
    0    2       2        ✓ nuclear magic
    1    6       8        ✓ nuclear magic
    2   12      20        ✓ nuclear magic
    3   20      40          (nuclear magic is 28, not 40)
    4   30      70          (nuclear magic is 50, not 70)

The gap between 20 and 28 arises in the standard model from spin-orbit
coupling that pushes 1f7/2 (8 states) down below the N=3 remainder.
In DST, this splitting is expected to emerge from the 8-hex / 6-sq
anisotropy of the Kelvin cell — not imposed, but geometrically inherent.
The degree to which the discrete lattice reproduces the nuclear magic
numbers beyond 20 is therefore a falsifiable prediction of DST.

Usage:
    from equilibrium.physical.atom.nuclear_kelvin import solve, magic_table
    eigs = solve(delta=0.3, radius=5.0, n_eig=80)
    magic_table(eigs)
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from equilibrium.math.geo.kelvin import KelvinLattice

# Nuclear magic numbers (empirical)
MAGIC_NUMBERS = [2, 8, 20, 28, 50, 82, 126]

# Harmonic oscillator shell closures (N=0,1,2,...):  cumΣ (N+1)(N+2)
HO_CLOSURES = [sum((n + 1) * (n + 2) for n in range(N + 1)) for N in range(7)]
# = [2, 8, 20, 40, 70, 112, 168]


def build_hamiltonian(delta: float, radius: float):
    """
    Build the nuclear Hamiltonian on a Kelvin cell lattice.

    T = -(1/2) ∇²_discrete      (kinetic, m_N = 1, ħ = 1)
    V = ½ r²                     (3D isotropic harmonic oscillator)

    Parameters
    ----------
    delta  : lattice spacing in oscillator units (→ 0 for continuum limit)
    radius : sphere radius in oscillator units

    Returns
    -------
    H      : scipy CSR sparse Hamiltonian
    points : (N, 3) lattice coordinates
    """
    lattice = KelvinLattice(lattice_spacing=delta)
    a = lattice.a
    a2 = a**2

    points = lattice.generate_sphere(radius)
    N = len(points)
    idx, key = lattice.point_index_map(points)

    offsets = lattice.all_offsets
    weights = lattice.laplacian_weights

    t = 1.0 / (2.0 * a2)  # hopping amplitude T = -(1/2m) ∇²

    rows, cols, vals = [], [], []

    for i, p in enumerate(points):
        r2 = float(np.dot(p, p))
        V = 0.5 * r2  # harmonic oscillator

        diag = V
        for off, w in zip(offsets, weights):
            nb = p + off
            k = key(nb)
            if k not in idx:
                continue
            j = idx[k]
            hop = t * w
            rows.append(i)
            cols.append(j)
            vals.append(-hop)
            diag += hop

        rows.append(i)
        cols.append(i)
        vals.append(diag)

    H = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return H, points


def solve(delta: float = 0.3, radius: float = 5.0, n_eig: int = 80) -> np.ndarray:
    """
    Solve the nuclear oscillator Hamiltonian on the Kelvin lattice.

    Returns the n_eig lowest eigenvalues sorted ascending (oscillator units,
    exact E_N = N + 3/2 in the continuum).
    """
    H, _ = build_hamiltonian(delta, radius)
    eigs = eigsh(H, k=n_eig, which="SA", return_eigenvectors=False)
    return np.sort(eigs)


def level_degeneracies(eigs: np.ndarray, tol_frac: float = 0.05) -> list:
    """
    Cluster eigenvalues into degenerate shells and return
    [(E_centre, count), ...] sorted by energy.
    """
    if len(eigs) == 0:
        return []
    shells, current = [], [eigs[0]]
    for e in eigs[1:]:
        spread = abs(e - current[0])
        ref = max(abs(current[0]), 1e-10)
        if spread / ref < tol_frac:
            current.append(e)
        else:
            shells.append((float(np.mean(current)), len(current)))
            current = [e]
    shells.append((float(np.mean(current)), len(current)))
    return shells


def magic_table(eigs: np.ndarray, tol_frac: float = 0.05):
    """
    Print the shell structure alongside harmonic-oscillator and nuclear
    magic numbers, and flag where the discrete Kelvin lattice deviates.
    """
    shells = level_degeneracies(eigs, tol_frac)
    ho_exact = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]  # E_N = N + 3/2 (ħω=1)

    print(
        f"\n{'Shell':>6}  {'E (lattice)':>13}  {'E (HO exact)':>13}  "
        f"{'deg':>5}  {'cumul':>7}  {'HO close':>10}  {'NM close':>10}"
    )
    print("-" * 80)

    cumul = 0
    for idx_s, (e, d) in enumerate(shells):
        cumul += d
        ho = ho_exact[idx_s] if idx_s < len(ho_exact) else float("nan")
        # flag closures
        ho_flag = f"← {cumul}" if cumul in HO_CLOSURES else ""
        nm_flag = f"← {cumul}" if cumul in MAGIC_NUMBERS else ""
        print(
            f"{idx_s:>6}  {e:>13.4f}  {ho:>13.4f}  "
            f"{d:>5}  {cumul:>7}  {ho_flag:>10}  {nm_flag:>10}"
        )

    print()
    matched = [
        m
        for m in MAGIC_NUMBERS
        if m <= cumul
        and m
        in {
            c
            for _, c in [
                (0, sum(d for _, d in shells[: i + 1])) for i in range(len(shells))
            ]
        }
    ]
    print(f"Nuclear magic numbers reproduced: {matched}")
    print(
        f"H.O. closures in spectrum:        "
        f"{[c for c in HO_CLOSURES if c <= cumul]}"
    )


def convergence_table(deltas=(0.4, 0.3, 0.2), radius=5.0, n_shells=4):
    """
    Show how the lowest shell energies converge toward E_N = N + 3/2
    as the lattice spacing delta → 0.
    """
    exact = [n + 1.5 for n in range(n_shells)]
    print(f"\n{'delta':>8}  {'N_pts':>8}  ", end="")
    for n in range(n_shells):
        print(f"  E_{n} (exact={exact[n]:.2f})", end="")
    print()

    for delta in deltas:
        H, pts = build_hamiltonian(delta, radius)
        N_pts = H.shape[0]
        eigs = np.sort(eigsh(H, k=4 * n_shells, which="SA", return_eigenvectors=False))
        shells = level_degeneracies(eigs)
        picked = [s[0] for s in shells[:n_shells]]
        print(f"{delta:>8.3f}  {N_pts:>8d}  ", end="")
        for e, e_ex in zip(picked, exact):
            print(f"  {e:+.5f} (err {abs(e-e_ex):.4f})", end="")
        print()
