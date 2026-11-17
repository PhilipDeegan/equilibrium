"""
Hydrogen Bohr levels from a discrete Planck-scale Kelvin cell lattice.

All quantities in Planck units (ℏ = c = G = k_B = 1, l_P = 1).

Physical constants:
    m_e  = 4.18542e-23  (electron mass / Planck mass)
    α    = 7.29735e-3   (fine-structure constant ≈ 1/137, dimensionless)
    a_0  = 1/(m_e * α)  ≈ 3.275e24 l_P  (Bohr radius in Planck lengths)

Hamiltonian on the Kelvin cell BCC lattice (l_P = 1 = nearest cell-centre distance):

    H = T + V
    T = -(1/2m_e) ∇²_discrete
    V_i = -α / max(r_i, r_min)     (Coulomb, regularised at the origin)

    BCC conventional cell side: a = 2/√3 l_P, so a² = 4/3
    Kinetic hopping: t = ℏ²/(2 m_e a²) = 3/(8 m_e)

Expected Bohr energies (Planck units):
    E_n = -(m_e α²) / (2 n²)

    n=1: E_1 = -m_e α²/2  ≈ -1.113e-27 E_P   (ground state)
    n=2: E_2 = E_1/4      ≈ -2.784e-28 E_P
    n=3: E_3 = E_1/9      ≈ -1.237e-28 E_P

Computational note:
    At true Planck scale a_0 ≈ 3.275e24 cells — impossible to simulate directly.
    We scale: set the numerical lattice spacing δ_num = N_cells * l_P and work
    in units of a_0 (set a_0 = 1), recovering Bohr levels as δ_num / a_0 → 0.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

# ---------------------------------------------------------------------------
# Planck-unit physical constants
# ---------------------------------------------------------------------------

M_E = 4.18542e-23  # electron mass in Planck masses
ALPHA = 7.29735e-3  # fine-structure constant
A_0 = 1.0 / (M_E * ALPHA)  # Bohr radius in Planck lengths ≈ 3.275e24

# Bohr energy in Planck units: E_n = -(m_e α²) / (2 n²)
E_BOHR = -M_E * ALPHA**2 / 2  # ground-state energy (n=1) in Planck units


def bohr_energy(n: int) -> float:
    """Exact Bohr energy level n in Planck units."""
    return E_BOHR / n**2


# ---------------------------------------------------------------------------
# Scaled solver  (works in units of a_0 to keep numbers O(1))
# ---------------------------------------------------------------------------


def build_hamiltonian(delta: float, radius: float):
    """
    Build the hydrogen Hamiltonian on a Kelvin cell lattice.

    Parameters
    ----------
    delta  : lattice spacing in units of a_0  (= δ_num / a_0)
             Conceptually δ_num / a_0 = l_P / a_0 ≈ 3e-25 for the true model.
    radius : sphere radius in units of a_0

    Returns
    -------
    H      : scipy CSR sparse Hamiltonian in units of E_hartree = m_e α² (= -2 E_BOHR)
    points : (N, 3) array of lattice coords in units of a_0
    """
    from equilibrium.cpp_dst import DSTLattice  # pylint: disable=no-name-in-module

    lat = DSTLattice(cell_size=delta, radius_cells=radius / delta)
    rows, cols, vals = lat.build_hamiltonian()
    N = lat.size()
    H = coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return H, lat.positions()


def solve(delta: float = 0.3, radius: float = 6.0, n_eig: int = 16) -> np.ndarray:
    """
    Solve hydrogen on the Kelvin lattice and return the n_eig lowest eigenvalues
    in Hartree atomic units (E_h = m_e α² = 27.211 eV, Bohr E_n = -1/(2n²) E_h).

    Parameters
    ----------
    delta  : lattice spacing / a_0   (smaller → more accurate, more expensive)
    radius : box radius / a_0
    n_eig  : number of eigenvalues

    Returns
    -------
    energies : sorted array of length n_eig, in Hartree
    """
    H, pts = build_hamiltonian(delta, radius)
    energies = eigsh(H, k=n_eig, which="SA", return_eigenvectors=False)
    return np.sort(energies)


def convergence_study(
    deltas=(0.3, 0.2, 0.15), box_factor: int = 3, n_levels: int = 2
) -> list:
    """
    Convergence of Kelvin lattice energy levels as delta → 0.

    Uses a separate eigensolve per level with radius = box_factor × n² to
    separate finite-box error from discretisation error.  With a shared box
    the n=2 error at radius=6 a₀ is dominated by box truncation, not δ.

    Default n_levels=2 (n=1 and n=2):
        n=1: radius = box_factor × 1 = 3 a₀  (compact, cheap)
        n=2: radius = box_factor × 4 = 12 a₀ (moderate)
        n=3: radius = box_factor × 9 = 27 a₀ (expensive — set n_levels=3 explicitly)

    Returns list of dicts:
        delta, N_per_level, energies (Hartree), errors_pct,
        lyman_alpha_nm, lyman_beta_nm
    """
    HAR_TO_NM = 1240.0 / 27.2114  # 1 E_h = 27.2114 eV; 1 eV ↔ 1240.0 nm

    exact = [-1.0 / (2 * n**2) for n in range(1, n_levels + 1)]

    results = []
    for delta in deltas:
        energies = []
        N_per_level = []
        for n in range(1, n_levels + 1):
            radius = float(max(6, box_factor * n**2))
            H, _ = build_hamiltonian(delta, radius)
            N = H.shape[0]
            N_per_level.append(N)
            print(f"  delta={delta}  n={n}  radius={radius}  N={N} ...", flush=True)
            n_eig = min(n + 10, H.shape[0] - 1)
            eigs = np.sort(eigsh(H, k=n_eig, which="SA", return_eigenvectors=False))
            energies.append(float(eigs[np.argmin(np.abs(eigs - exact[n - 1]))]))

        errors_pct = [100.0 * abs(p - e) / abs(e) for p, e in zip(energies, exact)]
        lyman_alpha = (
            HAR_TO_NM / (energies[1] - energies[0]) if n_levels >= 2 else float("nan")
        )
        lyman_beta = (
            HAR_TO_NM / (energies[2] - energies[0]) if n_levels >= 3 else float("nan")
        )

        results.append(
            {
                "delta": delta,
                "N_per_level": N_per_level,
                "energies": energies,
                "errors_pct": errors_pct,
                "lyman_alpha_nm": lyman_alpha,
                "lyman_beta_nm": lyman_beta,
            }
        )
    return results


def planck_extrapolation(deltas, errors_pct) -> tuple:
    """
    Fit error(δ) = c × δ² (BCC Laplacian converges as O(δ²)) and extrapolate
    to δ = l_P / a₀ ≈ 3.05×10⁻²⁵.

    Returns (c, extrapolated_error_pct).
    """
    d = np.asarray(deltas, dtype=float)
    e = np.asarray(errors_pct, dtype=float)
    c = float(np.dot(d**2, e) / np.dot(d**2, d**2))
    delta_planck = 1.0 / A_0
    return c, c * delta_planck**2


def print_convergence(results: list):
    """Print convergence_study results as a table."""
    n_levels = len(results[0]["energies"])
    exact = [-1.0 / (2 * n**2) for n in range(1, n_levels + 1)]

    header = f"{'delta':>8}  {'N(n=1)':>9}"
    for n in range(1, n_levels + 1):
        header += f"  E{n}(ex={exact[n-1]:+.4f})"
    if n_levels >= 2:
        header += "  λ_Lα(nm)"
    if n_levels >= 3:
        header += "  λ_Lβ(nm)"
    print(header)

    for r in results:
        row = f"{r['delta']:>8.3f}  {r['N_per_level'][0]:>9d}"
        for e, ep in zip(r["energies"], r["errors_pct"]):
            row += f"  {e:+.6f}({ep:.1f}%)"
        if n_levels >= 2:
            row += f"  {r['lyman_alpha_nm']:>9.3f}"
        if n_levels >= 3:
            row += f"  {r['lyman_beta_nm']:>9.3f}"
        print(row)


def convergence_table(deltas=(0.4, 0.3, 0.2), radius: float = 6.0, n_levels: int = 3):
    """Legacy fixed-box convergence table. Prefer convergence_study()."""
    exact = [-1.0 / (2 * n**2) for n in range(1, n_levels + 1)]
    print(f"\n{'delta':>8}  {'N_pts':>8}  ", end="")
    for n in range(1, n_levels + 1):
        print(f"  E_{n} (exact={exact[n-1]:+.4f})", end="")
    print()

    for delta in deltas:
        H, pts = build_hamiltonian(delta, radius)
        N = H.shape[0]
        eigs = np.sort(eigsh(H, k=3 * n_levels, which="SA", return_eigenvectors=False))
        picked = []
        for n in range(1, n_levels + 1):
            e_exact = exact[n - 1]
            closest = eigs[np.argmin(np.abs(eigs - e_exact))]
            picked.append(closest)
        print(f"{delta:>8.3f}  {N:>8d}  ", end="")
        for e, e_ex in zip(picked, exact):
            print(f"  {e:+.6f} (err {abs(e - e_ex):.4f})", end="")
        print()
