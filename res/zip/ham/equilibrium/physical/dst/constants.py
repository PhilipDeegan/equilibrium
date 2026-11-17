"""
CODATA/SI values used as external cross-checks for DST simulations.

In Planck units (l_P=1, t_P=1, K=1): c = 1 by definition.
The electromagnetic vacuum structure (Z₀, ε₀, μ₀) is not imposed —
it is a product of c and the BCC face geometry once c is recovered.
u₀ is pure gauge (only differences u_j − u_i enter the update rule)
and does not need to be specified.

These constants are guides: if a simulation predicts a length or energy,
compare against the values below to check we're on the right track.
"""

import numpy as np

# CODATA 2018 (SI)
C_SI = 2.99792458e8  # m/s  — speed of light (exact)
L_PLANCK_M = 1.616255e-35  # m    — Planck length
T_PLANCK_S = 5.391247e-44  # s    — Planck time
E_CHARGE_C = 1.602176634e-19  # C    — elementary charge (exact)
BOHR_RADIUS_M = 5.29177210903e-11  # m    — Bohr radius a₀
PROTON_RADIUS_FM = 0.8414  # fm   — CODATA proton charge radius
NEUTRON_RADIUS_FM = 0.8600  # fm   — neutron charge radius (rms)
ALPHA_FINE = 7.2973525693e-3  # —    — fine-structure constant

# BCC Kelvin lattice: sum of face-unit conductances over all 14 faces
# FU_k = A_k/d_k at unit L=1:  hex → √3/4,  sq → √3/12
# Σ_k FU_k = 8×(√3/4) + 6×(√3/12) = 5√3/2
FU_HEX = np.sqrt(3) / 4  # ≈ 0.4330
FU_SQ = np.sqrt(3) / 12  # ≈ 0.1443
FU_SUM = 8 * FU_HEX + 6 * FU_SQ  # ≈ 4.330  (5√3/2)


def cell_size_for_scale(physical_length_m: float, n_cells: float) -> float:
    """Cell size in Planck lengths to represent physical_length_m in n_cells."""
    return (physical_length_m / L_PLANCK_M) / n_cells
