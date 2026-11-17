G = 6.67430 * 1e-11  # m3⋅kg−1⋅s−2
e = 1.602176634 * 1e-19  # coulombs

# c as measured locally (Earth surface, inside Milky Way potential well)
c = 299_792_458  # m/s

# c_void: speed of light in true void (L_MAX = 1.0, outside all gravitational wells).
# Placeholder derived from Milky Way escape velocity at the solar radius:
#   v_esc ≈ 550 km/s  (Gaia DR2 / RAVE stellar kinematics, Monari et al. 2018,
#                       Bird et al. 2019; range 520–580 km/s across surveys)
#   Δc/c = v_esc² / (2 c²) ≈ 1.67×10⁻⁶  →  +502 m/s
# Solar and Earth contributions (~3 m/s, ~0.2 m/s) folded in but negligible.
# Uncertainty ~30–50% on the correction; dominant systematic is v_esc spread.
c_void = 299_792_960  # m/s  [placeholder — Gaia/RAVE anchor]
