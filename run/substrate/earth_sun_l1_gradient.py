"""
Earth-Sun substrate topography.

Sibling of earth_moon_gradient.py, same u(r)=GM/(rc^2) + rotation model, but
for the Sun-Earth pair instead of Earth-Moon. Moon and all other planets are
dropped entirely -- per the 2026-08-23 finding that the Sun's own u at
Earth's orbital distance (~9.87e-9) is ~14x BIGGER than Earth's own surface
term (~6.96e-10), this is the dominant unmodelled term from the Earth-Moon
version, not a negligible refinement. Earth-Moon's own contribution at this
scale is smaller again (Moon's u at 1 AU is utterly negligible), so folding
it back in here would be adding a small correction to a small correction --
not done, ignored the same way Jupiter/Venus/etc already were.

In DST:
    u(r) = GM/(rc²)           substrate potential (dimensionless)
    L(r) = L_vacuum × (1−u)   local throughput rate

The Sun-Earth L1 point is the throughput maximum (u minimum) on the segment
between the two bodies -- the saddle where the Sun's and Earth's depletion
fields cancel against the rotational (centrifugal) term. Real spacecraft
(SOHO, WIND, DSCOVR, Gaia, WMAP, JWST -- though JWST orbits L2, not L1) sit
in halo orbits around this exact point, so unlike the Earth-Moon "L1" (which
is a real saddle point but not officially designated L1), this one is a
genuine, independently-named, physically inhabited location.

Key results
-----------
u_sun_at_earth   Sun's own u at Earth's orbital distance (dominant, unmodelled
                 by earth_moon_gradient.py's u_at)
u_earth_surface  Earth's own u at its surface (for comparison -- ~14x smaller)
L1               point of minimum u (maximum throughput) between Earth and Sun
"""

import numpy as np
import scipy.constants as const

G = const.G
c = const.c
c2 = c**2

L_PLANCK_M = const.physical_constants["Planck length"][0]  # ~1.616e-35 m
L_VACUUM_LP = 1.0e30  # extragalactic reference throughput rate in Planck lengths

M_sun = 1.98892e30  # kg
M_earth = 5.9722e24  # kg
R_earth = 6.371e6  # m
R_sun = 6.957e8  # m
AU = 1.495978707e11  # m, Sun-Earth centre-to-centre distance (mean)

M_total = M_sun + M_earth
omega = np.sqrt(G * M_total / AU**3)  # orbital angular velocity  [rad/s]
x_bary = (
    M_earth * AU
) / M_total  # barycentre from Sun centre  [m] -- ~449 km, inside the Sun


# ---------------------------------------------------------------------------
# u-field (substrate potential)
# ---------------------------------------------------------------------------


def u_at(x: np.ndarray) -> np.ndarray:
    """
    DST substrate potential u at position x along the Sun-Earth axis (metres
    from Sun centre). Sun + Earth only -- no Moon, no other planets.

    u = GM_sun/(r_sun c²) + GM_earth/(r_earth c²) + ½ω²r_b²/c²
    """
    x = np.asarray(x, dtype=float)
    r_sun = np.maximum(np.abs(x), 1e-3)
    r_earth = np.maximum(np.abs(AU - x), 1e-3)
    r_b = x - x_bary

    u_grav = G * (M_sun / r_sun + M_earth / r_earth) / c2
    u_rot = 0.5 * omega**2 * r_b**2 / c2
    return u_grav + u_rot


def throughput_at(x: np.ndarray, L_vacuum_lP: float = L_VACUUM_LP) -> np.ndarray:
    """Local throughput rate L at position x along the Sun-Earth axis."""
    return L_vacuum_lP * (1.0 - u_at(x))


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def find_l1() -> dict:
    """
    Locate the Sun-Earth L1 point (throughput maximum / u minimum) between
    the two surfaces.
    """
    x_grid = np.linspace(R_sun, AU - R_earth, 2_000_000)
    u_grid = u_at(x_grid)
    L1_idx = int(np.argmin(u_grid))
    L1_m = float(x_grid[L1_idx])

    u_sun_surface = float(u_at(R_sun))
    u_earth_surface = float(u_at(AU - R_earth))
    u_sun_at_earth = float(
        G * M_sun / (AU * c2)
    )  # Sun's own u term, at Earth's distance

    return {
        "L1_m": L1_m,
        "L1_from_earth_km": (AU - L1_m) / 1e3,
        "L1_from_sun_km": L1_m / 1e3,
        "u_L1": float(u_grid[L1_idx]),
        "L_L1_lP": float(throughput_at(L1_m)),
        "u_sun_surface": u_sun_surface,
        "u_earth_surface": u_earth_surface,
        "u_sun_at_earth": u_sun_at_earth,
    }


# ---------------------------------------------------------------------------
# Clock gain, relative to Earth surface (matches earth_moon_gradient.py's
# original convention -- this script does not carry over the core-anchoring
# refinement, since the Sun's interior isn't a uniform sphere in any useful
# sense and isn't the point of this comparison)
# ---------------------------------------------------------------------------


def _clock_gain_us_per_day(x: float, u_ref: float) -> float:
    """DST daily clock gain in μs at x, relative to u_ref. f_DST = 1 - u."""
    return (u_ref - float(u_at(x))) * 86400 * 1e6


def _clock_gain_gr_us_per_day(x: float, u_ref: float) -> float:
    """GR (Schwarzschild) daily clock gain in μs at x, relative to u_ref."""
    u_r = float(u_at(x))
    return (np.sqrt(1.0 - 2.0 * u_r) / np.sqrt(1.0 - 2.0 * u_ref) - 1.0) * 86400 * 1e6


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------


def generate_substrate_topography():
    info = find_l1()

    points = {
        "Earth Surface": AU - R_earth,
        "Sun-Earth L1 (max throughput)": info["L1_m"],
        "Sun Surface": R_sun,
    }

    u_ref = info["u_earth_surface"]  # Earth surface as the comparison baseline

    print(
        "\nDST Sun-Earth substrate topography (Sun + Earth only, no Moon/other planets)"
    )
    print(f"  AU (Sun-Earth distance)     = {AU/1e3:,.1f} km")
    print(f"  u at Sun surface            = {info['u_sun_surface']:.6e}")
    print(f"  u at Earth surface          = {info['u_earth_surface']:.6e}")
    print(f"  Sun's own u at Earth's r    = {info['u_sun_at_earth']:.6e}")
    print(
        f"  (Earth's own surface term ~{float(G*M_earth/(R_earth*c2)):.6e}"
        f" -- Sun's term at 1 AU is ~"
        f"{info['u_sun_at_earth']/float(G*M_earth/(R_earth*c2)):.1f}x bigger)"
    )
    print(f"  L1 from Earth               = {info['L1_from_earth_km']:,.1f} km")
    print(f"  L1 from Sun                 = {info['L1_from_sun_km']:,.1f} km")
    print(f"  u at L1                     = {info['u_L1']:.6e}  (minimum u along axis)")
    print(
        f"  L at L1                     = {info['L_L1_lP']:.6e} l_P  (maximum throughput)"
    )

    L_P_fm = L_PLANCK_M * 1e15  # Planck length in fm

    name_w = max(len("Location"), max(len(name) for name in points))
    print(
        f"\n{'Location':<{name_w}} | {'Distance from Sun (km)':>23} | {'DST (μs/day)':>14}"
        f" | {'GR (μs/day)':>13} | {'Δ (ns/day)':>12}"
        f" | {'u (rel. Earth surf.)':>22} | {'ΔL rel. Earth surf. (fm)':>26}"
    )
    print("-" * (name_w + 133))
    for name, x_m in sorted(points.items(), key=lambda kv: kv[1]):
        dst_gain = _clock_gain_us_per_day(x_m, u_ref)
        gr_gain = _clock_gain_gr_us_per_day(x_m, u_ref)
        u_val = float(u_at(x_m)) - u_ref
        dL_fm = L_VACUUM_LP * u_val * L_P_fm
        diff_ns = (dst_gain - gr_gain) * 1e3
        print(
            f"{name:<{name_w}} | {x_m/1e3:>23,.1f} | {dst_gain:>14.4f}"
            f" | {gr_gain:>13.4f} | {diff_ns:>12.4f}"
            f" | {u_val:>22.6e} | {dL_fm:>26.4f}"
        )


if __name__ == "__main__":
    generate_substrate_topography()
