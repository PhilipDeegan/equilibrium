"""
Earth-Moon substrate topography.

Derives the DST u-field and throughput rate L along the Earth-Moon axis from
the superposition of two gravitational knot fields plus the system's rotational
potential.

In DST:
    u(r) = GM/(rc²)           substrate potential (dimensionless)
    L(r) = L_vacuum × (1−u)   local throughput rate

The gradient ∇u drives the advection of neighbouring cells, recovering Newton's
law.  The L1 point is the throughput maximum (u minimum) between Earth and Moon
— the saddle where the two depletion fields cancel.

Key results
-----------
u_surface     substrate potential at Earth surface
u_gps         substrate potential at GPS orbit
L1            point of minimum u (maximum throughput) along the Earth-Moon axis
"""

import numpy as np
import scipy.constants as const

G = const.G
c = const.c
c2 = c**2

L_PLANCK_M = const.physical_constants["Planck length"][0]  # ~1.616e-35 m
L_VACUUM_LP = 1.0e30  # extragalactic reference throughput rate in Planck lengths

M_earth = 5.9722e24  # kg
M_moon = 7.342e22  # kg
R_earth = 6.371e6  # m
R_moon = 1.737e6  # m
D_em = 384_400e3  # m  Earth-Moon centre-to-centre distance

M_total = M_earth + M_moon
omega = np.sqrt(G * M_total / D_em**3)  # orbital angular velocity  [rad/s]
x_bary = (M_moon * D_em) / M_total  # barycenter from Earth centre  [m]


# ---------------------------------------------------------------------------
# u-field (substrate potential)
# ---------------------------------------------------------------------------


def u_at(x: np.ndarray) -> np.ndarray:
    """
    DST substrate potential u at position x along the Earth-Moon axis
    (metres from Earth centre).

    u = GM_E/(r_E c²) + GM_M/(r_M c²) + ½ω²r_b²/c²
    """
    x = np.asarray(x, dtype=float)
    r_E = np.maximum(np.abs(x), 1e-3)
    r_M = np.maximum(np.abs(D_em - x), 1e-3)
    r_b = x - x_bary

    u_grav = G * (M_earth / r_E + M_moon / r_M) / c2
    u_rot = 0.5 * omega**2 * r_b**2 / c2
    return u_grav + u_rot


def throughput_at(x: np.ndarray, L_vacuum_lP: float = L_VACUUM_LP) -> np.ndarray:
    """
    Local throughput rate L at position x along the Earth-Moon axis.
    L(x) = L_vacuum × (1 − u(x))
    """
    return L_vacuum_lP * (1.0 - u_at(x))


def u_gradient_at(x: np.ndarray, dx: float = 1e3) -> np.ndarray:
    """du/dx at position x via central differences (dx in metres, default 1 km)."""
    return (u_at(x + dx) - u_at(x - dx)) / (2.0 * dx)


def u_gradient_analytic_at(x: np.ndarray) -> np.ndarray:
    """
    Exact du/dx along the Earth-Moon axis (0 < x < D_em), differentiating
    u_at directly instead of finite-differencing it. Needed because the
    physically meaningful step size (one lattice gap, ~L(x) itself, see
    dL_per_gap_at below) is many orders of magnitude smaller than any
    numerically safe central-difference dx at this coordinate scale.

    du/dx = G/c² * (-M_earth/x² + M_moon/(D_em-x)²) + ω²(x-x_bary)/c²
    """
    x = np.asarray(x, dtype=float)
    r_E = np.maximum(np.abs(x), 1e-3)
    r_M = np.maximum(np.abs(D_em - x), 1e-3)

    du_grav = G * (-M_earth / r_E**2 + M_moon / r_M**2) / c2
    du_rot = omega**2 * (x - x_bary) / c2
    return du_grav + du_rot


def L_gradient_at(x: np.ndarray, L_vacuum_lP: float = L_VACUUM_LP) -> np.ndarray:
    """dL/dx (Planck lengths per metre) from L(x) = L_vacuum*(1-u(x))."""
    return -L_vacuum_lP * u_gradient_analytic_at(x)


def dL_per_gap_at(x: np.ndarray, L_vacuum_lP: float = L_VACUUM_LP) -> np.ndarray:
    """
    Change in local cell size L across ONE lattice gap at position x, i.e.
    dL/dx evaluated over a step equal to the LOCAL cell size L(x) itself
    (converted to metres), not an arbitrary external length scale.

    This is the non-tautological replacement for
    `lattice_step_minimum_from_gps_gravitational_data()`: that function
    anchored on the externally-fixed Planck length (computed from a fixed
    c) and bolted a GPS clock-rate fraction onto it. Here the gap-to-gap
    step comes entirely from DST's own L(x)=L_vacuum*(1-u(x)) relation,
    driven by real Earth/Moon mass and orbit data -- c/L are treated as
    locally emergent throughout, never as one global anchor constant.
    """
    L_here_lP = throughput_at(x, L_vacuum_lP)
    dx_m = L_here_lP * L_PLANCK_M  # local gap width in metres
    return L_gradient_at(x, L_vacuum_lP) * dx_m


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def find_l1() -> dict:
    """
    Locate the L1 point (throughput maximum / u minimum) along the Earth-Moon
    axis between the two surfaces.

    Returns a dict with:
        L1_m        L1 position from Earth centre  [m]
        L1_km       L1 position  [km]
        u_L1        u at L1 (global minimum along axis)
        L_L1_lP     throughput rate at L1  [Planck lengths]
        u_surface   u at Earth surface
        u_gps       u at GPS orbit (Earth + 20 200 km)
        delta_u_gps u_surface − u_gps  (proportional to GPS clock gain)
    """
    x_grid = np.linspace(R_earth, D_em - R_moon, 1_000_000)
    u_grid = u_at(x_grid)
    L1_idx = int(np.argmin(u_grid))
    L1_m = float(x_grid[L1_idx])

    u_surface = float(u_at(R_earth))
    u_gps = float(u_at(R_earth + 20_200e3))

    return {
        "L1_m": L1_m,
        "L1_km": L1_m / 1e3,
        "u_L1": float(u_grid[L1_idx]),
        "L_L1_lP": float(throughput_at(L1_m)),
        "u_surface": u_surface,
        "u_gps": u_gps,
        "delta_u_gps": u_surface - u_gps,
    }


# ---------------------------------------------------------------------------
# Clock gain
# ---------------------------------------------------------------------------


def u_interior_at(x: np.ndarray) -> np.ndarray:
    """
    u(x) inside the Earth (0<=x<=R_earth): uniform-sphere interior potential,
    u_interior(r) = u_surface*(3-(r/R)^2)/2 -- the same GM/(rc^2) relation
    u_at uses, integrated over a uniform-density sphere instead of evaluated
    as a point mass. u_core = u_interior(0) = 1.5*u_surface falls straight
    out of this -- not an independent placeholder magnitude.
    """
    fraction = np.asarray(x, dtype=float) / R_earth
    u_surface = float(u_at(R_earth))
    return u_surface * (3.0 - fraction**2) / 2.0


def u_interior_gradient_at(x: np.ndarray) -> np.ndarray:
    """du/dx for u_interior_at -- zero at x=0 by symmetry."""
    u_surface = float(u_at(R_earth))
    return -u_surface * np.asarray(x, dtype=float) / R_earth**2


def _u_full_at(x: float) -> float:
    """u(x) for any x>=0, interior or exterior, one continuous function."""
    return float(u_interior_at(x)) if x < R_earth else float(u_at(x))


_U_CORE = None  # lazily computed -- avoids a second u_at(R_earth) call at import time


def _u_core() -> float:
    global _U_CORE
    if _U_CORE is None:
        _U_CORE = float(u_interior_at(0.0))
    return _U_CORE


def _clock_gain_us_per_day(x: float) -> float:
    """DST daily clock gain in μs at position x, relative to Earth's CORE
    (baseline=0 at the core, the system's own local minimum of throughput).
    f_DST(r) = 1 − u(r)   (linear in u)
    """
    return (_u_core() - _u_full_at(x)) * 86400 * 1e6


def _clock_gain_gr_us_per_day(x: float) -> float:
    """GR (Schwarzschild) daily clock gain in μs at position x, relative to
    Earth's CORE. f_GR(r) = sqrt(1 − 2u(r))   (weak-field Schwarzschild, no SR term)."""
    u_r = _u_full_at(x)
    u_c = _u_core()
    return (np.sqrt(1.0 - 2.0 * u_r) / np.sqrt(1.0 - 2.0 * u_c) - 1.0) * 86400 * 1e6


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------


# Milky Way escape-velocity correction (Gaia DR2/RAVE, v_esc~550 km/s at solar
# radius): Δc/c = v_esc^2/(2c^2) ~ 1.67e-6, i.e. C_VOID/C_LOCAL in
# DiscreteCell (inc/eq/core/data/lattice/cell/discrete_cell.hpp). Not modelled
# by u_at (Earth+Moon+rotation only) -- this is a much larger, separate-scale
# effect layered on top for the "Void" row. Guesswork/placeholder, per the
# same uncertainty caveat as C_VOID itself (~30-50% on the correction term).
MILKY_WAY_VOID_DU = 1.67e-6


GALILEO_ECCENTRIC_PERIGEE_ALT_M = (
    16_947e3  # GSAT0201/0202, real measured orbit (Delva et al. 2018)
)
GALILEO_ECCENTRIC_APOGEE_ALT_M = 26_252e3
GALILEO_GR_DEVIATION_1SIGMA = (
    2.48e-5  # measured |Δu_GR - Δu_predicted| / Δu_predicted, (0.19+/-2.48)e-5
)


def generate_substrate_topography():
    points = {
        "Earth Core": 0.0,
        "Earth Surface": R_earth,
        "LEO Orbit (model guess)": R_earth + 400e3,
        "GPS Orbit": R_earth + 20_200e3,
        "Galileo Perigee (measured)": R_earth + GALILEO_ECCENTRIC_PERIGEE_ALT_M,
        "Galileo Apogee (measured)": R_earth + GALILEO_ECCENTRIC_APOGEE_ALT_M,
        "GEO Orbit (model guess)": R_earth + 35_793e3,
        "HEO Orbit (model guess)": R_earth + 44_000e3,
        "Deep Space 100k (model guess)": 100_000e3,
        "Deep Space 200k (model guess)": 200_000e3,
        "Moon Surface (model guess)": D_em - R_moon,
        "Void (guesswork)": np.inf,
    }

    info = find_l1()
    points["L1 (max throughput)"] = info["L1_m"]

    # Earth's core (r=0, the same origin 1/r^2 is measured from) is the system's
    # actual local minimum of throughput -- its own discrete floor -- not the
    # classical GM/(rc^2)->0 convention at r->infinity. Anchor u=0 there instead.
    u_core = float(u_interior_at(0.0))

    u_peri = float(u_at(R_earth + GALILEO_ECCENTRIC_PERIGEE_ALT_M)) - u_core
    u_apo = float(u_at(R_earth + GALILEO_ECCENTRIC_APOGEE_ALT_M)) - u_core
    du_galileo_model = u_peri - u_apo

    print("\nDST Earth-Moon substrate topography")
    print(
        f"  Galileo perigee<->apogee Δu (model, GM/(rc²) only) = {du_galileo_model:.6e}\n"
        f"  Delva et al. 2018 measured deviation from GR       = "
        f"(0.19 ± {GALILEO_GR_DEVIATION_1SIGMA:.2e}) x Δu_GR, i.e. real orbit data,"
        f" not a formula extrapolation, constraining this Δu to ~{GALILEO_GR_DEVIATION_1SIGMA*100:.4f}%"
    )
    print(f"  u at Earth surface   = {info['u_surface']:.6e}")
    print(f"  u at GPS orbit       = {info['u_gps']:.6e}")
    print(
        f"  Δu (surface→GPS)     = {info['delta_u_gps']:.6e}"
        f"  ({info['delta_u_gps']*86400*1e6:.2f} μs/day)"
    )
    print(f"  L1 from Earth        = {info['L1_km']:,.1f} km")
    print(f"  u at L1              = {info['u_L1']:.6e}  (minimum u along axis)")
    print(
        f"  L at L1              = {info['L_L1_lP']:.6e} l_P" f"  (maximum throughput)"
    )

    # ΔL below vacuum in fm: ΔL = L_vacuum × u × L_Planck_m × 1e15
    L_P_fm = L_PLANCK_M * 1e15  # Planck length in fm

    name_w = max(len("Location"), max(len(name) for name in points))
    print(
        f"\n{'Location':<{name_w}} | {'Distance (km)':>20} | {'DST (μs/day)':>14}"
        f" | {'GR (μs/day)':>13} | {'Δ (ns/day)':>12}"
        f" | {'u (rel. core)':>14} | {'ΔL rel. core (fm)':>22} | {'dL per gap (l_P)':>18}"
    )
    print("-" * (name_w + 133))
    for name, x_m in sorted(points.items(), key=lambda kv: kv[1]):
        if np.isinf(x_m):
            # Void: Earth/Moon contribution vanishes (u->0); the Milky Way
            # escape-velocity term dominates and is layered on top here.
            dst_gain = _clock_gain_us_per_day(R_earth) + MILKY_WAY_VOID_DU * 86400e6
            gr_gain = _clock_gain_gr_us_per_day(R_earth) + MILKY_WAY_VOID_DU * 86400e6
            u_val = -MILKY_WAY_VOID_DU - u_core  # rel. to core, not to infinity
            dL_fm = L_VACUUM_LP * u_val * L_P_fm
            dL_gap = float(
                "nan"
            )  # not a point on the Earth-Moon axis -- no local gradient here
            dist_str = f"{'inf':>20}"
        else:
            dst_gain = _clock_gain_us_per_day(x_m)
            gr_gain = _clock_gain_gr_us_per_day(x_m)
            if x_m < R_earth:
                u_val = float(u_interior_at(x_m)) - u_core
                dudx = float(u_interior_gradient_at(x_m))
                L_here_lP = L_VACUUM_LP * (1.0 - u_val)
                dL_gap = (-L_VACUUM_LP * dudx) * (L_here_lP * L_PLANCK_M)
            else:
                u_val = float(u_at(x_m)) - u_core
                dL_gap = float(dL_per_gap_at(x_m))
            dL_fm = L_VACUUM_LP * u_val * L_P_fm
            dist_str = f"{x_m/1e3:>20,.1f}"
        diff_ns = (dst_gain - gr_gain) * 1e3  # μs → ns
        print(
            f"{name:<{name_w}} | {dist_str} | {dst_gain:>14.4f}"
            f" | {gr_gain:>13.4f} | {diff_ns:>12.4f}"
            f" | {u_val:>14.6e} | {dL_fm:>22.4f} | {dL_gap:>18.6e}"
        )


if __name__ == "__main__":
    generate_substrate_topography()
