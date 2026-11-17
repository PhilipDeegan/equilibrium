"""
Gravitational time dilation from DST local throughput rate.

In DST the equilibrium u-field outside a spherical mass satisfies ∇²u = 0,
giving u(r) = GM/(r c²)  (= rs / 2r in Planck units, rs = 2GM/c²).

Local throughput rate at radius r:
    L(r) = L_vacuum × (1 − u(r))  =  L_vacuum × (1 − GM/(rc²))

The throughput rate sets how rapidly the substrate at a given location can
exchange flux with its neighbours.  High u (near a mass) → reduced throughput
→ slower local clocks.  Low u (void) → maximum throughput → fastest clocks.

Local clock rate (relative to vacuum reference):
    f(r) = L(r) / L_vacuum = 1 − GM/(rc²)

This is the first-order Schwarzschild weak-field result.  For Earth
(GM/Rc² ≈ 7e-10), second-order corrections are ~2e-19 — completely negligible.

IMPORTANT — vacuum reference layers
────────────────────────────────────
"L_vacuum" is not a single universal value.  The substrate throughput is
suppressed by every gravitational source, giving a nested hierarchy of vacuum
states:

  L_extragalactic  (cosmic void between clusters)   ← true asymptotic reference
      − galaxy potential (φ ≈ 5.4e-7)  →  L_intragalactic  ≈ L_extragalactic − 8.7 pm
      − solar potential  (φ ≈ 9.9e-9)  →  L_solar_system   ≈ L_intragalactic − 160 fm
      − Earth potential  (φ ≈ 7.0e-10) →  L_earth_surface  ≈ L_solar_system  −  11 fm
      − altitude effect  (φ Δ ≈ 5.3e-10 sea→GPS)  →  8.6 fm variation

All Earth-based experiments (GPS, Pound-Rebka, atomic clocks) measure variations
relative to the local intragalactic+solar background — not the extragalactic void.

Experimental anchors:
    Pound-Rebka (1959)  : first direct gravitational redshift measurement
    Hafele-Keating 1971 : atomic clocks on aircraft, Δf/f ≈ g·h/c²
    GPS (ongoing)       : gravitational gain ≈ +45.9 μs/day at 20 200 km
"""

import numpy as np

# SI constants
_G = 6.674e-11  # m³ kg⁻¹ s⁻²
_C = 2.998e8  # m/s
_C2 = _C**2  # m²/s²

# Earth
M_EARTH = 5.972e24  # kg
R_EARTH = 6.371e6  # m  (mean radius)
G_SURFACE = 9.80665  # m/s²  (standard surface gravity)

# Gravitational length: GM_E/c²  (half Schwarzschild radius of Earth)
GM_EARTH_OVER_C2 = _G * M_EARTH / _C2  # ≈ 4.435e-3 m

# Sun
M_SUN = 1.989e30  # kg
AU = 1.496e11  # m  (1 astronomical unit — Earth–Sun distance)

# Milky Way: mass enclosed within solar orbit, derived from circular velocity
# v_circ ≈ 220 km/s at R_gc ≈ 8.5 kpc  →  M_enclosed = v²R/G
_V_CIRC_SUN = 220e3  # m/s
R_GALACTIC = 8.5e3 * 3.086e16  # m  (8.5 kpc in metres)
M_GALAXY = _V_CIRC_SUN**2 * R_GALACTIC / _G  # ≈ 1.9e41 kg ≈ 10^11 M_sun

#: Planck length in metres
L_PLANCK_M = 1.616e-35

#: CLAWD deep-space reference throughput rate (Planck lengths).
#: In the DST multi-resolution picture, vacuum throughput far from any mass
#: is expected to be ~ 10^30 l_P ≈ 16 μm.  This is the scale at which
#: altitude-dependent throughput variations fall in the femtometre range.
#: NOTE: this is the *extragalactic* reference.  Intragalactic vacuum is
#: already suppressed by ~8.7 pm relative to this.
L_VACUUM_CLAWD = 1.0e30


def phi(r_m: float, mass_kg: float = M_EARTH) -> float:
    """Dimensionless gravitational potential φ = GM/(rc²) at radius r (metres)."""
    return _G * mass_kg / (r_m * _C2)


def clock_rate(r_m: float, mass_kg: float = M_EARTH) -> float:
    """
    DST local clock rate at radius r from mass centre.
    f(r) = 1 − GM/(rc²)   [first-order Schwarzschild, c=1 in Planck units]
    Equivalently: L(r) / L_vacuum — the local throughput rate normalised to vacuum.
    """
    return 1.0 - phi(r_m, mass_kg)


def altitude_dilation(
    h_m: float,
    r_surface_m: float = R_EARTH,
    mass_kg: float = M_EARTH,
) -> float:
    """
    Fractional clock-rate gain between sea level and altitude h_m.
    Returns Δf/f = f(r+h)/f(r) − 1.  Positive → altitude clock runs faster.
    """
    f_sea = clock_rate(r_surface_m, mass_kg)
    f_alt = clock_rate(r_surface_m + h_m, mass_kg)
    return f_alt / f_sea - 1.0


def newtonian_approx(h_m: float) -> float:
    """Flat-Earth approximation: Δf/f ≈ g·h/c².  Valid for h << R_E."""
    return G_SURFACE * h_m / _C2


# ---------------------------------------------------------------------------
# Multi-source vacuum hierarchy
# ---------------------------------------------------------------------------


def phi_sources() -> dict:
    """
    Dimensionless gravitational potential contributions (φ = GM/rc²) at Earth's
    surface from each dominant source.

    The galaxy term uses mass enclosed within the solar orbit, derived from
    the circular velocity — it includes both baryonic mass and dark matter.
    """
    return {
        "galaxy": _G * M_GALAXY / (R_GALACTIC * _C2),
        "sun": _G * M_SUN / (AU * _C2),
        "earth": phi(R_EARTH),
    }


def phi_total_at_earth(h_m: float = 0.0) -> float:
    """
    Total gravitational potential at altitude h_m above Earth's surface,
    summing galaxy (background) + Sun (background) + Earth (varying with h).
    Galaxy and Sun are treated as constants over terrestrial altitudes.
    """
    src = phi_sources()
    return src["galaxy"] + src["sun"] + phi(R_EARTH + h_m)


def throughput_hierarchy(L_vacuum_lP: float = L_VACUUM_CLAWD) -> dict:
    """
    Throughput suppression at each level of the vacuum hierarchy, relative to
    the extragalactic reference (L_vacuum_lP).

    Returns dict of {label: delta_L_in_metres} — the reduction in throughput
    rate at each nesting level.
    """
    src = phi_sources()
    L_P = L_PLANCK_M
    phi_gps = phi(R_EARTH + 20_200e3)

    return {
        "galaxy (extragalactic→intragalactic)": L_vacuum_lP * src["galaxy"] * L_P,
        "sun (intragalactic→solar system)": L_vacuum_lP * src["sun"] * L_P,
        "earth (solar system→surface)": L_vacuum_lP * src["earth"] * L_P,
        "altitude (sea level→GPS orbit)": L_vacuum_lP * (src["earth"] - phi_gps) * L_P,
    }


def print_hierarchy(L_vacuum_lP: float = L_VACUUM_CLAWD):
    """Print the full vacuum-to-surface throughput suppression cascade."""
    src = phi_sources()
    L_P = L_PLANCK_M

    print(
        f"\nVacuum reference : L_extragalactic = {L_vacuum_lP:.2e} l_P"
        f" = {L_vacuum_lP * L_P * 1e6:.3f} μm"
    )
    print(f"\n{'Source':45s}  {'φ':>12}  {'ΔL (l_P)':>14}  {'ΔL':>12}")
    print("-" * 90)

    labels_phi = [
        ("galaxy  (extragal. → intragal.)", src["galaxy"]),
        ("sun     (intragal. → solar sys.)", src["sun"]),
        ("earth   (solar sys. → surface)", src["earth"]),
    ]
    for label, p in labels_phi:
        dL_lP = L_vacuum_lP * p
        dL_m = dL_lP * L_P
        if dL_m >= 1e-12:
            unit = f"{dL_m*1e12:.3f} pm"
        elif dL_m >= 1e-15:
            unit = f"{dL_m*1e15:.2f} fm"
        else:
            unit = f"{dL_m:.3e} m"
        print(f"  {label:43s}  {p:>12.4e}  {dL_lP:>14.3e}  {unit:>12}")

    # altitude sub-entry
    phi_sea = src["earth"]
    phi_gps = phi(R_EARTH + 20_200e3)
    dphi = phi_sea - phi_gps
    dL_lP = L_vacuum_lP * dphi
    dL_m = dL_lP * L_P
    print(
        f"    altitude (sea level → GPS)  "
        f"  Δφ={dphi:>12.4e}  {dL_lP:>14.3e}  {dL_m*1e15:>9.2f} fm"
    )


def print_table(altitudes_m=(1e3, 1e4, 2.02e7, 3.844e8)):
    """Compare DST prediction vs g·h/c² approximation at several altitudes."""
    print(f"\n{'altitude (m)':>14}  {'DST Δf/f':>14}  {'g·h/c²':>14}  {'ratio':>8}")
    print("-" * 58)
    for h in altitudes_m:
        dst = altitude_dilation(h)
        appr = newtonian_approx(h)
        rat = dst / appr if appr else float("nan")
        print(f"{h:>14.4g}  {dst:>14.4e}  {appr:>14.4e}  {rat:>8.5f}")


# ---------------------------------------------------------------------------
# Throughput rate profiles
# ---------------------------------------------------------------------------


def throughput_profile(
    altitudes_m,
    L_vacuum_lP: float = L_VACUUM_CLAWD,
    r_surface_m: float = R_EARTH,
    mass_kg: float = M_EARTH,
) -> np.ndarray:
    """
    DST local throughput rate L at each altitude above r_surface_m.

    Parameters
    ----------
    altitudes_m  : scalar or array of altitudes above the surface (metres)
    L_vacuum_lP  : vacuum (deep-space) throughput rate in Planck lengths

    Returns
    -------
    L : array of throughput rates in Planck lengths
        L(r) = L_vacuum × (1 − GM/(rc²))
    """
    alts = np.asarray(altitudes_m, dtype=float)
    r = r_surface_m + alts
    return L_vacuum_lP * (1.0 - _G * mass_kg / (r * _C2))


def throughput_gradient(
    L_vacuum_lP: float = L_VACUUM_CLAWD,
    r_surface_m: float = R_EARTH,
    mass_kg: float = M_EARTH,
) -> float:
    """
    Rate of change of throughput rate with altitude at the surface, in
    femtometres per km.

    dL/dr = L_vacuum × g / c²
    """
    g_eff = _G * mass_kg / r_surface_m**2  # m/s²
    dL_lP = L_vacuum_lP * g_eff / _C2  # l_P per metre
    return dL_lP * L_PLANCK_M * 1e15 * 1e3  # fm per km


def print_throughput(
    altitudes_m=(0, 1e3, 1e4, 100e3, 2.02e7, 3.844e8),
    L_vacuum_lP: float = L_VACUUM_CLAWD,
):
    """
    Print DST throughput rates at various altitudes, relative to sea level.

    Columns: altitude, L in Planck lengths, ΔL vs sea level in l_P and fm.
    """
    L_sea = throughput_profile(0.0, L_vacuum_lP)[()]
    L_arr = throughput_profile(np.asarray(altitudes_m), L_vacuum_lP)
    grad = throughput_gradient(L_vacuum_lP)

    print(
        f"\nVacuum throughput rate : {L_vacuum_lP:.2e} l_P = {L_vacuum_lP * L_PLANCK_M * 1e6:.3f} μm"
    )
    print(
        f"Sea-level deficit      : {L_vacuum_lP * phi(R_EARTH):.3e} l_P"
        f" = {L_vacuum_lP * phi(R_EARTH) * L_PLANCK_M * 1e15:.2f} fm below vacuum"
    )
    print(f"Surface gradient       : {grad:.4f} fm / km\n")

    print(
        f"{'altitude (m)':>14}  {'L (l_P)':>14}  {'ΔL vs sea (l_P)':>18}  {'ΔL (fm)':>10}"
    )
    print("-" * 65)
    labels = {
        0: "sea level",
        1e3: "1 km",
        1e4: "10 km",
        8848: "Everest",
        100e3: "100 km",
        2.02e7: "GPS",
        3.844e8: "Moon",
    }
    for h, L in zip(altitudes_m, L_arr):
        dL_lP = L - L_sea
        dL_fm = dL_lP * L_PLANCK_M * 1e15
        lab = labels.get(h, f"h={h:.3g}m")
        print(f"{h:>14.4g}  {L:>14.6e}  {dL_lP:>18.4e}  {dL_fm:>10.3f}" f"  {lab}")
