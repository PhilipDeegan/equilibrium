"""
Earth core vs surface -- dimensionless, unit-free, INTEGER-tick version.

No seconds, no metres, no G, no c anywhere below -- and no floats standing in
for the ratio either. The substrate only has whole ticks (L_MIN=1 is a
floor, not a limit of a continuum), so a true core/surface relationship has
to bottom out as p:q for some integers p, q. A decimal like
"1.000000000529191" is not a physical answer -- it is a sign that whatever
tick count it was measured over is not the resolving scale, or that
u_surface was fed in as an arbitrary real number instead of an exact
fraction of floor units.

Everything here uses fractions.Fraction so nothing gets silently rounded
into looking like a physical result before it actually is one.

Core/surface potential ratio (uniform sphere, classical 1/r assumption):
    u_core / u_surface = 3/2         exact, independent of scale

DST throughput ratio (surface ticks elapsed per core tick), DST's own
linear f = 1 - u:
    f_DST(u_surface) = (1 - u_surface) / (1 - u_core)

This stays an EXACT rational number for any rational u_surface (ratios of
polynomials in a rational are rational) -- so DST's own throughput law is
structurally compatible with a pure integer-tick substrate: for any
rational u_surface there is a finite, exact (core_ticks, surface_ticks)
integer pair.

GR (Schwarzschild) throughput ratio, for comparison:
    f_GR(u_surface) = sqrt(1 - 2*u_surface) / sqrt(1 - 2*u_core)

This is generically IRRATIONAL even for rational u_surface (a square root
of a rational is almost never rational) -- so the GR law has no exact
integer-tick representation at all, except at specially chosen u_surface.
That is a real structural mismatch with a discrete-tick substrate, not a
rounding detail, and is kept as a float below to make it visible rather
than hiding it behind more decimal places.
"""

from fractions import Fraction
import math

CORE_SURFACE_U_RATIO = Fraction(3, 2)  # u_core / u_surface, uniform sphere + 1/r, exact


def u_core_of(u_surface: Fraction) -> Fraction:
    """u_core, given u_surface, under the uniform-sphere/1/r assumption."""
    return CORE_SURFACE_U_RATIO * u_surface


def throughput_ratio_dst(u_surface: Fraction) -> Fraction:
    """Surface ticks per core tick, DST's own f = 1 - u. Exact rational."""
    u_core = u_core_of(u_surface)
    return (1 - u_surface) / (1 - u_core)


def throughput_ratio_gr(u_surface: Fraction) -> float:
    """Surface ticks per core tick, GR's f = sqrt(1 - 2u). Generically irrational."""
    u_core = u_core_of(u_surface)
    return math.sqrt(1.0 - 2.0 * float(u_surface)) / math.sqrt(
        1.0 - 2.0 * float(u_core)
    )


def resolving_tick_pair(ratio: Fraction) -> tuple[int, int]:
    """
    Smallest (core_ticks, surface_ticks) integer pair such that
    surface_ticks / core_ticks == ratio EXACTLY, with zero remainder.
    Fraction is always stored reduced to lowest terms, so this is just
    (denominator, numerator) -- no search needed, no leftover decimal.
    """
    return (ratio.denominator, ratio.numerator)


if __name__ == "__main__":
    print("Earth core vs surface -- unit-free, integer ticks only")
    print(
        f"  u_core / u_surface (exact, uniform sphere + 1/r) = {CORE_SURFACE_U_RATIO}"
    )
    print()
    print("Real Earth core/surface, anchored to the classical GM/(rc^2) result")
    print("(the one place a real number enters -- borrowed from the continuum")
    print("1/r formula in earth_moon_gradient.py, then rationalized exactly):")
    print()
    import earth_moon_gradient as _classical  # sibling module, same directory

    u_surface_real = _classical.u_at(_classical.R_earth)
    u_surface_frac = Fraction(u_surface_real).limit_denominator(10**12)
    ratio_real = throughput_ratio_dst(u_surface_frac)
    core_ticks, surface_ticks = resolving_tick_pair(ratio_real)
    print(f"  u_surface (real, float)             = {u_surface_real:.10e}")
    print(f"  u_surface (rationalized, floor units) = {u_surface_frac}")
    print(
        f"  resolving integer tick pair          = core {core_ticks:,} : surface {surface_ticks:,}"
    )
    print(
        f"  surplus                              = {surface_ticks - core_ticks:,} ticks"
    )
    print(
        "  caveat: this tick count is where float64's ~2.2e-16 precision runs out\n"
        "  approximating the continuum u_surface as a fraction -- it is a real\n"
        "  anchor, not an arbitrary illustrative one, but it is not yet a\n"
        "  lattice-derived integer in its own right."
    )
