"""
Double-slit interference on the Kelvin cell substrate.

Setup
-----
The domain is NOT vacuum with two special objects dropped in -- it is a
uniform background medium (gas_lam everywhere, low but non-zero: real
atmosphere has some proton density, just far less than a solid) with two
denser (higher-lam) solid objects sitting in it:

  - a slitted wall (wall_lam) at x=x_wall, pierced by one or two circular
    apertures. The apertures are NOT special "lam=0" holes -- they are
    simply the ABSENCE of the wall, i.e. ordinary background gas_lam, same
    as everywhere else in the domain,
  - a detector backplate (screen_lam) further along +x, a full solid slab
    with no apertures, representing the photographic-plate/sensor material
    that the propagating field is recorded against.

None of the three regions are imposed boundary conditions -- they are all
just different lam values feeding the same sigma = 1/(1+lam) update rule
everywhere. sigma scales both legs of step() (u from phi and phi from u), so
lam changes the local wave speed but not the impedance: a lam contrast is an
index contrast (a phase screen), not a mirror or an opaque body. In the
continuum no sigma profile reflects at normal incidence; what reflection a
thick high-lam slab shows on the lattice comes from the wave inside it being
unresolved (wavelength / (1 + lam) << one cell). Opacity and reflection in
real matter come from structure (free charge in a metal), not mass density.

At the defaults the slab is a single BCC layer (x=0) and each aperture is one
cell; the square-face links (dx = 1.155) cross a single layer directly.

The recorded pattern is the accumulated dose (integral of u^2 over the run)
at the entry face of the backplate slab.

Single slit gives a smooth diffraction envelope (no fringes) on the plate.
Two slits give the envelope modulated by interference fringes whose
spacing follows the same path-length-difference geometry as any other
wave theory -- no separate "quantum" rule is invoked, this is the same
local diffusion/wave update rule used everywhere else in this package.

Source: a continuous driven oscillator, not a single pulse
------------------------------------------------------------
A single Gaussian pulse (broadband -- one shot, many frequencies at once)
does not give a clean multi-fringe Young's pattern: there aren't enough
oscillation cycles reaching the screen for the path-length-difference
condition to pick out repeated minima/maxima, so a single-pulse run mostly
just shows a same-shape, uniformly-scaled-up two-slit envelope over the
one-slit one. Real Young's-fringe demonstrations always use continuous,
quasi-monochromatic illumination (a laser, or a small aperture fed from a
steady source) for exactly this reason. The source here is a thin slab
held at a forced, continuously oscillating u = A sin(omega t) for the
whole run -- a driven boundary (an external emitter), same role as an
antenna current in an EM simulation. Everything past the source (wall,
slits, propagation, the backplate) is still the ordinary, un-forced
update rule.
"""

import numpy as np

from .lattice import DampedLattice, SubstrateLattice, wave_speed


def _wall_cells(
    lat: SubstrateLattice,
    x_wall: float,
    thickness: float,
    slit_centres: list,
    slit_radius: float,
) -> np.ndarray:
    """Indices of cells forming the wall, excluding circular apertures
    (apertures are left at background gas_lam, not specially zeroed)."""
    in_slab = np.abs(lat.pos[:, 0] - x_wall) <= thickness / 2.0
    open_mask = np.zeros(lat.N, dtype=bool)
    for cy, cz in slit_centres:
        dy = lat.pos[:, 1] - cy
        dz = lat.pos[:, 2] - cz
        open_mask |= np.sqrt(dy**2 + dz**2) <= slit_radius
    return np.where(in_slab & ~open_mask)[0]


def _screen_cells(
    lat: SubstrateLattice, x_screen: float, thickness: float
) -> np.ndarray:
    """Indices of cells forming the (solid, no-aperture) detector backplate."""
    return np.where(np.abs(lat.pos[:, 0] - x_screen) <= thickness / 2.0)[0]


def _source_cells(lat: SubstrateLattice, x0: float, thickness: float) -> np.ndarray:
    """Indices of the thin slab acting as the driven (emitter) source."""
    return np.where(np.abs(lat.pos[:, 0] - x0) <= thickness / 2.0)[0]


def run(
    radius_cells: float = 18.0,
    spacing: float = 1.0,
    gas_lam: float = 0.05,
    x_wall: float = 0.0,
    wall_thickness: float = 1.0,
    wall_lam: float = 300.0,
    slit_separation: float = 6.0,
    slit_radius: float = 1.0,
    n_slits: int = 2,
    x_screen: float = 10.0,
    screen_thickness: float = 1.0,
    screen_lam: float = 300.0,
    pulse_x: float = -5.0,
    pulse_thickness: float = 1.0,
    drive_amplitude: float = 1.0,
    wavelength: float = 4.68,
    travel: float = 30.0,
    drive_period_steps: float | None = None,
    n_steps: int | None = None,
    sponge_fraction: float = 0.15,
    damp_max: float = 0.01,
):
    """
    Run one double-slit (or single-slit, n_slits=1) diffraction experiment.

    The source is a thin slab at x=pulse_x, forced every step to
    u = drive_amplitude * sin(2*pi*step/drive_period_steps) -- a continuous,
    quasi-monochromatic emitter (see module docstring for why a single
    pulse doesn't show clean fringes).

    The drive is set by its wavelength (lattice units) and the run length by
    `travel`, the distance a front covers (c = lattice.wave_speed()), so the
    experiment does not change with the face weights or dt. The defaults
    (4.68, 30) are what drive_period_steps = 250 / n_steps = 1600 gave at the
    pre-2026-09-28 W = 3:1 (omega ~ 4.9 there). omega must stay well below the
    dispersion cutoff (12.5 at W = 3:1, 6.93 at sqrt3:1) -- above it the wave
    is evanescent. drive_period_steps / n_steps, if given, override.

    The domain uses DampedLattice (sponge absorbing boundary) rather than
    a hard-reflecting outer edge -- with a continuously driven source run
    for many steps, reflections off a hard boundary would otherwise arrive
    back at the screen and contaminate the pattern.

    Returns (lat, dose) where `lat` is the SubstrateLattice after n_steps
    and `dose` is the per-cell accumulated exposure (integral of u^2 dt)
    at the backplate cells, aligned with lat.pos[screen_idx].
    """
    lat = DampedLattice(
        radius_cells=radius_cells,
        spacing=spacing,
        sponge_fraction=sponge_fraction,
        damp_max=damp_max,
    )
    lat.lam[:] = gas_lam

    if n_slits == 1:
        slit_centres = [(0.0, 0.0)]
    elif n_slits == 2:
        slit_centres = [(slit_separation / 2.0, 0.0), (-slit_separation / 2.0, 0.0)]
    else:
        raise ValueError("n_slits must be 1 or 2")

    wall = _wall_cells(lat, x_wall, wall_thickness, slit_centres, slit_radius)
    lat.lam[wall] = wall_lam

    x_max = lat.pos[:, 0].max()
    x_screen = x_screen if x_screen is not None else x_max * 0.8
    screen = _screen_cells(lat, x_screen, screen_thickness)
    lat.lam[screen] = screen_lam

    x_min = lat.pos[:, 0].min()
    px = pulse_x if pulse_x is not None else x_min * 0.6 + x_wall * 0.4
    source = _source_cells(lat, px, pulse_thickness)

    dt = lat.stable_dt()
    c = wave_speed(spacing)
    if drive_period_steps is None:
        drive_period_steps = wavelength / (c * dt)
    if n_steps is None:
        n_steps = int(np.ceil(travel / (c * dt)))
    omega = 2.0 * np.pi / (drive_period_steps * dt)
    dose = np.zeros(len(screen), dtype=np.float64)
    # source-side sum u^2, averaged over the last drive period (a snapshot
    # measures the standing-wave phase, not the reflected energy)
    src_side = lat.pos[:, 0] < x_wall - 3.0
    n_avg = max(1, int(round(drive_period_steps)))
    src_energy = 0.0
    for step_i in range(n_steps):
        t = step_i * dt
        lat.u[source] = drive_amplitude * np.sin(omega * t)
        lat.step(dt)
        dose += (lat.u[screen] ** 2) * dt
        if step_i >= n_steps - n_avg:
            src_energy += float(np.sum(lat.u[src_side] ** 2))
    lat._source_side_energy = src_energy / n_avg  # noqa: SLF001

    lat._screen_idx = screen  # noqa: SLF001 -- convenience for detector helpers below
    lat._wall_idx = wall
    lat._source_idx = source
    lat._slit_centres = slit_centres
    lat._slit_radius = slit_radius
    return lat, dose


def screen_dose_slice(
    lat: SubstrateLattice,
    dose: np.ndarray,
    z_thickness: float = 1.0,
    front_face_only: bool = True,
) -> tuple:
    """
    (y, dose) at the backplate, sorted by y -- the recorded fringe pattern.

    Restricted to |z| <= z_thickness so the result is a clean 1-D slice
    through the pattern's centre rather than every (y, z) cell scattered
    together (the backplate is a full 2-D plane, not a 1-D line).

    The BCC lattice interleaves depth sublattices at slightly different x
    within any finite-thickness slab (e.g. x=2.887 and x=3.464 both fall
    inside a screen_thickness=1.0 band around x_screen=3.0) -- these are
    genuinely different depths with very different dose (the far one has
    had the wave travel further after entering the plate), so mixing them
    by y alone gives a sawtooth that conflates two different slices, not
    real fringe structure. front_face_only restricts to the single x layer
    closest to the wall -- the plate's entry face, which is what actually
    first records the incident pattern.
    """
    pos = lat.pos[lat._screen_idx]
    mask = np.abs(pos[:, 2]) <= z_thickness
    if front_face_only and np.any(mask):
        x_front = pos[mask, 0].min()
        mask &= np.abs(pos[:, 0] - x_front) <= 1e-6
    y = pos[mask, 1]
    d = dose[mask]
    order = np.argsort(y)
    return y[order], d[order]


def screen_leakage(lat: SubstrateLattice, dose: np.ndarray) -> tuple:
    """
    (dose_behind_slit_line, dose_behind_wall_body): mean accumulated dose
    at backplate cells whose y,z falls behind an aperture vs behind the
    solid part of the wall. The latter is non-zero (the slab delays and
    transmits, it is not a hard fixed_u=0 block) but well below the former.
    """
    y = lat.pos[lat._screen_idx, 1]
    z = lat.pos[lat._screen_idx, 2]
    open_mask = np.zeros(len(y), dtype=bool)
    for cy, cz in lat._slit_centres:
        open_mask |= np.sqrt((y - cy) ** 2 + (z - cz) ** 2) <= lat._slit_radius
    behind_slit = float(np.mean(dose[open_mask])) if np.any(open_mask) else 0.0
    behind_body = float(np.mean(dose[~open_mask])) if np.any(~open_mask) else 0.0
    return behind_slit, behind_body


def reflected_slice(
    lat: SubstrateLattice,
    x_source_side: float,
    thickness: float = 2.0,
    z_thickness: float = 1.0,
) -> tuple:
    """
    u values on the SOURCE side of the wall, at x=x_source_side (< x_wall).
    Includes the outgoing drive, so non-zero here is not by itself
    evidence of reflection.
    """
    mask = (np.abs(lat.pos[:, 0] - x_source_side) <= thickness) & (
        np.abs(lat.pos[:, 2]) <= z_thickness
    )
    y = lat.pos[mask, 1]
    u = lat.u[mask]
    order = np.argsort(y)
    return y[order], u[order]
