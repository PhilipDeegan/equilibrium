"""
First pass at the new direction: a validated traveling-wave photon
transmitting through a schematic two-body (Sun+orbiting "Earth") lam
background, no hand-added coupling term, no sponge/damping -- the domain is
just sized so the transit completes before the wave could reach the outer
boundary. lam is recomputed each step from the orbiting body's current
position (the same lam0*R_body/r profile eddington.py/solar_system.py use,
just time-varying now).

Chirality test: run the identical pass twice, flipping the orbit's
rotation sense (omega -> -omega). A pure static-lensing effect is symmetric
under that flip; a genuine Lorentz/frame-dragging-type effect is not. This
is the sign-flip test every previous candidate mechanism in this project
has been missing, and it needs no picked coupling constant -- only the
already-validated lam->sigma->wave-speed mechanism.

Schematic/amplified scale (not real Sun-Earth SI values) -- first pass is
to see whether the mechanism exists in principle at all.
"""

import sys

sys.path.insert(0, ".")
import time

import numpy as np

from equilibrium.substrate.discrete import LAM_MAX
from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.traveling_wave import seed_traveling_wave

R = 15.0
HALF_LEN = 100.0
AXIS = 2

LAM0_SUN, R_SUN = 2.0, 3.0
LAM0_EARTH, R_EARTH = 0.5, 1.0
R_ORBIT = 6.0

Z0_WAVE = -60.0  # wave launched well before the Sun/Earth system (at z=0)
N_STEPS = 6000
LOG_EVERY = 100


def two_body_lam(pos, t, omega_orbit):
    r_sun = np.linalg.norm(pos, axis=1)
    safe_r_sun = np.where(r_sun > 0, r_sun, 1.0)
    lam_sun = np.where(r_sun > R_SUN, LAM0_SUN * R_SUN / safe_r_sun, LAM0_SUN)

    earth_pos = np.array([R_ORBIT * np.cos(omega_orbit * t), R_ORBIT * np.sin(omega_orbit * t), 0.0])
    r_earth = np.linalg.norm(pos - earth_pos[None, :], axis=1)
    safe_r_earth = np.where(r_earth > 0, r_earth, 1.0)
    lam_earth = np.where(r_earth > R_EARTH, LAM0_EARTH * R_EARTH / safe_r_earth, LAM0_EARTH)

    return np.clip(lam_sun + lam_earth, 0.0, LAM_MAX)


def run(omega_orbit, label):
    lat = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    p = seed_traveling_wave(lat, AXIS, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=4.0, z0=Z0_WAVE)
    dt = lat.stable_dt()
    v_phase = p["wavelength"] / p["T"]
    print(
        f"[{label}] N={lat.N} R={R} HALF_LEN={HALF_LEN} wavelength={p['wavelength']:.3f} "
        f"omega_wave={p['omega']:.4f} T={p['T']:.3f} v_phase={v_phase:.4f} dt={dt:.5f} "
        f"omega_orbit={omega_orbit:.5f} orbit_period={2*np.pi/abs(omega_orbit) if omega_orbit else float('inf'):.2f}",
        flush=True,
    )

    t = 0.0
    trace = []
    t_start = time.time()
    for step in range(N_STEPS):
        lat.lam = two_body_lam(lat.pos, t, omega_orbit)
        lat.step(dt)
        t += dt
        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            e = lat.u**2 + 0.5 * np.sum(lat.phi**2, axis=1)
            e_tot = float(np.sum(e))
            if e_tot > 0:
                cx = float(np.sum(e * lat.pos[:, 0]) / e_tot)
                cy = float(np.sum(e * lat.pos[:, 1]) / e_tot)
                cz = float(np.sum(e * lat.pos[:, 2]) / e_tot)
            else:
                cx = cy = cz = float("nan")
            trace.append((t, e_tot, cx, cy, cz))
            print(
                f"[{label}] step={step:6d} t={t:8.2f} E_tot={e_tot:.5f} "
                f"centroid=({cx:.4f},{cy:.4f},{cz:.4f}) wall={time.time()-t_start:6.1f}s",
                flush=True,
            )
    return np.array(trace)


def main():
    lat0 = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    p0 = seed_traveling_wave(lat0, AXIS, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=4.0, z0=Z0_WAVE)
    omega_wave = p0["omega"]
    omega_orbit = omega_wave / 30.0  # schematic: several orbits over the transit

    trace_pos = run(+omega_orbit, "omega=+")
    trace_neg = run(-omega_orbit, "omega=-")

    np.savez(
        "exp/2026/09/06/13_25_photon_rotating_background/chirality_traces.npz",
        trace_pos=trace_pos,
        trace_neg=trace_neg,
        omega_orbit=omega_orbit,
    )
    print("saved chirality_traces.npz", flush=True)


if __name__ == "__main__":
    main()
