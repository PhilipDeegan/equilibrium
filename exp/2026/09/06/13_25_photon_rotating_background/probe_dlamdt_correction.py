"""
Empirical test of the candidate "rate of induction" term derived by treating
u/sigma, phi/sigma as conserved (charge/flux-linkage) quantities:

    du/dt   +=  - u   * sigma * dlam/dt
    dphi/dt +=  - phi * sigma * dlam/dt

dlam/dt computed by plain finite difference (lam_next - lam_prev)/dt, the
form that would translate directly into a C++ addition (track lam_prev,
subtract, divide by dt) -- NOT an analytic derivative of the convenience
orbit formula, since a real substrate lam field wouldn't have one.

Philip's objection, not yet resolved analytically: lam/sigma is a substrate
metric (sigma_a/sigma_b IS the time-dilation ratio between cells,
equilibrium/substrate/discrete.py), so "the" conserved quantity is really
just total energy of the whole substrate -- not a borrowed circuit-charge
analogy. This script sidesteps that by testing empirically: does adding
this term (a) change total-energy conservation properties, and (b) produce
any effect on the wave beyond the pure-retardation baseline (no correction)?
Purely exploratory -- meant to give a positive/negative signal before
touching the C++ substrate at all.
"""

import sys

sys.path.insert(0, ".")
import numpy as np

from equilibrium.substrate.discrete import LAM_MAX
from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.traveling_wave import seed_traveling_wave

R = 12.0
HALF_LEN = 70.0
AXIS = 2

LAM0_SUN, R_SUN = 2.0, 3.0
LAM0_EARTH, R_EARTH = 0.5, 1.0
R_ORBIT = 6.0

Z0_WAVE = -40.0
N_STEPS = 3000
LOG_EVERY = 50


def two_body_lam(pos, t, omega_orbit):
    r_sun = np.linalg.norm(pos, axis=1)
    safe_r_sun = np.where(r_sun > 0, r_sun, 1.0)
    lam_sun = np.where(r_sun > R_SUN, LAM0_SUN * R_SUN / safe_r_sun, LAM0_SUN)

    earth_pos = np.array([R_ORBIT * np.cos(omega_orbit * t), R_ORBIT * np.sin(omega_orbit * t), 0.0])
    r_earth = np.linalg.norm(pos - earth_pos[None, :], axis=1)
    safe_r_earth = np.where(r_earth > 0, r_earth, 1.0)
    lam_earth = np.where(r_earth > R_EARTH, LAM0_EARTH * R_EARTH / safe_r_earth, LAM0_EARTH)

    return np.clip(lam_sun + lam_earth, 0.0, LAM_MAX)


def energy(lat):
    return float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))


def run(omega_orbit, apply_correction, label):
    lat = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    p = seed_traveling_wave(lat, AXIS, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=3.0, z0=Z0_WAVE)
    dt = lat.stable_dt()

    lam_prev = two_body_lam(lat.pos, 0.0, omega_orbit)
    lat.lam = lam_prev.copy()
    h0 = energy(lat)

    t = 0.0
    trace = []
    for step in range(N_STEPS):
        t_next = t + dt
        lam_next = two_body_lam(lat.pos, t_next, omega_orbit)

        if apply_correction:
            sigma = 1.0 / (1.0 + lat.lam)
            dlam_dt = (lam_next - lam_prev) / dt
            lat.u = lat.u - dt * sigma * dlam_dt * lat.u
            lat.phi = lat.phi - dt * (sigma * dlam_dt)[:, None] * lat.phi

        lat.lam = lam_next
        lat.step(dt)
        lam_prev = lam_next
        t = t_next

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            e = lat.u**2 + 0.5 * np.sum(lat.phi**2, axis=1)
            e_tot = float(np.sum(e))
            cx = float(np.sum(e * lat.pos[:, 0]) / e_tot) if e_tot > 0 else float("nan")
            cy = float(np.sum(e * lat.pos[:, 1]) / e_tot) if e_tot > 0 else float("nan")
            cz = float(np.sum(e * lat.pos[:, 2]) / e_tot) if e_tot > 0 else float("nan")
            trace.append((t, e_tot / h0, cx, cy, cz))

    trace = np.array(trace)
    print(
        f"[{label}] final: H_ratio={trace[-1,1]:.6f}  centroid=({trace[-1,2]:.4f},{trace[-1,3]:.4f},{trace[-1,4]:.4f})",
        flush=True,
    )
    return trace


def main():
    lat0 = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    p0 = seed_traveling_wave(lat0, AXIS, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=3.0, z0=Z0_WAVE)
    omega_orbit = p0["omega"] / 30.0
    print(f"N={lat0.N} omega_orbit={omega_orbit:.5f} orbit_period={2*np.pi/omega_orbit:.2f}", flush=True)

    results = {}
    for sign, sname in ((+1, "plus"), (-1, "minus")):
        for corr, cname in ((False, "baseline"), (True, "corrected")):
            trace = run(sign * omega_orbit, corr, f"{sname}/{cname}")
            results[f"{sname}_{cname}"] = trace

    np.savez("exp/2026/09/06/13_25_photon_rotating_background/dlamdt_traces.npz", **results)
    print("saved dlamdt_traces.npz", flush=True)

    print("\n=== summary: final centroid_y, by omega sign, with/without correction ===")
    for cname in ("baseline", "corrected"):
        cy_p = results[f"plus_{cname}"][-1, 3]
        cy_m = results[f"minus_{cname}"][-1, 3]
        print(f"{cname:>10}: cy(+omega)={cy_p:+.5f}  cy(-omega)={cy_m:+.5f}  sum(should be ~0 if antisymmetric)={cy_p+cy_m:+.6f}")


if __name__ == "__main__":
    main()
