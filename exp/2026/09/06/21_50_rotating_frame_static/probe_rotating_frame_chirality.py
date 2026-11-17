"""
Clean rotating-frame chirality test, replacing the confounded lab-frame
orbiting picture (probe_photon_rotating_lam.py, 2026-09-06 afternoon).

Sun+Earth are BOTH static, at fixed positions in the rotating frame -- no
orbit motion, no time-dependent lam at all. The only thing that changes
between the two runs is the sign of Omega in the two rotating-frame
correction terms (equilibrium.substrate.rotating_frame, derived in `now`,
2026-09-06 night entry):

    du[i]    += -dt * (Omega x x_i)      . grad(u)_i
    phi[i,k] += -dt * (Omega x Phi_vec_i) . offsets[k]

Since nothing about the IC/geometry moves, this is the "same position,
different sense" comparison the lab-frame orbiting picture could not
construct (there, flipping omega necessarily mirrored the source's
position too) -- a genuine Ampere/Lorentz-type coupling should show up as
an antisymmetric response under Omega -> -Omega with everything else held
fixed; a retardation/lensing artifact could not, since nothing here
propagates or lags.
"""

import sys

sys.path.insert(0, ".")

import numpy as np

from equilibrium.substrate.discrete import LAM_MAX
from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.rotating_frame import (
    apply_rotating_frame_correction,
    face_design,
    light_cylinder_radius,
)
from equilibrium.substrate.traveling_wave import face_offsets, seed_traveling_wave

R = 15.0
HALF_LEN = 100.0
AXIS = 2

LAM0_SUN, R_SUN = 2.0, 3.0
LAM0_EARTH, R_EARTH = 0.5, 1.0
R_ORBIT = 6.0  # Earth's FIXED rotating-frame position, on the +x axis

Z0_WAVE = -60.0
N_STEPS = 6000
LOG_EVERY = 100

C_LATTICE = np.sqrt(10.0)  # sqrt(W_HEX^2*8 + W_SQ^2*6)/... == sqrt(9+1), spacing=1


def static_two_body_lam(pos):
    r_sun = np.linalg.norm(pos, axis=1)
    safe_r_sun = np.where(r_sun > 0, r_sun, 1.0)
    lam_sun = np.where(r_sun > R_SUN, LAM0_SUN * R_SUN / safe_r_sun, LAM0_SUN)

    earth_pos = np.array([R_ORBIT, 0.0, 0.0])
    r_earth = np.linalg.norm(pos - earth_pos[None, :], axis=1)
    safe_r_earth = np.where(r_earth > 0, r_earth, 1.0)
    lam_earth = np.where(r_earth > R_EARTH, LAM0_EARTH * R_EARTH / safe_r_earth, LAM0_EARTH)

    return np.clip(lam_sun + lam_earth, 0.0, LAM_MAX)


def run(omega, label):
    lat = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    p = seed_traveling_wave(lat, AXIS, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=4.0, z0=Z0_WAVE)
    lat.lam = static_two_body_lam(lat.pos)  # set ONCE -- genuinely time-independent
    dt = lat.stable_dt()

    offsets, _ = face_offsets(lat)
    hop_len, pinv = face_design(offsets)
    omega_vec = np.array([0.0, 0.0, omega])

    t = 0.0
    trace = []
    for step in range(N_STEPS):
        apply_rotating_frame_correction(lat, offsets, hop_len, pinv, omega_vec, dt)
        lat.step_cpp(dt, n_threads=8)
        t += dt
        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            e = lat.u**2 + 0.5 * np.sum(lat.phi**2, axis=1)
            e_tot = float(np.sum(e))
            cx = float(np.sum(e * lat.pos[:, 0]) / e_tot) if e_tot > 0 else float("nan")
            cy = float(np.sum(e * lat.pos[:, 1]) / e_tot) if e_tot > 0 else float("nan")
            cz = float(np.sum(e * lat.pos[:, 2]) / e_tot) if e_tot > 0 else float("nan")
            trace.append((t, e_tot, cx, cy, cz))
            print(
                f"[{label}] step={step:6d} t={t:8.2f} E_tot={e_tot:.5f} "
                f"centroid=({cx:.4f},{cy:.4f},{cz:.4f})",
                flush=True,
            )
    return np.array(trace)


def main():
    lat0 = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    p0 = seed_traveling_wave(lat0, AXIS, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=4.0, z0=Z0_WAVE)
    omega = p0["omega"] / 30.0  # same schematic scale as the earlier (confounded) test

    r_lc = light_cylinder_radius(C_LATTICE, omega)
    print(f"N={lat0.N} R={R} omega={omega:.5f} r_lc={r_lc:.2f} (domain R={R}, {'OK' if R < r_lc else 'VIOLATED'})", flush=True)
    assert R < r_lc, f"domain radius {R} exceeds light-cylinder radius {r_lc:.2f} -- rotating-frame description invalid here"

    trace_pos = run(+omega, "omega=+")
    trace_neg = run(-omega, "omega=-")

    np.savez(
        "exp/2026/09/06/21_50_rotating_frame_static/chirality_traces.npz",
        trace_pos=trace_pos,
        trace_neg=trace_neg,
        omega=omega,
    )
    print("saved chirality_traces.npz", flush=True)

    cy_p, cy_m = trace_pos[-1, 3], trace_neg[-1, 3]
    cx_p, cx_m = trace_pos[-1, 2], trace_neg[-1, 2]
    print(f"\nfinal cy(+omega)={cy_p:+.6f} cy(-omega)={cy_m:+.6f} sum(should be ~0 if antisymmetric)={cy_p+cy_m:+.7f}")
    print(f"final cx(+omega)={cx_p:+.6f} cx(-omega)={cx_m:+.6f} diff(should be ~0 if symmetric)={cx_p-cx_m:+.7f}")


if __name__ == "__main__":
    main()
