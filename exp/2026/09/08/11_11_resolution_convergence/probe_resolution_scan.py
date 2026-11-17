"""
Is the ~10% v_B deficit found at c2=0.7 (weighted circ_D/scatter_DT,
exp/2026/09/08/10_32_weighted_circ_scatter) a discretization artifact or a
genuine resolution-independent property of this (phi,psi) coupling?

No medium/lam anywhere in this probe -- pure vacuum throughout, so there is
no literal "index of refraction" here; the candidate effect is just that
the (phi,psi) curl-sector (12-triangle circ_D/scatter_DT) and the (u,phi)
sector (plain div/grad) are different discrete operators on the same
lattice, so they could have different dispersion relations at a given
finite wavelength even in identical vacuum. Test: hold c2=0.7 fixed (the
already-calibrated value at n_cells_per_wavelength=8) and re-run the SAME
short-horizon gap-slope diagnostic at coarser discretizations (more cells
per wavelength = closer to the continuum limit for that wave). If the
residual slope shrinks toward zero as resolution increases, it's a
discretization artifact of the (u,phi) or (phi,psi) sector (or both) at
finite K; if it stays roughly constant, it's a genuine resolution-
independent mismatch worth its own investigation.
"""

import sys

sys.path.insert(0, ".")
import time

import numpy as np

from equilibrium.substrate.curl import b_vec_vectorized
from equilibrium.substrate.discrete_curl import triangle_triples
from equilibrium.substrate.lattice import SubstrateLattice, W_HEX, W_SQ
from equilibrium.substrate.traveling_wave import seed_traveling_wave
from tst.substrate.adaptive.adaptive_flux_tube import _MIRROR

R = 15.0
HALF_LEN = 90.0
AXIS = 2
Z0_WAVE = -50.0

N_STEPS = 1500
LOG_EVERY = 50
W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
N_FAMILIES = 12
C2 = 0.7
N_CELLS_PER_WAVELENGTH_VALUES = [8, 12, 16]  # 24+ would push the envelope
# (envelope_wavelengths=4 * wavelength, centered at z0=-50) past the
# HALF_LEN=90 domain boundary and contaminate the measurement with
# reflections -- stay resolution-safe rather than widen the domain for now.


def circ_D_weighted(phi, triples, N):
    out = np.zeros((N, N_FAMILIES))
    for f, ((h1, h2, sqk), a, b, c) in enumerate(triples):
        m = _MIRROR[sqk]
        out[a, f] = W_HEX * phi[a, h1] + W_HEX * phi[b, h2] + W_SQ * phi[c, m]
    return out


def scatter_DT_weighted(psi_dense, phi_shape, triples):
    inc = np.zeros(phi_shape)
    for f, ((h1, h2, sqk), a, b, c) in enumerate(triples):
        m = _MIRROR[sqk]
        v = psi_dense[a, f]
        np.add.at(inc, (a, np.full_like(a, h1)), W_HEX * v)
        np.add.at(inc, (b, np.full_like(b, h2)), W_HEX * v)
        np.add.at(inc, (c, np.full_like(c, m)), W_SQ * v)
    return inc


def symplectic_step_baseline(lat, dt, valid, safe_nb):
    div_phi = np.sum(np.where(valid, W[None, :] * lat.phi, 0.0), axis=1)
    u_new = lat.u - dt * div_phi
    u_nb_new = u_new[safe_nb]
    drive = np.where(valid, u_nb_new - u_new[:, None], 0.0)
    dphi = -dt * W[None, :] * drive
    return u_new, dphi


def run_case(n_cpw):
    lat = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    p = seed_traveling_wave(
        lat, AXIS, amplitude=1.0, n_cells_per_wavelength=n_cpw, envelope_wavelengths=4.0, z0=Z0_WAVE
    )
    v_phase = p["wavelength"] / p["T"]
    dt = lat.stable_dt()
    valid = lat.nb >= 0
    safe_nb = np.where(valid, lat.nb, 0)
    triples = triangle_triples(lat.nb)
    psi_dense = np.zeros((lat.N, N_FAMILIES))

    label = f"n_cpw={n_cpw}"
    print(
        f"[{label}] N={lat.N} wavelength={p['wavelength']:.3f} v_phase={v_phase:.4f} dt={dt:.6f}",
        flush=True,
    )

    trace = []
    blew_up = False
    t0 = time.time()
    for step in range(N_STEPS):
        u_new, dphi_base = symplectic_step_baseline(lat, dt, valid, safe_nb)
        vals_dense = circ_D_weighted(lat.phi, triples, lat.N)
        psi_new = psi_dense - dt * vals_dense
        extra = scatter_DT_weighted(psi_new, lat.phi.shape, triples)
        lat.u, lat.phi, psi_dense = u_new, lat.phi + dphi_base + dt * C2 * extra, psi_new

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            B = b_vec_vectorized(lat.pos, lat.nb, lat.phi)
            valid_rows = ~np.isnan(B).any(axis=1)
            Bmag2 = np.sum(B[valid_rows] ** 2, axis=1)
            pos_v = lat.pos[valid_rows]
            Bsum = float(np.sum(Bmag2))
            B_centroid_z = float(np.sum(Bmag2 * pos_v[:, AXIS]) / Bsum) if Bsum > 0 else float("nan")
            B_rms = float(np.sqrt(np.mean(Bmag2))) if Bmag2.size else float("nan")

            e = lat.u**2 + 0.5 * np.sum(lat.phi**2, axis=1)
            e_tot = float(np.sum(e))
            u_centroid_z = float(np.sum(e * lat.pos[:, AXIS]) / e_tot) if e_tot > 0 else float("nan")

            t_phys = (step + 1) * dt
            gap = B_centroid_z - u_centroid_z
            trace.append((t_phys, B_rms, B_centroid_z, u_centroid_z, gap))

            if not np.isfinite(B_rms) or B_rms > 1e6 or not np.isfinite(u_centroid_z):
                print(f"[{label}] BLEW UP at step={step}", flush=True)
                blew_up = True
                break
    trace = np.array(trace)
    np.save(f"exp/2026/09/08/11_11_resolution_convergence/trace_ncpw_{n_cpw}.npy", trace)
    slope = float(np.polyfit(trace[:, 0], trace[:, 4], 1)[0]) if len(trace) >= 2 else float("nan")
    v_B = v_phase + slope
    frac_deficit = (v_phase - v_B) / v_phase
    print(
        f"[{label}] wall={time.time()-t0:.1f}s full_slope={slope:+.4f} v_phase={v_phase:.4f} "
        f"v_B~{v_B:.4f} frac_deficit={frac_deficit:+.4f} blew_up={blew_up}",
        flush=True,
    )
    return n_cpw, v_phase, slope, v_B, frac_deficit, blew_up


def main():
    print(f"fixed c2={C2}, {N_STEPS} steps, scanning n_cells_per_wavelength over {N_CELLS_PER_WAVELENGTH_VALUES}\n", flush=True)
    results = [run_case(n) for n in N_CELLS_PER_WAVELENGTH_VALUES]
    print("\n=== summary ===", flush=True)
    for n_cpw, v_phase, slope, v_B, frac_deficit, blew_up in results:
        print(
            f"n_cpw={n_cpw:3d}: v_phase={v_phase:.4f} full_slope={slope:+.4f} v_B~{v_B:.4f} "
            f"frac_deficit={frac_deficit:+.4f} blew_up={blew_up}",
            flush=True,
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()
