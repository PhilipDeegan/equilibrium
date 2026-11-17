"""
Decisive follow-up to the c2 speed-matching sweep: the short (1500-step)
sweep found c2=2.0 minimizes the gap-growth rate (B_centroid_z lagging
u_centroid_z), a large improvement over the original c2=1e-3 (v_B~0 ->
v_B~0.8*v_phase). But a short window can't distinguish "genuine bounded
co-propagation (gap saturates/oscillates)" from "still an unbounded wake,
just growing more slowly". Run the calibrated c2=2.0 out to the SAME
5000-step horizon as last night's original probe_wave_b_field.py run,
alongside the original c2=1e-3 for direct comparison, and look at whether
the gap trend flattens in the second half of the run.
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

N_STEPS = 5000
LOG_EVERY = 100
W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
N_FAMILIES = 12
C2_VALUES = [1e-3, 2.0]


def circ_D_dense(phi, triples, N):
    out = np.zeros((N, N_FAMILIES))
    for f, ((h1, h2, sqk), a, b, c) in enumerate(triples):
        m = _MIRROR[sqk]
        out[a, f] = phi[a, h1] + phi[b, h2] + phi[c, m]
    return out


def scatter_DT_dense(psi_dense, phi_shape, triples):
    inc = np.zeros(phi_shape)
    for f, ((h1, h2, sqk), a, b, c) in enumerate(triples):
        m = _MIRROR[sqk]
        v = psi_dense[a, f]
        np.add.at(inc, (a, np.full_like(a, h1)), v)
        np.add.at(inc, (b, np.full_like(b, h2)), v)
        np.add.at(inc, (c, np.full_like(c, m)), v)
    return inc


def symplectic_step_baseline(lat, dt, valid, safe_nb):
    div_phi = np.sum(np.where(valid, W[None, :] * lat.phi, 0.0), axis=1)
    u_new = lat.u - dt * div_phi
    u_nb_new = u_new[safe_nb]
    drive = np.where(valid, u_nb_new - u_new[:, None], 0.0)
    dphi = -dt * W[None, :] * drive
    return u_new, dphi


def run(c2, label, lat_template, triples, dt, valid, safe_nb):
    lat = SubstrateLattice()
    lat.__dict__.update(lat_template.__dict__)
    lat.u = lat_template.u.copy()
    lat.phi = lat_template.phi.copy()
    psi_dense = np.zeros((lat.N, N_FAMILIES))

    trace = []
    t_start = time.time()
    for step in range(N_STEPS):
        u_new, dphi_base = symplectic_step_baseline(lat, dt, valid, safe_nb)
        vals_dense = circ_D_dense(lat.phi, triples, lat.N)
        psi_new = psi_dense - dt * vals_dense
        extra = scatter_DT_dense(psi_new, lat.phi.shape, triples)
        lat.u, lat.phi, psi_dense = u_new, lat.phi + dphi_base + dt * c2 * extra, psi_new

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
            print(
                f"[{label}] step={step:6d} t={t_phys:7.2f} |B|_rms={B_rms:.4e} "
                f"B_centroid_z={B_centroid_z:+8.3f} u_centroid_z={u_centroid_z:+8.3f} "
                f"gap={gap:+7.3f} wall={time.time()-t_start:6.1f}s",
                flush=True,
            )
            if not np.isfinite(B_rms) or B_rms > 1e6:
                print(f"[{label}] BLEW UP, stopping early", flush=True)
                break
    return np.array(trace)


def main():
    lat = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    p = seed_traveling_wave(lat, AXIS, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=4.0, z0=Z0_WAVE)
    v_phase = p["wavelength"] / p["T"]
    dt = lat.stable_dt()
    valid = lat.nb >= 0
    safe_nb = np.where(valid, lat.nb, 0)
    triples = triangle_triples(lat.nb)
    print(f"base (u,phi) wave: wavelength={p['wavelength']:.3f} v_phase={v_phase:.4f} dt={dt:.6f}", flush=True)

    for c2 in C2_VALUES:
        trace = run(c2, f"c2={c2:g}", lat, triples, dt, valid, safe_nb)
        np.save(f"exp/2026/09/08/09_34_c2_speed_matching_sweep/trace_long_c2_{c2:g}.npy", trace)

        t, gap = trace[:, 0], trace[:, 4]
        n = len(t)
        q = n // 4
        slope_first_half = float(np.polyfit(t[: n // 2], gap[: n // 2], 1)[0])
        slope_second_half = float(np.polyfit(t[n // 2 :], gap[n // 2 :], 1)[0])
        slope_last_quarter = float(np.polyfit(t[-q:], gap[-q:], 1)[0])
        print(
            f"[c2={c2:g}] SUMMARY: gap0={gap[0]:+.3f} gap_final={gap[-1]:+.3f} "
            f"slope_first_half={slope_first_half:+.4f} slope_second_half={slope_second_half:+.4f} "
            f"slope_last_quarter={slope_last_quarter:+.4f}",
            flush=True,
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()
