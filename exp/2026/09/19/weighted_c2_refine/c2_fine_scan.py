"""
Follow-up to exp/2026/09/08/10_32_weighted_circ_scatter (weighted
circ_D/scatter_DT, W_HEX/W_SQ-matched adjoint pair for the (phi,psi) B
co-propagation candidate). Round 2 there found a flat-ish full_slope
minimum around c2=0.7-1.5 (-0.31 to -0.35, vs -0.57 at c2=0.5 and -0.65 at
c2=0.4) but never swept BETWEEN 0.5 and 1.0 at finer resolution, and never
asked whether the long-horizon gap actually saturates (bounded
co-propagation) vs just slows its growth rate (still a real, unbounded
wake, only smaller). Two questions this script answers:

1. Finer c2 grid (0.5 to 1.2, step 0.05): is there a genuine minimum
   |full_slope|, or is it a flat plateau (suggesting a residual shape
   mismatch no single c2 can remove, not a mistuned scale)?
2. At the best c2 found, a long-horizon (5000-step) run: does the gap
   B_centroid_z - u_centroid_z asymptote to a constant (real co-propagation,
   just offset) or keep growing linearly forever (still a wake, slower)?

circ_D_weighted/scatter_DT_weighted are copied verbatim from
probe_weighted_sweep_round2.py (same W_HEX/W_SQ-matched adjoint
construction, not a new free parameter).
"""

import sys
import time

sys.path.insert(0, "/home/p/git/eq")

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

N_STEPS_SWEEP = 1500
LOG_EVERY_SWEEP = 50
N_STEPS_LONG = 5000
LOG_EVERY_LONG = 100

W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
N_FAMILIES = 12
C2_VALUES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10, 1.20]


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


def run(c2, label, lat_template, triples, dt, valid, safe_nb, n_steps, log_every):
    lat = SubstrateLattice()
    lat.__dict__.update(lat_template.__dict__)
    lat.u = lat_template.u.copy()
    lat.phi = lat_template.phi.copy()
    psi_dense = np.zeros((lat.N, N_FAMILIES))

    trace = []
    blew_up = False
    for step in range(n_steps):
        u_new, dphi_base = symplectic_step_baseline(lat, dt, valid, safe_nb)
        vals_dense = circ_D_weighted(lat.phi, triples, lat.N)
        psi_new = psi_dense - dt * vals_dense
        extra = scatter_DT_weighted(psi_new, lat.phi.shape, triples)
        lat.u, lat.phi, psi_dense = u_new, lat.phi + dphi_base + dt * c2 * extra, psi_new

        if step % log_every == 0 or step == n_steps - 1:
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
                print(f"[{label}] BLEW UP at step={step}, stopping early", flush=True)
                blew_up = True
                break
    return np.array(trace), blew_up


def main():
    lat = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    p = seed_traveling_wave(lat, AXIS, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=4.0, z0=Z0_WAVE)
    v_phase = p["wavelength"] / p["T"]
    dt = lat.stable_dt()
    valid = lat.nb >= 0
    safe_nb = np.where(valid, lat.nb, 0)
    triples = triangle_triples(lat.nb)
    print(f"base (u,phi) wave: wavelength={p['wavelength']:.3f} v_phase={v_phase:.4f} dt={dt:.6f}", flush=True)

    print(f"\n=== stage 1: fine c2 sweep, {N_STEPS_SWEEP} steps each ===", flush=True)
    sweep_results = {}
    for c2 in C2_VALUES:
        t0 = time.time()
        trace, blew_up = run(c2, f"c2={c2:g}", lat, triples, dt, valid, safe_nb, N_STEPS_SWEEP, LOG_EVERY_SWEEP)
        np.save(f"exp/2026/09/19/weighted_c2_refine/trace_c2_{c2:g}.npy", trace)
        full_slope = float("nan")
        gap_final = float("nan")
        if len(trace) >= 2:
            t, gap = trace[:, 0], trace[:, 4]
            full_slope = float(np.polyfit(t, gap, 1)[0])
            gap_final = float(gap[-1])
        sweep_results[c2] = (full_slope, gap_final, blew_up)
        print(
            f"c2={c2:6.3f}: blew_up={blew_up!s:5s} gap_final={gap_final:+8.3f} "
            f"full_slope={full_slope:+8.4f}  ({time.time()-t0:.1f}s)",
            flush=True,
        )

    best_c2 = min(
        (c2 for c2, (slope, _, blew_up) in sweep_results.items() if not blew_up and np.isfinite(slope)),
        key=lambda c2: abs(sweep_results[c2][0]),
    )
    print(f"\n--> best c2 by |full_slope|: {best_c2:.3f} (full_slope={sweep_results[best_c2][0]:+.4f})", flush=True)

    print(f"\n=== stage 2: long-horizon (5000-step) run at c2={best_c2:.3f} ===", flush=True)
    t0 = time.time()
    trace_long, blew_up_long = run(
        best_c2, f"long c2={best_c2:g}", lat, triples, dt, valid, safe_nb, N_STEPS_LONG, LOG_EVERY_LONG
    )
    np.save(f"exp/2026/09/19/weighted_c2_refine/trace_long_c2_{best_c2:g}.npy", trace_long)
    t, gap = trace_long[:, 0], trace_long[:, 4]
    n = len(t)
    q = n // 4
    slope_first_half = float(np.polyfit(t[: n // 2], gap[: n // 2], 1)[0])
    slope_second_half = float(np.polyfit(t[n // 2 :], gap[n // 2 :], 1)[0])
    slope_last_quarter = float(np.polyfit(t[-q:], gap[-q:], 1)[0])
    print(
        f"[c2={best_c2:.3f}] LONG SUMMARY: gap0={gap[0]:+.3f} gap_final={gap[-1]:+.3f} "
        f"slope_first_half={slope_first_half:+.4f} slope_second_half={slope_second_half:+.4f} "
        f"slope_last_quarter={slope_last_quarter:+.4f}  ({time.time()-t0:.1f}s)",
        flush=True,
    )
    print(
        "--> if slope_last_quarter is much smaller in magnitude than slope_first_half: "
        "the gap is saturating (genuine bounded co-propagation offset).\n"
        "--> if slopes stay roughly equal across the run: still an unbounded wake, "
        "just a slower one -- the residual is not fixable by c2 alone.",
        flush=True,
    )
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
