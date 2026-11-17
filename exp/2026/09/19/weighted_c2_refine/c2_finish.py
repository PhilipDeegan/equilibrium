"""
Resume script for c2_fine_scan.py, interrupted (clean shutdown, not a crash)
after 11/13 stage-1 c2 values. Missing: c2=1.10, 1.20 from the sweep, and
the whole stage-2 long-horizon (5000-step) saturation check.

Recovers already-done stage-1 full_slope values from the saved
trace_c2_*.npy files (no need to rerun them), runs the 2 missing sweep
points, then runs stage 2 at whichever c2 has the smallest |full_slope|
across ALL 13 values (not hardcoded -- in case 1.10/1.20 turn out to beat
the current leader, c2=0.75 at |full_slope|=0.289).
"""

import glob
import re
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
MISSING_C2 = [1.10, 1.20]
OUT_DIR = "exp/2026/09/19/weighted_c2_refine"


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


def full_slope_of(trace):
    if len(trace) < 2:
        return float("nan")
    t, gap = trace[:, 0], trace[:, 4]
    return float(np.polyfit(t, gap, 1)[0])


def main():
    lat = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    p = seed_traveling_wave(lat, AXIS, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=4.0, z0=Z0_WAVE)
    dt = lat.stable_dt()
    valid = lat.nb >= 0
    safe_nb = np.where(valid, lat.nb, 0)
    triples = triangle_triples(lat.nb)

    print("=== recovering already-done stage-1 c2 values from saved traces ===", flush=True)
    slopes = {}
    for fp in sorted(glob.glob(f"{OUT_DIR}/trace_c2_*.npy")):
        m = re.search(r"trace_c2_([\d.]+)\.npy", fp)
        if not m or "long" in fp:
            continue
        c2 = float(m.group(1))
        trace = np.load(fp)
        slopes[c2] = full_slope_of(trace)
        print(f"  recovered c2={c2:.3f}: full_slope={slopes[c2]:+.4f} (from {fp})", flush=True)

    print(f"\n=== running missing sweep values: {MISSING_C2} ===", flush=True)
    for c2 in MISSING_C2:
        if c2 in slopes:
            continue
        t0 = time.time()
        trace, blew_up = run(c2, f"c2={c2:g}", lat, triples, dt, valid, safe_nb, N_STEPS_SWEEP, LOG_EVERY_SWEEP)
        np.save(f"{OUT_DIR}/trace_c2_{c2:g}.npy", trace)
        slope = full_slope_of(trace) if not blew_up else float("nan")
        slopes[c2] = slope
        print(f"c2={c2:6.3f}: blew_up={blew_up!s:5s} full_slope={slope:+8.4f}  ({time.time()-t0:.1f}s)", flush=True)

    best_c2 = min((c2 for c2, s in slopes.items() if np.isfinite(s)), key=lambda c2: abs(slopes[c2]))
    print(f"\n--> best c2 across all {len(slopes)} values: {best_c2:.3f} (full_slope={slopes[best_c2]:+.4f})", flush=True)

    print(f"\n=== stage 2: long-horizon (5000-step) run at c2={best_c2:.3f} ===", flush=True)
    t0 = time.time()
    trace_long, blew_up_long = run(
        best_c2, f"long c2={best_c2:g}", lat, triples, dt, valid, safe_nb, N_STEPS_LONG, LOG_EVERY_LONG
    )
    np.save(f"{OUT_DIR}/trace_long_c2_{best_c2:g}.npy", trace_long)
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
