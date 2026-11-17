"""
Structural fix to the c2 speed-matching residual found in
exp/2026/09/08/09_34_c2_speed_matching_sweep: the calibrated c2~2.0
(unweighted circ_D/scatter_DT) cut the trailing-wake drift ~5x but left a
real oscillating-gap-plus-secular-drift residual. Diagnosis: circ_D/
scatter_DT there were geometrically UNWEIGHTED raw phi sums, unlike the
base (u,phi) system, whose two legs (du/dt=-sum W_k*phi_k, dphi/dt=-W_k*
grad(u)) BOTH carry the same W_HEX=3/W_SQ=1 face weights -- that matching
weight structure is exactly what makes (u,phi) a genuine transpose-
consistent conjugate pair with a single well-defined dispersion relation
(doc/dst.md's omega^2(K)). circ_D/scatter_DT as built for the 2026-09-07
diagnostic dropped those weights entirely, so the (phi,psi) subsystem's own
dispersion relation almost certainly has a different SHAPE (not just a
different scale) than the base system's -- a single scalar c2 can only
amplitude-match one wavenumber, not fix a shape mismatch, hence the
group-velocity beating seen in the long-horizon check.

Fix: every TRIANGLES family (h1, h2, sqk) has h1,h2 always hex (weight
W_HEX=3) and mirror(sqk) always sq (weight W_SQ=1) -- confirmed exactly
from discrete_curl.py's construction (h1,h2 in range(8), sqk offset by 8)
and adaptive_flux_tube's _MIRROR (permutes hex<->hex, sq<->sq only). So:

    circ_D_weighted(phi)[a,f]   = W_HEX*phi[a,h1] + W_HEX*phi[b,h2] + W_SQ*phi[c,m]
    scatter_DT_weighted(psi)    = the TRANSPOSE of circ_D_weighted, i.e. the
                                   same W_HEX/W_SQ multiplier is applied when
                                   depositing back onto each destination face.

This makes scatter_DT_weighted exactly D^T for D=circ_D_weighted (as linear
maps), which is the correct adjoint pairing -- reusing the SAME W_k values
the base system already uses, not a new free parameter. Re-sweep c2 (now
likely needing a different scale, since both operators are intrinsically
~3-7x stronger) and re-check the long-horizon co-propagation behavior
against the unweighted version's oscillating-plus-drifting result.
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
C2_VALUES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]


def circ_D_weighted(phi, triples, N):
    """circ_D with the SAME W_HEX/W_SQ weights the base system uses on its
    own two legs -- h1,h2 always hex (W_HEX), mirror(sqk) always sq (W_SQ)."""
    out = np.zeros((N, N_FAMILIES))
    for f, ((h1, h2, sqk), a, b, c) in enumerate(triples):
        m = _MIRROR[sqk]
        out[a, f] = W_HEX * phi[a, h1] + W_HEX * phi[b, h2] + W_SQ * phi[c, m]
    return out


def scatter_DT_weighted(psi_dense, phi_shape, triples):
    """Exact transpose of circ_D_weighted: same W_HEX/W_SQ multiplier applied
    to the destination face when scattering back."""
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


def run(c2, label, lat_template, triples, dt, valid, safe_nb):
    lat = SubstrateLattice()
    lat.__dict__.update(lat_template.__dict__)
    lat.u = lat_template.u.copy()
    lat.phi = lat_template.phi.copy()
    psi_dense = np.zeros((lat.N, N_FAMILIES))

    trace = []
    blew_up = False
    for step in range(N_STEPS):
        u_new, dphi_base = symplectic_step_baseline(lat, dt, valid, safe_nb)
        vals_dense = circ_D_weighted(lat.phi, triples, lat.N)
        psi_new = psi_dense - dt * vals_dense
        extra = scatter_DT_weighted(psi_new, lat.phi.shape, triples)
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

            if not np.isfinite(B_rms) or B_rms > 1e6 or not np.isfinite(u_centroid_z):
                print(f"[{label}] BLEW UP at step={step}, stopping early", flush=True)
                blew_up = True
                break
    trace = np.array(trace)
    if len(trace):
        print(
            f"[{label}] c2={c2:8.3g} final: t={trace[-1,0]:7.2f} |B|_rms={trace[-1,1]:.4e} "
            f"gap={trace[-1,4]:+8.3f} blew_up={blew_up}",
            flush=True,
        )
    return trace, blew_up


def main():
    lat = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    p = seed_traveling_wave(lat, AXIS, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=4.0, z0=Z0_WAVE)
    v_phase = p["wavelength"] / p["T"]
    dt = lat.stable_dt()
    valid = lat.nb >= 0
    safe_nb = np.where(valid, lat.nb, 0)
    triples = triangle_triples(lat.nb)

    print(f"base (u,phi) wave: wavelength={p['wavelength']:.3f} v_phase={v_phase:.4f} dt={dt:.6f}", flush=True)
    print(f"sweeping c2 over {C2_VALUES}, {N_STEPS} steps each\n", flush=True)

    results = {}
    for c2 in C2_VALUES:
        t0 = time.time()
        trace, blew_up = run(c2, f"c2={c2:g}", lat, triples, dt, valid, safe_nb)
        results[c2] = (trace, blew_up)
        np.save(f"exp/2026/09/08/10_32_weighted_circ_scatter/trace_c2_{c2:g}.npy", trace)
        print(f"  ({time.time()-t0:.1f}s)\n", flush=True)

    print("=== summary ===", flush=True)
    print(f"v_phase (base u,phi wave) = {v_phase:.4f}", flush=True)
    for c2 in C2_VALUES:
        trace, blew_up = results[c2]
        if len(trace) < 2:
            print(f"c2={c2:8.3g}: no usable trace (blew up immediately)", flush=True)
            continue
        t, gap = trace[:, 0], trace[:, 4]
        n_early = min(6, len(t))
        slope_early = float(np.polyfit(t[:n_early], gap[:n_early], 1)[0]) if n_early >= 2 else float("nan")
        slope_full = float(np.polyfit(t, gap, 1)[0]) if len(t) >= 2 else float("nan")
        print(
            f"c2={c2:8.3g}: blew_up={blew_up!s:5s} gap0={gap[0]:+7.3f} "
            f"gap_final={gap[-1]:+7.3f} early_slope={slope_early:+7.4f} full_slope={slope_full:+7.4f}",
            flush=True,
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()
