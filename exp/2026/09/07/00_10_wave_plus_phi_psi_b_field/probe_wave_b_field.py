"""
The direct, phenomenology-first test, per Philip's ontology correction
(`now`, 2026-09-06 night "standing correction" entry): stop asking "what
sources curl" (curl isn't a thing that gets sourced, it's a measurement of
phi's actual trajectory) and instead just look -- does adding the
(phi,psi) coupling to an ACTUAL PROPAGATING WAVE make DST's own validated
B=curl(phi) (equilibrium.substrate.curl.b_vec_vectorized, the operator
already confirmed to match classical wire/loop/dipole fields, NOT the
12-triangle circ_D metric which is contaminated by a traveling wave's own
sampling noise -- doc/dst/failures/per-triangle-curl-metric-doesnt-
distinguish-curl-full.txt) become genuinely nonzero and co-propagate/
oscillate with the wave, instead of staying frozen at exactly zero the
way it does under plain step() (doc/dst/failures/
b-field-frozen-under-plain-step.txt)?

No separate hand-built "bump" this time (that was the earlier, now-
understood-as-mistaken framing -- there is no second object that needs to
be dragged along; the only question is whether the (phi,psi) coupling
makes the WAVE'S OWN phi visit genuinely curl-full configurations that
travel with it, since it IS the same phi).

Accept/reject criterion, stated plainly: does |B| become nonzero and does
its spatial location track the wave's own u-energy envelope over time
(co-propagation), or does whatever B appears stay spatially disconnected
from where the wave actually is?
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

C2 = 1e-3
N_STEPS = 5000
LOG_EVERY = 100
W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
N_FAMILIES = 12


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


def run(c2, label):
    lat = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    p = seed_traveling_wave(lat, AXIS, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=4.0, z0=Z0_WAVE)
    v_phase = p["wavelength"] / p["T"]

    dt = lat.stable_dt()
    valid = lat.nb >= 0
    safe_nb = np.where(valid, lat.nb, 0)
    triples = triangle_triples(lat.nb)
    psi_dense = np.zeros((lat.N, N_FAMILIES))

    B0 = b_vec_vectorized(lat.pos, lat.nb, lat.phi)
    valid_rows0 = ~np.isnan(B0).any(axis=1)
    print(
        f"[{label}] N={lat.N} wavelength={p['wavelength']:.3f} v_phase={v_phase:.4f} dt={dt:.6f} "
        f"initial |B|_rms={np.sqrt(np.mean(np.linalg.norm(B0[valid_rows0],axis=1)**2)):.4e} (sanity, should be ~0)",
        flush=True,
    )

    t_start = time.time()
    trace = []
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
            B_rms = float(np.sqrt(np.mean(Bmag2)))

            e = lat.u**2 + 0.5 * np.sum(lat.phi**2, axis=1)
            e_tot = float(np.sum(e))
            u_centroid_z = float(np.sum(e * lat.pos[:, AXIS]) / e_tot) if e_tot > 0 else float("nan")

            t_phys = (step + 1) * dt
            trace.append((t_phys, B_rms, B_centroid_z, u_centroid_z))
            print(
                f"[{label}] step={step:6d} t={t_phys:7.2f} |B|_rms={B_rms:.4e} "
                f"B_centroid_z={B_centroid_z:+8.3f} u_centroid_z={u_centroid_z:+8.3f} "
                f"gap={B_centroid_z-u_centroid_z:+7.3f} wall={time.time()-t_start:6.1f}s",
                flush=True,
            )
            if not np.isfinite(B_rms) or B_rms > 1e6:
                print(f"[{label}] BLEW UP, stopping early", flush=True)
                break
    return np.array(trace)


def main():
    for c2, label in ((0.0, "c2=0_baseline"), (1e-3, "c2=1e-3_test")):
        trace = run(c2, label)
        np.save(f"exp/2026/09/07/00_10_wave_plus_phi_psi_b_field/trace_{label}.npy", trace)
    print("done", flush=True)


if __name__ == "__main__":
    main()
