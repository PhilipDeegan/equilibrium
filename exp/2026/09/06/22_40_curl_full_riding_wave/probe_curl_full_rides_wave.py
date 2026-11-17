"""
Does a curl-full ("B-like") component travel WITH a genuine propagating
wave once the validated (phi,psi) conjugate coupling is switched on, or
does it stay frozen in place the way DST's plain step() provably freezes
any B=curl(phi) content (doc/dst/failures/b-field-frozen-under-plain-step.txt)?

Recommendation #1 from the "missing terms for a genuine leapfrogging EM
wave" discussion (`now`, 2026-09-06): the (phi,psi) pair is the only
candidate mechanism already confirmed STABLE and non-destructive (bounded
oscillatory exchange, not runaway -- probe_long_energy_bound.py/
probe_long_ring_bound.py, exp/2026/09/06/10_52_phi_psi_ordering_long_control/)
-- but it has only ever been tested on STATIC/localized ICs (a curl-free
control bump, a real toroidal ring sitting still). Never on something that
actually propagates. This is that test.

Construction: seed a validated traveling wave (equilibrium.substrate.
traveling_wave, curl-free, real dispersion-relation solution) PLUS a small
LOCALIZED curl-full toroidal bump (equilibrium.substrate.curl_full_dipole_ic's
toroidal_envelope/_phihat, same construction probe_long_ring_bound.py uses,
just re-centred off-origin and given an independent, smaller envelope radius
-- ADDED to phi, not replacing it) co-located with the wave packet at t=0.
Run the EXACT validated (phi,psi) update loop (symplectic_step_baseline +
circ_D/scatter_DT, copied verbatim from probe_long_ring_bound.py) at C2=0
(baseline -- bump should stay frozen, reproducing the known identity) and
C2=1e-3 (test -- does the bump's own circulation-weighted centroid advance
along with the wave's u-energy centroid, instead of staying put?).
"""

import sys

sys.path.insert(0, ".")
import time

import numpy as np

from equilibrium.substrate.curl_full_dipole_ic import _phihat, toroidal_envelope
from equilibrium.substrate.discrete_curl import triangle_triples
from equilibrium.substrate.lattice import SubstrateLattice, W_HEX, W_SQ
from equilibrium.substrate.traveling_wave import seed_traveling_wave
from tst.substrate.adaptive.adaptive_flux_tube import _MIRROR

R = 20.0
HALF_LEN = 70.0
AXIS = 2
Z0_WAVE = -40.0
BUMP_SCALE = 10.0  # toroidal bump envelope radius, centred at (0,0,Z0_WAVE)
BUMP_AMP = 0.15  # small relative to the wave's own amplitude=1.0

N_STEPS = 4000
LOG_EVERY = 50
W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)


def build_phi_tor_local(lat, centre, scale, r_env):
    pos, nb = lat.pos, lat.nb
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    mid = 0.5 * (pos[:, None, :] + pos[safe_nb])
    diff = pos[safe_nb] - pos[:, None, :]
    N = pos.shape[0]
    A_mid = toroidal_envelope(mid.reshape(-1, 3), scale, centre=centre).reshape(N, 14)
    tang_mid = _phihat(mid.reshape(-1, 3), centre=centre).reshape(N, 14, 3)
    proj = np.einsum("ikd,ikd->ik", tang_mid, diff)
    phi = A_mid * proj
    r_cell = np.linalg.norm(pos - centre[None, :], axis=1)
    r_mid = np.linalg.norm(mid - centre[None, None, :], axis=2)
    phi = np.where(valid, phi, 0.0)
    phi[r_cell > r_env, :] = 0.0
    phi[r_mid > r_env] = 0.0
    return phi


def circ_D(phi, triples):
    out = []
    for (h1, h2, sqk), a, b, c in triples:
        m = _MIRROR[sqk]
        out.append(phi[a, h1] + phi[b, h2] + phi[c, m])
    return out


def scatter_DT(vals, phi_shape, triples):
    inc = np.zeros(phi_shape)
    for (fam, v) in zip(triples, vals):
        (h1, h2, sqk), a, b, c = fam
        m = _MIRROR[sqk]
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


def triangle_centres(lat, triples):
    """Fixed (M_total,3) positions (cell a's position per triangle instance)
    and (M_total,) family boundaries, for the circulation-weighted centroid."""
    return np.concatenate([lat.pos[a] for (_, a, b, c) in triples], axis=0)


def run(lat0, c2, label):
    lat = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    p = seed_traveling_wave(lat, AXIS, amplitude=1.0, n_cells_per_wavelength=8, envelope_wavelengths=4.0, z0=Z0_WAVE)
    centre = np.array([0.0, 0.0, Z0_WAVE])
    bump = BUMP_AMP * build_phi_tor_local(lat, centre, BUMP_SCALE, BUMP_SCALE)
    lat.phi = lat.phi + bump

    dt = lat.stable_dt()
    valid = lat.nb >= 0
    safe_nb = np.where(valid, lat.nb, 0)
    triples = triangle_triples(lat.nb)
    tri_pos_z = triangle_centres(lat, triples)[:, AXIS]

    psi = [np.zeros_like(a, dtype=np.float64) for (_, a, b, c) in triples]

    vals0 = circ_D(lat.phi, triples)
    circ_rms0 = float(np.sqrt(np.mean(np.concatenate(vals0) ** 2)))

    def total_h():
        return float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2) + 0.5 * c2 * sum(np.sum(psi_i**2) for psi_i in psi))

    print(
        f"[{label}] N={lat.N} R={R} HALF_LEN={HALF_LEN} wavelength={p['wavelength']:.3f} "
        f"dt={dt:.6f} c2={c2:.0e} circ_rms0={circ_rms0:.6f}",
        flush=True,
    )

    h0 = total_h()
    t_start = time.time()
    trace = []
    for step in range(N_STEPS):
        u_new, dphi_base = symplectic_step_baseline(lat, dt, valid, safe_nb)
        vals = circ_D(lat.phi, triples)
        psi_new = [pp - dt * v for pp, v in zip(psi, vals)]
        extra = scatter_DT(psi_new, lat.phi.shape, triples)
        lat.u, lat.phi, psi = u_new, lat.phi + dphi_base + dt * c2 * extra, psi_new

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            vals = np.concatenate(circ_D(lat.phi, triples))
            weight = vals**2
            wsum = float(np.sum(weight))
            circ_centroid_z = float(np.sum(weight * tri_pos_z) / wsum) if wsum > 0 else float("nan")
            circ_rms = float(np.sqrt(np.mean(vals**2)))

            e = lat.u**2 + 0.5 * np.sum(lat.phi**2, axis=1)
            e_tot = float(np.sum(e))
            u_centroid_z = float(np.sum(e * lat.pos[:, AXIS]) / e_tot) if e_tot > 0 else float("nan")

            hr = total_h() / h0
            t_phys = (step + 1) * dt
            trace.append((t_phys, circ_rms / circ_rms0, circ_centroid_z, u_centroid_z, hr))
            print(
                f"[{label}] step={step:6d} t={t_phys:7.2f} circ_rms/circ_rms0={circ_rms/circ_rms0:.4f} "
                f"circ_centroid_z={circ_centroid_z:+8.3f} u_centroid_z={u_centroid_z:+8.3f} "
                f"H_ratio={hr:.6f} wall={time.time()-t_start:6.1f}s",
                flush=True,
            )
            if not np.isfinite(hr) or hr > 1e6:
                print(f"[{label}] BLEW UP, stopping early", flush=True)
                break
    return np.array(trace)


def main():
    for c2, label in ((0.0, "c2=0_baseline"), (1e-3, "c2=1e-3_test")):
        trace = run(None, c2, label)
        np.save(f"exp/2026/09/06/22_40_curl_full_riding_wave/trace_{label}.npy", trace)
    print("done", flush=True)


if __name__ == "__main__":
    main()
