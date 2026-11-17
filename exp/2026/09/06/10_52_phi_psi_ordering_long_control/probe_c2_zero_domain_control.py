"""
Directly tests Philip's confound worry: the toroidal ring IC fills 90% of
the domain (sph_frac=0.9) -- is the observed circ_rms drain-then-recover
(probe_long_ring_bound.py) actually caused by the new (phi,psi) coupling, or
is it just this object relaxing/settling against domain confinement, with
nothing to do with the new term at all? The earlier "circulation stays
exactly frozen under plain step()" result ([[project_curl_full_dipole_spheromak]],
2026-09-02) was only confirmed at sph_frac filling ~27% of the domain
(R=30, r_sph=8) -- never at 90% fill, and never for the bare toroidal-only
half used here (only the full poloidal+toroidal object was tested then).

Method: EXACT same IC, EXACT same domain (R=50, sph_frac=0.9), EXACT same
step count (5050) and dt, but c2=0 -- plain step() only, no psi field at
all. If circ_rms stays frozen here (as it did for the full object at 27%
fill), the drain seen with c2=1e-3 is real and attributable to the new
term. If it drains the same way here too, the whole leak-vs-loss result is
confounded by domain size and needs retracting.
"""

import sys

sys.path.insert(0, ".")
import time

import numpy as np

from equilibrium.substrate.curl_full_dipole_ic import toroidal_envelope, _phihat
from equilibrium.substrate.discrete_curl import triangle_triples
from tst.substrate.adaptive.adaptive_flux_tube import _MIRROR
from equilibrium.substrate.lattice import SubstrateLattice, W_HEX, W_SQ

R = 50.0
SPH_FRAC = 0.9
N_STEPS = 5050
LOG_EVERY = 50
W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)


def build_phi_tor(lat, scale, r_sph):
    pos, nb = lat.pos, lat.nb
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    mid = 0.5 * (pos[:, None, :] + pos[safe_nb])
    diff = pos[safe_nb] - pos[:, None, :]
    N = pos.shape[0]
    A_mid = toroidal_envelope(mid.reshape(-1, 3), scale).reshape(N, 14)
    tang_mid = _phihat(mid.reshape(-1, 3)).reshape(N, 14, 3)
    proj = np.einsum("ikd,ikd->ik", tang_mid, diff)
    phi = A_mid * proj
    r_cell = np.linalg.norm(pos, axis=1)
    r_mid = np.linalg.norm(mid, axis=2)
    phi = np.where(valid, phi, 0.0)
    phi[r_cell > r_sph, :] = 0.0
    phi[r_mid > r_sph] = 0.0
    return phi


def circ_D(phi, triples):
    out = []
    for (h1, h2, sqk), a, b, c in triples:
        m = _MIRROR[sqk]
        out.append(phi[a, h1] + phi[b, h2] + phi[c, m])
    return out


def symplectic_step_baseline(lat, dt, valid, safe_nb):
    div_phi = np.sum(np.where(valid, W[None, :] * lat.phi, 0.0), axis=1)
    u_new = lat.u - dt * div_phi
    u_nb_new = u_new[safe_nb]
    drive = np.where(valid, u_nb_new - u_new[:, None], 0.0)
    dphi = -dt * W[None, :] * drive
    return u_new, dphi


def main():
    lat = SubstrateLattice(radius_cells=R)
    dt = lat.stable_dt()
    valid = lat.nb >= 0
    safe_nb = np.where(valid, lat.nb, 0)
    triples = triangle_triples(lat.nb)
    N = lat.N

    scale = SPH_FRAC * R
    r_sph = scale
    lat.phi = build_phi_tor(lat, scale, r_sph)
    lat.u = np.zeros(N)

    def circ_rms():
        vals = circ_D(lat.phi, triples)
        return float(np.sqrt(np.mean(np.concatenate(vals) ** 2)))

    def total_h():
        return float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))

    circ_rms0 = circ_rms()
    h0 = total_h()
    print(
        f"N={N} R={R} sph_frac={SPH_FRAC} dt={dt:.6f} c2=0 (PLAIN step() ONLY, no psi) "
        f"n_steps={N_STEPS} h0={h0:.6f} circ_rms0={circ_rms0:.6f}",
        flush=True,
    )

    t_start = time.time()
    for step in range(N_STEPS):
        u_new, dphi_base = symplectic_step_baseline(lat, dt, valid, safe_nb)
        lat.u, lat.phi = u_new, lat.phi + dphi_base
        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            cr = circ_rms() / circ_rms0
            hr = total_h() / h0
            t_phys = (step + 1) * dt
            print(
                f"step={step:6d} t_phys={t_phys:8.2f} circ_rms/circ_rms0={cr:.6f} "
                f"H_ratio={hr:.8f} wall={time.time()-t_start:6.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
