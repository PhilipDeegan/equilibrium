"""
Companion to probe_long_energy_bound.py: same long-horizon (t_phys up to
~150, vs the original 1500-step/~t=8 run) diagnostic, but on the REAL
toroidal-ring IC (curl-full, the actual object of interest) instead of the
curl-free control. The control run showed max|psi| and H_ratio both
BOUNDED/OSCILLATING (never secular) over t_phys~145 -- this checks whether
the ring's own circulation-RMS "decay" (previously measured as ~10% over
1500 steps / t_phys~8) is the start of a similar bounded oscillation
(benign, per Philip's leak-vs-loss framing) or continues decaying toward
zero (real loss).
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
C2 = 1e-3
N_STEPS = 8000
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


def main():
    lat = SubstrateLattice(radius_cells=R)
    dt = lat.stable_dt()
    valid = lat.nb >= 0
    safe_nb = np.where(valid, lat.nb, 0)
    triples = triangle_triples(lat.nb)

    scale = SPH_FRAC * R
    r_sph = scale
    lat.phi = build_phi_tor(lat, scale, r_sph)
    lat.u = np.zeros(lat.N)
    psi = [np.zeros_like(a, dtype=np.float64) for (_, a, b, c) in triples]

    vals0 = circ_D(lat.phi, triples)
    circ_rms0 = float(np.sqrt(np.mean(np.concatenate(vals0) ** 2)))

    def total_h():
        return float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2) + 0.5 * C2 * sum(np.sum(p**2) for p in psi))

    def circ_rms():
        vals = circ_D(lat.phi, triples)
        return float(np.sqrt(np.mean(np.concatenate(vals) ** 2)))

    h0 = total_h()
    print(
        f"N={lat.N} R={R} sph_frac={SPH_FRAC} dt={dt:.6f} c2={C2:.0e} "
        f"n_steps={N_STEPS} h0={h0:.6f} circ_rms0={circ_rms0:.6f}",
        flush=True,
    )

    t_start = time.time()
    for step in range(N_STEPS):
        u_new, dphi_base = symplectic_step_baseline(lat, dt, valid, safe_nb)
        vals = circ_D(lat.phi, triples)
        psi_new = [p - dt * v for p, v in zip(psi, vals)]
        extra = scatter_DT(psi_new, lat.phi.shape, triples)
        lat.u, lat.phi, psi = u_new, lat.phi + dphi_base + dt * C2 * extra, psi_new
        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            mp = max(float(np.max(np.abs(p))) for p in psi)
            hr = total_h() / h0
            cr = circ_rms() / circ_rms0
            t_phys = (step + 1) * dt
            print(
                f"step={step:6d} t_phys={t_phys:8.2f} circ_rms/circ_rms0={cr:.5f} "
                f"max|psi|={mp:.4e} H_ratio={hr:.8f} wall={time.time()-t_start:6.1f}s",
                flush=True,
            )
            if not np.isfinite(hr) or hr > 1e6:
                print("BLEW UP, stopping early", flush=True)
                break


if __name__ == "__main__":
    main()
