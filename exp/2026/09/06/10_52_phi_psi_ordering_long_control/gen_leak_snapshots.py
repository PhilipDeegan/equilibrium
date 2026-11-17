"""
Generates the field snapshots for the leak visualization: the same
toroidal-ring (phi,psi) run as probe_long_ring_bound.py, saving full
per-cell diagnostic arrays at t=0 (IC) and at step 5050 (t_phys~25.9, the
circulation-RMS trough measured in that run) so the plotting script can be
iterated on without re-running the (expensive, ~28min) simulation.
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
SNAPSHOT_STEP = 5050
OUT_DIR = "exp/2026/09/06/10_52_phi_psi_ordering_long_control"
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


def per_cell_phi_mag(phi, valid):
    return np.sqrt(np.sum(np.where(valid, phi**2, 0.0), axis=1))


def per_cell_circ_density(vals_list, triples, N):
    acc = np.zeros(N)
    cnt = np.zeros(N)
    for (fam, a, b, c), vals in zip(triples, vals_list):
        np.add.at(acc, a, np.abs(vals))
        np.add.at(cnt, a, 1.0)
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)


def per_cell_psi_density(psi_list, triples, N):
    return per_cell_circ_density(psi_list, triples, N)


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
    psi = [np.zeros_like(a, dtype=np.float64) for (_, a, b, c) in triples]

    pos = lat.pos.copy()

    vals0 = circ_D(lat.phi, triples)
    phi_mag0 = per_cell_phi_mag(lat.phi, valid)
    circd0 = per_cell_circ_density(vals0, triples, N)
    psid0 = per_cell_psi_density(psi, triples, N)
    print(f"N={N} R={R} sph_frac={SPH_FRAC} dt={dt:.6f} c2={C2:.0e} snapshot_step={SNAPSHOT_STEP}", flush=True)
    print(f"IC: max phi_mag={phi_mag0.max():.4f} max circ_density={circd0.max():.4f}", flush=True)

    t_start = time.time()
    for step in range(SNAPSHOT_STEP):
        u_new, dphi_base = symplectic_step_baseline(lat, dt, valid, safe_nb)
        vals = circ_D(lat.phi, triples)
        psi_new = [p - dt * v for p, v in zip(psi, vals)]
        extra = scatter_DT(psi_new, lat.phi.shape, triples)
        lat.u, lat.phi, psi = u_new, lat.phi + dphi_base + dt * C2 * extra, psi_new
        if step % 500 == 0:
            print(f"step={step} wall={time.time()-t_start:.1f}s", flush=True)

    t_phys = SNAPSHOT_STEP * dt
    vals1 = circ_D(lat.phi, triples)
    phi_mag1 = per_cell_phi_mag(lat.phi, valid)
    circd1 = per_cell_circ_density(vals1, triples, N)
    psid1 = per_cell_psi_density(psi, triples, N)
    print(f"snapshot @ t_phys={t_phys:.2f}: max phi_mag={phi_mag1.max():.4f} max circ_density={circd1.max():.4f} max psi_density={psid1.max():.4f}", flush=True)

    np.savez(
        f"{OUT_DIR}/leak_snapshots.npz",
        pos=pos,
        r_sph=r_sph,
        t_phys_end=t_phys,
        phi_mag0=phi_mag0,
        circd0=circd0,
        psid0=psid0,
        phi_mag1=phi_mag1,
        circd1=circd1,
        psid1=psid1,
    )
    print("saved", f"{OUT_DIR}/leak_snapshots.npz", flush=True)


if __name__ == "__main__":
    main()
