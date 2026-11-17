"""
Redesign of probe_psi_advection.py (exp/2026/09/06/23_10_psi_advection_by_u_gradient/),
per doc/dst/failures/psi-advection-fixed-u-background-breaks-baseline.txt:
that test froze u=U_SLOPE*z everywhere, which removes the (u,phi) system's
own restoring mechanism entirely -- phi/psi grew unboundedly even with the
new eta term OFF (eta=0 "control" was itself invalid).

Fix: u is NOT frozen anywhere. The "gravity-like" source is the EXACT same
curl-free Gaussian bump IC already validated as a BOUNDED control under the
(phi,psi) coupling (probe_long_energy_bound.py: u0=exp(-0.5*(r/4)^2),
phi=u_nb-u exactly, H_ratio in [0.9999,1.0042] over 60000 steps) -- a real,
bounded, self-consistently-evolving "mass," not an artificial infinite
ramp. A small toroidal curl-full bump is superposed off-centre (r0=8,
BUMP_SCALE=3). eta=0 must first reproduce that already-validated bounded
H_ratio behaviour (the real control this time); eta>0 tests whether the
toroidal bump's macro circulation centroid drifts toward the u0 peak
(the origin) on top of that valid baseline.

Still SCHEMATIC (see probe_psi_advection.py's own docstring) -- this only
tests whether the mechanism is even numerically viable once the baseline
is fixed, not whether it's a derived physical term.
"""

import sys

sys.path.insert(0, ".")
import time

import numpy as np

from equilibrium.substrate.curl_full_dipole_ic import _phihat, toroidal_envelope
from equilibrium.substrate.discrete_curl import triangle_triples
from equilibrium.substrate.lattice import SubstrateLattice, W_HEX, W_SQ
from equilibrium.substrate.rotating_frame import face_design, reconstruct_vector
from equilibrium.substrate.traveling_wave import face_offsets
from tst.substrate.adaptive.adaptive_flux_tube import _MIRROR

R = 22.0  # same domain as the already-validated probe_long_energy_bound.py control
BUMP_CENTRE = np.array([0.0, 0.0, -8.0])
BUMP_SCALE = 3.0
BUMP_AMP = 0.3

C2 = 1e-3
N_STEPS = 3000
LOG_EVERY = 100
W = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
N_FAMILIES = 12


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


def advect_psi_like_lam(psi_dense, u, valid, nb_idx, sigma, face_w, eta, dt):
    safe_nb = np.where(valid, nb_idx, 0)
    u_nb = np.where(valid, u[safe_nb], u[:, None])
    pull = np.maximum(u_nb - u[:, None], 0.0)
    new_psi = psi_dense.copy()
    for f in range(N_FAMILIES):
        flux = dt * eta * sigma[:, None] * face_w * pull * psi_dense[:, f : f + 1]
        dpsi = -flux.sum(axis=1)
        for k in range(14):
            v = valid[:, k]
            np.add.at(dpsi, nb_idx[v, k], flux[v, k])
        new_psi[:, f] += dpsi
    return new_psi


def run(eta, label):
    lat = SubstrateLattice(radius_cells=R)
    dt = lat.stable_dt()
    valid = lat.nb >= 0
    safe_nb = np.where(valid, lat.nb, 0)
    nb_idx = lat.nb
    triples = triangle_triples(lat.nb)
    offsets, _ = face_offsets(lat)
    hop_len, pinv = face_design(offsets)

    u0 = np.exp(-0.5 * (np.linalg.norm(lat.pos, axis=1) / 4.0) ** 2)
    lat.u = u0
    lat.phi = np.where(valid, u0[safe_nb] - u0[:, None], 0.0)
    lat.phi = lat.phi + BUMP_AMP * build_phi_tor_local(lat, BUMP_CENTRE, BUMP_SCALE, BUMP_SCALE)

    psi_dense = np.zeros((lat.N, N_FAMILIES))

    def total_h():
        return float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2) + 0.5 * C2 * np.sum(psi_dense**2))

    def macro_dist_from_origin():
        Phi_vec = reconstruct_vector(lat.phi, hop_len, pinv, valid=lat._valid)
        tang = _phihat(lat.pos, centre=BUMP_CENTRE)
        tor = np.einsum("ij,ij->i", Phi_vec, tang)
        weight = tor**2
        wsum = float(np.sum(weight))
        if wsum <= 0:
            return float("nan")
        centroid = np.array([np.sum(weight * lat.pos[:, d]) / wsum for d in range(3)])
        return float(np.linalg.norm(centroid))

    h0 = total_h()
    d0 = macro_dist_from_origin()
    print(f"[{label}] N={lat.N} R={R} eta={eta:.3g} dt={dt:.6f} h0={h0:.6f} initial_dist={d0:.4f}", flush=True)

    t_start = time.time()
    trace = []
    for step in range(N_STEPS):
        sigma = 1.0 / (1.0 + lat.lam)
        u_new, dphi_base = symplectic_step_baseline(lat, dt, valid, safe_nb)
        vals_dense = circ_D_dense(lat.phi, triples, lat.N)
        psi_new = psi_dense - dt * vals_dense
        psi_new = advect_psi_like_lam(psi_new, lat.u, valid, nb_idx, sigma, W, eta, dt)
        extra = scatter_DT_dense(psi_new, lat.phi.shape, triples)
        lat.u, lat.phi, psi_dense = u_new, lat.phi + dphi_base + dt * C2 * extra, psi_new

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            hr = total_h() / h0
            dist = macro_dist_from_origin()
            t_phys = (step + 1) * dt
            trace.append((t_phys, hr, dist))
            print(
                f"[{label}] step={step:6d} t={t_phys:7.2f} H_ratio={hr:.6f} "
                f"dist_from_origin={dist:.4f} (delta={dist-d0:+.4f}) wall={time.time()-t_start:6.1f}s",
                flush=True,
            )
            if not np.isfinite(hr) or hr > 1e6:
                print(f"[{label}] BLEW UP, stopping early", flush=True)
                break
    return np.array(trace)


def main():
    for eta, label in ((0.0, "eta=0_control"), (1.0, "eta=1_test")):
        trace = run(eta, label)
        np.save(f"exp/2026/09/06/23_45_psi_advection_bounded_source/trace_{label}.npy", trace)
    print("done", flush=True)


if __name__ == "__main__":
    main()
