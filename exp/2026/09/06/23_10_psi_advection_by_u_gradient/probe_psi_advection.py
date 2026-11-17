"""
Recommendation #2 from the "missing terms for a genuine leapfrogging EM
wave" thread (`now`, 2026-09-06 night; doc/dst/failures/
phi-psi-coupling-has-no-u-coupling-cannot-advect.txt): the validated
(phi,psi) pairing has NO coupling to u at all, so it structurally cannot
make curl-full content follow a wave. This gives psi a genuine u-coupled
advective term, built by DIRECT STRUCTURAL ANALOGY to DST's already-
validated gravity mechanism (SubstrateLattice.step_lam_force,
equilibrium/substrate/lattice.py:466 -- "force-driven lam advection along
grad(u), toward higher-u neighbours, flux = dt*scale*sigma_i*face_w*pull*
lam_i"): treat each of the 12 psi channels exactly like lam and apply the
IDENTICAL flux formula.

IMPORTANT, per this project's own "derive don't pick" discipline
(project_derive_dont_pick_parameters memory): this is SCHEMATIC, not a
first-principles derivation -- reusing lam's gravity-flux formula for psi
is an analogy (both are per-cell scalars that could plausibly transport
via the same phi-driven/u-driven flux machinery), not a term derived from
DST's own equations the way today's rotating-frame terms were. Treat any
positive result here as "worth deriving properly," not as validated
physics.

Cheapest possible test of the mechanism's core claim ("does psi drift
toward higher u"): a STATIC, FIXED u=k*z gradient (no wave, no dynamics
in u at all) with a small curl-full toroidal bump sitting off-centre.
Diagnostic is immune to the per-triangle circ_D sampling-noise confound
(doc/dst/failures/per-triangle-curl-metric-doesnt-distinguish-curl-full.txt)
because it uses a genuine MACRO measure instead: reconstruct phi as a
Cartesian vector per cell (equilibrium.substrate.rotating_frame, already
validated today against an analytic gradient) and project onto the
azimuthal direction about the bump's own axis (curl_full_dipole_ic._phihat)
-- a real geometric circulation-density proxy, not a local 3-cell loop.
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

R = 15.0
HALF_LEN = 30.0
AXIS = 2

U_SLOPE = 1.0  # static u = U_SLOPE * z, fixed_u everywhere -- higher u at +z
BUMP_CENTRE_Z = -10.0
BUMP_SCALE = 6.0
BUMP_AMP = 1.0

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
    """(N,12): dense per-cell circulation, zero where family f has no
    valid triangle at that cell -- same values as the compact circ_D,
    just scattered into a dense (N,12) array for lam-style advection."""
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
    u_new = np.where(lat.fixed_u, lat.u, lat.u - dt * div_phi)
    u_nb_new = u_new[safe_nb]
    drive = np.where(valid, u_nb_new - u_new[:, None], 0.0)
    dphi = -dt * W[None, :] * drive
    return u_new, dphi


def advect_psi_like_lam(psi_dense, u, valid, nb_idx, sigma, face_w, eta, dt):
    """Exact structural copy of SubstrateLattice.step_lam_force's flux
    formula (lattice.py:466), applied per psi channel instead of lam."""
    safe_nb = np.where(valid, nb_idx, 0)
    u_nb = np.where(valid, u[safe_nb], u[:, None])
    pull = np.maximum(u_nb - u[:, None], 0.0)  # (N,14)
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
    lat = SubstrateLattice.cylinder(R, HALF_LEN, axis=AXIS)
    lat.u = U_SLOPE * lat.pos[:, AXIS]
    lat.fixed_u = np.ones(lat.N, dtype=bool)  # u is a permanent, static background
    centre = np.array([0.0, 0.0, BUMP_CENTRE_Z])
    lat.phi = BUMP_AMP * build_phi_tor_local(lat, centre, BUMP_SCALE, BUMP_SCALE)

    dt = lat.stable_dt()
    valid = lat.nb >= 0
    safe_nb = np.where(valid, lat.nb, 0)
    nb_idx = lat.nb
    triples = triangle_triples(lat.nb)
    offsets, _ = face_offsets(lat)
    hop_len, pinv = face_design(offsets)

    psi_dense = np.zeros((lat.N, N_FAMILIES))

    def macro_centroid_z():
        Phi_vec = reconstruct_vector(lat.phi, hop_len, pinv, valid=lat._valid)
        tang = _phihat(lat.pos, centre=centre)
        tor = np.einsum("ij,ij->i", Phi_vec, tang)
        weight = tor**2
        wsum = float(np.sum(weight))
        return float(np.sum(weight * lat.pos[:, AXIS]) / wsum) if wsum > 0 else float("nan")

    z0 = macro_centroid_z()
    print(f"[{label}] N={lat.N} eta={eta:.3g} dt={dt:.6f} initial macro_centroid_z={z0:+.4f}", flush=True)

    t_start = time.time()
    trace = []
    for step in range(N_STEPS):
        sigma = 1.0 / (1.0 + lat.lam)  # lam stays 0 throughout -- sigma=1
        u_new, dphi_base = symplectic_step_baseline(lat, dt, valid, safe_nb)
        vals_dense = circ_D_dense(lat.phi, triples, lat.N)
        psi_new = psi_dense - dt * vals_dense
        psi_new = advect_psi_like_lam(psi_new, lat.u, valid, nb_idx, sigma, W, eta, dt)
        extra = scatter_DT_dense(psi_new, lat.phi.shape, triples)
        lat.u, lat.phi, psi_dense = u_new, lat.phi + dphi_base + dt * C2 * extra, psi_new

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            cz = macro_centroid_z()
            psi_rms = float(np.sqrt(np.mean(psi_dense**2)))
            phi_rms = float(np.sqrt(np.mean(lat.phi**2)))
            t_phys = (step + 1) * dt
            trace.append((t_phys, cz, psi_rms, phi_rms))
            print(
                f"[{label}] step={step:6d} t={t_phys:7.2f} macro_centroid_z={cz:+8.4f} "
                f"(drift={cz-z0:+7.4f}) psi_rms={psi_rms:.4e} phi_rms={phi_rms:.4e} wall={time.time()-t_start:6.1f}s",
                flush=True,
            )
            if not np.isfinite(phi_rms) or phi_rms > 1e8:
                print(f"[{label}] BLEW UP, stopping early", flush=True)
                break
    return np.array(trace)


def main():
    for eta, label in ((0.0, "eta=0_control"), (1.0, "eta=1_test")):
        trace = run(eta, label)
        np.save(f"exp/2026/09/06/23_10_psi_advection_by_u_gradient/trace_{label}.npy", trace)
    print("done", flush=True)


if __name__ == "__main__":
    main()
