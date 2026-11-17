"""
Does the poloidal-only dipole IC (equilibrium.substrate.dipole_mode_ic's
u=U, phi=grad(U), CURL-FREE by construction) actually behave as an "open
knot" -- radiating real energy out to the sponge under free step() dynamics
-- in contrast to the poloidal+toroidal "closed knot" (curl-full dipole/
spheromak, project_curl_full_dipole_spheromak.md's neutron candidate, which
was already found to stay confined: H/H0 in [0.9979, 1.0] over 200 steps at
cross_scale=1e-3, "visually unchanged topology")?

Per project_dst_neutron.md: "open knot (proton): u-flux radiates outward --
creates long-range field (charge)" vs "closed knot (neutron): toroidal
winding, zero net outward flux, charge-neutral by topology." The static IC
itself is hard-masked to exactly zero outside r_sph either way (both
seed_dipole_ic and seed_curl_full_dipole_ic mask both endpoints of every
boundary-crossing face) -- so "open vs closed" can only be a DYNAMICAL
distinction, not a t=0 snapshot one. This tests it directly: build both
configurations on the SAME domain, step them under plain step() only (no
step_cross -- matches the poloidal/tor lockstep probe's own convention of
testing structure-vs-structure under free wave dynamics before adding any
erosion mechanism), and compare how much energy leaves the r<=r_sph core /
crosses the sponge over time.

Adapted from equilibrium/substrate/dipole_mode_ic.py::seed_dipole_ic and
equilibrium/substrate/curl_full_dipole_ic.py::seed_curl_full_dipole_ic (both
built for the CellSlab/AdaptiveLattice leaf-batch API) onto a flat
DampedLattice (pos/nb arrays), same adaptation style already used in this
session's probe_curl_loop_dipole.py.
"""

import sys

import numpy as np
from scipy.special import spherical_jn

sys.path.insert(0, ".")
from equilibrium.substrate.dipole_mode_ic import K_FACTOR, U_and_gradU
from equilibrium.substrate.lattice import DampedLattice

R = 30.0
R_SPH = 8.0
N_STEPS = 8000
REPORT = 200
SPONGE = 0.15
DAMP_MAX = 0.003
N_THREADS = 8


def _phihat(pos, centre):
    p = pos - centre
    rho = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
    on_axis = rho <= 0
    rho_safe = np.where(on_axis, 1.0, rho)
    tang = np.zeros_like(p)
    tang[:, 0] = -p[:, 1] / rho_safe
    tang[:, 1] = p[:, 0] / rho_safe
    tang[on_axis] = 0.0
    return tang


def _toroidal_envelope(pos, scale, centre):
    p = (pos - centre) / scale
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    rho = np.sqrt(x**2 + y**2)
    r = np.sqrt(x**2 + y**2 + z**2)
    r_safe = np.where(r > 0, r, 1.0)
    sin_theta = np.where(r > 0, rho / r_safe, 0.0)
    return spherical_jn(1, K_FACTOR * r_safe) * sin_theta


def build_poloidal(lat, r_sph, centre=None, u_max=1.0, amplitude=1.0):
    """proton candidate: u=U, phi=grad(U), curl-free by construction."""
    if centre is None:
        centre = np.zeros(3)
    pos, nb = lat.pos, lat.nb
    r_cell = np.linalg.norm(pos - centre, axis=1) / r_sph
    inside = r_cell <= 1.0

    U_vals, _ = U_and_gradU(pos, r_sph, centre)
    u_max_val = float(np.abs(U_vals[inside]).max()) if inside.any() else 1.0
    lat.u = np.where(inside, u_max * U_vals / (u_max_val or 1.0), 0.0)

    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    mid = 0.5 * (pos[:, None, :] + pos[safe_nb])
    _, gU_mid = U_and_gradU(mid.reshape(-1, 3), r_sph, centre)
    gU_mid = gU_mid.reshape(pos.shape[0], 14, 3)
    diff = pos[safe_nb] - pos[:, None, :]
    dlen = np.linalg.norm(diff, axis=-1, keepdims=True)
    n_hat = diff / np.where(dlen > 0, dlen, 1.0)
    phi = np.einsum("nkd,nkd->nk", gU_mid, n_hat)
    phi_max = float(np.abs(phi).max()) or 1.0
    phi = phi * (amplitude / phi_max)

    nbr_r = np.linalg.norm(pos[safe_nb] - centre, axis=-1) / r_sph
    mask = valid & (nbr_r <= 1.0) & (r_cell[:, None] <= 1.0)
    lat.phi = np.where(mask, phi, 0.0)
    return inside


def add_toroidal(lat, r_sph, centre=None, amplitude=1.0):
    """extra ingredient for the neutron candidate: real, nonzero-circulation
    toroidal component added on top of the poloidal one (already seeded on
    lat.phi by build_poloidal), independently peak-normalized, same
    dual-zero masking convention."""
    if centre is None:
        centre = np.zeros(3)
    pos, nb = lat.pos, lat.nb
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    mid = 0.5 * (pos[:, None, :] + pos[safe_nb])
    env_mid = _toroidal_envelope(mid.reshape(-1, 3), r_sph, centre).reshape(pos.shape[0], 14)
    tang_mid = _phihat(mid.reshape(-1, 3), centre).reshape(pos.shape[0], 14, 3)
    diff = pos[safe_nb] - pos[:, None, :]
    dlen = np.linalg.norm(diff, axis=-1, keepdims=True)
    n_hat = diff / np.where(dlen > 0, dlen, 1.0)
    proj = np.einsum("nkd,nkd->nk", tang_mid, n_hat)
    phi_tor = env_mid * proj
    tor_max = float(np.abs(phi_tor).max()) or 1.0
    phi_tor = phi_tor * (amplitude / tor_max)

    r_cell = np.linalg.norm(pos - centre, axis=1) / r_sph
    nbr_r = np.linalg.norm(pos[safe_nb] - centre, axis=-1) / r_sph
    mask = valid & (nbr_r <= 1.0) & (r_cell[:, None] <= 1.0)
    lat.phi = lat.phi + np.where(mask, phi_tor, 0.0)


def run(label, with_toroidal):
    print(f"\n=== {label} ===")
    lat = DampedLattice(radius_cells=R, sponge_fraction=SPONGE, damp_max=DAMP_MAX)
    pos = lat.pos
    r_all = np.linalg.norm(pos, axis=1)
    core = r_all <= R_SPH
    build_poloidal(lat, R_SPH)
    if with_toroidal:
        add_toroidal(lat, R_SPH)
    dt = lat.stable_dt()
    H0 = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
    H0_core = float(np.sum(lat.u[core] ** 2) + 0.5 * np.sum(lat.phi[core] ** 2))
    print(f"N={lat.N}  dt={dt:.4e}  H0={H0:.6e}  H0_core={H0_core:.6e}")

    radiated_cum = 0.0
    history = []
    for step in range(N_STEPS + 1):
        if step % REPORT == 0:
            H = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
            H_core = float(np.sum(lat.u[core] ** 2) + 0.5 * np.sum(lat.phi[core] ** 2))
            history.append((step, H, H_core, radiated_cum))
            print(
                f"  {step:6d}  H/H0={H/H0:.6f}  H_core/H0_core={H_core/H0_core:.6f}"
                f"  rad_cum/H0={radiated_cum/H0:+.4e}",
                flush=True,
            )
        radiated_cum += -lat.bc_flux() * dt
        lat.step(dt, n_threads=N_THREADS)

    return history, H0, H0_core


def main():
    hist_pol, H0_pol, H0core_pol = run("Poloidal-only (proton candidate)", False)
    hist_neu, H0_neu, H0core_neu = run("Poloidal+toroidal (neutron candidate)", True)

    print("\n=== summary ===")
    final_pol = hist_pol[-1]
    final_neu = hist_neu[-1]
    print(
        f"proton candidate : H/H0={final_pol[1]/H0_pol:.6f}"
        f"  H_core/H0_core={final_pol[2]/H0core_pol:.6f}"
        f"  radiated/H0={final_pol[3]/H0_pol:+.4e}"
    )
    print(
        f"neutron candidate: H/H0={final_neu[1]/H0_neu:.6f}"
        f"  H_core/H0_core={final_neu[2]/H0core_neu:.6f}"
        f"  radiated/H0={final_neu[3]/H0_neu:+.4e}"
    )


if __name__ == "__main__":
    main()
