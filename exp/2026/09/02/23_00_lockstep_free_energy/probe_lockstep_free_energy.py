"""
Does the poloidal/toroidal "non-lockstep drift" already found in
project_curl_full_dipole_spheromak.md (pure step(), u_rms drifts upward
while macro circulation stays exactly frozen -- "not moving together in any
simple sense") behave like genuine free energy: does `step_cross` (DST's
"resistivity"/decay operator) consume it, driving the pair TOWARD lockstep
(a stable, non-decaying proton-like remnant) rather than eroding forever?

Physical framing (Philip, 2026-09-02): proton = two flux tubes (poloidal +
toroidal), both circulating, LOCKED together (curl != 0, but no free energy
to shed -- E=mc^2, stable). Neutron = the same structure plus an ADDITIONAL,
different-type circulation on top -- genuine stored free energy -- which
`step_cross` releases as it "decays" (real neutron-proton mass gap already
has a calibration number in project_neutron_decay_energetics.md, ~0.78 MeV).
This tests whether the ALREADY-FOUND drift (a real, unexplained non-lockstep
signal from the earlier lockstep probe) is a plausible carrier for that free
energy: track it under plain step() (reproduce the drift baseline) vs
step()+step_cross (does the crossing operator consume it, converging swing
toward 0, i.e. toward a locked/stable pair)?

Diagnostics, same as the original lockstep probe
(exp/2026/08/17/12_25_curl_full_dipole_ic_rebuild/probe_pol_tor_lockstep*.py):
u_rms (poloidal tracer -- nothing else seeds u) and macro shell circulation
(toroidal tracer, `_phi_circulation` -- phi_pol alone gives exactly 0
circulation, proven in test_curl_full_dipole_ic.py). swing := u_rms(t)/u_rms0
- circ(t)/circ0 -- zero means perfectly locked, nonzero means drift/free
energy present.
"""

import sys

import numpy as np
from scipy.special import spherical_jn

sys.path.insert(0, ".")
from equilibrium.substrate.dipole_mode_ic import K_FACTOR, U_and_gradU
from equilibrium.substrate.lattice import DampedLattice

R = 30.0
R_SPH = 8.0
CROSS_SCALE = 1e-3
N_STEPS = 6000
REPORT = 200
N_THREADS = 8
CIRC_R_HOP = 4.0


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


def add_toroidal(lat, r_sph, centre=None, amplitude=1.0):
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


def _phi_circulation(lat, centre, r_hop, dr=0.8):
    c = np.asarray(centre, dtype=float)
    nb_safe = np.where(lat.nb >= 0, lat.nb, 0)
    pos_j = lat.pos[nb_safe]
    pos_i = lat.pos[:, np.newaxis, :]
    mid = 0.5 * (pos_i + pos_j) - c
    r_mid = np.linalg.norm(mid, axis=-1)
    on_shell = (lat.nb >= 0) & (r_mid >= r_hop - dr) & (r_mid <= r_hop + dr)
    upper = on_shell & (np.arange(len(lat.pos))[:, None] < nb_safe)
    mx, my = mid[..., 0], mid[..., 1]
    rho = np.sqrt(mx**2 + my**2)
    rho_s = np.where(rho > 1e-10, rho, 1.0)
    diff = pos_j - pos_i
    d_len = np.linalg.norm(diff, axis=-1, keepdims=True)
    n_hat = diff / np.where(d_len > 0, d_len, 1.0)
    proj = n_hat[..., 0] * (-my / rho_s) + n_hat[..., 1] * (mx / rho_s)
    return float(np.sum(lat.phi[upper] * proj[upper]))


def run(label, use_cross):
    print(f"\n=== {label} ===")
    lat = DampedLattice(radius_cells=R, sponge_fraction=0.15, damp_max=0.003)
    build_poloidal(lat, R_SPH)
    add_toroidal(lat, R_SPH)
    dt = lat.stable_dt()

    u_rms0 = float(np.sqrt(np.mean(lat.u**2)))
    circ0 = _phi_circulation(lat, np.zeros(3), CIRC_R_HOP)
    H0 = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
    print(f"N={lat.N}  dt={dt:.4e}  u_rms0={u_rms0:.4e}  circ0={circ0:+.4e}  H0={H0:.4e}")

    history = []
    for step in range(N_STEPS + 1):
        if step % REPORT == 0:
            u_rms = float(np.sqrt(np.mean(lat.u**2)))
            circ = _phi_circulation(lat, np.zeros(3), CIRC_R_HOP)
            H = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
            ru = u_rms / u_rms0
            rc = circ / circ0 if abs(circ0) > 1e-30 else float("nan")
            swing = ru - rc
            history.append((step, ru, rc, swing, H / H0))
            print(
                f"  {step:6d}  u_ratio={ru:.6f}  circ_ratio={rc:.6f}"
                f"  swing={swing:+.6f}  H/H0={H/H0:.6f}",
                flush=True,
            )
        lat.step(dt, n_threads=N_THREADS)
        if use_cross:
            lat.step_cross(dt, CROSS_SCALE, n_threads=N_THREADS)

    return history


def main():
    hist_free = run("plain step() only (reproduce drift baseline)", False)
    hist_cross = run(f"step() + step_cross(scale={CROSS_SCALE}) (does decay consume the drift?)", True)

    print("\n=== summary ===")
    swings_free = [h[3] for h in hist_free]
    swings_cross = [h[3] for h in hist_cross]
    print(f"plain step():  swing min={min(swings_free):+.4f} max={max(swings_free):+.4f} final={swings_free[-1]:+.4f}")
    print(f"with cross():  swing min={min(swings_cross):+.4f} max={max(swings_cross):+.4f} final={swings_cross[-1]:+.4f}")


if __name__ == "__main__":
    main()
