"""
Re-test of exp/2026/09/02/23_00_lockstep_free_energy/probe_lockstep_free_energy.py
under `gate="barrier"` instead of `gate="v1"` (the default used there).

Why: that probe's "step_cross erodes circulation without converging swing
toward 0" finding (now.md, 2026-09-02 latest entry) carries a flagged
caveat -- gate="v1" is continuous Fickian diffusion, but doc/dst.md's own
theory describes crossing as a discrete all-or-nothing threshold gate
(project_diffusion_vs_gate_mismatch), and a prior R1/R2 audit found v1
"PROVEN inverted" on unrelated tests (project_step_cross_v2). This is
queued-next item 1 from now.md's 2026-09-02 SESSION END entry: same
lockstep/swing diagnostic, same IC, only the gate changes.

m0 derivation (doc/dst_next.md sec 8, not picked): m0 = scaling_factor *
void_limit, scaling_factor = (real knot radius in metres) / L_planck /
(probe's own scale unit). The neutron calibration there used r=2.51e-15m,
scale=4.8 -> m0=2.49. This probe's spheromak uses R_SPH=8.0 (not 4.8), so
m0 scales linearly (m0 ~ 1/scale for fixed real radius):
    m0 = 2.49 * (4.8 / 8.0) = 1.494
Hardcoded below with this derivation shown, not picked freehand.
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

# doc/dst_next.md sec 8: m0 = scaling_factor * void_limit, scaling_factor =
# r_real/L_planck/scale. Neutron anchor (r=2.51e-15m, scale=4.8) gives
# m0=2.49; rescaled linearly to this probe's scale=R_SPH=8.0.
M0_NEUTRON_REF = 2.49
SCALE_REF = 4.8
M0 = M0_NEUTRON_REF * SCALE_REF / R_SPH


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


def run(label, gate):
    print(f"\n=== {label} (gate={gate!r}, m0={M0 if gate == 'barrier' else None}) ===")
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
        if gate is not None:
            kwargs = {"gate": gate}
            if gate == "barrier":
                kwargs["m0"] = M0
            lat.step_cross(dt, CROSS_SCALE, n_threads=N_THREADS, **kwargs)

    return history


def main():
    print(f"M0 (derived, scale={R_SPH}) = {M0:.6f}")
    hist_free = run("plain step() only (reproduce drift baseline)", None)
    hist_v1 = run("step() + step_cross(gate='v1') (2026-09-02 baseline, for comparison)", "v1")
    hist_barrier = run(f"step() + step_cross(gate='barrier', m0={M0:.4f}) (theory-consistent gate)", "barrier")

    print("\n=== summary ===")
    for label, hist in (("plain step()", hist_free), ("v1", hist_v1), ("barrier", hist_barrier)):
        swings = [h[3] for h in hist]
        print(f"{label:12s}  swing min={min(swings):+.4f} max={max(swings):+.4f} final={swings[-1]:+.4f}")


if __name__ == "__main__":
    main()
