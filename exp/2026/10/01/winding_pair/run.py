"""Winding pair under self-sourced lam (next 2026-10-01 item 1; mem project_mass_as_decoupling_pressure).

Candidates are travelling waves u = A env cos(chi - omega t) on a torus (centre circle radius r0 in the xy plane, tube
width w), chi = n theta + q psi (theta toroidal, psi poloidal about the centre circle). n > 0 travels along +theta; the
helicity sign is sign(n q). Each component is seeded as a discrete travelling wave: u = A env cos chi, phi_k = A sigma_f w_k
(v_j - v_i) / omega with v = env sin chi, sigma_f from the seeded state's own lam (fixed point) and omega^2 the
sigma-weighted Rayleigh quotient of v, so u_t(0) = omega A env sin chi up to envelope corrections.

Cases (each scaled to the same u-energy as the single helix with peak u = 1):
  ring      (n, 0)                     planar control, no helicity
  helix     (n, +1)                    one winding
  pair_opp  (n, +1) + (n, -1)          opposite helicity, both travel +theta
  pair_cnt  (n, +1) + (-n, +1)         opposite helicity, counter-travelling in theta
  pair_same (n, +1) + (-n, -1)         same helicity, counter-travelling (standing knot)

Rule as exp/2026/09/30/electron_survival: lam = alpha (u^2 + 1/2 sum phi^2), implicit midpoint (exact H), DampedLattice
sponge. Records: tube energy / E0 (dist to the centre circle < 3 w), E-weighted tube sigma (proper time), tube centroid
and radius, and the complex mode amplitudes a_m(t) = sum env u e^{-i chi_m} / sum env^2 for chi_m in MODES (frequency
sign of a_m = travel direction; a zero-frequency a_m is frozen, not circulating).
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

from equilibrium.substrate.lattice import _W, DampedLattice, wave_speed

out = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--case", required=True)
p.add_argument("--alpha", type=float, default=36.0)
p.add_argument("--R", type=float, default=14.0)
p.add_argument("--r0", type=float, default=5.0)
p.add_argument("--w", type=float, default=1.2)
p.add_argument("--n", type=int, default=3)
p.add_argument("--crossings", type=float, default=40.0)
p.add_argument("--every", type=int, default=10)
p.add_argument("--tag", default="")
args = p.parse_args()

CASES = {
    "ring": [(1, 0)],
    "helix": [(1, 1)],
    "pair_opp": [(1, 1), (1, -1)],
    "pair_cnt": [(1, 1), (-1, 1)],
    "pair_same": [(1, 1), (-1, -1)],
}
MODES = [(1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1), (1, 2), (1, -2)]  # (sign of n, q)

c = wave_speed()
lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.25)
valid = lat.nb >= 0
nb = np.where(valid, lat.nb, 0)
x, y, z = lat.pos.T
rho = np.hypot(x, y)
theta = np.arctan2(y, x)
psi = np.arctan2(z, rho - args.r0)
d = np.hypot(rho - args.r0, z)
env = np.exp(-(d**2) / (2 * args.w**2))
tube = d < 3 * args.w
assert np.all(np.linalg.norm(lat.pos[tube], axis=1) < 0.75 * args.R), "tube reaches the sponge"


def chi(sn, q):
    return sn * args.n * theta + q * psi


def e_cell(u, phi):
    return 0.5 * u**2 + 0.25 * np.sum(np.where(valid, phi, 0.0) ** 2, axis=1)


def seed(comps, alpha, scale, n_iter=400, mix=0.3, tol=1e-10):
    """u = scale sum env cos chi; phi = scale sum sigma_f w (v_j - v_i) / omega_c, with sigma from the seeded state itself
    (fixed point) and omega_c^2 the sigma-weighted Rayleigh quotient of v_c -> u_t(0) = omega_c scale env sin chi_c."""
    u = scale * sum(env * np.cos(chi(sn, q)) for sn, q in comps)
    dvs = []
    for sn, q in comps:
        v = env * np.sin(chi(sn, q))
        dvs.append((v, np.where(valid, v[nb] - v[:, None], 0.0)))
    phi = np.zeros_like(lat.phi)
    for _ in range(n_iter):
        sig = 1.0 / (1.0 + alpha * 2.0 * e_cell(u, phi))
        sf = np.where(valid, 0.5 * (sig[:, None] + sig[nb]), 0.0) * _W[None, :]
        oms = [np.sqrt(0.5 * np.sum(sf**2 * dv**2) / np.sum(v**2)) for v, dv in dvs]
        new = scale * sum(sf * dv / om for (v, dv), om in zip(dvs, oms))
        dphi = np.abs(new - phi).max()
        phi = phi + mix * (new - phi)
        if dphi < tol:
            break
    return u, phi, oms, dphi


def rhs(alpha, u, phi):
    lam = alpha * 2.0 * e_cell(u, phi)
    sig = 1.0 / (1.0 + lam)
    sw = np.where(valid, 0.5 * (sig[:, None] + sig[nb]) * _W[None, :], 0.0)
    return -(sw * phi).sum(axis=1), -sw * (u[nb] - u[:, None]), lam


def step_midpoint(alpha, u0, p0, dt, tol=1e-13, max_iter=200):
    du, dp, _ = rhs(alpha, u0, p0)
    u1, p1 = u0 + dt * du, p0 + dt * dp
    for it in range(max_iter):
        du, dp, lam = rhs(alpha, 0.5 * (u0 + u1), 0.5 * (p0 + p1))
        un, pn = u0 + dt * du, p0 + dt * dp
        err = max(np.abs(un - u1).max(), np.abs(pn - p1).max())
        u1, p1 = un, pn
        if err < tol:
            return u1, p1, lam
    raise RuntimeError(f"midpoint did not converge: {err:.2e}")


# equal u-energy: the single helix with peak u = 1
u_ref = np.sum(np.cos(chi(1, 1)) ** 2 * env**2)
u_case = np.sum(sum(env * np.cos(chi(sn, q)) for sn, q in CASES[args.case]) ** 2)
s = np.sqrt(u_ref / u_case)
lat.u, lat.phi, omegas, seed_res = seed(CASES[args.case], args.alpha, s)
E0 = e_cell(lat.u, lat.phi)[tube].sum()

basis = np.array([np.exp(-1j * chi(sn, q)) * env for sn, q in MODES]) / np.sum(env**2)
dt = lat.stable_dt()
n_steps = int(round(args.crossings * 2 * args.R / c / dt))
print(f"{args.case} alpha={args.alpha:g} N={lat.N} omegas={['%.4f' % o for o in omegas]} scale={s:.4f} "
      f"E_tube0/E_tot={E0 / e_cell(lat.u, lat.phi).sum():.4f} E0={E0:.3f} seed_res={seed_res:.1e} steps={n_steps}", flush=True)

rec, amps = [], []
t0 = time.time()
lam = args.alpha * 2.0 * e_cell(lat.u, lat.phi)
for st in range(n_steps + 1):
    if st % args.every == 0:
        e = e_cell(lat.u, lat.phi)
        et = e[tube]
        sig = 1.0 / (1.0 + lam[tube])
        rec.append((st * dt, et.sum() / E0, np.sum(et * sig) / et.sum(), np.sum(e[tube] * z[tube]) / et.sum(),
                    np.sqrt(np.sum(et * rho[tube] ** 2) / et.sum()), lam.max(), e.sum()))
        amps.append(basis @ lat.u)
        if st % (args.every * 2000) == 0:
            print(f"  t={st * dt:8.2f} Etube/E0={rec[-1][1]:.4f} sigma={rec[-1][2]:.4f} rho_rms={rec[-1][4]:.3f} "
                  f"lam_max={rec[-1][5]:.3f} |a|={' '.join('%.3f' % abs(a) for a in amps[-1])} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if st < n_steps:
        lat.u, lat.phi, lam = step_midpoint(args.alpha, lat.u, lat.phi, dt)
        lat.apply_sponge(dt)

np.savez(out / f"{args.case}_a{args.alpha:g}{args.tag}.npz", rec=np.array(rec), amps=np.array(amps), modes=np.array(MODES),
         omegas=np.array(omegas), cols="t Etube/E0 sigma_tube z_c rho_rms lam_max Etot", u=lat.u, phi=lat.phi,
         pos=lat.pos, args=str(vars(args)))
print(f"done {time.time() - t0:.0f}s", flush=True)
