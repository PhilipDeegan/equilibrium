"""Winding pair at R20 (structure ~40 cells across) in an R56 domain -- verification run first (Philip 10-01).

Successor of exp/2026/10/01/winding_pair (invalid: tube 1.5 cells). Same construction: travelling waves
u = A env_q cos(chi - omega t) on a torus (env_q = (d/w)^|q| Gaussian, smooth on the centre line) (centre circle r0 in the xy plane, tube width w), chi = n theta + q psi; seeded
as travelling waves of the actual rule (phi = A sigma_f W (v_j - v_i) / omega_c, v = env sin chi, sigma from the
seeded state itself, damped fixed point); all cases scaled to the u-energy of the single helix with peak u = 1.
Rule (--lam_shape self): lam = alpha (u^2 + 1/2 sum phi^2), C++ implicit midpoint; otherwise a FIXED centred
depletion (core / ball / shell / sheath / tube, --lam0, --lam_s), linear step_cpp. Self rule: (equilibrium.substrate.self_lam.SelfLamMidpoint),
DampedLattice rate sponge (factors precomputed, applied in place).

Outputs: IC / final plots (meridional xz slice and equatorial xy slice of u, energy density, sigma) in this dir,
HDF5 snapshots (t, u, phi, lam, pos, nb + attrs) under /bk/dst or ~/tmp/dst, a record npz, a log.
"""

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium.substrate.lattice import _W, DampedLattice, wave_speed
from equilibrium.substrate.self_lam import SelfLamMidpoint, energy_cell

here = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--case", default="pair_opp")
p.add_argument("--alpha", type=float, default=8.0)
p.add_argument("--lam_shape", choices=["self", "core", "ball", "shell", "sheath", "tube"], default="self",
               help="self: lam = alpha 2 e (C++ midpoint); otherwise a FIXED centred depletion, alpha ignored, linear step")
p.add_argument("--lam0", type=float, default=4.0, help="peak of the fixed depletion")
p.add_argument("--lam_s", type=float, default=0.0, help="width of the fixed depletion (0: shape default)")
p.add_argument("--R", type=float, default=56.0)
p.add_argument("--r0", type=float, default=12.0)
p.add_argument("--w", type=float, default=8.0 / 3.0)
p.add_argument("--n", type=int, default=3)
p.add_argument("--steps", type=int, default=100)
p.add_argument("--every", type=int, default=10, help="steps between records")
p.add_argument("--snap_every", type=int, default=0, help="steps between HDF5 snapshots (0: IC and final only)")
p.add_argument("--threads", type=int, default=8)
p.add_argument("--dt_frac", type=float, default=1.0)
p.add_argument("--tag", default="")
args = p.parse_args()

CASES = {
    "ring": [(1, 0)],
    "helix": [(1, 1)],
    "pair_opp": [(1, 1), (1, -1)],
    "pair_cnt": [(1, 1), (-1, 1)],
    "pair_same": [(1, 1), (-1, -1)],
}
MODES = [(1, 0), (1, 1), (1, -1), (-1, 1)]  # (sign of n, q); (-1,-1) is the conjugate of (1,1) for real u

shape_tag = "" if args.lam_shape == "self" else f"_{args.lam_shape}{args.lam0:g}"
name = f"{args.case}{shape_tag}{'' if shape_tag else f'_a{args.alpha:g}'}_R{args.R:g}_r{args.r0:g}_w{args.w:.3g}{args.tag}"
bk_root = Path("/bk/dst") if Path("/bk/dst").is_dir() else Path.home() / "tmp/dst"
bk = bk_root / "2026/10/01/winding_pair_r20" / name
bk.mkdir(parents=True, exist_ok=True)
c = wave_speed()

t0 = time.time()
lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.25)
valid = lat.nb >= 0
nb = np.where(valid, lat.nb, 0)
x, y, z = lat.pos.T
rho = np.hypot(x, y)
theta = np.arctan2(y, x)
psi = np.arctan2(z, rho - args.r0)
d = np.hypot(rho - args.r0, z)
env = np.exp(-(d**2) / (2 * args.w**2))


def env_q(q):
    """(d/w)^|q| Gaussian: a phase winding q times around the centre circle must vanish there (d e^{i psi} =
    (rho - r0) + i z is smooth); a plain Gaussian makes cos(q psi) jump across the centre line."""
    return (d / args.w) ** abs(q) * env


tube = d < 3 * args.w
r_all = np.linalg.norm(lat.pos, axis=1)


def fixed_lam():
    """Static centred depletion (in situ only; remote / geometry uniform). Widths default to resolved scales."""
    sh, l0 = args.lam_shape, args.lam0
    if sh == "core":   # dense centre, point-like relative to the structure
        sw = args.lam_s or 3.0
        return l0 * np.exp(-(r_all**2) / (2 * sw**2))
    if sh == "ball":   # diffuse sphere filling the structure (smooth edge at r0 + 3w)
        sw = args.lam_s or 2.0
        return l0 * 0.5 * (1 - np.tanh((r_all - (args.r0 + 3 * args.w)) / sw))
    if sh == "shell":  # spherical shell through the tube centre
        sw = args.lam_s or args.w
        return l0 * np.exp(-((r_all - args.r0) ** 2) / (2 * sw**2))
    if sh == "sheath":  # toroidal sheath on the tube boundary (d = 2w)
        sw = args.lam_s or 0.75 * args.w
        return l0 * np.exp(-((d - 2 * args.w) ** 2) / (2 * sw**2))
    if sh == "tube":   # depletion filling the tube (static analogue of the self-sourced time average)
        sw = args.lam_s or args.w
        return l0 * np.exp(-(d**2) / (2 * sw**2))
    raise ValueError(sh)


lam_fix = None if args.lam_shape == "self" else fixed_lam()
alpha_eff = args.alpha if lam_fix is None else 0.0
r_inner = 0.75 * r_all.max()
assert r_all[tube].max() < r_inner, "tube reaches the sponge"
print(f"{name}: N={lat.N} build {time.time() - t0:.0f}s; structure radius {args.r0 + 3 * args.w:.1f}, "
      f"sponge from r={r_inner:.1f}, cells across tube (6w) {6 * args.w:.1f}", flush=True)


def chi(sn, q):
    return sn * args.n * theta + q * psi


def seed(comps, alpha, scale, n_iter=400, mix=0.3, tol=1e-10):
    u = scale * sum(env_q(q) * np.cos(chi(sn, q)) for sn, q in comps)
    dvs = []
    for sn, q in comps:
        v = env_q(q) * np.sin(chi(sn, q))
        dvs.append((v, np.where(valid, v[nb] - v[:, None], 0.0)))
    phi = np.zeros_like(lat.phi)
    for _ in range(n_iter):
        sig = 1.0 / (1.0 + (0.0 if lam_fix is None else lam_fix) + alpha * 2.0 * energy_cell(lat, u, phi))
        sf = np.where(valid, 0.5 * (sig[:, None] + sig[nb]), 0.0) * _W[None, :]
        oms = [np.sqrt(0.5 * np.sum(sf**2 * dv**2) / np.sum(v**2)) for v, dv in dvs]
        new = scale * sum(sf * dv / om for (v, dv), om in zip(dvs, oms))
        res = np.abs(new - phi).max()
        phi = phi + mix * (new - phi)
        if res < tol:
            break
    return u, phi, oms, res


u_ref = np.sum(np.cos(chi(1, 1)) ** 2 * env_q(1) ** 2)
u_case = np.sum(sum(env_q(q) * np.cos(chi(sn, q)) for sn, q in CASES[args.case]) ** 2)
scale = np.sqrt(u_ref / u_case)
t0 = time.time()
u, phi, omegas, seed_res = seed(CASES[args.case], alpha_eff, scale)
lat.u, lat.phi = u, phi
lat.lam[:] = alpha_eff * 2.0 * energy_cell(lat) if lam_fix is None else lam_fix
E_all0 = energy_cell(lat).sum()
E0 = energy_cell(lat)[tube].sum()
print(f"  seed {time.time() - t0:.0f}s: omegas {['%.4f' % o for o in omegas]}  residual {seed_res:.1e}  scale {scale:.4f}  "
      f"E0 tube {E0:.3f} ({E0 / E_all0:.4f} of total)  lam_max {lat.lam.max():.3f}", flush=True)

# sponge factors at this dt, applied in place (same as DampedLattice.apply_sponge with u_bg = phi_bg = 0)
dt = lat.stable_dt() * args.dt_frac
fu = np.exp(-lat._gamma * dt)
ff = np.exp(-lat._gamma_face * dt)
assert not lat.fixed_u.any() and not lat.fixed_phi.any()


def sponge():
    lat.u *= fu
    lat.phi *= ff


basis = np.array([np.exp(-1j * chi(sn, q)) * env_q(q) / np.sum(env_q(q) ** 2) for sn, q in MODES])
crossing = 2 * args.R / c


def snapshot(step):
    fn = bk / f"state_{step:07d}.h5"
    with h5py.File(fn, "w") as f:
        f.create_dataset("u", data=lat.u)
        f.create_dataset("phi", data=lat.phi, compression="gzip", compression_opts=1)
        f.create_dataset("lam", data=lat.lam)
        if step == 0:
            f.create_dataset("pos", data=lat.pos)
            f.create_dataset("nb", data=lat.nb.astype(np.int32))
        f.attrs.update(t=step * dt, step=step, dt=dt, alpha=args.alpha, case=args.case, args=json.dumps(vars(args)),
                       omegas=np.array(omegas), scale=scale, E0_tube=E0)
    return fn


def slices(tag):
    """xz and xy cuts of u, energy, sigma on ONE lattice plane (|.| < 0.3; planes are 1/sqrt3 apart), linearly
    interpolated onto a regular grid (thicker slabs mix phases -> false speckle)."""
    from scipy.interpolate import griddata

    e = energy_cell(lat)
    sig = 1.0 / (1.0 + lat.lam)
    lim = args.r0 + 4 * args.w
    g = np.linspace(-lim, lim, 321)
    G1, G2 = np.meshgrid(g, g)
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
    for row, (cut, a, b, la, lb) in enumerate([(np.abs(y) < 0.3, x, z, "x", "z"), (np.abs(z) < 0.3, x, y, "x", "y")]):
        m = cut & (np.abs(a) < lim + 2) & (np.abs(b) < lim + 2)
        for col, (val, lab, cmap, sym) in enumerate([(lat.u, "u", "RdBu_r", True), (e, "energy density", "viridis", False),
                                                     (sig, "sigma = 1/(1+lam)", "magma", False)]):
            V = griddata((a[m], b[m]), val[m], (G1, G2), method="linear")
            vmax = np.nanmax(np.abs(V)) if sym else None
            im = ax[row, col].imshow(V, origin="lower", extent=(-lim, lim, -lim, lim), cmap=cmap,
                                     vmin=-vmax if sym else None, vmax=vmax)
            ax[row, col].set_xlabel(la)
            ax[row, col].set_ylabel(lb)
            ax[row, col].set_title(f"{lab} ({'xz' if row == 0 else 'xy'} plane)")
            fig.colorbar(im, ax=ax[row, col], shrink=0.8)
    fig.suptitle(f"{name}  {tag}  t = {step_now * dt:.3f}  (= {step_now * dt / crossing:.3f} crossings)")
    fig.tight_layout()
    fn = here / f"{name}_{tag}.png"
    fig.savefig(fn, dpi=100)
    plt.close(fig)
    return fn


def grid_frac(u):
    """grid-scale fraction of u in the tube: residual against the mean of the 8 hex (nearest) neighbours."""
    hexm = np.where(valid[:, :8], u[nb[:, :8]], u[:, None]).mean(axis=1)
    r = (u - hexm)[tube]
    return np.sum(r**2) / np.sum(u[tube] ** 2)


step_now = 0
print(f"  IC plot {slices('ic')}  snapshot {snapshot(0)}", flush=True)

if lam_fix is None:
    stepper = SelfLamMidpoint(lat, alpha=args.alpha, n_threads=args.threads)
else:
    class _Fixed:  # lam static -> sigma state-independent -> symplectic Euler (step_cpp) is symplectic
        def step(self, dt):
            lat.step_cpp(dt, args.threads)
            return 1
    stepper = _Fixed()
rec, amps, iters, k2, gfr = [], [], [], [], []
t0 = time.time()
for st in range(args.steps + 1):
    step_now = st
    if st % args.every == 0 or st == args.steps:
        e = energy_cell(lat)
        et = e[tube]
        sg = 1.0 / (1.0 + lat.lam[tube])
        rec.append((st * dt, et.sum() / E0, np.sum(et * sg) / et.sum(), np.sum(et * z[tube]) / et.sum(),
                    np.sqrt(np.sum(et * rho[tube] ** 2) / et.sum()), lat.lam.max(), e.sum() / E_all0,
                    e[r_all >= r_inner].sum() / E_all0, e[r_all < args.r0 + 4 * args.w].sum() / E_all0))
        amps.append(basis @ lat.u)
        du = np.where(valid, lat.u[nb] - lat.u[:, None], 0.0)
        k2.append(0.5 * np.sum((_W**2 * du**2)[tube]) / np.sum(lat.u[tube] ** 2))
        gfr.append(grid_frac(lat.u))
    if args.snap_every and st and st % args.snap_every == 0 and st != args.steps:
        snapshot(st)
    if st < args.steps:
        iters.append(stepper.step(dt))
        sponge()
wall = time.time() - t0
print(f"  {args.steps} steps in {wall:.1f}s ({wall / max(args.steps, 1):.3f} s/step, {wall / max(args.steps, 1) * crossing / dt / 60:.1f} "
      f"min/crossing); midpoint iterations min/mean/max {min(iters)}/{np.mean(iters):.2f}/{max(iters)}", flush=True)

rec, amps = np.array(rec), np.array(amps)
print(f"  roughness k_eff^2 of u in the tube (1/2 sum W^2 du^2 / sum u^2): {k2[0]:.4f} -> {k2[-1]:.4f} "
      f"(x{k2[-1] / k2[0]:.3f}); series {' '.join('%.4f' % v for v in k2[:: max(1, len(k2) // 8)])}", flush=True)
print(f"  grid-scale fraction of u in the tube: {' '.join('%.2e' % v for v in gfr[:: max(1, len(gfr) // 10)])} (end {gfr[-1]:.2e})",
      flush=True)
print("  t  Etube/E0  sigma_tube  z_c  rho_rms  lam_max  Etot/E0tot  E_sponge/E0tot  E_struct/E0tot", flush=True)
for row in rec[:: max(1, len(rec) // 10)].tolist() + [rec[-1].tolist()]:
    print("   " + "  ".join(f"{v:.6e}" for v in row), flush=True)
tt = rec[:, 0]
for j, m in enumerate(MODES):
    ph = np.unwrap(np.angle(amps[:, j]))
    rate = np.polyfit(tt, ph, 1)[0] if len(tt) > 2 else np.nan
    print(f"  mode {m}: |a| {abs(amps[0, j]):.4f} -> {abs(amps[-1, j]):.4f}  phase rate {rate:+.4f} "
          f"(travel along +theta <-> negative)", flush=True)
print(f"  final plot {slices('final')}  snapshot {snapshot(args.steps)}", flush=True)
np.savez(here / f"{name}_rec.npz", rec=rec, amps=amps, modes=np.array(MODES), omegas=np.array(omegas), iters=np.array(iters), k2=np.array(k2), grid_frac=np.array(gfr),
         cols="t Etube/E0 sigma_tube z_c rho_rms lam_max Etot/E0tot Esponge/E0tot Estruct/E0tot", args=json.dumps(vars(args)))
