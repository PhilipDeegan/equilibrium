"""Poloidal winding around a toroidal loop on the continuous A-E model (C++ step_ae, DEC curl, iso weights, h = 1),
no lam (step_ae requires lam = 0): does a resolved coil of phi (= A) hold its shape?

phi circulates around the minor circle of a torus (centre circle r0 in the xy plane, tube width w):
    A = f(d) [ (r0 / rho) psi_hat + pitch theta_hat ],   f = (d/w) exp(-d^2 / 2 w^2) (1 - (d/d_cut)^2)^3
(d, psi): distance and angle from the tube centre line in the meridional plane, rho = cylindrical radius. rho A_psi
independent of psi -> div A = 0 for any pitch (pitch 0: closed poloidal loops; pitch != 0: helical lines, pitch
toroidal per poloidal). u = 0, E = 0 at t = 0: the coil starts at rest. B = curl A is toroidal for pitch 0.

Sponge: C++ quadratic_sponge on u, phi each step; E damped in Python with the same per-cell profile every
--e_every steps (the C++ sponge leaves E undamped, exp/2026/09/29/sponge_assess C).
Outputs: IC / snapshot / final slice plots in this dir, record npz, HDF5 snapshots under /bk/dst (or ~/tmp/dst).
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate import continuous as C

here = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--R", type=float, default=56.0)
p.add_argument("--r0", type=float, default=12.0)
p.add_argument("--w", type=float, default=8.0 / 3.0)
p.add_argument("--pitch", type=float, default=0.0)
p.add_argument("--crossings", type=float, default=3.0)
p.add_argument("--every", type=int, default=40, help="steps between records")
p.add_argument("--n_snap", type=int, default=6, help="HDF5 snapshots + slice plots over the run (besides IC)")
p.add_argument("--damp", type=float, default=0.02)
p.add_argument("--e_every", type=int, default=5)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--tag", default="")
args = p.parse_args()

name = f"coil_p{args.pitch:g}_R{args.R:g}_r{args.r0:g}_w{args.w:.3g}{args.tag}"
bk_root = Path("/bk/dst") if Path("/bk/dst").is_dir() else Path.home() / "tmp/dst"
bk = bk_root / "2026/10/01/poloidal_ae" / name
bk.mkdir(parents=True, exist_ok=True)

h = 1.0
t0 = time.time()
ac = C.anchor_cluster(args.R, h, n_threads=args.threads)
ac.set_curl("dec")
pos, nb = ac.pos, ac.nb
valid = nb >= 0
x, y, z = pos.T
r_all = np.linalg.norm(pos, axis=1)
rho_c = np.hypot(x, y)
d_c = np.hypot(rho_c - args.r0, z)
r_inner = 0.75 * r_all.max()
tube = d_c < 3 * args.w
struct = r_all < args.r0 + 4 * args.w
assert r_all[tube].max() < r_inner


D_CUT = min(4.5 * args.w, 0.95 * args.r0)


def env(d):
    """(d/w) Gaussian times a smooth cutoff at D_CUT (< r0), so r0/rho never multiplies a nonzero tail near the axis."""
    return (d / args.w) * np.exp(-(d**2) / (2 * args.w**2)) * np.clip(1 - (d / D_CUT) ** 2, 0, 1) ** 3


def A_fn(X):
    xx, yy, zz = X[..., 0], X[..., 1], X[..., 2]
    rho = np.hypot(xx, yy)
    th = np.arctan2(yy, xx)
    ps = np.arctan2(zz, rho - args.r0)
    d = np.hypot(rho - args.r0, zz)
    f = env(d)
    ap = f * args.r0 / np.maximum(rho, 1e-12)
    at = args.pitch * f
    ct, st = np.cos(th), np.sin(th)
    return np.stack([-np.sin(ps) * ap * ct - at * st, -np.sin(ps) * ap * st + at * ct, np.cos(ps) * ap], -1)


ac.phi = np.where(valid, np.einsum("k,nkc,kc->nk", C.W, A_fn(C.face_midpoints(ac)), C.D_HAT * h), 0.0)
ac.u = np.zeros(ac.N)
ac.e = np.zeros((ac.N, 14))
dt = C.stable_dt(ac)
c = C.C0
crossing = 2 * args.R / c
n_steps = int(round(args.crossings * crossing / dt))

BC = _lat.continuous.BoundaryCondition
ac.cl.set_bc(BC.quadratic_sponge(target=0.0, damp_max=args.damp, r_all=r_all, r_inner=r_inner))
s = np.clip((r_all - r_inner) / (r_all.max() - r_inner), 0, 1)
fe = ((1 - args.damp * s**2) ** args.e_every)[:, None]
sponge_rows = s > 0


def energy_cell(phi, e, u):
    """step_ae energy per cell (each face shared by two cells: 1/2 of 1/2 sum over the 14), 'dec' metric is
    Euclidean; u part as in step() at sigma = 1."""
    K = C.C0SQ * ac.curl_curl_dec(phi)
    return 0.25 * np.sum(e**2 + phi * K, axis=1) + 0.5 * u**2


def cell_vectors(phi, e):
    A = C.reconstruct_A(ac, phi)
    E = C.reconstruct_A(ac, e)
    B = C.curl_A(ac, phi)
    return A, E, B


def toroidal_poloidal(V):
    th = np.arctan2(y, x)
    vt = -V[:, 0] * np.sin(th) + V[:, 1] * np.cos(th)
    return vt, np.sqrt(np.maximum(np.sum(V**2, 1) - vt**2, 0.0))


def slices(tag, t):
    from scipy.interpolate import griddata

    phi, e, u = ac.phi, ac.e, ac.u
    A, E, B = cell_vectors(phi, e)
    en = energy_cell(phi, e, u)
    At, Ap = toroidal_poloidal(A)
    Bt, Bp = toroidal_poloidal(B)
    lim = args.r0 + 4 * args.w
    g = np.linspace(-lim, lim, 321)
    G1, G2 = np.meshgrid(g, g)
    panels = [(Ap, "|A| poloidal", "viridis", False), (At, "A toroidal", "RdBu_r", True), (Bt, "B toroidal", "RdBu_r", True),
              (Bp, "|B| poloidal", "viridis", False), (np.linalg.norm(E, axis=1), "|E|", "viridis", False),
              (en, "energy density", "magma", False)]
    fig, ax = plt.subplots(2, 6, figsize=(30, 9.5))
    for row, (cut, a, b, la, lb) in enumerate([(np.abs(y) < 0.3 * h, x, z, "x", "z"), (np.abs(z) < 0.3 * h, x, y, "x", "y")]):
        m = cut & (np.abs(a) < lim + 2) & (np.abs(b) < lim + 2)
        for col, (val, lab, cmap, sym) in enumerate(panels):
            V = griddata((a[m], b[m]), val[m], (G1, G2), method="linear")
            vmax = np.nanmax(np.abs(V)) if sym else None
            im = ax[row, col].imshow(V, origin="lower", extent=(-lim, lim, -lim, lim), cmap=cmap,
                                     vmin=-vmax if sym else None, vmax=vmax)
            ax[row, col].set_xlabel(la)
            ax[row, col].set_ylabel(lb)
            ax[row, col].set_title(f"{lab} ({'xz' if row == 0 else 'xy'})")
            fig.colorbar(im, ax=ax[row, col], shrink=0.8)
    fig.suptitle(f"{name}  {tag}  t = {t:.2f}  (= {t / crossing:.3f} crossings)")
    fig.tight_layout()
    fn = here / f"{name}_{tag}.png"
    fig.savefig(fn, dpi=80)
    plt.close(fig)
    return fn


def snapshot(step):
    fn = bk / f"state_{step:07d}.h5"
    with h5py.File(fn, "w") as f:
        f.create_dataset("u", data=ac.u)
        f.create_dataset("phi", data=ac.phi, compression="gzip", compression_opts=1)
        f.create_dataset("e", data=ac.e, compression="gzip", compression_opts=1)
        if step == 0:
            f.create_dataset("pos", data=pos)
            f.create_dataset("nb", data=nb.astype(np.int32))
        f.attrs.update(t=step * dt, step=step, dt=dt, args=json.dumps(vars(args)))
    return fn


en0 = energy_cell(ac.phi, ac.e, ac.u)
E_all0, E_tube0, E_struct0 = en0.sum(), en0[tube].sum(), en0[struct].sum()
th_c = np.arctan2(y, x)
ps_c = np.arctan2(z, rho_c - args.r0)


def a_psi_theta(A):
    """rho/r0 * A . psi_hat (constant along a poloidal loop for the seed) and A . theta_hat at the cells."""
    ar = A[:, 0] * np.cos(th_c) + A[:, 1] * np.sin(th_c)
    return (-np.sin(ps_c) * ar + np.cos(ps_c) * A[:, 2]) * rho_c / args.r0, -A[:, 0] * np.sin(th_c) + A[:, 1] * np.cos(th_c)


A0 = C.reconstruct_A(ac, ac.phi)
A0_psi, _ = a_psi_theta(A0)
A0_f = env(d_c)
print(f"{name}: N={ac.N} build+seed {time.time() - t0:.0f}s dt {dt:.5f} c {c:.3f} crossing {crossing:.1f} steps {n_steps}; "
      f"E0 {E_all0:.4f} tube {E_tube0 / E_all0:.4f} struct {E_struct0 / E_all0:.4f}; |A| max {np.abs(A0).max():.3f}", flush=True)
print(f"  IC plot {slices('ic', 0.0)}  snapshot {snapshot(0)}", flush=True)

snap_at = set(int(round(n_steps * (i + 1) / args.n_snap)) for i in range(args.n_snap))
rec = []
t0 = time.time()
for st in range(n_steps + 1):
    if st % args.every == 0 or st == n_steps:
        phi, e, u = ac.phi, ac.e, ac.u
        en = energy_cell(phi, e, u)
        et = en[tube]
        wc = np.where(tube, en, 0.0)
        rho_rms = np.sqrt(np.sum(wc * rho_c**2) / wc.sum())
        d_rms = np.sqrt(np.sum(wc * d_c**2) / wc.sum())
        a_psi, a_th = a_psi_theta(C.reconstruct_A(ac, phi))
        rec.append((st * dt, en.sum() / E_all0, et.sum() / E_all0, en[struct].sum() / E_all0,
                    en[sponge_rows].sum() / E_all0, rho_rms, d_rms, np.sum(a_psi * A0_psi) / np.sum(A0_psi**2),
                    np.sum(a_th * A0_f) / np.sum(A0_f**2), 0.25 * np.sum(e[tube] ** 2) / et.sum(),
                    np.sum(u**2), np.abs(u).max()))
    if st in snap_at:
        snapshot(st)
        if st != n_steps:
            print(f"  t {st * dt:.1f} plot {slices(f'step{st:07d}', st * dt)}", flush=True)
    if st < n_steps:
        ac.step_ae(dt)
        ac.alat.step_sponge(0)
        if (st + 1) % args.e_every == 0:
            ac.e = ac.e * fe
wall = time.time() - t0
rec = np.array(rec)
print(f"  {n_steps} steps in {wall:.0f}s ({wall / n_steps:.3f} s/step)", flush=True)
cols = "t Etot Etube Estruct Esponge rho_rms d_rms a_psi(proj on seed) a_theta(proj on f) Efrac_E_tube sum_u2 max_u"
print("  " + cols, flush=True)
for row in rec[:: max(1, len(rec) // 12)].tolist() + [rec[-1].tolist()]:
    print("   " + "  ".join(f"{v:.4e}" for v in row), flush=True)
print(f"  final plot {slices('final', n_steps * dt)}", flush=True)
np.savez(here / f"{name}_rec.npz", rec=rec, cols=cols, args=json.dumps(vars(args)), crossing=crossing)
