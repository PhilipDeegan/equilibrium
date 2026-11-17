"""Single-signed poloidal pulse on a torus (discrete step(), R56 domain, structure radius 20): net circulation carried by
a travelling pattern (Philip 10-01: travelling phi is the fundamental thing, travel is what makes it persist; a
zero-mean wave has no net circulation, so it is no vortex).

Seed, around the minor circle of a torus (centre circle r0 in xy, tube width w; d, psi = distance / angle from the
tube centre line in the meridional plane, rho = cylindrical radius):
    u = amp env(d) blob,   A = (r0 / rho) u / c psi_hat,   blob = Gaussian ring (width s) at (rho - r0, z) = (d0, 0)
i.e. a local 1D wave travelling along +psi at c on every poloidal loop (A = u / c), with rho A_psi independent of the
toroidal angle. sum_loop phi (net poloidal circulation) is nonzero and conserved by step() on any closed loop; its
uniform part alone would be frozen. Under a LINEAR rule (--rule none / fixed) the mean (frozen loop) and the zero-mean
travelling part superpose exactly, so only a nonlinear rule (--rule self: lam = alpha 2 e, C++ implicit midpoint)
can couple the circulation to its travel.

In a uniform medium a travelling pattern goes straight; only refraction bends it onto the minor circle. A fixed
tube depletion n = 1 + lam0 exp(-d^2/2w^2) has a light ring at the maximum of n d (lam0 4: d ~ 1.25 w ~ 3.3).
Scores: E_tube, E_struct; per poloidal loop circulation C0 (mean of rho/r0 A_psi over the tube, = net circulation)
and C1 = |first psi harmonic| (the travelling concentration) with its phase (travel rate); u's psi harmonics b0, b1;
grid-scale fraction.
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

from equilibrium.substrate import continuous as C
from equilibrium.substrate.lattice import DampedLattice
from equilibrium.substrate.self_lam import SelfLamMidpoint, energy_cell

C.set_weights("discrete")
here = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--rule", choices=["none", "fixed", "self"], default="none")
p.add_argument("--alpha", type=float, default=1.0, help="self rule: lam = alpha 2 e")
p.add_argument("--lam0", type=float, default=1.0, help="fixed rule: tube depletion peak (Gaussian width w)")
p.add_argument("--amp", type=float, default=1.0)
p.add_argument("--d0", type=float, default=3.3, help="blob: orbit radius in the meridional plane")
p.add_argument("--s", type=float, default=1.5, help="blob width (cells)")
p.add_argument("--dir", type=int, default=1, help="+1: travel along +psi (A along +psi); -1 reversed (A opposite u)")
p.add_argument("--R", type=float, default=56.0)
p.add_argument("--r0", type=float, default=12.0)
p.add_argument("--w", type=float, default=8.0 / 3.0)
p.add_argument("--crossings", type=float, default=3.0)
p.add_argument("--every", type=int, default=40)
p.add_argument("--n_snap", type=int, default=6)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--tag", default="")
args = p.parse_args()

rule_tag = {"none": "lin", "fixed": f"tube{args.lam0:g}", "self": f"self_a{args.alpha:g}"}[args.rule]
name = f"pulse_{rule_tag}_d{args.d0:g}_s{args.s:g}_amp{args.amp:g}{'' if args.dir > 0 else '_rev'}_R{args.R:g}{args.tag}"
bk_root = Path("/bk/dst") if Path("/bk/dst").is_dir() else Path.home() / "tmp/dst"
bk = bk_root / "2026/10/01/poloidal_pulse" / name
bk.mkdir(parents=True, exist_ok=True)

t0 = time.time()
lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.25)
assert abs(C.spacing_of(lat) - 1.0) < 1e-9
c = C.C0
valid = lat.nb >= 0
nb = np.where(valid, lat.nb, 0)
x, y, z = lat.pos.T
rho = np.hypot(x, y)
th = np.arctan2(y, x)
psi = np.arctan2(z, rho - args.r0)
d = np.hypot(rho - args.r0, z)
r_all = np.linalg.norm(lat.pos, axis=1)
r_inner = 0.75 * r_all.max()
tube = d < 3 * args.w
struct = r_all < args.r0 + 4 * args.w
D_CUT = min(4.5 * args.w, 0.95 * args.r0)


def env(dd):
    """Gaussian in d with a smooth cutoff at D_CUT (< r0), so r0/rho never multiplies a nonzero tail near the axis."""
    return np.exp(-(dd**2) / (2 * args.w**2)) * np.clip(1 - (dd / D_CUT) ** 2, 0, 1) ** 3


def blob(rr, zz):
    """single-signed toroidal ring at (rho - r0, z) = (d0, 0): smooth, resolved, starts at psi = 0."""
    return np.exp(-((rr - args.r0 - args.d0) ** 2 + zz**2) / (2 * args.s**2))


def A_fn(X):
    xx, yy, zz = X[..., 0], X[..., 1], X[..., 2]
    rr = np.hypot(xx, yy)
    tt, pp = np.arctan2(yy, xx), np.arctan2(zz, rr - args.r0)
    dd = np.hypot(rr - args.r0, zz)
    a = args.dir * args.amp * env(dd) * blob(rr, zz) / c * args.r0 / np.maximum(rr, 1e-12)
    return np.stack([-np.sin(pp) * np.cos(tt) * a, -np.sin(pp) * np.sin(tt) * a, np.cos(pp) * a], -1)


lat.u = args.amp * env(d) * blob(rho, z)
lat.phi = C.sample_A(lat, A_fn)
if args.rule == "fixed":
    lat.lam[:] = args.lam0 * np.exp(-(d**2) / (2 * args.w**2))
elif args.rule == "self":
    lat.lam[:] = args.alpha * 2.0 * energy_cell(lat)
dt = lat.stable_dt()
crossing = 2 * args.R / c
n_steps = int(round(args.crossings * crossing / dt))
fu, ff = np.exp(-lat._gamma * dt), np.exp(-lat._gamma_face * dt)
assert r_all[tube].max() < r_inner

if args.rule == "self":
    stepper = SelfLamMidpoint(lat, alpha=args.alpha, n_threads=args.threads)
    step = stepper.step
else:
    def step(dt_):
        lat.step_cpp(dt_, args.threads)
        return 1

w_t = np.where(tube, env(d), 0.0)
e_ipsi = np.exp(-1j * psi)


def loop_scores():
    """C0: mean net poloidal circulation (rho/r0 A_psi, tube-weighted); C1: its first psi harmonic (complex);
    b0, b1: same for u."""
    A = C.reconstruct_A(lat, lat.phi)
    ar = A[:, 0] * np.cos(th) + A[:, 1] * np.sin(th)
    a_psi = (-np.sin(psi) * ar + np.cos(psi) * A[:, 2]) * rho / args.r0
    W = w_t.sum()
    return (np.sum(w_t * a_psi) / W, 2 * np.sum(w_t * a_psi * e_ipsi) / W,
            np.sum(w_t * lat.u) / W, 2 * np.sum(w_t * lat.u * e_ipsi) / W)


def grid_frac(u):
    hexm = np.where(valid[:, :8], u[nb[:, :8]], u[:, None]).mean(axis=1)
    r = (u - hexm)[tube]
    return np.sum(r**2) / max(np.sum(u[tube] ** 2), 1e-300)


def slices(tag, t):
    from scipy.interpolate import griddata

    A = C.reconstruct_A(lat, lat.phi)
    ar = A[:, 0] * np.cos(th) + A[:, 1] * np.sin(th)
    a_psi = (-np.sin(psi) * ar + np.cos(psi) * A[:, 2]) * rho / args.r0
    e = energy_cell(lat)
    sig = 1.0 / (1.0 + lat.lam)
    lim = args.r0 + 4 * args.w
    gg = np.linspace(-lim, lim, 321)
    G1, G2 = np.meshgrid(gg, gg)
    panels = [(lat.u, "u", "RdBu_r", True), (a_psi, "rho/r0 A_psi (poloidal circulation)", "RdBu_r", True),
              (e, "energy density", "viridis", False), (sig, "sigma", "magma", False)]
    fig, ax = plt.subplots(2, 4, figsize=(20, 9.5))
    for row, (cut, a, b, la, lb) in enumerate([(np.abs(y) < 0.3, x, z, "x", "z"), (np.abs(z) < 0.3, x, y, "x", "y")]):
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


def snapshot(st):
    fn = bk / f"state_{st:07d}.h5"
    with h5py.File(fn, "w") as f:
        f.create_dataset("u", data=lat.u)
        f.create_dataset("phi", data=lat.phi, compression="gzip", compression_opts=1)
        f.create_dataset("lam", data=lat.lam)
        if st == 0:
            f.create_dataset("pos", data=lat.pos)
            f.create_dataset("nb", data=lat.nb.astype(np.int32))
        f.attrs.update(t=st * dt, step=st, dt=dt, args=json.dumps(vars(args)))
    return fn


e0 = energy_cell(lat)
E_all0 = e0.sum()
s0 = loop_scores()
print(f"{name}: N={lat.N} build+seed {time.time() - t0:.0f}s dt {dt:.5f} c {c:.3f} steps {n_steps}; E0 {E_all0:.3f} "
      f"tube {e0[tube].sum() / E_all0:.4f}; C0 {s0[0]:.4f} |C1| {abs(s0[1]):.4f} b0 {s0[2]:.4f} |b1| {abs(s0[3]):.4f} "
      f"lam_max {lat.lam.max():.3f}", flush=True)
print(f"  IC plot {slices('ic', 0.0)}  snapshot {snapshot(0)}", flush=True)
snap_at = set(int(round(n_steps * (i + 1) / args.n_snap)) for i in range(args.n_snap))
rec, iters = [], []
t0 = time.time()
for st in range(n_steps + 1):
    if st % args.every == 0 or st == n_steps:
        e = energy_cell(lat)
        C0, C1, b0, b1 = loop_scores()
        sg = 1.0 / (1.0 + lat.lam[tube])
        et = e[tube]
        rec.append((st * dt, e.sum() / E_all0, et.sum() / E_all0, e[struct].sum() / E_all0, C0 / s0[0],
                    abs(C1) / abs(s0[1]), np.angle(C1), b0 / s0[2], abs(b1) / abs(s0[3]), np.angle(b1),
                    np.sum(et * sg) / et.sum(), grid_frac(lat.u), lat.lam.max()))
    if st in snap_at:
        snapshot(st)
        if st != n_steps:
            print(f"  t {st * dt:.1f} plot {slices(f'step{st:07d}', st * dt)}", flush=True)
    if st < n_steps:
        iters.append(step(dt))
        lat.u *= fu
        lat.phi *= ff
wall = time.time() - t0
rec = np.array(rec)
cols = "t Etot Etube Estruct C0 |C1| argC1 b0 |b1| argb1 sigma_tube grid_frac lam_max"
print(f"  {n_steps} steps in {wall:.0f}s ({wall / n_steps:.3f} s/step); iterations mean {np.mean(iters):.2f}", flush=True)
print("  " + cols, flush=True)
for row in rec[:: max(1, len(rec) // 12)].tolist() + [rec[-1].tolist()]:
    print("   " + "  ".join(f"{v:.4e}" for v in row), flush=True)
for j, lab in ((6, "C1"), (9, "b1")):
    ph = np.unwrap(rec[:, j])
    n3 = len(ph) // 3
    print(f"  {lab} phase rate early {np.polyfit(rec[:n3, 0], ph[:n3], 1)[0]:+.4f} late "
          f"{np.polyfit(rec[-n3:, 0], ph[-n3:], 1)[0]:+.4f} (c / d0 = {c / args.d0:.4f})", flush=True)
print(f"  final plot {slices('final', n_steps * dt)}", flush=True)
np.savez(here / f"{name}_rec.npz", rec=rec, cols=cols, args=json.dumps(vars(args)), crossing=crossing)
