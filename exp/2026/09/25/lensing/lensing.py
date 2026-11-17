"""Weak lensing / Shapiro delay on continuous::step (level 0, face medium).

Depletion eps(r) = mu / sqrt(r^2 + a^2); local cell size (far view) ell = 1 - eps.
Kernel: sigma = tick rate, L = mesh hop measured in local cells; wave speed
~ sigma / L, index n = L / sigma.
  base : sigma = 1,       L = 1         n = 1
  T    : sigma = ell,     L = 1         n = 1/ell       (slower ticks only)
  GR   : sigma = ell,     L = 1/ell     n = 1/ell^2     (smaller cells + slower ticks, rate ~ size)
  N    : sigma = 1/ell,   L = 1/ell     n = 1           (smaller cells + faster ticks, Nordstrom-like)
sigma and L = h/<ell> (coarse size over the mean size of the discrete cells
it covers) are sampled at the face anchors from the continuum field, as
deviations from vacuum; the coarse cells stay uniform. Substrate dx is likely
quantized and unequal adjacent discrete cells distort their junction; the
medium is the average of that, not modelled per junction.
Plane u/phi pulse (step()'s longitudinal wave) in +x; arrival-time centroid
per detector cell at x = X1, minus base -> Delta T(b). Reference: straight-line
eikonal Delta T = (1/c0) int_{x0}^{X1} (n - 1) dx. Deflection alpha = c0 dT/db
(sign: delay larger near axis -> ray bends toward the lens).
GR asymptote for a point lens: alpha = 4 mu / b; here softened 4 mu b/(b^2+a^2).
"""
import argparse
import time

import numpy as np
from scipy.integrate import quad

from equilibrium import lattice as _lat
from equilibrium.substrate import continuous as C

C.set_weights("legacy")  # written for W = 3:1 on both legs (pre-2026-09-28; see continuous.WEIGHTS)

p = argparse.ArgumentParser()
p.add_argument("--mu", type=float, default=0.3)
p.add_argument("--a", type=float, default=3.0)
p.add_argument("--Lx", type=float, default=100.0)
p.add_argument("--Lt", type=float, default=60.0)
p.add_argument("--x0", type=float, default=-40.0)
p.add_argument("--X1", type=float, default=35.0)
p.add_argument("--w", type=float, default=3.0)
p.add_argument("--lam0", type=float, default=0.0, help="carrier wavelength (0 = bare gaussian)")
p.add_argument("--threads", type=int, default=8)
p.add_argument("--arms", default="base,T,GR,N")
args = p.parse_args()
c0 = C.C0
ct = _lat.continuous


def eps_fn(r):
    return args.mu / np.sqrt(r**2 + args.a**2)


def build(arm):
    alat = ct.AdaptiveLattice()
    ct.build_bcc_box(alat[0], args.Lx, args.Lt, args.Lt, d1=1.0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)
    ac = C.AnchorCluster(alat, 0)
    cl = alat[0]
    x = ac.pos
    dt = C.stable_dt(ac)
    k0 = 2 * np.pi / args.lam0 if args.lam0 > 0 else 0.0
    f = lambda s: np.exp(-0.5 * ((s - args.x0) / args.w) ** 2) * np.cos(k0 * (s - args.x0))
    C.seed(ac, *C.plane_wave([1, 0, 0], f), dt=dt)
    s = np.max(np.abs(x) / (0.5 * np.array([args.Lx, args.Lt, args.Lt])), axis=1)
    cl.set_bc(ct.BoundaryCondition.quadratic_sponge(target=0.0, damp_max=0.02, r_all=s, r_inner=0.85))
    eps = eps_fn(np.linalg.norm(C.face_midpoints(ac), axis=2))
    z, up = 0 * eps, eps / (1.0 - eps)
    dsig, dn = {"base": (z, z), "T": (-eps, z), "GR": (-eps, up), "N": (up, up)}[arm]
    ac.set_face_medium(dsig, dn)
    return alat, cl, x, dt


def run(arm):
    alat, cl, x, dt = build(arm)
    det = np.abs(x[:, 0] - args.X1) < 0.6
    t_c = (args.X1 - args.x0) / c0
    t_lo, t_hi = t_c - 4 * args.w / c0, t_c + 2.0 + 4 * args.w / c0
    n = int(np.ceil(t_hi / dt))
    s0 = np.zeros(det.sum())
    s1 = np.zeros(det.sum())
    t0 = time.time()
    for i in range(1, n + 1):
        alat.step(0, dt)
        alat.step_sponge(0)
        t = i * dt
        if t >= t_lo:
            u2 = np.array(cl.leaf_u)[det] ** 2
            s0 += u2
            s1 += t * u2
    print(f"  {arm}: n={n} dt={dt:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return x[det], s1 / np.maximum(s0, 1e-300)


def delay_ref(arm, b):
    n_of = {"T": lambda e: 1 / (1 - e), "GR": lambda e: 1 / (1 - e) ** 2, "N": lambda e: 1.0 + 0 * e}[arm]
    return quad(lambda xx: n_of(eps_fn(np.hypot(xx, b))) - 1.0, args.x0, args.X1, limit=200)[0] / c0


arms = args.arms.split(",")
res = {a: run(a) for a in arms}
xd, tb = res["base"]
b = np.hypot(xd[:, 1], xd[:, 2])
edges = np.arange(0.0, 0.85 * 0.5 * args.Lt * 0.8, 1.5)
bc = 0.5 * (edges[1:] + edges[:-1])
out = {"bc": bc}
print(f"\nmu={args.mu} a={args.a} x0={args.x0} X1={args.X1}")
for a in arms:
    if a == "base":
        continue
    dT = res[a][1] - tb
    prof = np.array([dT[(b >= lo) & (b < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])
    ref = np.array([delay_ref(a, bb) for bb in bc])
    out[a], out[a + "_ref"] = prof, ref
    m = bc > 2 * args.a
    alpha = -c0 * np.gradient(prof, bc)
    alpha_ref = -c0 * np.gradient(ref, bc)
    alpha_gr = 4 * args.mu * bc / (bc**2 + args.a**2)
    rel = prof[m] - prof[m][-1]
    rel_ref = ref[m] - ref[m][-1]
    k = float(np.dot(rel, rel_ref) / max(np.dot(rel_ref, rel_ref), 1e-300)) if a != "N" else float("nan")
    print(f"{a:>3}: dT(b~0)={prof[0]:+.4f} (ref {ref[0]:+.4f})  shape fit dT/ref = {k:.3f}")
    print("     b     : " + " ".join(f"{v:6.1f}" for v in bc[m][::2]))
    print("     alpha : " + " ".join(f"{v:+6.3f}" for v in alpha[m][::2]))
    print("     ref   : " + " ".join(f"{v:+6.3f}" for v in alpha_ref[m][::2]))
    print("     GR asy: " + " ".join(f"{v:+6.3f}" for v in alpha_gr[m][::2]))
np.savez(f"lensing_mu{args.mu}_a{args.a}_lam{args.lam0}.npz", **out)
