"""10-08 vs: the walk against step() on matched ICs, no structure (Philip: quantify the difference).
Same periodic BCC box (cpp_box). step(): u = eps G(r), phi = 0 (A = 0), G = exp(-r^2 / 2 w^2); continuum system
u_t = -c^2 div A, A_t = -grad u (c = 2, iso weights). Walk: background rho0 = 1, seed amplitude sqrt(1 + eps G), phase 0
(no flow); linearised, its density obeys the same wave equation at long wavelength with the Bogoliubov sound speed
c_s = n / (m xi sqrt18) (g from --walk_xi). Both release an outgoing spherical pulse. Compared: radial profiles of
u / eps and (rho - 1) / eps at matched front radius r_f = c t (each in its own units), the measured front speed, and the
normalised L2 difference of r x profile. Plus the rotational sector: a smooth divergence-free A (poloidal flow around a
ring, A = curl(f(s) e_theta)) is exactly static under step() (u_t = -div A = 0 -> A_t = 0); checked over 200 steps."""
import argparse

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import CppWalk, cpp_box, healing_g

p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=220)
p.add_argument("--m", type=float, default=0.8)
p.add_argument("--xi", type=float, default=3.9)
p.add_argument("--w", type=float, nargs="+", default=[4.0, 8.0, 16.0])
p.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.1, 0.3])
p.add_argument("--rf", type=float, nargs="+", default=[20.0, 35.0, 50.0])
p.add_argument("--threads", type=int, default=8)
args = p.parse_args()
n = np.sqrt(1 - args.m**2)
g = healing_g(args.m, args.xi, walk=True)
cs = np.sqrt(g * n / (3 * args.m))
side = args.P / np.sqrt(3)


def radial(pos, f, rmax):
    r = np.linalg.norm((pos + side / 2) % side - side / 2, axis=1)
    k = (r / 1.0).astype(int)
    m_ = k < rmax
    s = np.bincount(k[m_], f[m_], minlength=rmax)
    c = np.bincount(k[m_], minlength=rmax)
    return np.arange(rmax) + 0.5, s / np.maximum(c, 1)


def front(rr, prof):
    """outermost radius where |r prof| reaches half its max (pulse front)."""
    a = np.abs(rr * prof)
    return rr[np.nonzero(a >= 0.5 * a.max())[0].max()]


rmax = int(side / 2) - 1
print(f"P {args.P} side {side:.1f}; walk m {args.m} xi {args.xi} g {g:.5f} c_s {cs:.4f} / tick; step c {C.C0:.3f}")
res = {}
for w in args.w:
    # step()
    alat = cpp_box(args.P, n_threads=args.threads)
    ac = C.AnchorCluster(alat)
    pos = ac.pos
    wrap = lambda x: (x + side / 2) % side - side / 2  # noqa: E731
    ac.u = np.exp(-np.sum(wrap(pos)**2, 1) / (2 * w * w))
    ac.phi = np.zeros((ac.N, 14))
    dt = 0.4 / float(np.sum(C.W**2))  # stable_dt with h = 1 (spacing_of reads a wrapped neighbour in the periodic box)
    t, prof_s = 0.0, {}
    for rf in sorted(args.rf):
        nst = int(round((rf / C.C0 - t) / dt))
        for _ in range(nst):
            alat.step(0, dt)
        t += nst * dt
        prof_s[rf] = radial(pos, ac.u, rmax)
    del alat, ac
    for eps in args.eps:
        alat = cpp_box(args.P, n_threads=args.threads)
        wk = CppWalk(alat, 0, m=args.m, g=g, parity=True)
        G = np.exp(-np.sum(wrap(pos)**2, 1) / (2 * w * w))
        wk.seed(np.sqrt(1 + eps * G).astype(complex))
        t = 0
        for rf in sorted(args.rf):
            nt = int(round(rf / cs)) - t
            wk.step(nt)
            t += nt
            rr, pw = radial(pos, (wk.density() - 1) / eps, rmax)
            rr, ps = prof_s[rf]
            a, b = rr * ps, rr * pw
            sel = rr > 3
            diff = np.sqrt(np.sum((a - b)[sel]**2) / np.sum(a[sel]**2))
            res[(w, eps, rf)] = (front(rr, ps), front(rr, pw), diff, np.abs(b).max() / np.abs(a).max())
            print(f"w {w:4.1f} (w/xi {w / args.xi:4.2f}) eps {eps:4.2f} r_f {rf:4.0f}: front step {res[(w, eps, rf)][0]:5.1f}"
                  f" walk {res[(w, eps, rf)][1]:5.1f}; peak ratio walk/step {res[(w, eps, rf)][3]:5.2f}; L2 diff {diff:5.2f}",
                  flush=True)
        del alat, wk

# rotational sector under step(): smooth divergence-free A
alat = cpp_box(args.P, n_threads=args.threads)
ac = C.AnchorCluster(alat)
R0, wt = 20.0, 4.0


def A_fn(x):
    x = (x + side / 2) % side - side / 2
    rc = np.maximum(np.hypot(x[:, 0], x[:, 1]), 1.0)
    s2 = (rc - R0)**2 + x[:, 2]**2
    f = np.exp(-s2 / (2 * wt * wt))
    dfr, dfz = -(rc - R0) / wt**2 * f, -x[:, 2] / wt**2 * f
    Ar, Az = -dfz, f / rc + dfr
    return np.stack([Ar * x[:, 0] / rc, Ar * x[:, 1] / rc, Az], 1)


ac.u = np.zeros(ac.N)
mid = (ac.pos[:, None, :] + 0.5 * C.D_HAT[None]).reshape(-1, 3)
phi_s = C.W[None, :] * np.einsum("nkd,kd->nk", A_fn(mid).reshape(ac.N, 14, 3), C.D_HAT)
phi0 = C.project_div_free(ac, phi_s)
print(f"rotational IC: sampled smooth A, discrete-divergence-free projection changed it by "
      f"{np.linalg.norm(phi0 - phi_s) / np.linalg.norm(phi_s):.2e} (relative L2)")
ac.phi = phi0
dt = 0.4 / float(np.sum(C.W**2))
for _ in range(200):
    alat.step(0, dt)
print(f"\nrotational: smooth divergence-free ring flow (R {R0}, tube {wt}): after 200 step() ticks max |d phi| / max |phi| "
      f"{np.abs(ac.phi - phi0).max() / np.abs(phi0).max():.2e}, max |u| {np.abs(ac.u).max():.2e} / max |phi| {np.abs(phi0).max():.2e}")
