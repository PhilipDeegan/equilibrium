"""10-08 vs: does a state-dependent, non-gradient term in phi's update let step() carry circulation?
step() alone: phi_t = -grad u -> d/dt curl phi = 0 exactly (discrete curl of a discrete gradient), so no change to u's
equation (e.g. density-weighted flux) can ever move circulation. Minimal local candidate for 'flow carried by flow':
after each C++ step(), phi_k += dt kappa W_k (F_f . d_k), F = A x B per cell (A = reconstruct_A, B = curl_A, the
continuous.py anchor operators), F_f = (F_i + F_j) / 2 at the face (shared -> phi stays antisymmetric). The gradient part
of that relation (-grad |A|^2 / 2) is left out: it cannot touch circulation.
IC: the smooth divergence-free poloidal ring flow of pulse.py (R0, tube wt, amplitude a0), u = 0. Tracked: centroid
(rho, z) of |B|, total |B|, energy-like sum (u^2 + sum phi^2 / 2)."""
import argparse

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

C.spacing_of = lambda lat: 1.0  # periodic box: the helper reads a wrapped neighbour; hex distance is 1
p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=120)
p.add_argument("--kappa", type=float, nargs="+", default=[0.0, 1.0, -1.0])
p.add_argument("--a0", type=float, default=1.0)
p.add_argument("--R0", type=float, default=12.0)
p.add_argument("--wt", type=float, default=3.0)
p.add_argument("--T", type=float, default=10.0)
p.add_argument("--every", type=float, default=2.0)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--dt_scale", type=float, default=1.0, help="dt = dt_scale x the step() CFL dt")
p.add_argument("--centred", action="store_true", help="time-centre the added term on step()'s staggered phi: phi_new = "
               "phi* + dt kappa N((phi_prev + phi_new) / 2), phi_prev = phi before step(), phi* = after (implicit, --midpoint iterations)")
p.add_argument("--strang", action="store_true", help="half the added term before step(), half after (2nd order splitting)")
p.add_argument("--no_step", action="store_true", help="diagnostic: the added term alone, no step()")
p.add_argument("--flow", default="wsum", help="cell flow in A x B: 'wsum' = sum_k W_k phi_k d_k / (2 C0SQ) (the energy's own "
               "pairing -> sum phi dphi = 0 exactly at first order), 'ls' = reconstruct_A (least squares, 10-08 first run)")
p.add_argument("--midpoint", type=int, default=0, help="implicit midpoint, this many fixed-point iterations: phi1 = phi0 + dt "
               "kappa N(phi_m), phi_m = (phi0 + phi1) / 2 -> with wsum, phi_m . N(phi_m) = 0 makes |phi|^2 exact (0 = explicit)")
args = p.parse_args()
side = args.P / np.sqrt(3)


def wrap(x):
    return (x + side / 2) % side - side / 2


def A_fn(x):
    x = wrap(x)
    rc = np.maximum(np.hypot(x[:, 0], x[:, 1]), 1.0)
    f = args.a0 * np.exp(-((rc - args.R0)**2 + x[:, 2]**2) / (2 * args.wt**2))
    dfr, dfz = -(rc - args.R0) / args.wt**2 * f, -x[:, 2] / args.wt**2 * f
    Ar, Az = -dfz, f / rc + dfr
    return np.stack([Ar * x[:, 0] / rc, Ar * x[:, 1] / rc, Az], 1)


WD = C.W[:, None] * C.D_HAT


def lam(ph):
    """W_k (F_f . d_k), F = A x B per cell, face average."""
    A = ph @ WD / (2 * C.C0SQ) if args.flow == "wsum" else C.reconstruct_A(ac, ph)
    F = np.cross(A, C.curl_A(ac, ph))
    return C.W[None, :] * np.einsum("nkd,kd->nk", 0.5 * (F[:, None, :] + F[nb]), C.D_HAT)


def carry(a, ref=None):
    """phi += a N(phi_m) by fixed point (--midpoint iterations; 0 = explicit); phi_m = (ref + phi_new) / 2, ref = phi
    before the update (plain midpoint) or before step() (--centred)."""
    ph0 = ac.phi
    ref = ph0 if ref is None else ref
    d = lam(0.5 * (ref + ph0)) if ref is not ph0 else lam(ph0)
    for _ in range(args.midpoint):
        d = lam(0.5 * (ref + ph0 + a * d))
    ac.phi = ph0 + a * d


for kappa in args.kappa:
    alat = cpp_box(args.P, n_threads=args.threads)
    ac = C.AnchorCluster(alat)
    pos, nb = ac.pos, ac.nb
    mid = (pos[:, None, :] + 0.5 * C.D_HAT[None]).reshape(-1, 3)
    phi = C.project_div_free(ac, C.W[None, :] * np.einsum("nkd,kd->nk", A_fn(mid).reshape(ac.N, 14, 3), C.D_HAT))
    ac.u, ac.phi = np.zeros(ac.N), phi
    dt = args.dt_scale * 0.4 / float(np.sum(C.W**2))
    x = wrap(pos)
    rho_c = np.hypot(x[:, 0], x[:, 1])
    nst, ev = int(round(args.T / dt)), max(1, int(round(args.every / dt)))
    ph_prev = ac.phi
    print(f"flow {args.flow}{f' midpoint x{args.midpoint}' if args.midpoint else ''}{' strang' if args.strang else ''}{' centred' if args.centred else ''} dt x{args.dt_scale:g} kappa {kappa:+.1f} a0 {args.a0} R0 {args.R0} wt {args.wt} dt {dt:.4f} ({nst} ticks)", flush=True)
    for s in range(nst + 1):
        if s % ev == 0:
            B = C.curl_A(ac)
            bm = np.linalg.norm(B, axis=1)
            wgt = bm * (rho_c < side / 2.5)
            zc = np.angle(np.sum(wgt * np.exp(2j * np.pi * x[:, 2] / side))) * side / (2 * np.pi)
            u, ph = ac.u, ac.phi
            Elf = np.sum(u**2) + 0.5 * np.sum(ph_prev * ph) if s else np.nan
            print(f"  t {s * dt:6.2f}: |B| centroid rho {np.sum(wgt * rho_c) / wgt.sum():6.2f} z {zc:+7.3f}; sum|B| "
                  f"{bm.sum():9.2f}; max |u| {np.abs(u).max():.3e}; E {np.sum(u**2) + 0.5 * np.sum(ph**2):10.3f} E_lf {Elf:10.3f}", flush=True)
        if s == nst:
            break
        ph_prev = ac.phi
        h_ = 0.5 if args.strang else 1.0
        if kappa and args.strang:
            carry(h_ * dt * kappa)
        if not args.no_step:
            alat.step(0, dt)
        if kappa:
            carry(h_ * dt * kappa, ph_prev if args.centred else None)
