"""10-11 lie: self-transport of phi as a commutator of cell-to-cell exchange matrices (Pavlov et al. 2011 form), no
vectors. a_ij = phi_ik / W_k (edge integral, circulation along the hop) and v_ij = W_k phi_ik (flux, = dH/da for step()'s
H = sum phi^2 / 2) are antisymmetric N x N matrices on the faces S. Euler-Poincare: da/dt = kappa P_S [M, v], M = a
extended to 2-hop pairs by the mean of a over the 2-hop paths (path dependence = loop circulation). The 1-hop parts of
M v and v M cancel exactly, so only the 2-hop extension moves anything. sum v da = tr(v^T [M, v]) = 0 for any M:
step()'s energy is untouched by the term at first order (exact with implicit midpoint).
IC as 10-08 vs/carry.py (smooth div-free poloidal ring flow R0, tube wt, a0; u = 0). --flow wsum runs carry.py's
A x B term on the same harness for comparison."""
import argparse

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

C.spacing_of = lambda lat: 1.0
p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=120)
p.add_argument("--kappa", type=float, nargs="+", default=[0.0, 1.0, -1.0])
p.add_argument("--a0", type=float, default=1.0)
p.add_argument("--R0", type=float, default=12.0)
p.add_argument("--wt", type=float, default=3.0)
p.add_argument("--T", type=float, default=10.0)
p.add_argument("--every", type=float, default=2.0)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--flow", default="lie", help="'lie' (commutator) or 'wsum' (10-08 A x B)")
p.add_argument("--midpoint", type=int, default=3)
p.add_argument("--diag", action="store_true", help="evaluate the terms on the IC and exit")
p.add_argument("--expect", type=float, default=None, help="expected ring speed (z / time) at kappa; progress lines print the ratio")
p.add_argument("--light", type=int, default=0, help="light progress line every this many ticks (0: nst // 40)")
args = p.parse_args()
side = args.P / np.sqrt(3)
D, W, MIR = C.D_HAT, C.W, C.K_MIRROR


def wrap(x):
    return (x + side / 2) % side - side / 2


def A_fn(x):
    x = wrap(x)
    rc = np.maximum(np.hypot(x[:, 0], x[:, 1]), 1.0)
    f = args.a0 * np.exp(-((rc - args.R0)**2 + x[:, 2]**2) / (2 * args.wt**2))
    dfr, dfz = -(rc - args.R0) / args.wt**2 * f, -x[:, 2] / args.wt**2 * f
    Ar, Az = -dfz, f / rc + dfr
    return np.stack([Ar * x[:, 0] / rc, Ar * x[:, 1] / rc, Az], 1)


def key(v):
    return tuple(np.round(v, 6))


hop1 = {key(D[a]) for a in range(14)}
paths = {}
for p_ in range(14):
    for q in range(14):
        s = D[p_] + D[q]
        if np.linalg.norm(s) > 1e-9 and key(s) not in hop1:
            paths.setdefault(key(s), []).append((p_, q))
KEYS = list(paths)
KIDX = {k: n for n, k in enumerate(KEYS)}
MA = [[(b, KIDX[key(D[k] + D[b])]) for b in range(14) if key(D[k] + D[b]) in KIDX] for k in range(14)]
AM = [[(a, KIDX[key(D[k] - D[a])]) for a in range(14) if key(D[k] - D[a]) in KIDX] for k in range(14)]


def lie(ph):
    """W_k (P_S [M, v])_ik with a = ph / W, v = W ph."""
    a, v = ph / W, ph * W
    M2 = np.empty((ac.N, len(KEYS)))
    for n, k_ in enumerate(KEYS):
        M2[:, n] = sum(a[:, p_] + a[nb[:, p_], q] for p_, q in paths[k_]) / len(paths[k_])
    T = np.zeros_like(ph)
    for k in range(14):
        j = nb[:, k]
        for b, n in MA[k]:
            T[:, k] -= M2[:, n] * v[j, b]
        for a_, n in AM[k]:
            T[:, k] -= v[:, a_] * M2[nb[:, a_], n]
    return W * T


WD = W[:, None] * D


def wsum(ph):
    A = ph @ WD / (2 * C.C0SQ)
    F = np.cross(A, C.curl_A(ac, ph))
    return W[None, :] * np.einsum("nkd,kd->nk", 0.5 * (F[:, None, :] + F[nb]), D)


N_FN = lie if args.flow == "lie" else wsum


def carry(a_, ref):
    ph0 = ac.phi
    d = N_FN(0.5 * (ref + ph0))
    for _ in range(args.midpoint):
        d = N_FN(0.5 * (ref + ph0 + a_ * d))
    ac.phi = ph0 + a_ * d


for kappa in args.kappa:
    alat = cpp_box(args.P, n_threads=args.threads)
    ac = C.AnchorCluster(alat)
    pos, nb = ac.pos, ac.nb
    mid = (pos[:, None, :] + 0.5 * D[None]).reshape(-1, 3)
    phi = C.project_div_free(ac, W[None, :] * np.einsum("nkd,kd->nk", A_fn(mid).reshape(ac.N, 14, 3), D))
    ac.u, ac.phi = np.zeros(ac.N), phi
    if args.diag:
        L, Wc = lie(phi), wsum(phi)
        anti = np.abs(L + L[nb, MIR[None, :]]).max()
        print(f"N {ac.N}  |lie| {np.abs(L).max():.3e} antisym resid {anti:.1e}  sum phi.lie {np.sum(phi * L):.2e} "
              f"(vs sum|phi||lie| {np.sum(np.abs(phi * L)):.2e})  |wsum| {np.abs(Wc).max():.3e}")
        cc = np.sum(L * Wc) / np.sqrt(np.sum(L**2) * np.sum(Wc**2))
        print(f"corr(lie, wsum) {cc:+.3f}  best scale {np.sum(L * Wc) / np.sum(Wc**2):+.3f}")
        divL = np.sum(W * L, 1)
        print(f"div of lie (sum W out) max {np.abs(divL).max():.3e}")
        Lt, Wt = C.project_div_free(ac, L), C.project_div_free(ac, Wc)
        cc = np.sum(Lt * Wt) / np.sqrt(np.sum(Lt**2) * np.sum(Wt**2))
        print(f"div-free parts: |lie_T|/|lie| {np.linalg.norm(Lt) / np.linalg.norm(L):.3f}  |wsum_T|/|wsum| "
              f"{np.linalg.norm(Wt) / np.linalg.norm(Wc):.3f}  corr {cc:+.3f}  best scale {np.sum(Lt * Wt) / np.sum(Wt**2):+.3f}")
        break
    dt = 0.4 / float(np.sum(W**2))
    x = wrap(pos)
    rho_c = np.hypot(x[:, 0], x[:, 1])
    nst, ev = int(round(args.T / dt)), max(1, int(round(args.every / dt)))
    def centroid():
        bm = np.linalg.norm(C.curl_A(ac), axis=1)
        wgt = bm * (rho_c < side / 2.5)
        return (np.sum(wgt * rho_c) / wgt.sum(), np.angle(np.sum(wgt * np.exp(2j * np.pi * x[:, 2] / side))) * side / (2 * np.pi),
                bm)

    def vs_expect(v):
        return f" ({v / (args.expect * np.sign(kappa) / 0.132 * kappa):.3f} x expected)" if args.expect and kappa else ""

    ph_prev = ac.phi
    if kappa:
        _, z0, _ = centroid()
        ac.phi = phi + kappa * N_FN(phi)
        _, z1, _ = centroid()
        ac.phi = phi
        print(f"t 0 instantaneous speed {z1 - z0:+.5f}{vs_expect(z1 - z0)} (term only, unit time, linear estimate)", flush=True)
    import time
    light = args.light or max(1, nst // 40)
    t_wall, E0 = time.time(), None
    print(f"flow {args.flow} midpoint x{args.midpoint} kappa {kappa:+.1f} a0 {args.a0} R0 {args.R0} wt {args.wt} P {args.P} "
          f"dt {dt:.4f} ({nst} ticks)", flush=True)
    for s in range(nst + 1):
        if s % ev == 0 or (s and s % light == 0):
            rc_, zc, bm = centroid()
            u, ph = ac.u, ac.phi
            Elf = np.sum(u**2) + 0.5 * np.sum(ph_prev * ph) if s else np.nan
            if s and E0 is None:
                E0 = Elf
            if s % ev == 0:
                print(f"  t {s * dt:6.2f}: |B| centroid rho {rc_:6.2f} z {zc:+7.3f}; sum|B| "
                      f"{bm.sum():9.2f}; max |u| {np.abs(u).max():.3e}; E_lf {Elf:12.6f}", flush=True)
            else:
                w = (time.time() - t_wall) / s
                print(f"    . t {s * dt:6.2f} z {zc:+7.4f} running speed {zc / (s * dt):+.5f}{vs_expect(zc / (s * dt))} "
                      f"E_lf drift {Elf / E0 - 1:+.1e}  {w:.1f} s/tick ETA {w * (nst - s) / 60:.0f} min", flush=True)
        if s == nst:
            break
        ph_prev = ac.phi
        alat.step(0, dt)
        if kappa:
            carry(dt * kappa, ph_prev)
