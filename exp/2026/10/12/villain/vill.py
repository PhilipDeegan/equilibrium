"""10-12 villain: (b) + remainder, dynamic. phi = W (dtheta - 2 pi n + r) with the n <-> r transfer phi-neutral, so the
dynamics IS step() + kappa P_S [M, v] (11/lie) on a real phi; n is the label round(Gamma / 2 pi) of each elementary
triangle's loop sum Gamma of a = phi / W. IC: the 10-08 tick ring (theta = Omega / 2, phi = W wrap(dtheta): every
triangle carries exactly 0 or +-2 pi). Question: does the commutator move the concentrated ring while keeping each
triangle's circulation whole (no smearing into fractions), count fixed, energy exact? kappa 0: gradient-only, frozen."""
import argparse
import time

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box, solid_angle

C.spacing_of = lambda lat: 1.0
p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=120)
p.add_argument("--R0", type=float, default=15.0)
p.add_argument("--kappa", type=float, default=0.132)
p.add_argument("--T", type=float, default=10.0)
p.add_argument("--every", type=float, default=1.0)
p.add_argument("--midpoint", type=int, default=3)
p.add_argument("--threads", type=int, default=1)
p.add_argument("--dt_scale", type=float, default=1.0)
p.add_argument("--wt", type=float, default=0.0, help="0: tick ring (one-triangle core); >0: smooth ring tube width (11/lie IC)")
p.add_argument("--a0", type=float, default=1.0)
p.add_argument("--g", type=float, default=0.0, help="face rate sigma_f = 1 + g sin(2 pi x / side) on step() and the commutator")
p.add_argument("--gmode", default="both", help="both | step | lie: where the face rate applies")
p.add_argument("--smooth", type=float, default=0.0, help=">0: diffuse the tick ring vorticity (phi -= eps curl curl phi) for this long, then div-free project: smooth core, whole turn linked")
p.add_argument("--eps", type=float, default=0.02)
p.add_argument("--off", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="ring centre offset (break lattice symmetry: exact +-pi offsets)")
p.add_argument("--t0only", action="store_true", help="instantaneous d rms / dt from one commutator evaluation, then exit (fast fail)")
p.add_argument("--circ", type=float, default=0.0, help=">0: rescale the smooth ring so the circulation linked on the axis = circ x 2 pi")
args = p.parse_args()
side = args.P / np.sqrt(3)
D, W = C.D_HAT, C.W
TWO_PI = 2 * np.pi
wrap = lambda a: (a + np.pi) % TWO_PI - np.pi
wrapx = lambda x: (x + side / 2) % side - side / 2

alat = cpp_box(args.P, n_threads=args.threads)
ac = C.AnchorCluster(alat)
pos, nb, N = ac.pos, ac.nb, ac.N

key = lambda v: tuple(np.round(v, 6))
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
    a, v = ph / W, ph * W
    M2 = np.empty((N, len(KEYS)))
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


def carry(a_, ref):
    ph0 = ac.phi
    d = SIGF * lie(0.5 * (ref + ph0))
    for _ in range(args.midpoint):
        d = SIGF * lie(0.5 * (ref + ph0 + a_ * d))
    ac.phi = ph0 + a_ * d


tri = []
for ka in range(8):
    for kb in range(8):
        for ks in range(8, 14):
            if np.allclose(D[kb] - D[ka], D[ks]):
                tri.append((ka, kb, ks))
x = wrapx(pos)
kz = TWO_PI / side


def tri_stats(ph):
    a = ph / W
    G = np.stack([a[:, ka] + a[nb[:, ka], ks] - a[:, kb] for ka, kb, ks in tri], 1) / TWO_PI
    cen = np.stack([x + (D[ka] + D[kb]) / 3 for ka, kb, _ in tri], 1)
    n = np.rint(G)
    m = n != 0
    absG = np.abs(G)
    frac = np.abs(G - n)
    cd = wrapx(cen[m])
    rho = np.hypot(cd[:, 0], cd[:, 1])
    zc = np.angle(np.mean(np.exp(1j * kz * cd[:, 2]))) / kz
    cw = wrapx(cen.reshape(-1, 3))
    w = absG.ravel()
    zw = np.angle(np.sum(w * np.exp(1j * kz * cw[:, 2]))) / kz
    global XW
    XW = np.angle(np.sum(w * np.exp(1j * kz * cw[:, 0]))) / kz
    dcore = np.hypot(np.hypot(cw[:, 0], cw[:, 1]) - args.R0, cw[:, 2] - zw)
    leak = w[dcore > 3 + 2 * args.wt].sum() / w.sum()
    rms = np.sqrt(np.sum(w * dcore**2) / w.sum())
    return (int(m.sum()), int(n.sum()), rho.mean() if m.any() else np.nan, zc if m.any() else np.nan, zw,
            frac[m].max() if m.any() else 0.0, absG[~m].max(), absG[~m].sum() / max(1, m.sum()), leak, rms, w.sum())


s_ = np.linspace(0, TWO_PI, 400, endpoint=False)
ring = np.stack([args.R0 * np.cos(s_), args.R0 * np.sin(s_), 0 * s_], 1) + np.array(args.off)
shifts = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float) * side
phase = lambda X: 0.5 * solid_angle(ring, X, shifts=shifts, n_threads=args.threads)
rng = np.random.default_rng(0)
jumps = []
for ax in range(3):
    smp = (rng.random((400, 3)) - 0.5) * side
    smp[:, ax] = -side / 2
    hi = smp.copy()
    hi[:, ax] += side
    jumps.append(np.angle(np.mean(np.exp(1j * (phase(hi) - phase(smp))))))
th = (phase(pos) - sum(jumps[a] * pos[:, a] / side for a in range(3))) % TWO_PI
phi = W[None, :] * wrap(th[nb] - th[:, None])
if args.wt > 0:
    def A_fn(X):
        X = wrapx(X)
        rc = np.maximum(np.hypot(X[:, 0], X[:, 1]), 1.0)
        f = args.a0 * np.exp(-((rc - args.R0)**2 + X[:, 2]**2) / (2 * args.wt**2))
        dfr, dfz = -(rc - args.R0) / args.wt**2 * f, -X[:, 2] / args.wt**2 * f
        Ar, Az = -dfz, f / rc + dfr
        return np.stack([Ar * X[:, 0] / rc, Ar * X[:, 1] / rc, Az], 1)
    mid = (pos[:, None, :] + 0.5 * D[None]).reshape(-1, 3)
    phi = C.project_div_free(ac, W[None, :] * np.einsum("nkd,kd->nk", A_fn(mid).reshape(N, 14, 3), D))


def zchain(r):
    i = int(np.argmin(np.sum((x - [r, 0, 0])**2, 1)))
    ch = [i]
    for _ in range(args.P // 2 - 1):
        ch.append(nb[ch[-1], 13])
    assert nb[ch[-1], 13] == ch[0]
    return np.array(ch)


CH = {r: zchain(r) for r in (0.0, args.R0 / 2, args.R0 + 3 * max(args.wt, 1) + 3)}
linked = lambda ph: [np.sum(ph[c, 13] / W[13]) / TWO_PI for c in CH.values()]
if args.circ > 0:
    phi = phi * args.circ / linked(phi)[0]
if args.smooth > 0:
    print(f"linked before smoothing {' '.join(f'{v:+.6f}' for v in linked(phi))}", flush=True)
    for it in range(int(round(args.smooth / args.eps))):
        phi = phi - args.eps * ac.curl_curl_dec(phi)
    phi = C.project_div_free(ac, phi)
    print(f"smoothed tau {args.smooth}: linked {' '.join(f'{v:+.6f}' for v in linked(phi))}", flush=True)
ac.u, ac.phi = np.zeros(N), phi
xm = wrapx(pos[:, None, :] + 0.5 * D[None])[..., 0]
DSIG = args.g * np.sin(TWO_PI * xm / side)
DSIG = 0.5 * (DSIG + DSIG[nb, C.K_MIRROR[None, :]])
SIGF = 1.0 + DSIG if args.gmode in ("both", "lie") else 1.0
if args.g and args.gmode in ("both", "step"):
    ac.set_face_medium(DSIG, np.zeros_like(DSIG))
print("linked-circulation lines at r", " ".join(f"{r:.1f}" for r in CH), flush=True)

if args.t0only:
    t_ = time.time()
    r0 = tri_stats(phi)[9]
    L0 = lie(phi)
    eps_ = 1e-3
    rp, rm = tri_stats(phi + eps_ * args.kappa * L0)[9], tri_stats(phi - eps_ * args.kappa * L0)[9]
    print(f"t0only: rms {r0:.4f}  d rms / dt {(rp - rm) / (2 * eps_):+.4f}  ({time.time() - t_:.0f} s)", flush=True)
    raise SystemExit
dt = args.dt_scale * 0.4 / float(np.sum(W**2))
nst, ev = int(round(args.T / dt)), max(1, int(round(args.every / dt)))
st = tri_stats(phi)
d1 = tri_stats(phi + args.kappa * lie(phi)) if args.kappa else st
print(f"P {args.P} N {N} R0 {args.R0} kappa {args.kappa} dt {dt:.4f} ticks {nst} midpoint x{args.midpoint}", flush=True)
print(f"t 0 instantaneous (term x unit time): |Gamma|-weighted z {d1[4] - st[4]:+.5f}; defect-label z {d1[3] - st[3]:+.5f}; "
      f"max off-integer at defects {d1[5]:.3e}; max |Gamma| off-defect {d1[6]:.3e}", flush=True)
hdr = "count sum_n rho z_label z_|G| max|G-n|@def max|G|@non sum|G|@non/count"
print(f"  {'t':>6s} {hdr}  E_lf", flush=True)
ph_prev, t_wall = ac.phi, time.time()
for s in range(nst + 1):
    if s % ev == 0:
        ph, u = ac.phi, ac.u
        c, sn, rho, zl, zw, fdef, gnon, snon, leak, rms, tot = tri_stats(ph)
        Elf = np.sum(u**2) + 0.5 * np.sum(ph_prev * ph)
        wt = (time.time() - t_wall) / max(1, s)
        print(f"  {s * dt:6.2f} {c:5d} {sn:+5d} {rho:6.2f} {zl:+7.4f} {zw:+7.4f} {fdef:.3e} {gnon:.3e} {snon:.3e}  {Elf:.6f}"
              f"  leak {leak:.3e} rms {rms:.3f} sum|G| {tot:.4e} linked {' '.join(f'{v:+.6f}' for v in linked(ph))} x_|G| {XW:+.5f}"
              f"  ({wt:.1f} s/tick, ETA {wt * (nst - s) / 60:.0f} min)", flush=True)
    if s == nst:
        break
    ph_prev = ac.phi
    alat.step(0, dt)
    if args.kappa:
        carry(dt * args.kappa, ph_prev)
