"""10-16 geom: 10-14 vill.py + --geom plain | twist | tokamak | hopf (same step() + Cartan rules, vary geometry).
twist: toroidal swirl v_phi = v0 exp(-d^2 / 2 wsw^2) through the core (helical vorticity lines).
tokamak: whole-turn core + toroidal flow v0 inside a shell d < rshell (poloidal vorticity sheet = windings around the core).
hopf: a = Hopf-fibre tangent field (2(xz - y), 2(yz + x), 1 - rho^2 + z^2) / (1 + r^2)^3 at scale R0: every vorticity line
links every other once, a || da (Lamb vector 0, steady Euler). v0 = wind / R0: wind = toroidal-loop circulation in turns.
10-13 cartan: vill.py (10-12) + --flow cartan: da_e/dt = -kappa i_v(da)_e, vector-free:
i(e) = -(3/4) / W_e^2 sum_{triangles t containing e} omega_t (a_next - a_prev) (loop-oriented, omega_t = loop sum of a).
Antisymmetric per triangle -> sum v . term = 0 exactly; zero wherever every triangle touching e has omega 0 (no
circulation created in irrotational flow). 3/4 and hex-hex / hex-square = 1/2 fixed by fit.py (exact i_u omega).
10-12 villain: (b) + remainder, dynamic. phi = W (dtheta - 2 pi n + r) with the n <-> r transfer phi-neutral, so the
dynamics IS step() + kappa P_S [M, v] (11/lie) on a real phi; n is the label round(Gamma / 2 pi) of each elementary
triangle's loop sum Gamma of a = phi / W. IC: the 10-08 tick ring (theta = Omega / 2, phi = W wrap(dtheta): every
triangle carries exactly 0 or +-2 pi). Question: does the commutator move the concentrated ring while keeping each
triangle's circulation whole (no smearing into fractions), count fixed, energy exact? kappa 0: gradient-only, frozen."""
import argparse
import time
from pathlib import Path

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
p.add_argument("--flow", default="cartan", help="cartan | lie")
p.add_argument("--cpp", action="store_true", help="flow cartan: step() + transport in C++ (alat.step_cartan)")
p.add_argument("--circ", type=float, default=0.0, help=">0: rescale the smooth ring so the circulation linked on the axis = circ x 2 pi")
p.add_argument("--seed", default="", help="npz: load the post-smoothing phi if present, else save it")
p.add_argument("--ckpt", default="", help="npz: resume from it if present; rewritten at every print")
p.add_argument("--rcut", type=float, nargs="+", default=[4.0, 6.0], help="energy fraction in cells beyond these distances from the core line")
p.add_argument("--geom", default="plain")
p.add_argument("--wind", type=float, default=1.0, help="twist / tokamak: circulation along the core loop, in turns")
p.add_argument("--wsw", type=float, default=3.0)
p.add_argument("--rshell", type=float, default=6.0)
p.add_argument("--wshell", type=float, default=1.5)
args = p.parse_args()
side = args.P / np.sqrt(3)
D, W, MIR = C.D_HAT, C.W, C.K_MIRROR
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


def cartan(ph):
    a = ph / W
    T = np.zeros_like(ph)
    for ka, kb, ks in tri:
        j = nb[:, ka]
        a1, a2, a3 = a[:, ka], a[j, ks], -a[:, kb]
        w = a1 + a2 + a3
        T[:, ka] += w * (a2 - a3)
        T[j, ks] += w * (a3 - a1)
        T[:, kb] -= w * (a1 - a2)
    T = 0.5 * (T - T[nb, MIR[None, :]])
    return 0.75 * T / W


def carry(a_, ref):
    ph0 = ac.phi
    d = SIGF * FLOW(0.5 * (ref + ph0))
    for _ in range(args.midpoint):
        d = SIGF * FLOW(0.5 * (ref + ph0 + a_ * d))
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
    global RMS2
    RMS2 = np.sqrt(np.sum(w**2 * dcore**2) / np.sum(w**2))
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
if args.seed and Path(args.seed).exists():
    phi = np.load(args.seed)["phi"]
    print(f"seed loaded {args.seed}: linked {' '.join(f'{v:+.6f}' for v in linked(phi))}", flush=True)
elif args.smooth > 0:
    print(f"linked before smoothing {' '.join(f'{v:+.6f}' for v in linked(phi))}", flush=True)
    for it in range(int(round(args.smooth / args.eps))):
        phi = phi - args.eps * ac.curl_curl_dec(phi)
    phi = C.project_div_free(ac, phi)
    print(f"smoothed tau {args.smooth}: linked {' '.join(f'{v:+.6f}' for v in linked(phi))}", flush=True)
    if args.seed:
        Path(args.seed).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.seed, phi=phi)
mid = wrapx(pos[:, None, :] + 0.5 * D[None] - np.array(args.off)).reshape(-1, 3)


def tor_field(X, prof):
    rho = np.maximum(np.hypot(X[:, 0], X[:, 1]), 1e-9)
    f = prof(np.hypot(rho - args.R0, X[:, 2]))
    return np.stack([-X[:, 1] / rho * f, X[:, 0] / rho * f, 0 * f], 1)


def hopf(X):
    x_, y_, z_ = (X / args.R0).T
    r2 = x_**2 + y_**2 + z_**2
    return np.stack([2 * (x_ * z_ - y_), 2 * (y_ * z_ + x_), 1 - x_**2 - y_**2 + z_**2], 1) / (1 + r2[:, None])**3


v0 = args.wind / args.R0
prof = {"twist": lambda d: v0 * np.exp(-d**2 / (2 * args.wsw**2)),
        "tokamak": lambda d: v0 * 0.5 * (1 - np.tanh((d - args.rshell) / args.wshell))}
if args.geom in prof:
    phi = C.project_div_free(ac, phi + W[None, :] * np.einsum("nkd,kd->nk", tor_field(mid, prof[args.geom]).reshape(N, 14, 3), D))
elif args.geom == "hopf":
    phi = C.project_div_free(ac, W[None, :] * np.einsum("nkd,kd->nk", hopf(mid).reshape(N, 14, 3), D))
    phi = phi * max(args.circ, 1.0) / linked(phi)[0]
cl = np.stack([args.R0 * np.cos(s_), args.R0 * np.sin(s_), 0 * s_], 1)
tl = np.stack([-np.sin(s_), np.cos(s_), 0 * s_], 1)
vl = {"hopf": lambda X: hopf(X) * max(args.circ, 1.0) / linked_raw, "plain": lambda X: 0 * X}
if args.geom == "hopf":
    linked_raw = linked(W[None, :] * np.einsum("nkd,kd->nk", hopf(mid).reshape(N, 14, 3), D))
    linked_raw = linked_raw[0] / 1.0
    vcore = vl["hopf"](cl)
elif args.geom in prof:
    vcore = tor_field(cl, prof[args.geom])
else:
    vcore = 0 * cl
print(f"geom {args.geom}: core-loop circulation (analytic seed winding) {np.sum(vcore * tl, 1).mean() * args.R0:+.4f} turns", flush=True)
ac.u, ac.phi = np.zeros(N), phi
FLOW = cartan if args.flow == 'cartan' else lie
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
    L0 = FLOW(phi)
    eps_ = 1e-3
    rp, rm = tri_stats(phi + eps_ * args.kappa * L0)[9], tri_stats(phi - eps_ * args.kappa * L0)[9]
    print(f"t0only: rms {r0:.4f}  d rms / dt {(rp - rm) / (2 * eps_):+.4f}  ({time.time() - t_:.0f} s)", flush=True)
    raise SystemExit
dt = args.dt_scale * 0.4 / float(np.sum(W**2))
nst, ev = int(round(args.T / dt)), max(1, int(round(args.every / dt)))
st = tri_stats(phi)
d1 = tri_stats(phi + args.kappa * FLOW(phi)) if args.kappa else st
print(f"P {args.P} N {N} R0 {args.R0} kappa {args.kappa} dt {dt:.4f} ticks {nst} midpoint x{args.midpoint}", flush=True)
print(f"t 0 instantaneous (term x unit time): |Gamma|-weighted z {d1[4] - st[4]:+.5f}; defect-label z {d1[3] - st[3]:+.5f}; "
      f"max off-integer at defects {d1[5]:.3e}; max |Gamma| off-defect {d1[6]:.3e}", flush=True)
hdr = "count sum_n rho z_label z_|G| max|G-n|@def max|G|@non sum|G|@non/count"
print(f"  {'t':>6s} {hdr}  E_lf", flush=True)
ph_prev, t_wall, s0 = ac.phi, time.time(), 0
if args.ckpt and Path(args.ckpt).exists():
    ck = np.load(args.ckpt)
    ac.phi, ac.u, ph_prev, s0 = ck["phi"], ck["u"], ck["ph_prev"], int(ck["s"])
    print(f"resumed {args.ckpt} at t {s0 * dt:.2f}", flush=True)
for s in range(s0, nst + 1):
    if s % ev == 0:
        ph, u = ac.phi, ac.u
        c, sn, rho, zl, zw, fdef, gnon, snon, leak, rms, tot = tri_stats(ph)
        Elf = np.sum(u**2) + 0.5 * np.sum(ph_prev * ph)
        ec = u**2 + 0.5 * np.sum(ph**2, 1)
        dc = np.hypot(np.hypot(x[:, 0], x[:, 1]) - args.R0, wrapx(x[:, 2] - zw))
        fo = " ".join(f"{ec[dc > rc_].sum() / ec.sum():.6e}" for rc_ in args.rcut)
        wt = (time.time() - t_wall) / max(1, s - s0)
        print(f"  {s * dt:6.2f} {c:5d} {sn:+5d} {rho:6.2f} {zl:+7.4f} {zw:+7.4f} {fdef:.3e} {gnon:.3e} {snon:.3e}  {Elf:.6f}"
              f"  leak {leak:.3e} rms {rms:.3f} rms2 {RMS2:.3f} sum|G| {tot:.4e} linked {' '.join(f'{v:+.6f}' for v in linked(ph))} x_|G| {XW:+.5f} Eout {fo} Eu {np.sum(u**2) / ec.sum():.6e}"
              f"  ({wt:.1f} s/tick, ETA {wt * (nst - s) / 60:.0f} min)", flush=True)
        if args.ckpt and s > s0:
            np.savez(args.ckpt + ".tmp.npz", phi=ph, u=u, ph_prev=ph_prev, s=s)
            Path(args.ckpt + ".tmp.npz").replace(args.ckpt)
    if s == nst:
        break
    ph_prev = ac.phi
    if args.cpp and args.flow == "cartan" and not args.g:
        alat.step_cartan(0, dt, args.kappa, args.midpoint)
        continue
    alat.step(0, dt)
    if args.kappa:
        carry(dt * args.kappa, ph_prev)
