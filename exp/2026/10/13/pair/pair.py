"""10-13 pair: two coaxial rings (z = -+d / 2) under step() + continuous.cartan_transport (kappa 1). --kind turn: each
links a whole turn (tick phase 0.5 Omega, smoothed tau, then div-free project; sign s on the second); --kind zero:
11/lie net-zero smooth tubes (wt, a0; sign s). --single: first ring only (reference). Per-ring |Gamma|-weighted z /
rho (triangles nearest each ring's last z), energy, linked circulation on the axis. Question (Philip 10-13: imbalance
is what bonds): do whole-turn rings (exterior flow) interact where net-zero rings (no exterior) do not?"""
import argparse
import time

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box, solid_angle

C.spacing_of = lambda lat: 1.0
p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=120)
p.add_argument("--R0", type=float, default=9.0)
p.add_argument("--d", type=float, default=9.0)
p.add_argument("--s", type=float, default=1.0)
p.add_argument("--kind", default="turn")
p.add_argument("--single", action="store_true")
p.add_argument("--tau", type=float, default=1.0)
p.add_argument("--eps", type=float, default=0.02)
p.add_argument("--wt", type=float, default=3.0)
p.add_argument("--a0", type=float, default=1.0)
p.add_argument("--kappa", type=float, default=1.0)
p.add_argument("--T", type=float, default=30.0)
p.add_argument("--every", type=float, default=1.0)
p.add_argument("--threads", type=int, default=2)
p.add_argument("--midpoint", type=int, default=3)
p.add_argument("--off", type=float, nargs=3, default=[0.13, 0.07, 0.11])
args = p.parse_args()
side = args.P / np.sqrt(3)
D, W = C.D_HAT, C.W
TWO_PI = 2 * np.pi
wrap = lambda a: (a + np.pi) % TWO_PI - np.pi
wrapx = lambda x: (x + side / 2) % side - side / 2
t_wall = time.time()
alat = cpp_box(args.P, n_threads=args.threads)
ac = C.AnchorCluster(alat)
pos, nb, N = ac.pos, ac.nb, ac.N
x = wrapx(pos)
off = np.array(args.off)
zs = [-args.d / 2, args.d / 2] if not args.single else [0.0]
sg = [1.0, args.s]
if args.kind == "turn":
    s_ = np.linspace(0, TWO_PI, 400, endpoint=False)
    shifts = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float) * side
    def phase(X):
        th = 0
        for z, g in zip(zs, sg):
            ring = np.stack([args.R0 * np.cos(s_), args.R0 * np.sin(s_), 0 * s_ + z], 1) + off
            th = th + g * 0.5 * solid_angle(ring, X, shifts=shifts, n_threads=args.threads)
        return th
    rng = np.random.default_rng(0)
    jumps = []
    for ax in range(3):
        smp = (rng.random((200, 3)) - 0.5) * side
        smp[:, ax] = -side / 2
        hi = smp.copy()
        hi[:, ax] += side
        jumps.append(np.angle(np.mean(np.exp(1j * (phase(hi) - phase(smp))))))
    th = (phase(pos) - sum(jumps[a] * pos[:, a] / side for a in range(3))) % TWO_PI
    phi = W[None, :] * wrap(th[nb] - th[:, None])
    for _ in range(int(round(args.tau / args.eps))):
        phi = phi - args.eps * ac.curl_curl_dec(phi)
    phi = C.project_div_free(ac, phi)
else:
    def A_fn(X):
        X = wrapx(X - off)
        rc = np.maximum(np.hypot(X[:, 0], X[:, 1]), 1.0)
        A = 0
        for z, g in zip(zs, sg):
            dz = wrapx(X[:, 2:3] - z)[:, 0]
            f = g * args.a0 * np.exp(-((rc - args.R0)**2 + dz**2) / (2 * args.wt**2))
            dfr, dfz = -(rc - args.R0) / args.wt**2 * f, -dz / args.wt**2 * f
            Ar, Az = -dfz, f / rc + dfr
            A = A + np.stack([Ar * X[:, 0] / rc, Ar * X[:, 1] / rc, Az], 1)
        return A
    mid = (pos[:, None, :] + 0.5 * D[None]).reshape(-1, 3)
    phi = C.project_div_free(ac, W[None, :] * np.einsum("nkd,kd->nk", A_fn(mid).reshape(N, 14, 3), D))
ac.u, ac.phi = np.zeros(N), phi
print(f"IC built ({time.time() - t_wall:.0f} s)", flush=True)

tri = [(ka, kb, ks) for ka in range(8) for kb in range(8) for ks in range(8, 14) if np.allclose(D[kb] - D[ka], D[ks])]
cen = np.stack([wrapx(x + (D[ka] + D[kb]) / 3 - off) for ka, kb, _ in tri], 1).reshape(-1, 3)
rho_t = np.hypot(cen[:, 0], cen[:, 1])
kz = TWO_PI / side
ich = int(np.argmin(np.sum((x - off)**2 * [1, 1, 0], 1)))
ch = [ich]
for _ in range(args.P // 2 - 1):
    ch.append(nb[ch[-1], 13])
ch = np.array(ch)


def stats(ph, zprev):
    a = ph / W
    w = np.abs(np.stack([a[:, ka] + a[nb[:, ka], ks] - a[:, kb] for ka, kb, ks in tri], 1).ravel())
    dist = np.stack([np.abs(wrapx(cen[:, 2] - z)) for z in zprev], 1)
    own = np.argmin(dist, 1)
    out = []
    for k in range(len(zprev)):
        m = (own == k) & (w > 1e-3 * w.max())
        wk = w[m]
        zc = zprev[k] + np.angle(np.sum(wk * np.exp(1j * kz * wrapx(cen[m, 2] - zprev[k])))) / kz
        out.append((wrapx(zc), np.sum(wk * rho_t[m]) / wk.sum(), wk.sum()))
    return out, np.sum(ph[ch, 13] / W[13]) / TWO_PI


def carry(a_, ref):
    ph0 = ac.phi
    d = C.cartan_transport(ac, 0.5 * (ref + ph0))
    for _ in range(args.midpoint):
        d = C.cartan_transport(ac, 0.5 * (ref + ph0 + a_ * d))
    ac.phi = ph0 + a_ * d


dt = 0.4 / float(np.sum(W**2))
nst, ev = int(round(args.T / dt)), max(1, int(round(args.every / dt)))
print(f"kind {args.kind} single {args.single} P {args.P} N {N} R0 {args.R0} d {args.d} s {args.s} kappa {args.kappa} "
      f"dt {dt:.4f} ticks {nst}", flush=True)
zprev = list(zs)
ph_prev, t0 = ac.phi, time.time()
for st in range(nst + 1):
    if st % ev == 0:
        ph = ac.phi
        rs, lk = stats(ph, zprev)
        zprev = [r[0] for r in rs]
        E = np.sum(ac.u**2) + 0.5 * np.sum(ph_prev * ph)
        wtk = (time.time() - t0) / max(1, st)
        print(f"  t {st * dt:6.2f} " + " | ".join(f"z {z:+8.3f} rho {r:6.3f} sum|G| {g:8.2f}" for z, r, g in rs)
              + f" | gap {(wrapx(zprev[1] - zprev[0]) if len(zprev) > 1 else 0):+7.3f} linked {lk:+.5f} E {E:.6f}"
              f" ({wtk:.2f} s/tick ETA {wtk * (nst - st) / 60:.0f} min)", flush=True)
    if st == nst:
        break
    ph_prev = ac.phi
    alat.step(0, dt)
    if args.kappa:
        carry(dt * args.kappa, ph_prev)
