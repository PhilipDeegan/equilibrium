"""step() with a compression-dependent face rate and a one-sided compression asymptote, no Cartan.
H = sum_i e(u_i) + 1/4 sum_{i,k} s_ik phi_ik^2,  s_ik = (s(u_i) + s(u_nb)) / 2 (relational, at the face)
  e(u) = um^2 (-x - ln(1 - x)), x = u / um   (~ u^2 / 2 small; push-back -> infinity as u -> um; soft for u < 0)
  s(u) = exp(-beta x)                        (beta > 0: compression lowers the cost of circulation -> pinch candidate)
u_t = -sum_k W_k s_ik phi_ik,   phi_ik_t = -W_k (g_nb - g_i),   g_i = e'(u_i) + 1/4 s'(u_i) sum_k phi_ik^2
Energy-exact in continuous time (RK4 here, drift printed). phi changes by gradients only -> every loop sum of phi
is frozen (circulation cannot move or change); what can respond is compression u and the gradient part of phi.
um = inf, beta = 0 is plain step() (div-free phi static, u stays 0).
usage: pinch.py --seed S --P 96 --a 6 --beta 0.5 --um 0.3 --T 60"""
import argparse
import time
from pathlib import Path

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

C.spacing_of = lambda lat: 1.0
p = argparse.ArgumentParser()
p.add_argument("--seed", required=True)
p.add_argument("--P", type=int, default=96)
p.add_argument("--a", type=float, default=6.0)
p.add_argument("--beta", type=float, default=0.0)
p.add_argument("--um", type=float, default=np.inf)
p.add_argument("--T", type=float, default=60.0)
p.add_argument("--dt", type=float, default=0.05)
p.add_argument("--every", type=float, default=2.0)
p.add_argument("--save", default="")
args = p.parse_args()
ac = C.AnchorCluster(cpp_box(args.P, n_threads=4))
N, nb = ac.N, ac.nb
valid = nb >= 0
nbs = np.where(valid, nb, 0)
W = C.W[None, :]
side = args.P / np.sqrt(3)
OFF = np.array([0.13, 0.07, 0.11])
x = (ac.pos - OFF + side / 2) % side - side / 2
r = np.linalg.norm(x, axis=1)
um, beta = args.um, args.beta


def e(u):
    if np.isinf(um):
        return 0.5 * u**2
    xx = u / um
    return um**2 * (-xx - np.log1p(-xx))


def de(u):
    return u if np.isinf(um) else u / (1 - u / um)


def s(u):
    return np.ones_like(u) if beta == 0 or np.isinf(um) else np.exp(-beta * u / um)


def ds(u):
    return np.zeros_like(u) if beta == 0 or np.isinf(um) else -beta / um * s(u)


def sface(u):
    sv = s(u)
    return np.where(valid, 0.5 * (sv[:, None] + sv[nbs]), 0.0)


def rhs(u, phi):
    ut = -np.sum(W * sface(u) * phi, 1)
    g = de(u) + 0.25 * ds(u) * np.sum(phi**2, 1)
    pt = np.where(valid, -W * (g[nbs] - g[:, None]), 0.0)
    return ut, pt


def energy(u, phi):
    return e(u).sum() + 0.25 * np.sum(sface(u) * phi**2)


phi = np.load(args.seed)["phi"]
u = np.zeros(N)
E0 = energy(u, phi)
loop0 = None
SH = np.array([0.5, 1.0, 1.5, 2.5]) * args.a
nst, ev = int(round(args.T / args.dt)), max(1, int(round(args.every / args.dt)))
print(f"P {args.P} a {args.a} N {N} beta {beta} um {um} dt {args.dt} steps {nst}  E0 {E0:.6e}", flush=True)
print("   t     dE/E       umax     umin    <u>r<a/2  <u>a/2..a  <u>a..1.5a  <u>1.5a..2.5a  Eu/E   circE in/out(r>1.5a)")
tw = time.time()
t, nxt, st = 0.0, 0.0, 0
while True:
    if t >= nxt - 1e-12:
        nxt += args.every
        E = energy(u, phi)
        Eu = e(u).sum()
        bins = np.searchsorted(SH, r)
        mu = [u[bins == b].mean() for b in range(4)]
        ce = 0.25 * np.sum(sface(u) * phi**2, 1)
        xm = u.max() / um if np.isfinite(um) else 0.0
        print(f"{t:6.1f}  {E / E0 - 1:+.2e}  {u.max():+.4f}  {u.min():+.4f}  "
              + "  ".join(f"{m:+.5f}" for m in mu)
              + f"  {Eu / E:.4f}  {ce[r < 1.5 * args.a].sum() / E:.4f}/{ce[r > 1.5 * args.a].sum() / E:.4f}"
              + f"  x_max {xm:.4f}  steps {st}  ({time.time() - tw:.0f} s)", flush=True)
        if not np.isfinite(E):
            break
    if t >= args.T - 1e-12 or not np.isfinite(t):
        break
    xm = u.max() / um if np.isfinite(um) else 0.0
    dt = min(args.dt * min(1.0, 4 * (1 - xm)), args.T - t)
    k1 = rhs(u, phi)
    k2 = rhs(u + 0.5 * dt * k1[0], phi + 0.5 * dt * k1[1])
    k3 = rhs(u + 0.5 * dt * k2[0], phi + 0.5 * dt * k2[1])
    k4 = rhs(u + dt * k3[0], phi + dt * k3[1])
    u = u + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    phi = phi + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    t += dt
    st += 1
if args.save:
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.save, u=u, phi=phi, r=r)
