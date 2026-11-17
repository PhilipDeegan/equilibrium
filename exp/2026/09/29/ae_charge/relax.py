"""Continuous AE (C++ step_ae, iso weights, h = 1): seed only a charge -- E with div E = +1 at the
core, -1 on the outer shell (closed lattice), u = A = 0 -- and let the dynamics build the potential.
Static state: A_t = -grad u - E = 0 -> e = -W (u_n - u_i) (h = 1) -> u = psi, the discrete
Poisson solution (equilibrium/substrate/charge_exterior.py). Nothing held, no sponge: the
transient keeps ringing, so compare the time-averaged u with psi. div E must stay exactly
constant with the DEC curl (charge conservation)."""

import argparse
from pathlib import Path

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.charge_exterior import capacitor_source, divergence, seed_charge

p = argparse.ArgumentParser()
p.add_argument("--R", type=float, default=8.0)
p.add_argument("--curl", default="dec")
p.add_argument("--crossings", type=float, default=40.0)
p.add_argument("--threads", type=int, default=4)
args = p.parse_args()

ac = C.anchor_cluster(args.R, 1.0, n_threads=args.threads)
ac.set_curl(args.curl)
s, core, shell = capacitor_source(ac.pos)
e0, psi = seed_charge(ac.nb, C.W, s)
ac.e = e0
ac.u = np.zeros(ac.N)
ac.phi = np.zeros((ac.N, 14))
dt = C.stable_dt(ac)
n = int(round(args.crossings * 2 * args.R / C.C0 / dt))
r = np.linalg.norm(ac.pos, axis=1)
acc = np.zeros(ac.N)
n_avg = 0
d0 = divergence(ac.e, ac.nb, C.W)
for i in range(n):
    ac.step_ae(dt)
    if i >= n // 2:
        acc += ac.u
        n_avg += 1
u_avg = acc / n_avg
dd = np.abs(divergence(ac.e, ac.nb, C.W) - d0).max()
m = ~shell
x, y = psi[m] - psi[m].mean(), u_avg[m] - u_avg[m].mean()
a = float(x @ y / (x @ x))
res = np.sqrt(np.mean((y - a * x) ** 2)) / np.abs(y).max()
inner = (r > 1.5) & (r < args.R - 2)
print(f"curl={args.curl} R={args.R} N={ac.N} steps={n}: max |div E change| {dd:.2e}; "
      f"<u> = a psi + c: a = {a:.4f}, rel rms resid {res:.3e}; core <u> - shell <u> = "
      f"{u_avg[core] - u_avg[shell].mean():.4f} vs psi {psi[core] - psi[shell].mean():.4f}; "
      f"inner rel resid {np.sqrt(np.mean((u_avg[inner]-u_avg[inner].mean() - a*(psi[inner]-psi[inner].mean()))**2))/np.abs(psi[inner]-psi[inner].mean()).max():.3e}")
np.savez(Path(__file__).parent / f"relax_{args.curl}_R{args.R:g}.npz", u_avg=u_avg, psi=psi, pos=ac.pos)
