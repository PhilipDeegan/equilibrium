"""Force proxy for two flux sources on step(): F_2 = <q_2(t) g_2 . A_1(t)>,
the time-averaged coupling of source 2's charge to source 1's radial field
(step()'s static charge field is A = phi-reconstruction; DC energy
E_int = a/d gives F = -dE/dd = +a/d^2 along +x for like sources, and
q_2 A_1,x(x_2) reproduces that). Single-source runs only (superposition);
F(anti) = -F(like) exactly. Positive = pushes source 2 away (+x).
Continuum point-source reference (u_tt = c^2 lap u + qdot g, A_t = -grad u):
  A_r = Q/(4 pi c^2) [cos(w tau)/r^2 - k sin(w tau)/r],  tau = t - r/c
  <cos(w t + D) A_r(d)> = Q^2/(8 pi c^2) [cos(kd + D)/d^2 + k sin(kd + D)/d]
DC: Q^2/(4 pi c^2 d^2). Gaussian source width s: radiative term x
exp(-(ks)^2/2) per source (form factor); near-field term ~unchanged for d >> s.
"""
import argparse

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate import continuous as C

C.set_weights("legacy")  # written for W = 3:1 on both legs (pre-2026-09-28; see continuous.WEIGHTS)

p = argparse.ArgumentParser()
p.add_argument("--R", type=float, default=40.0)
p.add_argument("--s", type=float, default=1.5)
p.add_argument("--w", type=float, default=1.911)
p.add_argument("--ds", default="3,4.5,6,9,12")
p.add_argument("--n-periods", type=int, default=6)
p.add_argument("--T-dc", type=float, default=30.0)
p.add_argument("--threads", type=int, default=8)
args = p.parse_args()
ct = _lat.continuous
c0, c2 = C.C0, C.C0SQ
V = (2 / 3**0.5) ** 3 / 2

ac = C.anchor_cluster(args.R, 1.0, args.threads)
x = ac.pos
r = np.linalg.norm(x, axis=1)
ac.cl.set_bc(ct.BoundaryCondition.quadratic_sponge(target=0.0, damp_max=0.02, r_all=r, r_inner=0.8 * r.max()))


def gauss(x0):
    g = np.exp(-0.5 * ((x - x0) ** 2).sum(1) / args.s**2)
    return g / (g.sum() * V)


g1 = gauss(np.zeros(3))
ds = [float(v) for v in args.ds.split(",")]
g2s = [gauss(np.array([d, 0, 0])) for d in ds]
dt = C.stable_dt(ac)
if args.w > 0:
    per = 2 * np.pi / args.w
    t_spin = 2 * args.R / c0 + per
    T = t_spin + args.n_periods * per
else:
    t_spin, T = 0.66 * args.T_dc, args.T_dc
n = int(np.ceil(T / dt))
acc_c = np.zeros(len(ds))
acc_s = np.zeros(len(ds))
cnt = 0
for i in range(n):
    t = i * dt
    q = np.cos(args.w * t) if args.w > 0 else min(1.0, t / (0.33 * args.T_dc))
    u = ac.u
    u += dt * q * g1
    ac.u = u
    ac.alat.step(0, dt)
    ac.alat.step_sponge(0)
    if t >= t_spin and i % 10 == 0:
        Ax = C.reconstruct_A(ac)[:, 0]
        tp = t + 1.5 * dt
        for j, g2 in enumerate(g2s):
            a = float((g2 * Ax).sum() * V)
            acc_c[j] += np.cos(args.w * tp) * a if args.w > 0 else a
            acc_s[j] += -np.sin(args.w * tp) * a if args.w > 0 else 0.0
        cnt += 1
F0 = acc_c / cnt
Fpi = -F0
k = args.w / c0
print(f"w={args.w} k={k:.4f} s={args.s} R={args.R}")
for j, d in enumerate(ds):
    if args.w > 0:
        ref = (1 / (8 * np.pi * c2)) * (np.cos(k * d) / d**2 + k * np.sin(k * d) / d)
        ff = np.exp(-(k * args.s) ** 2)
        ref_ff = (1 / (8 * np.pi * c2)) * (np.cos(k * d) / d**2 + ff * k * np.sin(k * d) / d)
    else:
        ref = ref_ff = 1 / (4 * np.pi * c2 * d**2)
    print(f"d={d:5.1f} kd={k*d:5.2f}  F_like={F0[j]:+.4e}  point ref={ref:+.4e}  ref w/ form factor={ref_ff:+.4e}"
          f"  -> like {'REPEL' if F0[j] > 0 else 'ATTRACT'}")
np.save(f"force_w{args.w:g}.npy", np.array([ds, F0]))
