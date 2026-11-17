"""Interaction of two periodic flux sources on step() (level 0): does the
substrate make like-phase sources attract or repel?

Sources inject at a prescribed rate: u_t = -div phi + q_i(t) g_i(x),
q_i = Q cos(w t + ph_i) (w = 0: DC, ramped), g a normalised gaussian. In a
static steady state div phi = q g (the source IS a flux divergence, the
Kirchhoff charge), with u uniform -- step()'s static exterior field is
radial phi. Flux is held fixed, not u, so F = -dE_int/dd with
E_int(d) = <H_both> - <H_1> - <H_2>, H = step()'s energy (u^2/2sigma +
1/4 sum phi^2/sigma_f) over the interior (r < r_meas, sponge excluded),
time-averaged over whole periods after spin-up.
Driven sources are pinned in amplitude: this measures the SIGN of the
interaction only, not evidence for a real particle.
"""
import argparse
import time

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate import continuous as C

C.set_weights("legacy")  # written for W = 3:1 on both legs (pre-2026-09-28; see continuous.WEIGHTS)

p = argparse.ArgumentParser()
p.add_argument("--R", type=float, default=40.0)
p.add_argument("--s", type=float, default=1.5)
p.add_argument("--Q", type=float, default=1.0)
p.add_argument("--w", type=float, default=0.0, help="angular frequency (0 = DC)")
p.add_argument("--ds", default="4,6,8,12")
p.add_argument("--n-periods", type=int, default=4)
p.add_argument("--T-dc", type=float, default=30.0)
p.add_argument("--threads", type=int, default=8)
args = p.parse_args()
ct = _lat.continuous
c0 = C.C0


def make():
    ac = C.anchor_cluster(args.R, 1.0, args.threads)
    r = np.linalg.norm(ac.pos, axis=1)
    ac.cl.set_bc(ct.BoundaryCondition.quadratic_sponge(target=0.0, damp_max=0.02, r_all=r, r_inner=0.8 * r.max()))
    return ac, r


V_CELL = (2 / 3**0.5) ** 3 / 2


def gauss(ac, x0):
    d2 = ((ac.pos - x0) ** 2).sum(1)
    g = np.exp(-0.5 * d2 / args.s**2)
    return g / (g.sum() * V_CELL)


def run(centres, phases):
    ac, r = make()
    gs = [gauss(ac, c) for c in centres]
    dt = C.stable_dt(ac)
    if args.w > 0:
        per = 2 * np.pi / args.w
        t_spin = 2 * args.R / c0 + per
        T = t_spin + args.n_periods * per
    else:
        t_spin, T = 0.66 * args.T_dc, args.T_dc
    n = int(np.ceil(T / dt))
    meas = r < 0.8 * 0.8 * r.max()
    acc = []
    for i in range(n):
        t = i * dt
        if args.w > 0:
            q = [args.Q * np.cos(args.w * t + ph) for ph in phases]
        else:
            ramp = min(1.0, t / (0.33 * args.T_dc))
            q = [args.Q * ramp * np.cos(ph) for ph in phases]
        u = ac.u
        for qi, g in zip(q, gs):
            u += dt * qi * g
        ac.u = u
        ac.alat.step(0, dt)
        ac.alat.step_sponge(0)
        if t >= t_spin and i % 20 == 0:
            acc.append(float(C.energy_density(ac)[meas].sum()))
    return float(np.mean(acc))


tag = f"w{args.w:g}"
print(f"R={args.R} s={args.s} w={args.w} " + (f"k={args.w/c0:.4f} lambda={2*np.pi*c0/args.w:.1f}" if args.w > 0 else "DC"), flush=True)
rows = []
t0 = time.time()
for d in [float(v) for v in args.ds.split(",")]:
    c1, c2 = np.array([-d / 2, 0, 0]), np.array([d / 2, 0, 0])
    H1 = run([c1], [0.0])
    H2 = run([c2], [0.0])
    Hin = run([c1, c2], [0.0, 0.0])
    Hanti = run([c1, c2], [0.0, np.pi])
    Ein, Eanti = Hin - H1 - H2, Hanti - H1 - H2
    rows.append((d, H1, H2, Hin, Hanti, Ein, Eanti))
    print(f"d={d:5.1f}  H1={H1:.5e} H2={H2:.5e}  E_int in-phase={Ein:+.5e}  anti={Eanti:+.5e}  "
          f"d*E_in={d*Ein:+.4e} ({time.time()-t0:.0f}s)", flush=True)
rows = np.array(rows)
np.save(f"phase_interaction_{tag}.npy", rows)
d, Ein, Eanti = rows[:, 0], rows[:, 5], rows[:, 6]
for name, E in (("in-phase", Ein), ("anti-phase", Eanti)):
    F = -np.gradient(E, d)
    sgn = "REPEL" if np.all(F > 0) else ("ATTRACT" if np.all(F < 0) else "mixed")
    slope = np.polyfit(np.log(d), np.log(np.abs(E)), 1)[0] if np.all(E != 0) else np.nan
    print(f"{name:>10}: F=-dE/dd = " + " ".join(f"{v:+.3e}" for v in F) + f"  -> {sgn};  |E_int| ~ d^{slope:.2f}")
