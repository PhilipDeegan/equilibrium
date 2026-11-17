"""Does a disturbance feel a settled background? (Philip 11-03: photons ride on a magnetically settled background.)

Continuous level-0 sphere R, h 1. Background: uniform circulation about z, phi = sample_A(A = B/2 z x r) (phi ~ A),
div-free by construction (projected). Disturbance: small Gaussian u pulse at the origin. Response = run(bg + pulse)
- run(bg) (background held fixed like a partial derivative), compared with run(pulse) on no background.
step(): linear, response must equal the bare pulse exactly (control). step_cartan (kappa, 3 iterations): the
transport couples circulation to flow, first order in B.
Measures: ||resp - bare|| / ||bare|| (u, phi), the share of the response's face content in the div-free (circulation)
part, and u at probes r 12 on +x, +y, -x, +z.
"""

import argparse
import time

import numpy as np

from equilibrium.substrate import continuous as C

p = argparse.ArgumentParser()
p.add_argument("--R", type=float, default=30.0)
p.add_argument("--B", type=float, default=0.05)
p.add_argument("--amp", type=float, default=1e-3)
p.add_argument("--s", type=float, default=2.0)
p.add_argument("--T", type=float, default=20.0)
p.add_argument("--kappa", type=float, default=1.0)
p.add_argument("--rule", choices=["step", "cartan"], default="cartan")
p.add_argument("--settle", action="store_true", help="u_bg = kappa psi balancing the gradient part of the transport")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--out", default="")
args = p.parse_args()

t0 = time.time()
ac = C.anchor_cluster(args.R, 1.0, n_threads=args.threads)
pos = ac.pos
bg_phi = C.project_div_free(ac, C.sample_A(ac, lambda x: 0.5 * args.B * np.stack([-x[:, 1], x[:, 0], 0 * x[:, 0]], 1)))
u_bg = np.zeros(ac.N)
if args.settle and args.rule == "cartan":
    u_bg = args.kappa * C.project_div_free(ac, ac.cartan_transport(bg_phi), return_psi=True)[1]
pulse_u = args.amp * np.exp(-np.sum(pos**2, 1) / (2 * args.s**2))
dt = C.stable_dt(ac)
n = int(round(args.T / dt))
probes = {k: int(np.argmin(np.sum((pos - np.array(v)) ** 2, 1))) for k, v in
          dict(px=(12, 0, 0), py=(0, 12, 0), mx=(-12, 0, 0), pz=(0, 0, 12)).items()}
print(f"N {ac.N} dt {dt:.4f} steps {n} rule {args.rule} B {args.B} kappa {args.kappa} setup {time.time() - t0:.1f}s", flush=True)


def run(u0, phi0):
    ac.u, ac.phi = u0, phi0
    tr = np.empty((n, len(probes)))
    for i in range(n):
        if args.rule == "step":
            ac.alat.step(ac.i, dt)
        else:
            ac.alat.step_cartan(ac.i, dt, args.kappa, 3)
        u = ac.cl.leaf_u
        tr[i] = [u[j] for j in probes.values()]
    return ac.u, ac.phi, tr


t1 = time.time()
ub, pb, trb = run(u_bg.copy(), np.zeros((ac.N, 14)) + bg_phi)
print(f"  bg run {time.time() - t1:.0f}s", flush=True)
uf, pf, trf = run(u_bg + pulse_u, bg_phi.copy())
u0_, p0_, tr0 = run(pulse_u, np.zeros((ac.N, 14)))
ur, pr, trr = uf - ub, pf - pb, trf - trb


def curl_share(ph):
    tot = np.sum(ph**2)
    return float(np.sum(C.project_div_free(ac, ph) ** 2) / tot) if tot > 0 else 0.0


nu, npf = np.linalg.norm(u0_), np.linalg.norm(p0_)
print(f"||resp-bare||/||bare||: u {np.linalg.norm(ur - u0_) / nu:.3e}  phi {np.linalg.norm(pr - p0_) / npf:.3e}")
print(f"circulation share of face content: response {curl_share(pr):.3e}  bare {curl_share(p0_):.3e}")
print(f"bg drift: ||bg_phi(T) - bg_phi(0)||/||bg_phi|| {np.linalg.norm(pb - bg_phi) / np.linalg.norm(bg_phi):.3e}  "
      f"max|u_bg(T) - u_bg(0)| {np.abs(ub - u_bg).max():.3e}")
t = np.arange(1, n + 1) * dt
for k, (name, _) in enumerate(probes.items()):
    a, b = trr[:, k], tr0[:, k]
    ia, ib = np.argmax(np.abs(a)), np.argmax(np.abs(b))
    print(f"  {name}: peak resp {a[ia]:+.4e} at t {t[ia]:.2f}   bare {b[ib]:+.4e} at t {t[ib]:.2f}   "
          f"max|resp-bare| / peak {np.abs(a - b).max() / np.abs(b).max():.3e}")
if args.out:
    np.savez(args.out, t=t, trr=trr, tr0=tr0, ur=ur, pr=pr, u0=u0_, p0=p0_, pos=pos, args=str(vars(args)))
print(f"done {time.time() - t0:.0f}s")
