"""Phase speed of plane-wave packets on the EXPERIMENTAL A-E model (C++ step_ae).

Packet f(xi) = cos(k chi) exp(-chi^2 / 2 s^2), chi = n.x - c t - xi0, travels `travel` along n.
  T mode: A = p f, E = c p f', B = f' n x p, u = 0          (exact continuum solution)
  L mode: u = f, A = n f / c, E = 0                         (step() sector)
Measured in a cylinder rho < rho_m about the n axis (the sphere wall's disturbance never reaches it):
F(t) = sum_i q_i exp(-i k n.x_i) for numerical and exact q on the same cells; arg(F_num / F_ex) =
-(omega_num - omega) t (the e^{-ikx} projection picks e^{+i omega t}), so dc/c = -slope / omega. u, E sit at t_i = i dt, phi (A, B) at t_i + dt/2.
L mode also gets the exact step() (leapfrog, 14-face) prediction for u.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from equilibrium.substrate import continuous as C

C.set_weights("legacy")  # written for W = 3:1 on both legs (pre-2026-09-28; see continuous.WEIGHTS)

p = argparse.ArgumentParser()
p.add_argument("--h", type=float, default=0.25)
p.add_argument("--R", type=float, default=14.0)
p.add_argument("--lam", type=float, default=4.0)
p.add_argument("--s", type=float, default=1.2)
p.add_argument("--travel", type=float, default=6.0)
p.add_argument("--rho", type=float, default=4.0)
p.add_argument("--dir", default="1,0,0")
p.add_argument("--mode", choices=["T", "L"], default="T")
p.add_argument("--threads", type=int, default=8)
p.add_argument("--n-meas", type=int, default=6)
args = p.parse_args()

c = C.C0
k = 2 * np.pi / args.lam
om = c * k
n = np.array([float(v) for v in args.dir.split(",")])
n /= np.linalg.norm(n)
ref = np.eye(3)[int(np.argmin(np.abs(n)))]
pol = ref - (ref @ n) * n
pol /= np.linalg.norm(pol)
nxp = np.cross(n, pol)
xi0 = -args.travel / 2
s2 = args.s**2


def f(xi):
    ch = xi - xi0
    return np.cos(k * ch) * np.exp(-0.5 * ch**2 / s2)


def fp(xi):
    ch = xi - xi0
    return (-k * np.sin(k * ch) - ch / s2 * np.cos(k * ch)) * np.exp(-0.5 * ch**2 / s2)


def fields(x, t):
    xi = x @ n - c * t
    if args.mode == "T":
        return dict(A=f(xi)[:, None] * pol, E=(c * fp(xi))[:, None] * pol, B=fp(xi)[:, None] * nxp, u=0 * xi)
    return dict(A=(f(xi) / c)[:, None] * n, E=0 * x, B=0 * x, u=f(xi))


lat = C.anchor_cluster(args.R, args.h, args.threads)
h = args.h
dt = C.stable_dt(lat)
T = args.travel / c
nst = int(np.ceil(T / dt))
dt = T / nst
x = lat.pos
xn = x @ n
rho = np.linalg.norm(x - xn[:, None] * n, axis=1)
idx = np.nonzero((rho < args.rho) & (np.abs(xn) < args.travel / 2 + 4 * args.s))[0]
ph = np.exp(-1j * k * xn[idx])
print(f"mode={args.mode} dir={args.dir} h={h} kh={k * h:.4f} N={lat.N} meas={len(idx)} dt={dt:.3e} steps={nst}", flush=True)

lat.u = C.sample_u(lat, lambda y: fields(y, 0.0)["u"])
lat.phi = C.sample_A(lat, lambda y: fields(y, 0.5 * dt)["A"])
lat.e = C.sample_A(lat, lambda y: fields(y, 0.0)["E"])

comps = dict(T=dict(A=pol, E=pol, B=nxp), L=dict(A=n, u=None))[args.mode]


def measure(i):
    t_ue, t_phi = i * dt, (i + 0.5) * dt
    xs = x[idx]
    ex_ue, ex_phi = fields(xs, t_ue), fields(xs, t_phi)
    num = {}
    if "A" in comps:
        num["A"] = C.reconstruct_A_at(lat, idx, lat.phi_rows(idx)) @ comps["A"]
    if "B" in comps:
        num["B"] = C.curl_A_at(lat, idx, lat.phi_rows) @ comps["B"]
    E_num = C.reconstruct_A_at(lat, idx, lat.e_rows(idx))
    if "E" in comps:
        num["E"] = E_num @ comps["E"]
    if "u" in comps:
        num["u"] = np.asarray(lat.u)[idx]
    out = dict(t=t_ue)
    for key, q in num.items():
        ex = ex_ue if key in ("E", "u") else ex_phi
        qe = ex[key] if key == "u" else ex[key] @ comps[key]
        Fn, Fe = (q * ph).sum(), (qe * ph).sum()
        tt = t_ue if key in ("E", "u") else t_phi
        out[key] = dict(phase=float(np.angle(Fn / Fe)), amp=float(abs(Fn) / abs(Fe)), tt=tt,
                        resid=float(np.linalg.norm(q - qe) / np.linalg.norm(qe)))
    if args.mode == "L":
        out["E_leak"] = float(np.sqrt((E_num**2).sum(1).mean()) / (k * np.sqrt((ex_ue["u"] ** 2).mean())))
    return out


meas_steps = sorted({int(round(nst * q / args.n_meas)) for q in range(args.n_meas + 1)})
rows = []
t0 = time.time()
for i in range(nst + 1):
    if i in meas_steps:
        r = measure(i)
        rows.append(r)
        print(f"  step {i}/{nst} t={r['t']:.3f} " + " ".join(
            f"{key}: ph={r[key]['phase']:+.5f} amp={r[key]['amp']:.5f} res={r[key]['resid']:.2e}"
            for key in comps) + (f" E_leak={r['E_leak']:.2e}" if "E_leak" in r else "") + f" ({time.time() - t0:.0f}s)",
            flush=True)
    if i < nst:
        lat.step_ae(dt)

res = dict(mode=args.mode, dir=args.dir, h=h, kh=k * h, lam=args.lam, N=lat.N, steps=nst, dt=dt)
for key in comps:
    tt = np.array([r[key]["tt"] for r in rows])
    phs = np.unwrap([r[key]["phase"] for r in rows])
    slope = np.polyfit(tt, phs, 1)[0]
    res[key] = dict(dc_c=float(-slope / om), amp_end=rows[-1][key]["amp"], resid_end=rows[-1][key]["resid"])
if args.mode == "L":
    D = C.D_HAT
    S = float(np.sum(C.W**2 * (1 - np.cos(h * k * (D @ n))))) / h**2
    res["u_pred_dc_c"] = float((2 / dt) * np.arcsin(0.5 * dt * np.sqrt(S)) / om - 1)
    res["E_leak_end"] = rows[-1]["E_leak"]
print(json.dumps(res), flush=True)
with open(Path(__file__).parent / "results.jsonl", "a") as fh:
    fh.write(json.dumps(res) + "\n")
