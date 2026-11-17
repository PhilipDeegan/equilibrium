"""Hertzian dipole on the EXPERIMENTAL A-E model (continuous.AE).

J = z I(t) g(x), g gaussian (int g = 1), I = I0 sin(wt) with a one-period
ramp. From A_tt = c^2 lap A + J (Lorenz): A = pdot(t - r/c) z / (4 pi c^2 r),
far field |S| = pddot^2 sin^2(th) / (16 pi^2 c^3 r^2), <P> = I0^2 w^2/(12 pi c^3).
Measure <S.rhat> over a window that closes before the wall reflection
(E held 0 near the edge) returns to the outermost shell.
"""
import argparse
import time

import numpy as np

from equilibrium.substrate import continuous as C

C.set_weights("legacy")  # written for W = 3:1 on both legs (pre-2026-09-28; see continuous.WEIGHTS)
from equilibrium.substrate.lattice import SubstrateLattice

p = argparse.ArgumentParser()
p.add_argument("--h", type=float, default=0.5)
p.add_argument("--R", type=float, default=20.0)
p.add_argument("--lam", type=float, default=4.0, help="wavelength")
p.add_argument("--src-w", type=float, default=0.6)
p.add_argument("--every", type=int, default=8)
p.add_argument("--out", default="")
p.add_argument("--cpp", action="store_true")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--snaps", type=int, default=4, help="full-state snapshots over the window (cpp)")
args = p.parse_args()

c = C.C0
c2 = C.C0SQ
w = 2 * np.pi * c / args.lam
per = 2 * np.pi / w
I0 = 1.0

lat = C.anchor_cluster(args.R, args.h, args.threads) if args.cpp else SubstrateLattice(radius_cells=args.R, spacing=args.h)
x = lat.pos
r = np.linalg.norm(x, axis=1)
V = (args.h * 2 / 3**0.5) ** 3 / 2
rhat = x / np.maximum(r, 1e-12)[:, None]
cth = np.where(r > 0, x[:, 2] / np.maximum(r, 1e-12), 0.0)


def g_fn(p_):
    s = args.src_w
    return np.exp(-0.5 * (p_**2).sum(1) / s**2) / ((2 * np.pi) ** 1.5 * s**3)


Jz = C.sample_A(lat, lambda p_: np.stack([0 * p_[:, 0], 0 * p_[:, 0], g_fn(p_)], 1))
print("int g dV =", float(g_fn(x).sum() * V))

r_shells = np.array([5.0, 6.5, 8.0, 9.5])
dr = 2 * args.h
t_start = r_shells[-1] / c + 2 * per
t_end = (2 * args.R - r_shells[-1]) / c - 3 * args.h / c
n_per = int(np.floor((t_end - t_start) / per))
t_end = t_start + n_per * per
print(f"w={w:.3f} period={per:.3f} window=[{t_start:.2f},{t_end:.2f}] ({n_per} periods) N={lat.N}", flush=True)

ae = None if args.cpp else C.AE(lat)
# measured cells only: the shells (E, curl A there) and the source support (J.E)
sh = np.nonzero(np.abs(r[:, None] - r_shells[None, :]).min(1) < dr / 2)[0]
src = np.nonzero(r < 8 * args.src_w)[0]
Jz_src = Jz[src]
if args.cpp:
    JzT = lat.faces_T(Jz)
    del Jz
    step = lambda dt, s_: lat.step_ae(dt, scale=s_, J_T=JzT)
    e_rows, phi_rows = lat.e_rows, lat.phi_rows
else:
    step = lambda dt, s_: ae.step(dt, s_ * Jz)
    e_rows, phi_rows = (lambda j: ae.e[j]), (lambda j: lat.phi[j])
dt = C.stable_dt(lat)
n = int(np.ceil(t_end / dt))
dt = t_end / n

th_edges = np.linspace(0, np.pi, 10)
th = np.arccos(np.clip(cth, -1, 1))
acc = np.zeros((len(r_shells), len(th_edges) - 1))
cnt = np.zeros_like(acc)
n_meas = 0
Pin_acc, Pin_n = 0.0, 0
from pathlib import Path

snap_dir = Path(f"/bk/dst/2026/09/25/dipole_R{args.R / args.h:g}_h{args.h}")
snap_steps = set()
if args.cpp and args.snaps:
    snap_dir.mkdir(parents=True, exist_ok=True)
    np.save(snap_dir / "pos.npy", lat.pos)
    i0 = int(np.ceil(t_start / dt))
    snap_steps = {int(round(i0 + (n - i0) * q / args.snaps)) for q in range(args.snaps + 1)}


def snapshot(i):
    np.savez(snap_dir / f"state_{i:06d}.npz", t=i * dt, u=np.asarray(lat.cl.leaf_u),
             phi=np.asarray(lat.cl.leaf_phi), e=np.asarray(lat.alat.leaf_e(lat.i)))


t0 = time.time()
for i in range(1, n + 1):
    t = (i - 1) * dt
    ramp = min(1.0, t / per)
    step(dt, I0 * np.sin(w * t) * ramp)
    t_now = i * dt
    if t_now >= t_start and i % args.every == 0:
        S = c2 * np.cross(C.reconstruct_A_at(lat, sh, e_rows(sh)), C.curl_A_at(lat, sh, phi_rows))
        Sr = (S * rhat[sh]).sum(1)
        for a, Rs in enumerate(r_shells):
            m = np.abs(r[sh] - Rs) < dr / 2
            idx = np.digitize(th[sh][m], th_edges) - 1
            np.add.at(acc[a], np.clip(idx, 0, len(th_edges) - 2), Sr[m])
            np.add.at(cnt[a], np.clip(idx, 0, len(th_edges) - 2), 1)
        n_meas += 1
        Pin_acc += -float((I0 * np.sin(w * t) * ramp * Jz_src * e_rows(src)).sum()) * V * 3.0 / 80.0
        Pin_n += 1
    if i in snap_steps:
        snapshot(i)
    if i % 400 == 0:
        print(f"  step {i}/{n} t={t_now:.2f} |u|max={np.abs(lat.u).max():.2e} ({time.time()-t0:.0f}s)", flush=True)

Sbar = acc / np.maximum(cnt, 1)
thc = 0.5 * (th_edges[1:] + th_edges[:-1])
P_th = I0**2 * w**2 / (12 * np.pi * c**3)
S_amp_th = lambda R_: I0**2 * w**2 / 2 / (16 * np.pi**2 * c**3 * R_**2)
print(f"\nmeasurements={n_meas}  P_theory={P_th:.4e}  source work -<J.E> / P_th = {Pin_acc/max(Pin_n,1)/P_th:.3f}")
for a, Rs in enumerate(r_shells):
    ring = 2 * np.pi * Rs**2 * np.sin(thc) * (th_edges[1] - th_edges[0])
    P = float((Sbar[a] * ring).sum())
    fit = float((Sbar[a] * np.sin(thc) ** 2).sum() / (np.sin(thc) ** 4).sum())
    resid = float(np.sqrt(np.mean((Sbar[a] - fit * np.sin(thc) ** 2) ** 2)) / max(abs(fit), 1e-30))
    print(f"r={Rs:4.1f} P/P_th={P/P_th:.3f}  r^2*S_eq={Rs**2*Sbar[a, len(thc)//2]:.4e}"
          f"  sin^2 fit amp/theory={fit/S_amp_th(Rs):.3f}  pattern resid={resid:.3f}"
          f"  pole/eq={Sbar[a,0]/Sbar[a,len(thc)//2]:+.3f}")
    print("      S(th)/S_th_eq: " + " ".join(f"{v/S_amp_th(Rs):+.3f}" for v in Sbar[a]))
eq = Sbar[:, len(thc) // 2]
slope = np.polyfit(np.log(r_shells), np.log(np.abs(eq)), 1)[0]
print(f"equatorial slope d log S / d log r = {slope:.3f} (theory -2)")
if args.out:
    np.savez(args.out, r_shells=r_shells, th_edges=th_edges, Sbar=Sbar, h=args.h, R=args.R,
             w=w, P_th=P_th, **({"snap_dir": str(snap_dir)} if args.cpp and args.snaps else
                                dict(u=lat.u, phi=lat.phi, e=ae.e, pos=lat.pos)))
