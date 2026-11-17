"""10-24 reference ladder on the a 9 DST geometry (periodic box side P / sqrt3, Taylor spheromak inside a, smoothed).
  --case vacuum : rung 1, source-free Maxwell from A = curl^-1 B0 (exact at each output time)
  --case diffuse: rung 4a, B_t = eta lap B (exact)
  --case mhd    : rung 4b/c, incompressible resistive MHD from v = 0, B = amp * seed
Per output: magnetic energy EB, kinetic EV, helicity H, lam_eff a / 4.493, overlap <B, B0> / |B||B0|, energy share in
r < a / a..1.5a / beyond, centroid of |B|^2, and grid-scale share (energy at k > k_max / 2)."""
import argparse
import time
from pathlib import Path

import numpy as np

from equilibrium.reference import spectral as S

p = argparse.ArgumentParser()
p.add_argument("--case", choices=["vacuum", "diffuse", "mhd"], required=True)
p.add_argument("--P", type=float, default=96)
p.add_argument("--n", type=int, default=64)
p.add_argument("--a", type=float, default=9.0)
p.add_argument("--smooth", type=float, default=1.0)
p.add_argument("--amp", type=float, default=0.3)
p.add_argument("--eta", type=float, default=0.0)
p.add_argument("--nu", type=float, default=0.0)
p.add_argument("--T", type=float, default=300.0)
p.add_argument("--every", type=float, default=10.0)
p.add_argument("--noise", type=float, default=1e-4, help="rms of a random smooth div-free v seed (instability seed)")
p.add_argument("--pair", choices=["", "co", "counter"], default="", help="two spheromaks on the z axis (counter = second mirrored)")
p.add_argument("--sep", type=float, default=0.0, help="centre distance (default 2 a, touching)")
p.add_argument("--push", type=float, default=0.0, help="approach speed of each body (pair runs): v_z = -push sign(z) inside each body, projected div-free")
p.add_argument("--save", default="")
args = p.parse_args()
box = S.Box(args.n, args.P / np.sqrt(3))
if args.pair:
    B0 = S.spheromak_pair(box, args.a, args.sep or 2 * args.a, args.smooth, args.amp, hel2=1 if args.pair == "co" else -1)
else:
    B0 = S.taylor_spheromak(box, args.a, smooth=args.smooth, amp=args.amp)
rho_ = np.maximum(np.hypot(box.x[0], box.x[1]), 1e-12)
ephi = np.stack([-box.x[1] / rho_, box.x[0] / rho_, 0 * rho_])
upper = box.x[2] > 0
n0 = np.linalg.norm(B0)
edges = np.array([1.0, 1.5]) * args.a
kfine = box.k2 > (np.pi / box.dx / 2) ** 2


def diag(t, B, v=None):
    eb = 0.5 * np.sum(B**2, 0)
    sh = box.shells(eb, edges)
    EB = sh.sum()
    Bh = box.fft(B)
    w = np.where(box.k[2] > 0, 2.0, 1.0)
    fine = np.sum(w * kfine * np.sum(np.abs(Bh) ** 2, 0)) / np.sum(w * np.sum(np.abs(Bh) ** 2, 0))
    cen = np.sum(box.x * eb, axis=(1, 2, 3)) / eb.sum()
    row = dict(t=t, EB=EB, EV=0.0 if v is None else S.energy(box, v), H=S.helicity(box, B),
               lam=S.lam_eff(box, B) * args.a / S.LAM_A, ov=np.sum(B * B0) / (np.linalg.norm(B) * n0),
               s_in=sh[0] / EB, s_sh=sh[1] / EB, s_out=sh[2] / EB, cz=cen[2], cr=np.hypot(cen[0], cen[1]), fine=fine)
    row["Etor"] = 0.5 * np.sum(np.sum(B * ephi, 0) ** 2) * box.dV
    row["up"] = np.sum(eb[upper]) * box.dV / EB
    czu = np.sum(box.x[2][upper] * eb[upper]) / eb[upper].sum()
    czl = np.sum(box.x[2][~upper] * eb[~upper]) / eb[~upper].sum()
    row["gap"] = czu - czl
    return row


hist = []


def out(row, eta_s=None):
    hist.append(row)
    r0 = hist[0]
    print(f"  t {row['t']:7.2f}  EB/EB0 {row['EB'] / r0['EB']:.5f}  EV/EB0 {row['EV'] / r0['EB']:.3e}  H/|H0| {row['H'] / abs(r0['H']) if abs(r0['H']) > 1e-6 * r0['EB'] else row['H']:.5f}"
          f"  lam a/4.49 {row['lam']:.4f}  overlap {row['ov']:.4f}  in/shell/out {row['s_in']:.4f} {row['s_sh']:.4f} {row['s_out']:.4f}"
          f"  cz {row['cz']:+.3f} cr {row['cr']:.3f}  fine {row['fine']:.2e}  Etor/EB {row['Etor'] / row['EB']:.4f}  upper {row['up']:.4f}  dz {row['gap']:.3f}" + ("" if eta_s is None else f"  (ETA {eta_s / 60:.1f} min)"),
          flush=True)


print(f"case {args.case} box side {box.side:.3f} n {args.n} dx {box.dx:.3f} a {args.a} smooth {args.smooth} amp {args.amp} "
      f"eta {args.eta} nu {args.nu}  seed lam a/4.49 {S.lam_eff(box, B0) * args.a / S.LAM_A:.4f}", flush=True)
times = np.arange(0, args.T + 1e-9, args.every)
t_wall = time.time()
if args.case == "vacuum":
    A0 = box.ifft(box.uncurl_h(box.fft(B0)))
    for t in times:
        A, E = S.maxwell_vacuum(box, A0, t)
        row = diag(t, box.curl(A))
        row["EE"] = S.energy(box, E)
        out(row)
elif args.case == "diffuse":
    for t in times:
        out(diag(t, S.diffuse(box, B0, args.eta, t)))
else:
    rng = np.random.default_rng(1)
    vh = box.project_h(box.fft(rng.standard_normal(B0.shape)) * np.exp(-box.k2 * 4.0)[None])
    v0 = box.ifft(vh)
    v0 *= args.noise / np.sqrt(np.mean(np.sum(v0**2, 0))) if args.noise > 0 else 0.0
    if args.pair and args.push > 0:
        sep = args.sep or 2 * args.a
        for zc in (sep / 2, -sep / 2):
            d = np.sqrt(box.x[0] ** 2 + box.x[1] ** 2 + (box.x[2] - zc) ** 2)
            v0[2] += -args.push * np.sign(zc) * 0.5 * (1 - np.tanh((d - 0.8 * args.a) / 1.5))
    m = S.MHD(box, v0, B0, nu=args.nu, eta=args.eta)
    out(diag(0.0, m.B, m.v))
    for t_next in times[1:]:
        while m.t < t_next - 1e-9:
            m.step(min(m.cfl_dt(), t_next - m.t))
        out(diag(m.t, m.B, m.v), (time.time() - t_wall) / m.t * (args.T - m.t))
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, B=m.B, v=m.v, B0=B0, args=str(vars(args)))
r0, r1 = hist[0], hist[-1]
print(f"RESULT {args.case}  EB {r1['EB'] / r0['EB']:.4f}  H {r1['H'] / abs(r0['H']) if abs(r0['H']) > 1e-6 * r0['EB'] else r1['H']:.4f}  overlap {r1['ov']:.4f}  in {r0['s_in']:.4f}->{r1['s_in']:.4f}"
      f"  lam {r0['lam']:.4f}->{r1['lam']:.4f}  fine {r0['fine']:.1e}->{r1['fine']:.1e}  wall {time.time() - t_wall:.0f} s")
if args.save:
    np.save(args.save.replace(".npz", "_hist.npy"), hist)
