"""Item B, step 2: is a v1 / v2 relative flow unstable on the exact walk (spin branch has zero sound speed, spin_symbol.py)?

Periodic BCC box (cpp_box, parity walk). Background = plane waves on positive-frequency doublet states of the two-tick
symbol: w1 in D(q1) at wavevector q1 z, w2 in D(q2) at q2 z, w1 orthogonal to w2 (amplitudes cos b, sin b),
q = 2 pi n / side. Two copies run side by side, one with complex noise eps per cell; growth = log |noisy - clean|.
Cases: rest (n 0), counterflow (+n, -n), one-sided (v1 +n, v2 0 = the lone-ring situation), coflow (+n, +n).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.walk import CppWalk, healing_g, symbol_dirac, to_real

here = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--m", type=float, default=0.8)
p.add_argument("--xi", type=float, default=3.9)
p.add_argument("--P", type=int, default=96)
p.add_argument("--Pz", type=int, default=0, help="axial period (flow direction); default P")
p.add_argument("--mix", type=float, default=45.0)
p.add_argument("--case", action="append", required=True, help="n1,n2 (v1, v2 wavenumbers along z, units 2 pi / side)")
p.add_argument("--T", type=int, default=4000)
p.add_argument("--every", type=int, default=50)
p.add_argument("--eps", type=float, default=1e-7)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--tag", default="")
args = p.parse_args()
m, g = args.m, healing_g(args.m, args.xi, 1.0, walk=True)
B = np.block([[np.zeros((2, 2)), np.eye(2)], [np.eye(2), np.zeros((2, 2))]])


def U2(k):
    k = np.asarray(k, float)
    return B @ symbol_dirac(-k, m) @ B @ symbol_dirac(k, m)


def doublet(k):
    lam, V = np.linalg.eig(U2(k))
    i = np.argsort(np.angle(lam))[:2]
    Q, _ = np.linalg.qr(V[:, i])
    return Q, -np.angle(lam[i]) / 2


D0, _ = doublet(np.zeros(3))
v1, v2 = D0[:, 0], D0[:, 1]
Pz = args.Pz or args.P


def box():
    al = _lat.continuous.AdaptiveLattice()
    _lat.continuous.build_bcc_box_periodic(al[0], args.P, args.P, Pz, d1=1.0)
    al.rebuild(0)
    al.set_n_threads(args.threads)
    return al


alats = [box() for _ in range(2)]
pos = np.array(alats[0][0].leaf_pos)
side = Pz / np.sqrt(3)
z = pos[:, 2]
b = np.radians(args.mix)
rng = np.random.default_rng(1)
out = {}
for case in args.case:
    n1, n2 = (int(s) for s in case.split(","))
    q1, q2 = 2 * np.pi * n1 / side, 2 * np.pi * n2 / side
    Da, oma = doublet([0, 0, q1])
    Db, omb = doublet([0, 0, q2])
    w1 = Da @ (Da.conj().T @ v1)
    w1 /= np.linalg.norm(w1)
    c = Db.conj().T @ w1
    w2 = Db @ np.array([-c[1].conj(), c[0].conj()]) if np.linalg.norm(c) > 1e-12 else Db @ (Db.conj().T @ v2)
    w2 /= np.linalg.norm(w2)
    if n1 == n2:
        w2 = Da @ (Da.conj().T @ v2) - w1 * (w1.conj() @ (Da @ (Da.conj().T @ v2)))
        w2 /= np.linalg.norm(w2)
    psi = np.cos(b) * np.exp(1j * q1 * z)[:, None] * w1[None, :] + np.sin(b) * np.exp(1j * q2 * z)[:, None] * w2[None, :]
    noise = args.eps * (rng.standard_normal(psi.shape) + 1j * rng.standard_normal(psi.shape))
    walks = []
    for k, al in enumerate(alats):
        w = CppWalk(al, 0, m=m, g=g, rho0=1.0, parity=True)
        ps = psi + (noise if k else 0)
        w.set_state(to_real(ps[:, :2]), to_real(ps[:, 2:]), t=0)
        walks.append(w)
    vg = (np.sqrt(1 - m * m) / (3 * m))
    ts, dn = [], []
    for t in range(0, args.T + 1, args.every):
        (L0, R0), (L1, R1) = walks[0].state(), walks[1].state()
        d = np.sqrt(((L1 - L0) ** 2).sum() + ((R1 - R0) ** 2).sum()) / np.sqrt(len(pos))
        ts.append(t)
        dn.append(d)
        if t < args.T:
            for w in walks:
                w.step(args.every)
    ts, dn = np.array(ts), np.array(dn)
    lin = (dn > 30 * args.eps) & (dn < 1e-2)
    gam = np.polyfit(ts[lin], np.log(dn[lin]), 1)[0] if lin.sum() >= 3 else np.nan
    rho0 = walks[0].density()
    print(f"case n1 {n1:+d} n2 {n2:+d}: q1 {q1:.4f} q2 {q2:.4f}  w1.w2 {abs(w1.conj() @ w2):.1e}  rel velocity "
          f"{vg * abs(q1 - q2):.4f}  clean rho sd at T {rho0.std():.2e}; |d| {dn[0]:.1e} -> {dn[-1]:.1e}, "
          f"growth (linear stage) {gam:.2e} / tick ({lin.sum()} pts)", flush=True)
    out[case] = (ts, dn, gam)
fig, ax = plt.subplots(figsize=(9, 6))
for case, (ts, dn, gam) in out.items():
    ax.semilogy(ts, dn, label=f"n1,n2 = {case}  gamma {gam:.1e}")
ax.set_xlabel("tick")
ax.set_ylabel("|noisy - clean| per cell")
ax.set_title(f"v1/v2 relative flow, P {args.P} Pz {Pz} mix {args.mix:g} xi {args.xi}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(here / f"counterflow_P{args.P}_Pz{Pz}_mix{args.mix:g}{args.tag}.png", dpi=80)
