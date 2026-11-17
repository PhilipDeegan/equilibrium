"""Packet check of the real-form walk on our lattice against the exact symbol: centroid velocity vs group velocity
d omega / dk of the seeded branch, norm conservation away from the sponge, packet width growth."""

import time
from pathlib import Path

import numpy as np

from equilibrium.substrate.lattice import DampedLattice
from walk import DiracWalk, symbol_dirac, to_real

here = Path(__file__).parent
lat = DampedLattice(radius_cells=40, sponge_fraction=0.25)
pos = lat.pos
s0 = 5.0
rows = []
for m in (0.0, 0.2):
    for lab, kd, kmag in (("[100]", [1, 0, 0], 0.6), ("[111]", [1, 1, 1], 0.6), ("[-1-1-1]", [-1, -1, -1], 0.6),
                          ("[100] k 1.2", [1, 0, 0], 1.2), ("rest k 0", [1, 0, 0], 0.0)):
        n = np.array(kd, float) / np.linalg.norm(kd)
        k0 = kmag * n
        D = symbol_dirac(k0, m)
        lam, V = np.linalg.eig(D)
        om = -np.angle(lam)
        b = int(np.argmax(om))  # positive-frequency branch with the largest omega
        v = V[:, b]

        def om_b(k):
            l2, V2 = np.linalg.eig(symbol_dirac(k, m))
            return -np.angle(l2[int(np.argmax(np.abs(V2.conj().T @ v)))])

        eps = 1e-5
        vg = np.array([(om_b(k0 + eps * e) - om_b(k0 - eps * e)) / (2 * eps) for e in np.eye(3)])
        env = np.exp(-np.sum(pos**2, 1) / (2 * s0**2)) * np.exp(1j * pos @ k0)
        w = DiracWalk(lat, m=m)
        psi = env[:, None] * v[None, :]
        w.L, w.R = to_real(psi[:, :2]), to_real(psi[:, 2:])
        rho0 = w.density()
        N0 = rho0.sum()
        T = 40
        t0 = time.time()
        cs, ws = [], []
        for t in range(T + 1):
            rho = w.density()
            c = rho @ pos / rho.sum()
            cs.append(c)
            ws.append(np.sqrt(rho @ np.sum((pos - c) ** 2, 1) / rho.sum()))
            if t < T:
                w.step()
        cs = np.array(cs)
        vel = np.polyfit(np.arange(T + 1), cs, 1)[0]
        rows.append(f"m {m:.1f} {lab:12s} k {kmag}: omega {om[b]:+.4f}  v_group {np.round(vg, 4).tolist()} |{np.linalg.norm(vg):.4f}|"
                    f"  v_centroid {np.round(vel, 4).tolist()} |{np.linalg.norm(vel):.4f}|  width {ws[0]:.2f} -> {ws[-1]:.2f}"
                    f"  norm {rho.sum() / N0:.6f}  ({time.time() - t0:.0f}s)")
        print(rows[-1], flush=True)
print(f"c = 1/sqrt3 = {1 / np.sqrt(3):.4f} hex hops per tick")
(here / "packet.log").write_text("\n".join(rows) + "\n")
