"""toroidal / poloidal flow of a ring in the periodic box: phase gradient of the dominant rest component a = psi . v1*
(8 hex-diagonal differences, grad = sum_h h dphi_h / 8 in grid units), in the meridional half-plane |y| < 1 (x > 0)
and over the whole box: v_phi (toroidal) map + (v_rho, v_z) arrows; net toroidal flux int rho v_phi over the
half-plane; also |v2|^2 and negative-frequency weight in the slab. usage: toroidal.py state.npz P out.png"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium.substrate.walk import HS, symbol_dirac, to_complex

f, P, out = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
lam, V = np.linalg.eig(symbol_dirac(np.zeros(3), 0.8))
pf, nf = np.flatnonzero(-np.angle(lam) > 0), np.flatnonzero(-np.angle(lam) < 0)
v1, v2 = V[:, pf[0]], V[:, pf[1]]
pos = np.load(f.parent / "pos.npy", mmap_mode="r")
ix = np.rint(np.asarray(pos) * np.sqrt(3)).astype(np.int32)
H = P // 2
slab = np.flatnonzero((np.abs(ix[:, 1]) <= 1) & (ix[:, 0] > 0))
lut = np.full((P,) * 3, -1, np.int32)
lut[ix[:, 0] + H, ix[:, 1] + H, ix[:, 2] + H] = np.arange(len(ix), dtype=np.int32)
nbr = np.stack([lut[(ix[slab, 0] + int(h[0]) + H) % P, (ix[slab, 1] + int(h[1]) + H) % P, (ix[slab, 2] + int(h[2]) + H) % P]
                for h in HS], 1)
del lut
ck = np.load(f)
L, R = ck["L"], ck["R"]
need = np.unique(np.concatenate([slab, nbr.ravel()]))
psi = np.concatenate([to_complex(L[need]), to_complex(R[need])], 1)
t = int(ck["t"])
del L, R, ck
m = {g: i for i, g in enumerate(need)}
loc = np.vectorize(m.get)
ps, pn = psi[loc(slab)], psi[loc(nbr)]
a, an = ps @ v1.conj(), pn @ v1.conj()
rho = (np.abs(ps) ** 2).sum(1)
dph = np.angle(an * a.conj()[:, None])
g = (dph[:, :, None] * HS[None]).sum(1) / 8 * np.sqrt(3)  # per hex distance
p = np.asarray(pos[slab])
x, z = p[:, 0], p[:, 2]
v_phi, v_rho, v_z = g[:, 1], g[:, 0], g[:, 2]  # y = 0, x > 0: phi-hat = y-hat
w2 = np.abs(ps @ v2.conj()) ** 2
wn = (np.abs(ps @ V[:, nf].conj()) ** 2).sum(1)
dA = (1 / np.sqrt(3)) ** 2 * 2  # each grid point in a |y|<=1 slab of width ~3 grid -> area per point ~ h^2 * 2 (BCC parity)
flux = np.sum(rho * v_phi) / 3 * dA
print(f"{f.parent.name[:40]} t {t}: slab cells {len(slab)}; net toroidal flux int rho v_phi dA {flux:+.3e}; "
      f"<|v_phi|> {np.mean(np.abs(v_phi)):.2e} <|v_pol|> {np.mean(np.hypot(v_rho, v_z)):.2e}; max |v2|^2 {w2.max():.2e}; "
      f"max neg {wn.max():.2e}", flush=True)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
lim = np.percentile(np.abs(v_phi), 99)
sc = ax[0].scatter(x, z, c=v_phi, s=2, cmap="RdBu_r", vmin=-lim, vmax=lim)
ax[0].tricontour(x, z, rho, levels=[0.25], colors="k", linewidths=0.8)
k = np.arange(0, len(x), max(1, len(x) // 1500))
ax[0].quiver(x[k], z[k], v_rho[k], v_z[k], color="0.3", scale=None, width=0.0015)
ax[0].set_aspect("equal"); ax[0].set_xlabel("x (rho)"); ax[0].set_ylabel("z")
ax[0].set_title(f"toroidal v_phi (colour), poloidal (arrows), core rho=0.25; t {t}")
plt.colorbar(sc, ax=ax[0])
sc = ax[1].scatter(x, z, c=rho, s=2, cmap="viridis"); plt.colorbar(sc, ax=ax[1]); ax[1].set_aspect("equal")
ax[1].set_title("density")
fig.suptitle(f"{f.parent.name}  net toroidal flux {flux:+.2e}")
fig.tight_layout(); fig.savefig(out, dpi=80)
