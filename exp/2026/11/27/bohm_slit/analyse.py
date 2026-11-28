import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

here = Path(__file__).parent
run = here / (sys.argv[1] if len(sys.argv) > 1 else "run_g.npz")
d = np.load(run)
a = ast.literal_eval(str(d["args"]))
yc, kx = int(d["yc"]), float(d["kx"])
lam = 2 * np.pi / kx
L = a["xs"] - a["xw"]
hy = d["hit_y"]
hit = np.isfinite(hy)
N = len(hy)
flux = d["flux"]
y = np.arange(len(flux))
lo, hi = a["sponge"], a["ny"] - a["sponge"]
bw = 2.0
edges = np.arange(lo, hi + bw, bw)
cnt, _ = np.histogram(hy[hit], edges)
pd = cnt / (N * bw)
fb = np.array([flux[(y >= e0) & (y < e1)].mean() for e0, e1 in zip(edges[:-1], edges[1:])])
err = np.sqrt(np.maximum(cnt, 1)) / (N * bw)
chi2 = (((pd - fb) / err) ** 2)[fb > 0].sum()
dof = (fb > 0).sum()

pk, _ = find_peaks(flux, prominence=flux.max() * 0.02)
sp = np.diff(y[pk])
print(f"lambda={lam:.2f} L={L} d={a['slit_d']} fraunhofer spacing={lam * L / a['slit_d']:.2f}")
print(f"flux peaks at y-yc={y[pk] - yc}  spacings={sp}")
print(f"transmitted: particles={hit.mean():.4f} flux={flux.sum():.4f} (inside screen window {flux[int(lo):int(hi)].sum():.4f})")
print(f"histogram vs flux: chi2/dof={chi2:.1f}/{dof} L1(shape)={np.abs(pd / pd.sum() - fb / fb.sum()).sum():.4f}")
print(f"crossed symmetry axis: {d['crossed'].sum()} of {N}")
upper = hit & (d["side0"] > 0)
lower = hit & (d["side0"] < 0)
print(f"hits from upper half below axis: {(hy[upper] < yc).sum()}, lower half above: {(hy[lower] > yc).sum()}")
print(f"hops hex/sq: {d['nhops'].sum(0)}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
dt_ = np.load(here / sys.argv[2]) if len(sys.argv) > 2 else d
tr = dt_["traj"]
for i in range(tr.shape[1]):
    ax[0].plot(tr[:, i, 0], tr[:, i, 1] - yc, lw=0.6, color="C0" if d["side0"][dt_["traj_ids"][i]] > 0 else "C1")
w = d["wall_A"]
wy, wx = np.nonzero(w.T)
ax[0].scatter(wx, wy - yc, s=1, c="k")
ax[0].axvline(a["xs"], color="gray", ls=":")
ax[0].set(xlim=(a["sponge"], a["xs"] + 5), ylim=(-70, 70), xlabel="x (cells)", ylabel="y - yc",
          title="Bohm trajectories (blue: start above axis, orange: below)")
ax[1].step(edges[:-1] - yc, pd, where="post", label=f"particles ({hit.sum()} hits of {N})")
ax[1].plot(y - yc, flux, "k", lw=1, label="time-integrated flux J_x (Born)")
ax[1].set(xlim=(lo - yc, hi - yc), xlabel="y - yc at screen", ylabel="arrivals per unit y (fraction of all)",
          title=f"screen at x={a['xs']}: chi2/dof={chi2:.0f}/{dof}")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(run.parent / "bohm_slit.png", dpi=110)
