import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

run = Path(__file__).parent / (sys.argv[1] if len(sys.argv) > 1 else "run_a4.npz")
d = np.load(run)
a = ast.literal_eval(str(d["args"]))
Z, n = a["Z"], a["n"]
lu = 1 / Z
c = n // 2
ext = (np.array([-c, n - c, -c, n - c]) - 0.5) / lu
E, al = d["E"], d["alpha"]

fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
vmax = d["rho0c"].max()
for j, (k, title) in enumerate([("rho0c", "Copenhagen |psi|^2, F = 0"), ("rho0s", f"swarm density, F = 0 (N = {a['N']})")]):
    im = ax[0, j].imshow(d[k].T, origin="lower", extent=ext, vmin=0, vmax=vmax, cmap="magma")
    ax[0, j].set(title=title, xlabel="x / a0", ylabel="z / a0", xlim=(-5, 5), ylim=(-5, 5))
fig.colorbar(im, ax=ax[0, :2], shrink=0.8)

r = np.abs(np.arange(n) - c) / lu
ax[0, 2].semilogy(r, d["rho0c"][c] / lu**-3 * 1, "k-", lw=1, label="Copenhagen (lattice)")
ax[0, 2].semilogy(r, d["rho0s"][c] / lu**-3, ".", ms=3, label="swarm")
rr = np.linspace(0, 8, 200)
ax[0, 2].semilogy(rr, np.exp(-2 * rr) / np.pi, "C3--", lw=1, label="exact e^{-2r}/pi")
ax[0, 2].axhline(a["eps_n"] / a["N"] * lu**3, color="gray", ls=":", label="floor (one constituent)")
ax[0, 2].set(xlabel="|z| / a0 along the axis", ylabel="density (a0^-3)", ylim=(1e-7, 1), title="radial profile", xlim=(0, 8))
ax[0, 2].legend(fontsize=8)

dc = d["rho1c"] - d["rho0c"]
ds = (d["rho1s"] - d["rhoms"]) / 2 if "rhoms" in d else d["rho1s"] - d["rho0s"]
lim = np.abs(dc).max()
for j, (D, title) in enumerate([(dc, "Copenhagen: on - off"), (ds, "swarm: (+F - -F)/2")]):
    ax[1, j].imshow(D.T, origin="lower", extent=ext, vmin=-lim, vmax=lim, cmap="RdBu_r")
    ax[1, j].set(title=f"{title}, F = {a['F']} au along +z", xlabel="x / a0", ylabel="z / a0", xlim=(-5, 5), ylim=(-5, 5))

z = (np.arange(n) - c) / lu
ax[1, 2].plot(z, dc[c] / lu**-3, "k-", lw=1, label="Copenhagen")
ax[1, 2].plot(z, ds[c] / lu**-3, ".", ms=3, label="swarm")
ax[1, 2].set(xlabel="z / a0 (x = 0)", ylabel="density change (a0^-3)", xlim=(-6, 6),
             title=f"alpha: Copenhagen {al[0]:.2f}, swarm {al[2]:.2f}, exact 4.5")
ax[1, 2].legend(fontsize=8)
fig.suptitle(f"Hydrogen 1s on the Kelvin lattice (a0 = {lu:g} cells): E Copenhagen {E[0]:.4f}, swarm {E[2]:.4f}, exact -0.5")
fig.savefig(run.with_suffix(".png"), dpi=100)
print(f"wrote {run.with_suffix('.png')}")
