"""Plots of the SCF window means (avg_c*.npy): density in the ring plane z = zc, density and phase (vs the far mean) in
the meridional cut y = 0, and the radial l = 0 deficit / l = 0 phase / l = 1 phase profiles about (0, 0, zc), cycles
overlaid. Output: <here>/<run name split on '_'>/cNN.png + profiles.png."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from equilibrium.substrate.walk import cpp_box

run = Path(sys.argv[1])
P = int(sys.argv[2]) if len(sys.argv) > 2 else 300
here = Path(__file__).parent
odir = here / Path(*run.name.split("_"))
odir.mkdir(parents=True, exist_ok=True)
pos = np.array(cpp_box(P)[0].leaf_pos)
x, y, z = pos.T
rho_c = np.hypot(x, y)
side = P / np.sqrt(3)
lim = side / 2
files = sorted(run.glob("avg_c*.npy"), key=lambda f: int(f.stem[5:]))
fig_p, axp = plt.subplots(1, 3, figsize=(18, 5))
cmap = plt.get_cmap("viridis")
for i, f in enumerate(files):
    a = np.load(f)
    rho = np.abs(a) ** 2
    core = rho < 0.25
    wt = (1 - rho)[core]
    rc, zc = np.sum(wt * rho_c[core]) / wt.sum(), np.sum(wt * z[core]) / wt.sum()
    dt = np.hypot(rho_c - rc, z - zc)
    a = a * np.conj(np.mean(a[dt > 45])) / np.abs(np.mean(a[dt > 45]))
    ph = np.angle(a)
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    zp = z[np.argmin(np.abs(z - zc))]
    m = np.abs(z - zp) < 0.05
    gg = np.linspace(-lim, lim, 401)
    G1, G2 = np.meshgrid(gg, gg)
    im = ax[0].imshow(griddata((x[m], y[m]), rho[m], (G1, G2)), origin="lower", extent=(-lim, lim, -lim, lim),
                      cmap="viridis", vmin=0, vmax=1.1)
    ax[0].set_title(f"density, ring plane z = {zp:+.1f}")
    fig.colorbar(im, ax=ax[0], shrink=0.8)
    m2 = np.abs(y) < 0.3
    H1, H2 = np.meshgrid(gg, zc + gg)
    im = ax[1].imshow(griddata((x[m2], z[m2]), rho[m2], (H1, H2)), origin="lower",
                      extent=(-lim, lim, zc - lim, zc + lim), cmap="viridis", vmin=0, vmax=1.1)
    ax[1].set_title("density, meridional cut y = 0")
    fig.colorbar(im, ax=ax[1], shrink=0.8)
    im = ax[2].imshow(griddata((x[m2], z[m2]), ph[m2], (H1, H2)), origin="lower",
                      extent=(-lim, lim, zc - lim, zc + lim), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax[2].set_title("phase vs far mean, meridional cut y = 0")
    fig.colorbar(im, ax=ax[2], shrink=0.8)
    fig.suptitle(f"{run.name} {f.stem}: rc {rc:.1f} zc {zc:+.1f}")
    fig.tight_layout()
    fig.savefig(odir / f"c{int(f.stem[5:]):02d}.png", dpi=80)
    plt.close(fig)
    rel = pos - [0, 0, zc]
    r = np.linalg.norm(rel, axis=1)
    ct = rel[:, 2] / np.maximum(r, 1e-9)
    rs = np.arange(45, lim - 2, 5.0)
    prof = []
    for r0 in rs:
        sh = np.abs(r - r0) < 2.5
        prof.append([np.mean(1 - rho[sh]), np.angle(np.mean(a[sh])), 3 * np.mean(ph[sh] * ct[sh])])
    prof = np.array(prof)
    c = cmap(i / max(1, len(files) - 1))
    for j in range(3):
        axp[j].plot(rs, prof[:, j], color=c, label=f.stem)
for j, t in enumerate(["l = 0 deficit <1 - rho>", "l = 0 phase", "l = 1 phase (cos theta)"]):
    axp[j].set_title(t)
    axp[j].set_xlabel("r about (0, 0, zc)")
    axp[j].axhline(0, color="k", lw=0.5)
axp[0].legend(fontsize=7)
fig_p.suptitle(run.name)
fig_p.tight_layout()
fig_p.savefig(odir / "profiles.png", dpi=80)
print(odir)
