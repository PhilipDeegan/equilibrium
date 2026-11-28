import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).parent
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for f in sorted(here.glob("run_k*.npz")):
    d = np.load(f)
    t, x, p0, x0, k0, vg, s0, m = d["t"], d["x"], d["p0"], d["x0"], d["k0"], d["vg"], float(d["sigma0"]), float(d["m"])
    kh = k0 / np.linalg.norm(k0)
    vc = (x[-1, 0] - x[0, 0]) / t[-1]
    cont = np.sqrt(1 + (t / (2 * m * s0**2)) ** 2)
    rel = np.linalg.norm(x[:, 1:7] - x[:, :1], axis=2) / np.linalg.norm(p0[1:7] - p0[:1], axis=1)
    para = rel[:, :2].mean(1)
    perp = rel[:, 2:6].mean(1)
    evar = d["ens_var"] / d["S2"]
    print(
        f"{f.stem}: |k0|={np.linalg.norm(k0):.4f} centre v.khat={vc @ kh:.4f} |vg|={np.linalg.norm(vg):.4f} "
        f"|k0|/m={np.linalg.norm(k0) / m:.4f} v_perp={np.linalg.norm(vc - (vc @ kh) * kh):.2e}\n"
        f"   spread t_end: para={para[-1]:.4f} perp={perp[-1]:.4f} continuum={cont[-1]:.4f} "
        f"ens var/rho var t0={evar[0]:.4f} t_end={evar[-1]:.4f} drift={evar[-1] / evar[0] - 1:+.2e} "
        f"ens mean-X drift={np.linalg.norm((d['ens_mean'][-1] - d['X'][-1]) - (d['ens_mean'][0] - d['X'][0])):.3e}\n"
        f"   hops hex/sq total={d['hops'].sum(0)}"
    )
    lab = f.stem.replace("run_k", "k=")
    ax[0].plot(t, para, label=f"{lab} along k")
    ax[0].plot(t, perp, "--", label=f"{lab} across k")
    ax[1].plot(t, (x[:, 0] - x[0, 0]) @ kh - np.linalg.norm(k0) / m * t, label=lab)
    ax[2].plot(t, evar / evar[0] - 1, label=lab)
ax[0].plot(t, cont, "k:", label="continuum")
ax[0].set(xlabel="t", ylabel="|x_i - x_centre| / initial", title="Bohm spreading of particle offsets")
ax[1].set(xlabel="t", ylabel="centre particle lag along k (cells)", title="lag vs continuum k0/m")
ax[2].set(xlabel="t", ylabel="ensemble var / rho var, rel. to t=0", title="equivariance (500 particles)")
for a in ax:
    a.legend(fontsize=7)
fig.tight_layout()
fig.savefig(here / "bohm_free.png", dpi=110)
print("wrote", here / "bohm_free.png", file=sys.stderr)
