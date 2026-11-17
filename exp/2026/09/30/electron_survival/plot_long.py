import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out = Path(__file__).parent
names = ["gauss_g36_w1.5", "gauss_g100_w1.5", "dipole_g36_w1.5", "dipole_g100_w1.5"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for n in names:
    rec = np.load(out / f"long_{n}.npz")["rec"]
    t = rec[:, 0]
    ax[0].plot(t, rec[:, 1], label=n)
    ax[1].plot(t, rec[:, 6] / rec[0, 6], label=n)
    ax[2].semilogy(t, rec[:, 2], label=n)
    i = np.searchsorted(t, [t[-1] * q for q in (0.25, 0.5, 0.75)])
    print(n, "Etot/Etot(t0):", " ".join(f"{rec[j, 6] / rec[0, 6]:.4f}" for j in [0, *i, -1]),
          " max step-to-step rise", f"{np.max(np.diff(rec[:, 6])) / rec[0, 6]:.2e}")
for a, tl in zip(ax, ["core energy / initial (r<4.5)", "total lattice energy / first sample", "E-weighted core sigma"]):
    a.set_title(tl); a.set_xlabel("t"); a.legend(fontsize=7)
fig.tight_layout()
fig.savefig(out / "long.png", dpi=110)
