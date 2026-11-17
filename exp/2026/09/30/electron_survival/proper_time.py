"""Item 2 (next 10-01): retention vs core proper time tau = int sigma_core dt (E-weighted core sigma, saved records).

If Ecore/E0 vs tau collapses across g, the 'retention' at fixed t is clock slowdown, not trapping.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium.substrate.lattice import wave_speed

here = Path(__file__).parent
cross = 2 * 12.0 / wave_speed()


def load(f):
    r = np.load(f)["rec"]
    t = r[:, 0]
    tau = np.concatenate([[0.0], np.cumsum(0.5 * (r[1:, 2] + r[:-1, 2]) * np.diff(t))])
    return t / cross, tau / cross, r[:, 1]


def at(x, y, x0):
    return np.interp(x0, x, y) if x[-1] >= x0 else np.nan


taus = (0.5, 1.0, 2.0, 3.0)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for row, ic in enumerate(("gauss", "dipole")):
    files = sorted(here.glob(f"scan_{ic}_w1.5_g*_{ic}.npz"), key=lambda f: float(re.search(r"_g([0-9.]+)_", f.name)[1]))
    print(f"{ic} w1.5 scan: g | t_end  tau_end | Ecore/E0 at tau = {taus} | at t = t_end")
    cmap = plt.get_cmap("viridis")
    for i, f in enumerate(files):
        g = float(re.search(r"_g([0-9.]+)_", f.name)[1])
        t, tau, E = load(f)
        col = cmap(i / max(len(files) - 1, 1))
        axes[row, 0].plot(t, E, color=col, lw=0.8, label=f"g={g:g}" if i % 3 == 0 else None)
        axes[row, 1].plot(tau, E, color=col, lw=0.8)
        print(f"  {g:6g} | {t[-1]:5.1f} {tau[-1]:6.2f} | " + " ".join(f"{at(tau, E, x):.3f}" for x in taus)
              + f" | {E[-1]:.3f}")
    for f in sorted(here.glob(f"long_{ic}_g*_w1.5*.npz")):
        t, tau, E = load(f)
        axes[row, 2].plot(tau, E, lw=0.8, label=f.stem[len("long_"):])
        print(f"  long {f.stem[5:]:28s} t_end {t[-1]:6.1f} tau_end {tau[-1]:6.2f} | "
              + " ".join(f"{at(tau, E, x):.3f}" for x in taus) + f" | {E[-1]:.3f}")
    axes[row, 0].set_xlabel("t (crossings)")
    axes[row, 1].set_xlabel("tau_core (proper crossings)")
    axes[row, 2].set_xlabel("tau_core (proper crossings)")
    axes[row, 1].set_xlim(0, 3)
    for ax in axes[row]:
        ax.set_ylabel(f"{ic}: Ecore / E0")
        ax.set_yscale("log")
    axes[row, 0].legend(fontsize=7)
    axes[row, 2].legend(fontsize=6)
fig.tight_layout()
fig.savefig(here / "proper_time.png", dpi=120)
