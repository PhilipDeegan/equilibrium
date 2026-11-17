"""alpha scan of run.py (--tag _scan): grid-scale fraction, tube energy, winding amplitude vs t and proper time."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).parent
runs = []
for fn in here.glob("pair_opp_a*_R56_r12_w2.67_scan_rec.npz"):
    d = np.load(fn)
    runs.append((json.loads(str(d["args"]))["alpha"], d))
runs.sort(key=lambda r: r[0])

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for a, d in runs:
    r = d["rec"]
    t, E, sig = r[:, 0], r[:, 1], r[:, 2]
    tau = np.concatenate([[0], np.cumsum(0.5 * (sig[1:] + sig[:-1]) * np.diff(t))])
    gf, amp = d["grid_frac"], np.abs(d["amps"][:, 1]) / np.abs(d["amps"][0, 1])
    for row, xx, xl in ((0, t, "t"), (1, tau, "proper time (tube, E-weighted)")):
        ax[row, 0].semilogy(xx, gf, label=f"alpha {a:g}")
        ax[row, 1].plot(xx, E, label=f"alpha {a:g}")
        ax[row, 2].semilogy(xx, amp, label=f"alpha {a:g}")
        for c, yl in enumerate(["grid-scale fraction of u (tube)", "E_tube / E0", "|a_(n,+1)| / initial"]):
            ax[row, c].set_xlabel(xl)
            ax[row, c].set_ylabel(yl)
    k = np.argmax(gf > 0.1) if np.any(gf > 0.1) else None
    print(f"alpha {a:5g}: sigma_tube(0) {sig[0]:.3f}  grid frac end {gf[-1]:.3e}"
          + (f"  > 0.1 at t = {t[k]:.2f} (tau {tau[k]:.2f})" if k else "  never > 0.1")
          + f"  E_tube end {E[-1]:.3f}  |a| end {amp[-1]:.3f}  lam_max end {r[-1, 5]:.2f}")
ax[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(here / "alpha_scan.png", dpi=100)
