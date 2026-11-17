"""run.py records: energy in tube / structure / domain, and the seed projections of A, vs t (crossings)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).parent
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for k, fn in enumerate(sorted(here.glob("coil_*_rec.npz"))):
    d = np.load(fn)
    r, tc = d["rec"], d["rec"][:, 0] / float(d["crossing"])
    lab = fn.name.split("_R")[0]
    for j, (col, s) in enumerate(((2, "E_tube"), (3, "E_struct"), (1, "E_domain"))):
        ax[0].semilogy(tc, r[:, col], color=f"C{k}", ls=["-", "--", ":"][j], label=f"{lab} {s}")
    ax[1].plot(tc, r[:, 7], color=f"C{k}", label=f"{lab} a_psi")
    if "p0" not in lab:
        ax[1].plot(tc, r[:, 8], color=f"C{k}", ls="--", label=f"{lab} a_theta")
    ax[2].plot(tc, r[:, 6], color=f"C{k}", label=f"{lab} d_rms (tube, E-weighted)")
ax[0].set_ylabel("energy / initial")
ax[1].set_ylabel("projection of A on the seed")
ax[2].set_ylabel("d_rms")
for a in ax:
    a.set_xlabel("t / crossing")
    a.legend(fontsize=8)
fig.suptitle("poloidal coil on step_ae (DEC, no lam), R56 h = 1, r0 12, w 2.67")
fig.tight_layout()
fig.savefig(here / "coil_rec.png", dpi=100)
