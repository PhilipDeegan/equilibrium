"""run.py records: tube / structure energy, net circulation C0, travelling concentration |C1| and its phase, vs t."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).parent
fig, ax = plt.subplots(2, 3, figsize=(18, 9))
for k, fn in enumerate(sorted(here.glob("pulse_*_rec.npz"))):
    d = np.load(fn)
    r = d["rec"]
    t = r[:, 0]
    lab = fn.name.split("_R")[0].replace("pulse_", "")
    kw = dict(color=f"C{k}", label=lab)
    ax[0, 0].semilogy(t, r[:, 2], **kw)
    ax[0, 0].semilogy(t, r[:, 3], color=f"C{k}", ls="--")
    ax[0, 1].plot(t, r[:, 4], **kw)
    ax[0, 2].semilogy(t, r[:, 5], **kw)
    ax[1, 0].plot(t, -np.unwrap(r[:, 6]), **kw)
    ax[1, 1].semilogy(t, np.maximum(r[:, 11], 1e-6), **kw)
    ax[1, 2].plot(t, r[:, 7], **kw)
    ax[1, 2].plot(t, r[:, 8], color=f"C{k}", ls="--")
for a, yl in zip(ax.flat, ["E_tube (solid), E_struct (dashed)", "net poloidal circulation C0 / initial",
                           "|C1| / initial (travelling concentration)", "phase of C1 (orbit angle, rad)",
                           "grid-scale fraction of u (tube)", "u: b0 (solid), |b1| (dashed) / initial"]):
    a.set_xlabel("t")
    a.set_ylabel(yl)
ax[0, 1].legend(fontsize=8)
fig.suptitle("single-signed poloidal pulse (discrete step(), R56, r0 12, w 2.67)")
fig.tight_layout()
fig.savefig(here / "pulse_rec.png", dpi=100)
