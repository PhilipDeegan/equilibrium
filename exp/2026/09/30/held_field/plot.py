from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium.substrate.lattice import wave_speed

here = Path(__file__).parent
cr = 24.0 / wave_speed()
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
for E in ("0.001", "0.003", "0.01"):
    r = np.load(here / f"rec_gauss_g100_E{E}z.npz")["rec"]
    ax[0].plot(r[:, 0] / cr, r[:, 8] / float(E), lw=0.8, label=f"E={E}")
    ax[1].plot(r[:, 0] / cr, r[:, 8] / float(E), lw=0.8, label=f"E={E}")
    r = np.load(here / f"rec_dipole_g100_E{E}x.npz")["rec"]
    ax[2].plot(r[:, 0] / cr, r[:, 9] / float(E), lw=0.8, label=f"E={E}")
for ic, col, lab in (("gauss_g100_E0z", 11, "gauss |p_z|"), ("dipole_g100_E0x", 9, "dipole |p_x|"),
                     ("dipole_g100_E0x", 6, "dipole |x_c core|")):
    r = np.load(here / f"rec_{ic}.npz")["rec"]
    ax[3].semilogy(r[:, 0] / cr, np.abs(r[:, col]) + 1e-20, lw=0.8, label=lab)
ax[0].set_ylabel("gauss z_c(core) / E")
ax[1].set_xlim(0, 6)
ax[1].set_ylabel("gauss z_c(core) / E (early)")
ax[2].set_ylabel("dipole sum x u / E")
ax[3].set_ylabel("E = 0: symmetry-forbidden components")
for a in ax:
    a.set_xlabel("t (crossings)")
    a.legend(fontsize=7)
fig.tight_layout()
fig.savefig(here / "held_field.png", dpi=110)
