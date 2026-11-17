import sys

sys.path.insert(0, ".")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("exp/2026/08/20/17_21_sponge_ablation/result.npz")
half_len = float(d["half_len"])

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax = axes[0]
ax.plot(d["z_sponge"], d["u0_sponge"], "k--", lw=1, label="u(z, t=0)")
ax.plot(d["z_sponge"], d["uT_sponge"], "r-", lw=1.2, label="u(z, t=T)  [sponge]")
ax.axvspan(0.85 * half_len, half_len, color="red", alpha=0.08, label="sponge zone")
ax.axvspan(-half_len, -0.85 * half_len, color="red", alpha=0.08)
ax.set_ylabel("u")
ax.set_title(f"DampedLattice (sponge on)   corr_u={float(d['corr_u_sponge']):.5f}")
ax.legend(loc="upper right", fontsize=8)

ax = axes[1]
ax.plot(d["z_plain"], d["u0_plain"], "k--", lw=1, label="u(z, t=0)")
ax.plot(d["z_plain"], d["uT_plain"], "b-", lw=1.2, label="u(z, t=T)  [no sponge]")
ax.set_ylabel("u")
ax.set_xlabel("z (axial position, cells)")
ax.set_title(f"Plain SubstrateLattice (no BC)   corr_u={float(d['corr_u_plain']):.5f}")
ax.legend(loc="upper right", fontsize=8)

fig.suptitle("Traveling-wave one-period phase test: sponge vs no-sponge ablation")
fig.tight_layout()
out = "exp/2026/08/20/17_21_sponge_ablation/sponge_ablation.png"
fig.savefig(out, dpi=130)
print(f"saved {out}")
