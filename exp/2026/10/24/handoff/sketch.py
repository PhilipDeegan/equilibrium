"""Schematic of the receiver test (meridional slice): structure = two opposite toroidal lobes + near field; panels:
isolated (no receiver), receiver shell at distance d, internal receivers."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

x = np.linspace(-3, 3, 400)
X, Z = np.meshgrid(x, x)
R = np.hypot(X, Z)
lobes = np.exp(-((np.abs(X) - 0.6) ** 2 + Z**2) / 0.12) * np.sign(X)
near = np.exp(-np.maximum(R - 1.0, 0) / 0.35) * (R > 1.0) * 0.25


def base(ax, title):
    ax.imshow(lobes, extent=[-3, 3, -3, 3], origin="lower", cmap="RdBu_r", vmin=-1.2, vmax=1.2)
    ax.contour(X, Z, near, levels=[0.02, 0.06, 0.12], colors="0.55", linewidths=0.8, linestyles="--")
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ls=":", color="0.3", lw=0.8))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlim(-3, 3), ax.set_ylim(-3, 3)


def arrow(ax, p, q, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.2, **kw))


fig, axs = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

ax = axs[0]
base(ax, "A  isolated: emitter, no receiver")
ax.text(0, -2.55, "near field held (dashed), nothing offloads\nexpect: no decay (today's photon-pair run)", ha="center", fontsize=8.5)
ax.text(2.1, 2.5, "vacuum\n(not a receiver)", ha="center", fontsize=8, color="0.4")

ax = axs[1]
base(ax, "B  receiver shell at distance d")
d = 1.9
ax.add_patch(Circle((0, 0), d, fill=False, color="#2ca02c", lw=6, alpha=0.35))
for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
    c, s = np.cos(th), np.sin(th)
    arrow(ax, (1.1 * c, 1.1 * s), ((d - 0.15) * c, (d - 0.15) * s), color="#d62728")
    arrow(ax, ((d - 0.2) * c + 0.12 * s, (d - 0.2) * s - 0.12 * c), (1.15 * c + 0.12 * s, 1.15 * s - 0.12 * c), color="#1f77b4", alpha=0.6)
ax.annotate("", xy=(d, -2.2), xytext=(0, -2.2), arrowprops=dict(arrowstyle="<->", color="0.2"))
ax.text(d / 2, -2.45, "d", ha="center", fontsize=9)
ax.text(0, 2.55, "offload (red) and return (blue) through the near field\nmeasure: rate vs d, cycling vs settling", ha="center", fontsize=8.5)
ax.text(2.35, 1.2, "receptive\ncells", color="#2ca02c", fontsize=8, ha="center")

ax = axs[2]
base(ax, "C  internal receivers (collision analogue)")
rng = np.random.default_rng(3)
pts = rng.uniform(-0.95, 0.95, (40, 2))
pts = pts[np.hypot(*pts.T) < 0.95]
ax.scatter(pts[:, 0], pts[:, 1], s=14, color="#2ca02c", edgecolor="k", lw=0.3, zorder=3)
for p_ in pts[::4]:
    q = p_ + rng.normal(0, 0.25, 2)
    arrow(ax, tuple(p_), tuple(q), color="#d62728", alpha=0.8)
ax.text(0, -2.55, "fine-scale content inside the structure takes / returns\nexpect: resistive-like decay (MHD reference)", ha="center", fontsize=8.5)

fig.suptitle("Receiver test, meridional slice: structure (two opposite toroidal lobes), r = a dotted, near field dashed",
             fontsize=11)
fig.savefig("/home/p/git/eq/exp/2026/10/24/handoff/sketch.png", dpi=110)
