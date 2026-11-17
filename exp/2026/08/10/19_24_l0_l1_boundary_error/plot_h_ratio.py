"""H/H0 trajectory comparison: uniform L0 baseline vs the single L0/L1
junction case. Recomputes H directly from the saved checkpoints (not from
log text) so the plot is reproducible from the .h5 files alone.
"""

import glob
import os
import sys

sys.path.insert(0, ".")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium import lattice as _lat


def H_of(path):
    alat = _lat.AdaptiveLattice.load(path)
    cl = alat[0]
    u = np.array(cl.leaf_u)
    phi = np.array(cl.leaf_phi)
    return float(np.sum(u**2) + 0.5 * np.sum(phi**2))


def series(ic_path, step_dir):
    H0 = H_of(ic_path)
    steps = sorted(glob.glob(os.path.join(step_dir, "step_*.h5")))
    xs = [0]
    ys = [1.0]
    for p in steps:
        i = int(os.path.basename(p).split("_")[1].split(".")[0])
        xs.append(i)
        ys.append(H_of(p) / H0)
    return xs, ys


def main():
    x_u, y_u = series("/bk/uniform_l0_ic220.h5", "/bk/uniform_l0_stepped")
    x_b, y_b = series("/bk/l0_l1_boundary_ic220.h5", "/bk/l0_l1_boundary_stepped")

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_lin.plot(x_u, y_u, "o-", label="uniform L0 (no junction)", color="#1f77b4")
    ax_lin.plot(x_b, y_b, "s-", label="R60 L0 / L1 @ R40 (1 junction)", color="#d62728")
    ax_lin.set_xlabel("step")
    ax_lin.set_ylabel("H/H0")
    ax_lin.set_title("H/H0 (linear)")
    ax_lin.legend(fontsize=8)
    ax_lin.grid(alpha=0.3)

    ax_log.plot(
        x_u,
        np.abs(np.array(y_u) - 1.0) + 1e-16,
        "o-",
        label="uniform L0 (no junction)",
        color="#1f77b4",
    )
    ax_log.plot(
        x_b,
        np.abs(np.array(y_b) - 1.0) + 1e-16,
        "s-",
        label="R60 L0 / L1 @ R40 (1 junction)",
        color="#d62728",
    )
    ax_log.set_yscale("log")
    ax_log.set_xlabel("step")
    ax_log.set_ylabel("|H/H0 - 1|")
    ax_log.set_title("|H/H0 - 1| (log)")
    ax_log.legend(fontsize=8)
    ax_log.grid(alpha=0.3, which="both")

    fig.suptitle("Level-border error: uniform L0 vs single L0/L1 junction", fontsize=11)
    plt.tight_layout()
    out = "exp/2026/08/10/19_24_l0_l1_boundary_error/plots/h_ratio_comparison.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")
    print("uniform:", list(zip(x_u, y_u)))
    print("boundary:", list(zip(x_b, y_b)))


if __name__ == "__main__":
    main()
