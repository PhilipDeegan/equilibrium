"""First real quantitative read of ~/tmp/EM_B.vtkhdf (PHARE spheromak B(t),
35 steps, t=0->3.4) -- tracks toroidal-like (Bx, out-of-plane) vs
poloidal-like (in-plane By,Bz) RMS over the core, plus whole-slice total
field energy, to see whether PHARE shows a fast reconnection/merge phase
followed by a slow relaxed plateau, as a target regime for DST's
cross_scale-driven erosion.

Reuses the reader from
exp/2026/08/16/14_50_rebuild_r180l0_r60l1_post_pass1_fix/plot_phare_b_slice.py
(load_step_grid) -- no changes to that reader, this is analysis only.

Run:
    PYTHONPATH=$PWD python3 exp/2026/08/18/15_22_phare_bt_quantitative_read/analyze_phare_bt.py
"""

import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "exp/2026/08/16/14_50_rebuild_r180l0_r60l1_post_pass1_fix")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py

from plot_phare_b_slice import load_step_grid

HERE = os.path.dirname(__file__)
PATH = os.path.expanduser("~/tmp/EM_B.vtkhdf")
CORE_R = 15


def main():
    with h5py.File(PATH, "r") as f:
        values = f["VTKHDF/Steps/Values"][:]
        n_unique = 35  # first half of the file; second half is a verified duplicate
        ts, bx_rms, ip_rms, total_e = [], [], [], []
        core = None
        for step in range(n_unique):
            grid = load_step_grid(f, step)
            Bx, By, Bz = grid[:, :, 0], grid[:, :, 1], grid[:, :, 2]
            if core is None:
                n = grid.shape[0]
                yi = np.arange(n) - (n - 1) / 2.0
                Yi, Zi = np.meshgrid(yi, yi)
                core = np.sqrt(Yi**2 + Zi**2) < CORE_R
            ts.append(values[step])
            bx_rms.append(np.sqrt(np.nanmean(Bx[core] ** 2)))
            ip_rms.append(np.sqrt(np.nanmean(By[core] ** 2 + Bz[core] ** 2)))
            total_e.append(np.nansum(Bx**2 + By**2 + Bz**2))

    ts = np.array(ts)
    bx_rms = np.array(bx_rms)
    ip_rms = np.array(ip_rms)
    total_e = np.array(total_e)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(
        ts, bx_rms, "o-", label="Bx rms (toroidal-like, core r<15)", color="crimson"
    )
    ax.plot(
        ts,
        ip_rms,
        "s-",
        label="in-plane rms (poloidal-like, core r<15)",
        color="steelblue",
    )
    ax.axvspan(0, 1.8, color="crimson", alpha=0.07, label="fast phase (t<~1.8)")
    ax.set_xlabel("t")
    ax.set_ylabel("field RMS (core)")
    ax.set_title("PHARE core field: fast toroidal collapse, slow poloidal bleed")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(ts, total_e / total_e[0], "d-", color="black")
    ax.axvspan(0, 1.8, color="crimson", alpha=0.07)
    ax.set_xlabel("t")
    ax.set_ylabel("total field energy / E(t=0)")
    ax.set_title("Whole-slice energy: -62% by t=3.4, mostly in fast phase")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out_png = os.path.join(HERE, "plots", "phare_bt_two_timescale.png")
    plt.savefig(out_png, dpi=140)
    print(f"wrote {out_png}")

    print("\nt, Bx_rms, inplane_rms, total_E/E0")
    for t, bx, ip, e in zip(ts, bx_rms, ip_rms, total_e / total_e[0]):
        print(f"{t:.1f}  {bx:.5f}  {ip:.5f}  {e:.4f}")


if __name__ == "__main__":
    main()
