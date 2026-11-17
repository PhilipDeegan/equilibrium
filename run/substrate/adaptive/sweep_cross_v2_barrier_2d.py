"""
2D (m0 x cross_scale) sweep for the step_cross v2 `barrier` gate.

Follow-up to probe_cross_v2_gates.py's single-cross_scale (0.01, fixed)
m0 scan (doc note, `now` 2026-08-21): m0~=3 was the best separation point
found there, but cross_scale itself was never varied -- this sweep checks
whether that conclusion holds once cross_scale moves too, or whether it was
an artifact of the one fixed value.

R1 (electron double-ring, must NOT erode) / R2 (neutron, MUST decay) are
read the same way as probe_cross_v2_gates.py: final-step winding survival,
separation = surv_e / surv_n (want >1, i.e. electron survives more than
neutron decays).

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/sweep_cross_v2_barrier_2d.py
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/sweep_cross_v2_barrier_2d.py \
      --m0-list 2.49,3.0,5.0 --cs-list 1e-3,3e-3,1e-2,3e-2,1e-1 --n-steps 2000
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from run.substrate.adaptive.probe_cross_v2_gates import (
    build_lattice,
    run_arm,
    seed_ic,
)

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=15.0)
parser.add_argument("--sph-frac", type=float, default=0.2)
parser.add_argument("--B0", type=float, default=1.0)
parser.add_argument("--lam-target", type=float, default=1.0)
parser.add_argument("--u-max", type=float, default=1.0)
parser.add_argument("--m0-list", type=str, default="1.6e-1,3.0e-1,2.49,3.0,5.0")
parser.add_argument("--cs-list", type=str, default="1e-3,3e-3,1e-2,3e-2,1e-1")
parser.add_argument("--n-steps", type=int, default=2000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=8)
parser.add_argument("--eng-frac", type=float, default=1e-3)
parser.add_argument("--data-out", type=str, default="sweep_cross_v2_barrier_2d.npz")
parser.add_argument("--plot-out", type=str, default="sweep_cross_v2_barrier_2d.png")
parser.add_argument("--no-plot", action="store_true")


def main():
    args = parser.parse_args()
    m0_list = [float(v) for v in args.m0_list.split(",")]
    cs_list = [float(v) for v in args.cs_list.split(",")]
    scale = args.R * args.sph_frac

    print(f"R={args.R}  scale={scale:.1f}  m0_list={m0_list}  cs_list={cs_list}")
    print(
        f"grid size = {len(m0_list)} x {len(cs_list)} x 2 ICs = "
        f"{len(m0_list)*len(cs_list)*2} runs"
    )

    sep_grid = np.full((len(m0_list), len(cs_list)), np.nan)
    surv_e_grid = np.full((len(m0_list), len(cs_list)), np.nan)
    surv_n_grid = np.full((len(m0_list), len(cs_list)), np.nan)

    for i, m0 in enumerate(m0_list):
        for j, cs in enumerate(cs_list):
            args.cross_scale = cs
            res_e = run_arm("electron", "barrier", m0, args, scale)
            res_n = run_arm("neutron", "barrier", m0, args, scale)
            se = float(res_e["hist"][-1, 1])
            sn = float(res_n["hist"][-1, 1])
            sep = se / sn if sn > 0 else np.nan
            surv_e_grid[i, j] = se
            surv_n_grid[i, j] = sn
            sep_grid[i, j] = sep
            print(
                f"[m0={m0:.3e} cs={cs:.3e}] surv_e={se:.4f} surv_n={sn:.4f}"
                f" sep={sep:.3f}",
                flush=True,
            )

    print(f"\n{'m0 \\ cs':>10}" + "".join(f"{cs:>12.2e}" for cs in cs_list))
    for i, m0 in enumerate(m0_list):
        print(
            f"{m0:>10.3e}"
            + "".join(f"{sep_grid[i,j]:>12.3f}" for j in range(len(cs_list)))
        )

    best_i, best_j = np.unravel_index(np.nanargmax(sep_grid), sep_grid.shape)
    print(
        f"\nbest separation = {sep_grid[best_i, best_j]:.3f} at"
        f" m0={m0_list[best_i]:.3e}, cross_scale={cs_list[best_j]:.3e}"
        f"  (surv_e={surv_e_grid[best_i,best_j]:.4f}, surv_n={surv_n_grid[best_i,best_j]:.4f})"
    )

    if args.data_out:
        np.savez(
            args.data_out,
            m0_list=np.array(m0_list),
            cs_list=np.array(cs_list),
            sep_grid=sep_grid,
            surv_e_grid=surv_e_grid,
            surv_n_grid=surv_n_grid,
        )
        print(f"data saved -> {args.data_out}")

    if args.no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(
        sep_grid,
        aspect="auto",
        origin="lower",
        extent=[0, len(cs_list), 0, len(m0_list)],
        cmap="RdYlGn",
        vmin=0.5,
        vmax=max(2.0, float(np.nanmax(sep_grid))),
    )
    ax.set_xticks(np.arange(len(cs_list)) + 0.5)
    ax.set_xticklabels([f"{cs:.1e}" for cs in cs_list], rotation=45)
    ax.set_yticks(np.arange(len(m0_list)) + 0.5)
    ax.set_yticklabels([f"{m0:.2e}" for m0 in m0_list])
    ax.set_xlabel("cross_scale")
    ax.set_ylabel("m0")
    ax.set_title("barrier gate: separation surv_e/surv_n")
    fig.colorbar(im, ax=ax, label="separation")
    for i in range(len(m0_list)):
        for j in range(len(cs_list)):
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{sep_grid[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"plot saved -> {args.plot_out}")


if __name__ == "__main__":
    main()
