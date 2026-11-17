"""Plot the Hopf-linked ring pair geometry for both curl-full ICs
(build_curlfull_uniform_r180.py / build_curlfull_l1_wider_margin.py), t=0
only -- no stepping (that comes later, separately).

plot_telescope_xz.plot_cluster (u-based) would show nothing here, since u is
never seeded in this IC; all content lives on ~332 ring/plaquette leaves'
phi faces out of 30M+ total. Ring identity (ring_A_js/ring_B_js) isn't
stored in the .h5 checkpoint, so this re-derives it deterministically via
find_square_ring_on_leaves + find_linked_loop_anchored on the loaded lattice
(same build, same result).
"""

import sys

sys.path.insert(0, ".")
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equilibrium import lattice as _lat
from equilibrium.substrate.curl_full_spheromak_ic import (
    find_square_ring_on_leaves,
    find_linked_loop_anchored,
)

HOPS = 41


def raw_circ(cl, ring_js, ring_faces):
    phi = np.array(cl.leaf_phi)
    return float(sum(phi[ring_faces[idx], ring_js[idx]] for idx in range(len(ring_js))))


def H_of(cl):
    u = np.array(cl.leaf_u)
    phi = np.array(cl.leaf_phi)
    return float(np.sum(u**2) + 0.5 * np.sum(phi**2))


def plot_ring(cl, ring_A_js, ring_B_js, label, out_png, clip_r):
    pos = np.array(cl.leaf_pos)
    phi = np.array(cl.leaf_phi)
    max_phi = np.max(np.abs(phi), axis=0)
    nonzero = max_phi > 1e-12

    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pos[ring_A_js, 0],
        pos[ring_A_js, 1],
        pos[ring_A_js, 2],
        c="tab:red",
        s=10,
        label=f"Ring A ({len(ring_A_js)} tiles)",
    )
    ax.scatter(
        pos[ring_B_js, 0],
        pos[ring_B_js, 1],
        pos[ring_B_js, 2],
        c="tab:blue",
        s=40,
        label=f"Ring B ({len(ring_B_js)} tiles)",
    )
    other = nonzero.copy()
    other[ring_A_js] = False
    other[ring_B_js] = False
    if other.any():
        ax.scatter(
            pos[other, 0],
            pos[other, 1],
            pos[other, 2],
            c="gray",
            s=2,
            alpha=0.3,
            label="other nonzero-phi",
        )
    ax.set_xlim(-clip_r, clip_r)
    ax.set_ylim(-clip_r, clip_r)
    ax.set_zlim(-clip_r, clip_r)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{label}: Hopf-linked ring pair geometry (t=0)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[{label}] plotted -> {out_png}", flush=True)


def process(label, ic_path, out_dir, leaf_mask_level=None, clip_r=20.0):
    t0 = time.time()
    alat = _lat.AdaptiveLattice.load(ic_path)
    cl = alat[0]
    print(
        f"[{label}] loaded {ic_path}  n_leaves={cl.n_leaves}  t={time.time()-t0:.1f}s",
        flush=True,
    )

    leaf_mask = None
    if leaf_mask_level is not None:
        levels = np.array(cl.leaf_level)
        leaf_mask = levels == leaf_mask_level

    A_js, A_faces = find_square_ring_on_leaves(
        cl, axis="z", hops=HOPS, leaf_mask=leaf_mask
    )
    B_js, B_faces = find_linked_loop_anchored(cl, A_js, A_faces, anchor_idx=0)
    print(
        f"[{label}] re-derived ring A={len(A_js)} B={len(B_js)}  t={time.time()-t0:.1f}s",
        flush=True,
    )

    circA0 = raw_circ(cl, A_js, A_faces)
    circB0 = raw_circ(cl, B_js, B_faces)
    H0 = H_of(cl)
    print(
        f"[{label}] t=0: circA={circA0:.6f} circB={circB0:.6f} H={H0:.6e}", flush=True
    )

    os.makedirs(out_dir, exist_ok=True)
    plot_ring(cl, A_js, B_js, label, f"{out_dir}/{label}_ring_t0.png", clip_r)
    print(f"[{label}] done  total_t={time.time()-t0:.1f}s", flush=True)


def main():
    exp_dir = "exp/2026/08/16/18_56_curlfull_hopf_spheromak_ic"
    process(
        "curlfull_uniform_r180",
        "/bk/curlfull_uniform_r180_ic220.h5",
        f"{exp_dir}/plots",
        leaf_mask_level=None,
        clip_r=60.0,
    )
    process(
        "curlfull_l1_wider_margin",
        "/bk/curlfull_l1_wider_margin_ic220.h5",
        f"{exp_dir}/plots",
        leaf_mask_level=1,
        clip_r=20.0,
    )


if __name__ == "__main__":
    main()
