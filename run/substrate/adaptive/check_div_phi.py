"""
Discrete divergence diagnostic and Helmholtz cleaning for a phi IC snapshot.

  div_i = sum_k phi_k(i)  [unweighted, 14 faces]

Non-zero div drives spurious u in pass 1.  Proper cleaning: solve the
graph-Laplacian Poisson equation  L psi = div,  then subtract the gradient:
  phi_k(i) -= (psi[nbr_k] - psi[i])
This projects phi onto its divergence-free component globally.

Usage:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/check_div_phi.py \
      shell_profile_R100_s10_cs_0.h5 --out div_phi.png

  # Helmholtz clean and save:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/check_div_phi.py \
      shell_profile_R100_s10_cs_0.h5 --correct --save corrected_ic.h5 --out div_phi_after.png
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat


def report_div(phi, label):
    div = phi.sum(axis=1)
    phi_rms = float(np.sqrt((phi**2).mean()))
    div_rms = float(np.sqrt((div**2).mean()))
    print(
        f"  [{label}]  div rms={div_rms:.4e}  phi rms={phi_rms:.4e}"
        f"  ratio={div_rms/phi_rms:.4f}",
        flush=True,
    )
    return div


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("--out", default="div_phi.png")
    parser.add_argument("--correct", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--thickness", type=float, default=2.0)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--cg-maxiter", type=int, default=500)
    args = parser.parse_args()

    print(f"Loading {args.snapshot} ...", flush=True)
    alat = _lat.AdaptiveLattice.load(args.snapshot)
    cl = alat[0]
    phi = np.array(cl.leaf_phi).T  # (N, 14)
    pos = np.array(cl.leaf_pos)  # (N, 3)
    nb = np.array(cl.leaf_nb)  # (N, 14) global neighbour indices (-1 = missing)
    N = phi.shape[0]
    print(f"  N={N}", flush=True)

    report_div(phi, "before")

    if args.correct:
        from scipy.sparse.linalg import cg

        print("  Building graph Laplacian ...", flush=True)
        from scipy.sparse import csr_matrix

        # Vectorised COO: off-diagonal -1 entries for valid neighbours
        i_idx = np.repeat(np.arange(N), 14)  # (N*14,)
        j_idx = nb.ravel()  # (N*14,)
        valid = (j_idx >= 0) & (j_idx < N)  # exclude boundary and non-leaf
        off_r = i_idx[valid]
        off_c = j_idx[valid]
        deg = (nb >= 0).sum(axis=1).astype(np.float64)  # (N,)
        rows = np.concatenate([np.arange(N), off_r])
        cols = np.concatenate([np.arange(N), off_c])
        vals = np.concatenate([deg, -np.ones(off_r.size)])
        L = csr_matrix((vals, (rows, cols)), shape=(N, N))
        print(f"  Laplacian built  nnz={L.nnz}", flush=True)

        div = phi.sum(axis=1)  # (N,) — rhs

        # Fix psi[0]=0 to remove null space (constant shift)
        L = L.tolil()
        L[0, :] = 0
        L[0, 0] = 1.0
        div[0] = 0.0
        L = L.tocsr()

        print("  Solving Poisson L psi = div (CG) ...", flush=True)
        psi, info = cg(L, div, rtol=args.cg_tol, maxiter=args.cg_maxiter)
        print(f"  CG info={info}  |psi| max={np.abs(psi).max():.4e}", flush=True)

        # Subtract gradient: phi_k(i) -= psi[nbr_k] - psi[i]
        for k in range(14):
            valid = (nb[:, k] >= 0) & (nb[:, k] < N)
            phi[valid, k] -= psi[nb[valid, k]] - psi[valid]
            # boundary/non-leaf faces: leave unchanged

        report_div(phi, "after")

        cl.set_leaf_phi(phi.T)
        if args.save:
            alat.save(args.save)
            print(f"  saved → {args.save}", flush=True)

    # Visualise divergence
    div_plot = phi.sum(axis=1)
    t = args.thickness
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Discrete div(phi) — {args.snapshot}", fontsize=12)
    slices = [
        ("XY (z≈0)", np.abs(pos[:, 2]) < t, 0, 1, "x", "y"),
        ("XZ (y≈0)", np.abs(pos[:, 1]) < t, 0, 2, "x", "z"),
        ("YZ (x≈0)", np.abs(pos[:, 0]) < t, 1, 2, "y", "z"),
    ]
    for ax, (title, mask, ix, iy, xl, yl) in zip(axes, slices):
        px, py = pos[mask, ix], pos[mask, iy]
        vm = np.abs(div_plot[mask]).max() or 1.0
        hb = ax.hexbin(
            px,
            py,
            C=div_plot[mask],
            reduce_C_function=np.mean,
            gridsize=70,
            cmap="seismic",
            vmin=-vm,
            vmax=vm,
        )
        plt.colorbar(hb, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"  saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
