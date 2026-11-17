"""
DST Harris sheet visualisation — layout matches PHARE harris_2d output.

Quantities shown (init left, final right):
  Bx  — field reversal component (phi_x proxy)
  By  — reconnecting component   (phi_y proxy)
  Jz  — current density proxy    (curl: ∂By/∂x − ∂Bx/∂y on interpolated grid)
  u   — scalar potential          (density / pressure analogue)

Usage:
  PYTHONPATH=$PWD python3 vis/substrate/harris_sheet.py harris_init.h5 harris_final.h5
  PYTHONPATH=$PWD python3 vis/substrate/harris_sheet.py harris_final.h5          # single file
  PYTHONPATH=$PWD python3 vis/substrate/harris_sheet.py harris_init.h5 harris_final.h5 --save out.png
"""

import argparse
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

# BCC face x/y components (faces 0-7 hex, 8-13 sq)
# hex_dir lex order: 0=(-,-,-) 1=(-,-,+) 2=(-,+,-) 3=(-,+,+)
#                    4=(+,-,-) 5=(+,-,+) 6=(+,+,-) 7=(+,+,+)
# sq: 8=-x 9=+x 10=-y 11=+y 12=-z 13=+z
HEX_DX = np.array([-1, -1, -1, -1, +1, +1, +1, +1], dtype=float)
HEX_DY = np.array([-1, -1, +1, +1, -1, -1, +1, +1], dtype=float)
# B_x proxy: (mean phi on +x faces) − (mean phi on −x faces)
POS_X = [4, 5, 6, 7, 9]
NEG_X = [0, 1, 2, 3, 8]
# B_y proxy: (mean phi on +y faces) − (mean phi on −y faces)
POS_Y = [2, 3, 6, 7, 11]
NEG_Y = [0, 1, 4, 5, 10]


# ---------------------------------------------------------------------------
def load(fname):
    import h5py

    with h5py.File(fname, "r") as f:
        pos = f["pos"][:]
        u = f["u"][:]
        phi = f["phi"][:]
        meta = dict(f.attrs)
    return pos, u, phi, meta


def midlayer(pos, *arrays):
    z = pos[:, 2]
    zmid = (z.max() + z.min()) / 2.0
    ztol = (z.max() - z.min()) / 6.0
    m = np.abs(z - zmid) < ztol
    out = []
    for a in arrays:
        out.append(a[m] if (a.ndim == 1 or a.shape[0] == len(m)) else a[:, m])
    return m, *out


def to_grid(x, y, v, nr):
    from scipy.interpolate import griddata

    xi = np.linspace(x.min(), x.max(), nr)
    yi = np.linspace(y.min(), y.max(), nr)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), v, (Xi, Yi), method="linear")
    return Xi, Yi, Zi


def compute_fields(pos, phi, grid_res):
    _, pos2, phi2 = midlayer(pos, pos, phi)
    x, y = pos2[:, 0], pos2[:, 1]
    bx = phi2[POS_X, :].mean(0) - phi2[NEG_X, :].mean(0)
    by = phi2[POS_Y, :].mean(0) - phi2[NEG_Y, :].mean(0)
    Xg, Yg, Bx = to_grid(x, y, bx, grid_res)
    _, _, By = to_grid(x, y, by, grid_res)
    # Jz = dBy/dx - dBx/dy  (finite differences on the regular grid)
    dx = Xg[0, 1] - Xg[0, 0]
    dy = Yg[1, 0] - Yg[0, 0]
    dBy_dx = np.gradient(np.nan_to_num(By), dx, axis=1)
    dBx_dy = np.gradient(np.nan_to_num(Bx), dy, axis=0)
    Jz = dBy_dx - dBx_dy
    return Xg, Yg, Bx, By, Jz


def compute_u_grid(pos, u, grid_res):
    _, pos2, u2 = midlayer(pos, pos, u)
    x, y = pos2[:, 0], pos2[:, 1]
    Xg, Yg, Ug = to_grid(x, y, u2, grid_res)
    return Xg, Yg, Ug


def _vlim(arr, B0):
    vm = float(np.nanpercentile(np.abs(np.nan_to_num(arr)), 98))
    return (-max(vm, B0 * 0.01), max(vm, B0 * 0.01))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--save", default=None)
    parser.add_argument("--grid-res", type=int, default=400)
    parser.add_argument(
        "--B0",
        type=float,
        default=None,
        help="Harris B0 for fixed colour limits (read from h5 attrs if not given)",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load all files and compute
    # -----------------------------------------------------------------------
    datasets = []
    B0_val = args.B0
    for fname in args.files:
        pos, u, phi, meta = load(fname)
        Xg, Yg, Bx, By, Jz = compute_fields(pos, phi, args.grid_res)
        _, _, Ug = compute_u_grid(pos, u, args.grid_res)
        label = f"step={meta.get('step','?')}"
        datasets.append(
            dict(Xg=Xg, Yg=Yg, Bx=Bx, By=By, Jz=Jz, Ug=Ug, label=label, meta=meta)
        )
        if B0_val is None:
            B0_val = float(meta.get("B0", 0.05))

    # Colour limits: use the LAST (most evolved) dataset's 98th-percentile amplitude,
    # shared across all datasets so init and final are on the same scale.
    B0 = B0_val
    CMAPS = {"Bx": "RdBu_r", "By": "RdBu_r", "Jz": "RdBu_r", "u": "inferno"}
    LABELS = {"Bx": "B_x (phi_x)", "By": "B_y (phi_y)", "Jz": "J_z", "u": "u"}
    GRID_KEY = {"Bx": "Bx", "By": "By", "Jz": "Jz", "u": "Ug"}

    for d in datasets:
        d["vlims"] = {qty: _vlim(d[GRID_KEY[qty]], B0) for qty in ["Bx", "By", "Jz"]}
        d["vlims"]["u"] = None

    ncols = len(datasets)
    QTYS = ["Bx", "By", "Jz", "u"]
    nrows = len(QTYS)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 3 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for col, d in enumerate(datasets):
        grid_data = {"Bx": d["Bx"], "By": d["By"], "Jz": d["Jz"], "u": d["Ug"]}
        y0_approx = (d["Yg"].max() - d["Yg"].min()) / 4.0
        for row, qty in enumerate(QTYS):
            ax = axes[row, col]
            V = grid_data[qty]
            if qty == "u":
                vlo = float(np.nanpercentile(np.nan_to_num(V), 2))
                vhi = float(np.nanpercentile(np.nan_to_num(V), 98))
                norm = mcolors.Normalize(vmin=vlo, vmax=vhi)
            else:
                lims = d["vlims"][qty]
                if lims is None:
                    lims = (-B0, B0)
                vlo, vhi = lims
                norm = mcolors.TwoSlopeNorm(0, vmin=vlo, vmax=vhi)
            im = ax.pcolormesh(
                d["Xg"],
                d["Yg"],
                V,
                cmap=CMAPS[qty],
                norm=norm,
                shading="auto",
                rasterized=True,
            )
            fig.colorbar(im, ax=ax, shrink=0.9, label=LABELS[qty])
            ax.axhline(y0_approx, color="w", lw=0.8, ls="--", alpha=0.6)
            ax.axhline(-y0_approx, color="w", lw=0.8, ls="--", alpha=0.6)
            ax.set_title(f"{LABELS[qty]}  —  {d['label']}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

    fig.suptitle("DST Harris Sheet", fontsize=14)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
