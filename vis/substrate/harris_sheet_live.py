"""
DST Harris sheet — live z-midplane view during simulation.

Shows 4 panels updated every N steps:
  phi_x  — x-projection of phi (Bx proxy, reconnecting field)
  phi_z  — z-projection of phi (out-of-plane field proxy)
  u      — pressure / density   (compresses at O-points)
  phi_y  — E_y proxy            (phi[+y faces] − phi[−y faces]; inflow driver)

Triangulates the z-midplane leaf mesh once; subsequent renders re-interpolate
values only — no re-triangulation, no artist recreation.
If AMR changes the leaf count between renders the triangulation is refreshed.

Usage (from run/substrate/adaptive/harris_sheet.py)::

    from vis.substrate.harris_sheet_live import HarrisLive
    live = HarrisLive(cl, y0, B0, every=args.report)
    # inside the step loop:
    live.update(cl, step)
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from equilibrium.plot import LiveFig

# BCC face direction components (hex 0-7, sq 8-13 = -x +x -y +y -z +z)
_HEX = np.array(
    [
        [-1, -1, -1],
        [-1, -1, +1],
        [-1, +1, -1],
        [-1, +1, +1],
        [+1, -1, -1],
        [+1, -1, +1],
        [+1, +1, -1],
        [+1, +1, +1],
    ],
    dtype=float,
)
_SQ = np.array(
    [[-1, 0, 0], [+1, 0, 0], [0, -1, 0], [0, +1, 0], [0, 0, -1], [0, 0, +1]],
    dtype=float,
)
_FACE = np.vstack([_HEX, _SQ])  # (14, 3)
FACE_DX = _FACE[:, 0]  # (14,)
FACE_DY = _FACE[:, 1]
FACE_DZ = _FACE[:, 2]

POS_Y = [2, 3, 6, 7, 11]
NEG_Y = [0, 1, 4, 5, 10]


class HarrisLive:
    """
    Live z-midplane slice view for the Harris sheet DST simulation.

    Parameters
    ----------
    cl     : AdaptiveCellCluster — initial cluster state
    y0     : float               — Harris null-line half-separation (dashed guides at ±y0)
    B0     : float               — Harris amplitude (sets colour floor)
    every  : int                 — update every N steps
    grid_n : int                 — interpolation grid points per axis
    """

    def __init__(self, cl, y0, B0, *, every=500, grid_n=300):
        self._y0 = y0
        self._B0 = B0
        self._every = every
        self._grid_n = grid_n

        self._ims = {}  # panel-key → imshow handle
        self._cbs = {}  # panel-key → colorbar handle
        self._n_leaves = 0

        # 2×2 panel layout, dark theme
        self._lf = LiveFig(n_rows=2, n_cols=2, figsize=(14, 8), every=every)
        fig = self._lf.fig
        fig.patch.set_facecolor("#0a0a1a")
        for ax in self._lf.axes.flat:
            ax.set_facecolor("#0a0a1a")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

        self._setup_geometry(cl)
        self._render(cl, step=0)

    # ------------------------------------------------------------------
    def _setup_geometry(self, cl):
        """Build z-midplane mask and Delaunay triangulation."""
        pos = np.array(cl.leaf_pos)
        z = pos[:, 2]
        zmid = (z.max() + z.min()) / 2.0
        ztol = (z.max() - z.min()) / 4.0
        self._mask = np.abs(z - zmid) < ztol
        px = pos[self._mask, 0]
        py = pos[self._mask, 1]
        self._px = px
        self._py = py
        self._n_leaves = len(pos)

        xi = np.linspace(px.min(), px.max(), self._grid_n)
        yi = np.linspace(py.min(), py.max(), self._grid_n)
        self._Xi, self._Yi = np.meshgrid(xi, yi)
        self._extent = [xi.min(), xi.max(), yi.min(), yi.max()]

        pts = np.column_stack([px, py])
        self._tri = Delaunay(pts)

    # ------------------------------------------------------------------
    def update(self, cl, step):
        if step % self._every != 0:
            return
        if len(cl.leaf_pos) != self._n_leaves:
            self._setup_geometry(cl)
            self._ims.clear()
            self._cbs.clear()
        self._render(cl, step)

    # ------------------------------------------------------------------
    def _interp(self, vals):
        return LinearNDInterpolator(self._tri, vals[self._mask])(self._Xi, self._Yi)

    # ------------------------------------------------------------------
    def _sym_lim(self, Z, floor):
        vm = float(np.nanpercentile(np.abs(np.nan_to_num(Z)), 99))
        return max(vm, floor)

    # ------------------------------------------------------------------
    def _render(self, cl, step):
        phi = np.asarray(cl.leaf_phi)  # (14, N)
        u = np.asarray(cl.leaf_u)  # (N,)

        # Project phi onto each Cartesian axis — proxy for the magnetic field components
        phi_x = (FACE_DX[:, None] * phi).sum(0)  # Bx proxy (reconnecting field)
        phi_z = (FACE_DZ[:, None] * phi).sum(0)  # Bz proxy (out-of-plane)
        phi_y = phi[POS_Y].mean(0) - phi[NEG_Y].mean(0)  # E_y proxy (inflow)

        Zbx = self._interp(phi_x)
        Zbz = self._interp(phi_z)
        Zpy = self._interp(phi_y)
        Zu = self._interp(u)

        bx_lim = self._sym_lim(Zbx, self._B0 * 0.01)
        bz_lim = self._sym_lim(Zbz, self._B0 * 0.001)
        py_lim = self._sym_lim(Zpy, self._B0 * 0.001)
        u_lo = float(np.nanpercentile(np.nan_to_num(Zu), 2))
        u_hi = float(np.nanpercentile(np.nan_to_num(Zu), 98))

        panels = [
            (
                0,
                0,
                "bx",
                "phi_x  — Bx proxy (reconnecting field)",
                Zbx,
                "RdBu_r",
                mcolors.TwoSlopeNorm(0, vmin=-bx_lim, vmax=bx_lim),
            ),
            (
                0,
                1,
                "bz",
                "phi_z  — Bz proxy (out-of-plane)",
                Zbz,
                "RdBu_r",
                mcolors.TwoSlopeNorm(0, vmin=-bz_lim, vmax=bz_lim),
            ),
            (
                1,
                0,
                "u",
                "u  (pressure / density)",
                Zu,
                "inferno",
                mcolors.Normalize(vmin=u_lo, vmax=u_hi),
            ),
            (
                1,
                1,
                "phy",
                "phi_y  — E_y proxy (inflow driver)",
                Zpy,
                "RdBu_r",
                mcolors.TwoSlopeNorm(0, vmin=-py_lim, vmax=py_lim),
            ),
        ]

        for r, c, key, title, Z, cmap, norm in panels:
            ax = self._lf.ax(r, c)
            if key not in self._ims:
                im = ax.imshow(
                    Z,
                    extent=self._extent,
                    origin="lower",
                    cmap=cmap,
                    norm=norm,
                    aspect="auto",
                    interpolation="bilinear",
                )
                cb = plt.colorbar(im, ax=ax, pad=0.02)
                cb.ax.yaxis.set_tick_params(color="white")
                plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
                ax.axhline(self._y0, color="white", lw=0.8, ls="--", alpha=0.5)
                ax.axhline(-self._y0, color="white", lw=0.8, ls="--", alpha=0.5)
                ax.set_title(title, color="white", fontsize=9)
                ax.set_xlabel("x", color="white", fontsize=8)
                ax.set_ylabel("y", color="white", fontsize=8)
                self._ims[key] = im
                self._cbs[key] = cb
            else:
                self._ims[key].set_data(Z)
                self._ims[key].set_norm(norm)

        self._lf.fig.suptitle(f"Harris sheet — step {step}", color="white", fontsize=10)
        self._lf.flush()
