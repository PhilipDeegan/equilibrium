"""
Live equatorial slice view for the solar system DST simulation.

Triangulates the leaf mesh once on init (positions are static for a fixed
lattice); subsequent renders only re-interpolate values and update imshow
artists in-place — no re-triangulation, no artist recreation.

Usage (from run/substrate/solar_system.py)::

    from vis.substrate.solar_system_live import SolarSystemLive
    live = SolarSystemLive(cl, body_params, every=500)
    # inside step loop:
    live.update(cl, step)
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from equilibrium.plot import LiveFig


class SolarSystemLive:
    """
    Live equatorial (z≈0) slice view showing chi terrain and u field.

    Parameters
    ----------
    cl          : AdaptiveCellCluster  — initial cluster state
    body_params : dict                 — body layout from solar_system.py
    every       : int                  — update every N steps
    z_thick     : float                — half-thickness of equatorial slice
    grid_n      : int                  — interpolation grid resolution per axis
    """

    def __init__(self, cl, body_params, *, every=500, z_thick=3.0, grid_n=300):
        self._every = every
        self._body_params = body_params

        # --- equatorial slice mask (static — lattice doesn't change during run) ---
        pos = np.array(cl.leaf_pos)
        mask = np.abs(pos[:, 2]) < z_thick
        self._mask = mask
        px = pos[mask, 0]
        py = pos[mask, 1]
        self._px = px
        self._py = py

        # Regular interpolation grid
        xi = np.linspace(px.min(), px.max(), grid_n)
        yi = np.linspace(py.min(), py.max(), grid_n)
        self._Xi, self._Yi = np.meshgrid(xi, yi)
        self._xi, self._yi = xi, yi

        # Triangulate once — reused for every subsequent render
        points = np.column_stack([px, py])
        self._tri = Delaunay(points)

        # Artist handles (None until first render)
        self._im_chi = None
        self._im_u = None
        self._dot_centroids = {}  # name → Line2D marker

        # Build figure
        self._lf = LiveFig(n_rows=1, n_cols=2, figsize=(16, 7), every=every)
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

        self._render(cl, step=0)

    # ------------------------------------------------------------------
    def update(self, cl, step):
        if step % self._every != 0:
            return
        self._render(cl, step)

    # ------------------------------------------------------------------
    def _interpolate(self, values):
        """Re-interpolate a new value array using the pre-built triangulation."""
        interp = LinearNDInterpolator(self._tri, values)
        Z = interp(self._Xi, self._Yi)
        return Z

    def _planet_centroids(self, cl):
        """Chi-weighted centroid of each planet's neighbourhood."""
        pos = np.array(cl.leaf_pos)
        u = np.array(cl.leaf_u)
        r_sun = self._body_params["Sun"]["radius_cells"] * 3.0
        result = {}
        for name, p in self._body_params.items():
            if name == "Sun":
                continue
            centre = p["centre"]
            window = max(p["orbit_cells"] * 0.5, p["radius_cells"] * 10, 5.0)
            dist = np.linalg.norm(pos - centre, axis=1)
            r_from_sun = np.linalg.norm(pos, axis=1)
            m = (dist < window) & (r_from_sun > r_sun)
            w = np.maximum(u[m], 0.0)
            wsum = float(w.sum())
            if wsum > 1e-20:
                result[name] = (pos[m] * w[:, None]).sum(axis=0) / wsum
            else:
                result[name] = np.full(3, float("nan"))
        return result

    def _render(self, cl, step):
        ell = np.array(cl.leaf_ell)[self._mask]
        u = np.array(cl.leaf_u)[self._mask]
        chi = 1.0 - ell

        Zchi = self._interpolate(chi)
        Zu = self._interpolate(u)

        chi_plot = np.where(np.isfinite(Zchi), np.log1p(Zchi * 1000), np.nan)
        u_plot = np.where(np.isfinite(Zu), Zu, np.nan)

        extent = [self._xi.min(), self._xi.max(), self._yi.min(), self._yi.max()]

        ax0 = self._lf.ax(0, 0)
        ax1 = self._lf.ax(0, 1)

        if self._im_chi is None:
            # First render — create all artists
            self._chi_vmin = float(np.nanmin(chi_plot))
            self._chi_vmax = float(np.nanmax(chi_plot))
            valid_u = u_plot[np.isfinite(u_plot) & (u_plot > 0)]
            self._u_vmax = (
                float(np.percentile(valid_u, 99.5)) if len(valid_u) > 0 else 1.0
            )

            self._im_chi = ax0.imshow(
                chi_plot,
                extent=extent,
                origin="lower",
                cmap="inferno",
                aspect="equal",
                interpolation="bilinear",
                vmin=self._chi_vmin,
                vmax=self._chi_vmax,
            )
            cb0 = plt.colorbar(self._im_chi, ax=ax0, pad=0.02)
            cb0.set_label("log(1 + chi×1000)", color="white")
            cb0.ax.yaxis.set_tick_params(color="white")
            plt.setp(cb0.ax.yaxis.get_ticklabels(), color="white")
            ax0.set_title("chi terrain", color="white")
            ax0.set_xlabel("x [cells]", color="white")
            ax0.set_ylabel("y [cells]", color="white")

            self._im_u = ax1.imshow(
                u_plot,
                extent=extent,
                origin="lower",
                cmap="plasma",
                aspect="equal",
                interpolation="bilinear",
                norm=mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=self._u_vmax),
            )
            cb1 = plt.colorbar(self._im_u, ax=ax1, pad=0.02)
            cb1.set_label("u (pressure)", color="white")
            cb1.ax.yaxis.set_tick_params(color="white")
            plt.setp(cb1.ax.yaxis.get_ticklabels(), color="white")
            ax1.set_title("u field — planet blobs", color="white")
            ax1.set_xlabel("x [cells]", color="white")

            # Static body markers at initial positions
            for name, p in self._body_params.items():
                cx, cy = p["centre"][0], p["centre"][1]
                for ax in (ax0, ax1):
                    ax.plot(cx, cy, "o", color="white", ms=3, alpha=0.4, zorder=3)
                    ax.annotate(
                        name[:3],
                        (cx, cy),
                        color="white",
                        fontsize=6,
                        xytext=(4, 4),
                        textcoords="offset points",
                        zorder=3,
                    )

            # Dynamic centroid markers (will be updated each render)
            colors = plt.cm.tab10(np.linspace(0, 1, len(self._body_params)))
            for (name, _), col in zip(self._body_params.items(), colors):
                if name == "Sun":
                    continue
                (dot,) = ax1.plot([], [], "+", color=col, ms=10, mew=1.5, zorder=4)
                self._dot_centroids[name] = dot
        else:
            # In-place update
            self._im_chi.set_data(chi_plot)
            self._im_u.set_data(u_plot)

        # Update centroid dots
        centroids = self._planet_centroids(cl)
        for name, dot in self._dot_centroids.items():
            c = centroids.get(name, np.full(3, float("nan")))
            if np.all(np.isfinite(c)):
                dot.set_data([c[0]], [c[1]])
            else:
                dot.set_data([], [])

        self._lf.fig.suptitle(f"step {step}", color="white", fontsize=10)
        self._lf.flush()
