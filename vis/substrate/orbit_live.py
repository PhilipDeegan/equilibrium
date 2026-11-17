"""
Live view for the DST orbit sandbox.

Three panels:
  1. xy (top-down): chi excess above Sun background — shows planet knot
  2. xz (side view): chi excess — shows out-of-plane knot motion
  3. Angular momentum: Lx, Ly, Lz vs step — precession / inter-axis transfer

Chi excess (planet chi above Sun 1/r background) is the durable structure.
The tide-locked centre-of-mass is its centroid, tracked by the cyan + marker.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from equilibrium.plot import LiveFig


class OrbitLive:
    """
    Parameters
    ----------
    cl            : initial cluster
    planet_centre : initial planet position (3,)
    R_central     : central knot radius — excluded from centroid
    chi_central   : central knot chi0 — used to compute Sun background
    orbit_normal  : unit normal of orbital plane
    galactic_axis : unit vector toward galactic centre
    every         : update every N steps
    z_thick       : half-thickness of xy / xz slices
    grid_n        : interpolation grid resolution
    """

    def __init__(
        self,
        cl,
        planet_centre,
        R_central,
        chi_central,
        *,
        orbit_normal=None,
        galactic_axis=None,
        every=200,
        z_thick=3.0,
        grid_n=200,
    ):
        self._every = every
        self._planet_centre = np.array(planet_centre)
        self._r_orbit = float(np.linalg.norm(planet_centre))
        self._R_central = R_central
        self._chi_central = chi_central
        self._last_centre = np.array(planet_centre)

        # L history
        self._L_steps: list = []
        self._L_vals: list = []

        pos = np.array(cl.leaf_pos)

        # --- xy slice (|z| < z_thick) ---
        m_xy = np.abs(pos[:, 2]) < z_thick
        self._mask_xy = m_xy
        px, py = pos[m_xy, 0], pos[m_xy, 1]
        xi = np.linspace(px.min(), px.max(), grid_n)
        yi = np.linspace(py.min(), py.max(), grid_n)
        self._Xi_xy, self._Yi_xy = np.meshgrid(xi, yi)
        self._xi_xy, self._yi_xy = xi, yi
        self._tri_xy = Delaunay(np.column_stack([px, py]))
        self._ext_xy = [xi.min(), xi.max(), yi.min(), yi.max()]
        # Sun background on the xy slice grid
        r_grid_xy = np.sqrt(self._Xi_xy**2 + self._Yi_xy**2)
        self._chi_sun_xy = chi_central * R_central / np.maximum(r_grid_xy, 1e-9)

        # --- xz slice (|y| < z_thick) ---
        m_xz = np.abs(pos[:, 1]) < z_thick
        self._mask_xz = m_xz
        px2, pz2 = pos[m_xz, 0], pos[m_xz, 2]
        xi2 = np.linspace(px2.min(), px2.max(), grid_n)
        zi2 = np.linspace(pz2.min(), pz2.max(), grid_n)
        self._Xi_xz, self._Zi_xz = np.meshgrid(xi2, zi2)
        self._xi_xz, self._zi_xz = xi2, zi2
        self._tri_xz = Delaunay(np.column_stack([px2, pz2]))
        self._ext_xz = [xi2.min(), xi2.max(), zi2.min(), zi2.max()]
        r_grid_xz = np.sqrt(self._Xi_xz**2 + self._Zi_xz**2)
        self._chi_sun_xz = chi_central * R_central / np.maximum(r_grid_xz, 1e-9)

        # Artist handles
        self._im_xy = self._im_xz = None
        self._dot_xy = self._dot_xz = None
        self._lines_L = None

        # Figure
        self._lf = LiveFig(n_rows=1, n_cols=3, figsize=(21, 6), every=every)
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

    def record_L(self, step, L):
        if np.all(np.isfinite(L)):
            self._L_steps.append(step)
            self._L_vals.append(np.array(L))

    # ------------------------------------------------------------------
    def _interp(self, tri, vals, Xi, Yi):
        return LinearNDInterpolator(tri, vals)(Xi, Yi)

    def _chi_excess(self, cl):
        """Chi excess above Sun background for xy and xz slice masks."""
        ell = np.array(cl.leaf_ell)
        chi = 1.0 - ell
        pos = np.array(cl.leaf_pos)
        r = np.linalg.norm(pos, axis=1)
        chi_bg = self._chi_central * self._R_central / np.maximum(r, 1e-9)
        exc = np.maximum(chi - chi_bg, 0.0)
        return exc

    def _centroid(self, cl):
        pos = np.array(cl.leaf_pos)
        exc = self._chi_excess(cl)
        r_sun = np.linalg.norm(pos, axis=1)
        dist = np.linalg.norm(pos - self._last_centre, axis=1)
        window = max(self._r_orbit * 0.6, 5.0)
        mask = (dist < window) & (r_sun > self._R_central * 2.0)
        w = exc[mask]
        wsum = float(w.sum())
        if wsum < 1e-20:
            return np.full(3, float("nan"))
        c = (w[:, None] * pos[mask]).sum(axis=0) / wsum
        self._last_centre = c
        return c

    def _render(self, cl, step):
        exc = self._chi_excess(cl)

        # xy interpolation → chi excess on grid, subtract Sun background
        Z_xy = self._interp(self._tri_xy, exc[self._mask_xy], self._Xi_xy, self._Yi_xy)
        exc_xy = np.where(np.isfinite(Z_xy), Z_xy, np.nan)

        # xz interpolation
        Z_xz = self._interp(self._tri_xz, exc[self._mask_xz], self._Xi_xz, self._Zi_xz)
        exc_xz = np.where(np.isfinite(Z_xz), Z_xz, np.nan)

        ax0, ax1, ax2 = self._lf.ax(0, 0), self._lf.ax(0, 1), self._lf.ax(0, 2)

        if self._im_xy is None:
            # --- first render: create all artists ---
            vmax_xy = (
                float(np.nanpercentile(exc_xy[np.isfinite(exc_xy)], 99.5))
                if np.any(np.isfinite(exc_xy))
                else 1e-3
            )
            vmax_xz = (
                float(np.nanpercentile(exc_xz[np.isfinite(exc_xz)], 99.5))
                if np.any(np.isfinite(exc_xz))
                else 1e-3
            )

            self._im_xy = ax0.imshow(
                exc_xy,
                extent=self._ext_xy,
                origin="lower",
                cmap="inferno",
                aspect="equal",
                interpolation="bilinear",
                vmin=0,
                vmax=vmax_xy,
            )
            cb0 = plt.colorbar(self._im_xy, ax=ax0, pad=0.02)
            cb0.set_label("chi excess (planet)", color="white")
            cb0.ax.yaxis.set_tick_params(color="white")
            plt.setp(cb0.ax.yaxis.get_ticklabels(), color="white")
            ax0.set_title("chi excess — xy (top-down)", color="white")
            ax0.set_xlabel("x [cells]", color="white")
            ax0.set_ylabel("y [cells]", color="white")
            ax0.plot(0, 0, "o", color="yellow", ms=5, zorder=4, label="Sun")

            self._im_xz = ax1.imshow(
                exc_xz,
                extent=self._ext_xz,
                origin="lower",
                cmap="inferno",
                aspect="equal",
                interpolation="bilinear",
                vmin=0,
                vmax=vmax_xz,
            )
            cb1 = plt.colorbar(self._im_xz, ax=ax1, pad=0.02)
            cb1.set_label("chi excess (planet)", color="white")
            cb1.ax.yaxis.set_tick_params(color="white")
            plt.setp(cb1.ax.yaxis.get_ticklabels(), color="white")
            ax1.set_title("chi excess — xz (side view)", color="white")
            ax1.set_xlabel("x [cells]", color="white")
            ax1.set_ylabel("z [cells]", color="white")
            ax1.plot(0, 0, "o", color="yellow", ms=5, zorder=4)

            # Initial planet position marker
            ax0.plot(
                self._planet_centre[0],
                self._planet_centre[1],
                "o",
                color="white",
                ms=3,
                alpha=0.4,
                zorder=3,
            )
            ax1.plot(
                self._planet_centre[0],
                self._planet_centre[2],
                "o",
                color="white",
                ms=3,
                alpha=0.4,
                zorder=3,
            )

            # Dynamic centroid marker (tide-locked CoM)
            (self._dot_xy,) = ax0.plot(
                [], [], "+", color="cyan", ms=14, mew=2, zorder=5
            )
            (self._dot_xz,) = ax1.plot(
                [], [], "+", color="cyan", ms=14, mew=2, zorder=5
            )

            # L time series
            ax2.set_title("Angular momentum (r_cm × v_cm)", color="white")
            ax2.set_xlabel("step", color="white")
            ax2.set_ylabel("L  [arb.]", color="white")
            ax2.axhline(0, color="#444", lw=0.6, ls=":")
            self._lines_L = []
            for col, lab in zip(["#ff6666", "#66ff99", "#6699ff"], ["Lx", "Ly", "Lz"]):
                (ln,) = ax2.plot([], [], color=col, lw=1.2, label=lab)
                self._lines_L.append(ln)
            ax2.legend(
                fontsize=9, facecolor="#111", labelcolor="white", loc="upper left"
            )
        else:
            self._im_xy.set_data(exc_xy)
            self._im_xz.set_data(exc_xz)

        # Update centroid dots
        c = self._centroid(cl)
        if np.all(np.isfinite(c)):
            self._dot_xy.set_data([c[0]], [c[1]])
            self._dot_xz.set_data([c[0]], [c[2]])
        else:
            self._dot_xy.set_data([], [])
            self._dot_xz.set_data([], [])

        # Update L time series
        if self._L_steps:
            steps = np.array(self._L_steps)
            Lvals = np.array(self._L_vals)
            for i, ln in enumerate(self._lines_L):
                ln.set_data(steps, Lvals[:, i])
            ax2.relim()
            ax2.autoscale_view()

        self._lf.fig.suptitle(f"step {step}", color="white", fontsize=10)
        self._lf.flush()
