from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from equilibrium.log import getLogger
from equilibrium.utils import ensure_dir

logger = getLogger(__name__)


def savefig(path, fig=None, **kwargs):
    """Save a figure, creating its parent directory first if needed.

    Long sweeps that only call plt.savefig() at the very end (after the
    entire computation) can lose all that work to a plain FileNotFoundError
    if the output directory doesn't exist. Use this instead of
    fig.savefig()/plt.savefig() directly wherever the directory isn't
    already guaranteed to exist.
    """
    path = Path(path)
    ensure_dir(path)
    target = fig if fig is not None else plt
    target.savefig(path, **kwargs)
    logger.info("saved -> %s", path)
    return path


def plot_series_panels(
    results, keys, panels, out, figsize=None, cmap_name="viridis", label_fmt=str
):
    """Shared sweep-summary plot: one color-coded line per key (e.g. per
    cross_scale value) across a row of side-by-side panels sharing a common
    x-axis ("t" in each result dict).

    results  : dict key -> dict of 1D arrays, each containing at least "t"
    keys     : ordered sequence of results' keys to plot (one color each)
    panels   : sequence of dicts, one per subplot:
                 ylabel : str
                 title  : str
                 y      : callable(h) -> 1D array (same length as h["t"])
                 log    : bool, plot semilogy(|y|) instead of plot(y) (default False)
    out      : output path (see savefig)
    label_fmt: callable(key) -> str legend label (default str(key))
    """
    fig, axes = plt.subplots(1, len(panels), figsize=figsize or (6 * len(panels), 5.5))
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap(cmap_name)
    for i, key in enumerate(keys):
        h = results[key]
        color = cmap(i / max(len(keys) - 1, 1))
        label = label_fmt(key)
        for ax, panel in zip(axes, panels):
            y = panel["y"](h)
            if panel.get("log", False):
                ax.semilogy(h["t"], np.abs(y) + 1e-30, color=color, label=label)
            else:
                ax.plot(h["t"], y, color=color, label=label)
    for ax, panel in zip(axes, panels):
        ax.set_xlabel("time")
        ax.set_ylabel(panel["ylabel"])
        ax.set_title(panel["title"])
        ax.legend(fontsize=6)
    fig.tight_layout()
    savefig(out, fig=fig, dpi=130)
    return fig


class Fig:
    _plotters: dict = None

    def __call__(self, *args, **kwargs):
        return self.fig

    def __init__(self, cols=1, rows=1, scale=1, **kwargs):
        if not kwargs.get("FORCE_INTERACTIVE", False):
            try:
                import matplotlib

                get_ipython().__class__.__name__

                if matplotlib.get_backend() != "inline":
                    logger.warning(
                        "matplotlib not set to inline - multiple outputs may appear, "
                        'put "%%matplotlib inline" at the top of your ipynb file'
                    )

                plt.ioff()
            except NameError:
                ...

        self.cols = cols
        self.rows = rows
        self.scale = scale
        figsize = (self.cols * scale, self.rows * scale)
        self.fig = plt.figure(figsize=figsize)
        self.plots = []

    def sub_plot(self, data=None, title=None, ax_set: dict = None, *args, **kwargs):
        args = args if args else (self.rows, self.cols, len(self.plots) + 1)
        ax = self.fig.add_subplot(*args, **kwargs)
        self.plots.append(ax)
        if title:
            ax.set_title(title)
        self.plot_data(ax, data)
        if ax_set:
            ax.set(**ax_set)
        return ax

    def cla(self, axs):
        for ax in axs:
            ax.cla()

    def plot_datum(self, ax, datum):
        if datum is None:
            return
        if type(datum) in self.plotters:  # pylint: disable=E1135
            self.plotters[type(datum)](ax, datum)  # pylint: disable=E1136
        else:
            try_plot_unknown(ax, datum)

    def plot_data(self, ax, data):
        try:
            for e in data:
                self.plot_datum(ax, e)
        except TypeError:
            self.plot_datum(ax, data)

    def animation(self, *args, **kwargs):
        return Animation(self.fig, *args, **kwargs)

    @property
    def plotters(self) -> dict:
        """Delay settings to guard against weird import issues"""
        if not Fig3D._plotters:
            from equilibrium.math import geo

            Fig3D._plotters = {geo.line.Line: plot_line, geo.line.Vector: plot_vector}
        return Fig3D._plotters

    def show(self, *args, **kwargs):
        return plt.show()


class Fig3D(Fig):
    # _plotters: dict = None

    def __init__(self, cols, rows, scale=5):
        super().__init__(cols, rows, scale)

    def sub_plot_3d(self, data=None, elev=30, azim=50, roll=0, title=None):
        ax = self.sub_plot(data, projection="3d", aspect="equal", title=title)
        ax.invert_yaxis()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev, azim, roll, vertical_axis="y")
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.zaxis.set_major_locator(plt.MaxNLocator(4))
        return ax

    def axes_view(self, data=None):
        def sub_plot(elev=30, azim=50, roll=0):
            return self.sub_plot_3d(data, elev, azim, roll)

        return [sub_plot(90, 90), sub_plot(0, 40, 0), sub_plot(0), sub_plot(0, 5)]


def plot_line(ax, v):
    ax.plot(*v.xyz())


def plot_vector(ax, v):
    from equilibrium.math import geo

    plot_line(ax, geo.line.Line([0] * v.c.shape[0], v.c))


def try_plot_unknown(ax, v):
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    x_vals = v[:, 0:1]
    y_vals = v[:, 1:2]
    z_vals = v[:, 2:3]

    # Plot the values
    ax.scatter(x_vals, y_vals, z_vals, c="b", marker="o")
    return ax


class Animation:
    def __init__(self, fig, **kwargs):
        from matplotlib import animation

        if isinstance(fig, Fig):
            fig = fig.fig
        self.ani = animation.FuncAnimation(fig=fig, **kwargs)

    def show(self):
        try:
            get_ipython().__class__.__name__

            from IPython.display import HTML

            plt.ioff()

            return HTML(self.ani.to_html5_video())
        except (NameError, ImportError):
            ...

        return plt.show()

    def write_to_file(self, file, fps=10):
        from matplotlib import animation

        FFwriter = animation.FFMpegWriter(fps=fps)
        self.ani.save(file, writer=FFwriter)


class LiveFig:
    """
    Generic wrapper for matplotlib interactive-mode (ion) figures.

    Creates the figure once; subsequent updates call flush() which issues
    draw_idle + pause so the window stays responsive during a simulation loop.

    Usage::

        lf = LiveFig(n_rows=1, n_cols=2, figsize=(14, 6), every=500)
        # ... create artists on lf.ax(0,0), lf.ax(0,1) ...
        for step in range(n_steps):
            sim.step()
            if lf.should_update(step):
                im.set_data(new_array)
                lf.flush()
    """

    def __init__(self, n_rows=1, n_cols=1, figsize=None, every=1):
        plt.ion()
        if figsize is None:
            figsize = (n_cols * 6, n_rows * 5)
        self.fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        self._axes = axes
        self.every = every
        plt.show(block=False)
        plt.pause(0.05)

    @property
    def axes(self):
        return self._axes

    def ax(self, row=0, col=0):
        return self._axes[row, col]

    def should_update(self, step):
        return step % self.every == 0

    def flush(self, pause=0.05):
        self.fig.canvas.draw_idle()
        plt.pause(pause)


class style:
    @staticmethod
    def use(s):
        plt.style.use(s)
