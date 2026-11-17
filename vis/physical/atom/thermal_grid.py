import argparse
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

STENCILS = {
    4: [(-1, 0), (1, 0), (0, -1), (0, 1)],
    8: [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)],
}


def initial(size=3, hot=11.5, jitter=0.0, seed=42):
    grid = np.ones((size, size))
    if jitter:
        grid += np.random.RandomState(seed).uniform(-jitter, jitter, grid.shape)
    grid[size // 2, size // 2] += hot
    return grid


def step_downhill(grid, k=0.1, stencil=8):
    n = grid.shape[0]
    new = grid.copy()
    for i, j in np.ndindex(grid.shape):
        for di, dj in STENCILS[stencil]:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and grid[ni, nj] > grid[i, j]:
                delta = k * (grid[ni, nj] - grid[i, j])
                new[i, j] += delta
                new[ni, nj] -= delta
    return new


def step_symmetric(grid, k=0.1, stencil=4, emit_threshold=2.0, emit_loss=0.05):
    n = grid.shape[0]
    new = grid.copy()
    for i, j in np.ndindex(grid.shape):
        for di, dj in STENCILS[stencil]:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n:
                delta = k * (grid[ni, nj] - grid[i, j])
                new[i, j] += delta
                new[ni, nj] -= delta
        if new[i, j] > emit_threshold:
            new[i, j] -= emit_loss
    return new


def run(grid, step, steps, stop_at_equilibrium=True):
    history = [grid]
    c = grid.shape[0] // 2
    for s in range(steps):
        grid = step(grid)
        history.append(grid)
        if (
            stop_at_equilibrium
            and np.isclose(grid, grid[c, c], rtol=0, atol=1e-15).all()
        ):
            print("THERMAL EQUILIBRIUM ACHIEVED AT TIMESTEP", s)
            break
    return history


def run_downhill(steps=251, k=0.1, stencil=8, hot=11.5):
    return run(initial(hot=hot), partial(step_downhill, k=k, stencil=stencil), steps)


def run_emitting(steps=10, k=0.1, stencil=4, hot=1.5, jitter=0.1):
    step = partial(step_symmetric, k=k, stencil=stencil)
    return run(initial(hot=hot, jitter=jitter), step, steps, stop_at_equilibrium=False)


def figure(grid):
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(grid, annot=True, cmap="hot", cbar=True, ax=ax)
    ax.set_title(
        f"Final Temperature Distribution in {grid.shape[0]}x{grid.shape[1]} Grid"
    )
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--k", type=float, default=0.1)
    a = ap.parse_args(argv)
    if a.emit:
        history = run_emitting(a.steps or 10, a.k)
    else:
        history = run_downhill(a.steps or 251, a.k)
    print(history[-1])
    figure(history[-1])
    plt.show()


if __name__ == "__main__":
    main()
