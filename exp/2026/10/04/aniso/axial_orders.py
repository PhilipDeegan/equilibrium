"""ordering variants of the axial split-step tick (axial_mix.py): which cycle of C ticks restores 90-deg symmetry."""
from itertools import permutations

import numpy as np

from axial_mix import B, Sc, measure

P6 = list(permutations(range(3)))


def seq(ticks):
    def f(k):
        U = np.eye(4)
        for g in ticks:
            U = g(k) @ U
        return U
    return f, len(ticks)


def c(order, par=False):
    if par:
        return lambda k: B @ Sc(-np.asarray(k), order) @ B
    return lambda k: Sc(k, order)


V = {
    "xyz,zyx": seq([c((0, 1, 2)), c((2, 1, 0))]),
    "xyz,par zyx": seq([c((0, 1, 2)), c((2, 1, 0), True)]),
    "xyz,par xyz": seq([c((0, 1, 2)), c((0, 1, 2), True)]),
    "6 perms": seq([c(o) for o in P6]),
    "6 perms + par": seq([g for o in P6 for g in (c(o), c(o[::-1], True))]),
    "6 perms, par same": seq([g for o in P6 for g in (c(o), c(o, True))]),
}
for name, (f, t) in V.items():
    for kk in [0.1, 0.3]:
        kin, sp, two = measure(f, t, kk)
        print(f"  {name:18s} |k| {kk}: kinetic/k^2 {kin / kk**2:.4f} rel spread {sp:.2e} 90-deg {two:.1e}", flush=True)
