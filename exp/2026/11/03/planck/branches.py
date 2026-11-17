"""A1 step 1 precondition: how many of step()'s degrees of freedom propagate? One-step map of the periodic BCC box
(n^3 unit cells, 2 u + 14 face values each) built column by column; eigenvalues 1 = static, else oscillating."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "10/25/defect"))
from bcc import Box  # noqa: E402

n, dt = int(sys.argv[1]) if len(sys.argv) > 1 else 4, 0.05
b = Box((n, n, n))
fl = b.fields()
sizes = [f.size for f in fl]
dim = sum(sizes)


def setv(v):
    o = 0
    for f, s in zip(b.fields(), sizes):
        f.reshape(-1)[:] = v[o:o + s]
        o += s


def getv():
    return np.concatenate([f.reshape(-1) for f in b.fields()])


M = np.empty((dim, dim))
for j in range(dim):
    e = np.zeros(dim)
    e[j] = 1
    setv(e)
    b.step(dt)
    M[:, j] = getv()
ev = np.linalg.eigvals(M)
static = np.sum(np.abs(ev - 1) < 1e-9)
print(f"n {n}: dof {dim} (per unit cell 16), static {static} ({static / n**3:.3f} per cell), "
      f"oscillating {dim - static} ({(dim - static) / n**3:.3f} per cell), |ev| max-1 {np.abs(ev).max() - 1:.1e}")
