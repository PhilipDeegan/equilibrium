import os
import sys

import numpy as np

os.environ.setdefault("PERTURB", "0")
sys.argv = ["x"]
sys.path.insert(0, "/home/p/git/phare/mkn/tests/functional")
src = open("/home/p/git/phare/mkn/tests/functional/emhd_spheromak.py").read().split("def config")[0]
src = src.replace("from pyphare import cpp", "").replace("import pyphare.pharein as ph", "").replace(
    "from pyphare.pharesee.run import Run", "").replace("from pyphare.simulator.simulator import Simulator, startMPI", "").replace("ph.NO_GUI()", "")
exec(src)
Bs = np.load("B_steps.npy")
g = np.arange(65) * 0.125
X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
bx, by, bz = spheromak(X, Y, Z)
Ba = np.stack([bx, by, bz], -1)
for s, B in enumerate(Bs):
    for c in range(3):
        print(f"step {s} comp {'xyz'[c]}: max|phare| {np.abs(B[..., c]).max():.3f} max|analytic| {np.abs(Ba[..., c]).max():.3f}"
              f"  rel L2 diff {np.linalg.norm(B[..., c] - Ba[..., c]) / np.linalg.norm(Ba[..., c]):.3f}")
    # try axis permutations in case of layout mismatch
    for perm in [(0, 1, 2), (2, 1, 0)]:
        Bp = B.transpose(*perm, 3)
        print("  perm", perm, "total rel diff", f"{np.linalg.norm(Bp - Ba) / np.linalg.norm(Ba):.3f}")
m = 32
print("analytic by at y=mid, row z=mid:", np.round(Ba[::8, m, m, 1], 3))
print("phare    by at y=mid, row z=mid:", np.round(Bs[0][::8, m, m, 1], 3))
print("analytic bz along x at y=z=mid:", np.round(Ba[::8, m, m, 2], 3))
print("phare    bz along x at y=z=mid:", np.round(Bs[0][::8, m, m, 2], 3))
