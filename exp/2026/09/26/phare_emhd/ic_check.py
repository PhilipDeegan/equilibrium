"""Load emhd_spheromak_p0 EM_B.vtkhdf, compare step 0 (and the last step) with the analytic IC."""
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "/home/p/git/phare/mkn/tests/functional")
d = Path("/home/p/git/phare/mkn/phare_outputs/test/emhd_spheromak_p0")
f = h5py.File(d / "EM_B.vtkhdf")
L0 = f["VTKHDF/Level0"]
S = f["VTKHDF/Steps"]
boxes = L0["AMRBox"][:]
nbox = S["Level0/NumberOfAMRBox"][:]
boff = S["Level0/AMRBoxOffset"][:]
poff = S["Level0/PointDataOffset/data"][:]
print("times", S["Values"][:], "boxes/step", nbox, "\n", boxes)
data = L0["PointData/data"]
N = 65
out = []
for s in range(len(nbox)):
    B = np.full((N, N, N, 3), np.nan)
    p = poff[s]
    for b in boxes[boff[s]:boff[s] + nbox[s]]:
        lo, hi = b[0::2], b[1::2]
        sh = hi - lo + 2
        n = int(np.prod(sh))
        blk = data[p:p + n].reshape(sh[2], sh[1], sh[0], 3).transpose(2, 1, 0, 3)
        B[lo[0]:lo[0] + sh[0], lo[1]:lo[1] + sh[1], lo[2]:lo[2] + sh[2]] = blk
        p += n
    out.append(B)
    print("step", s, "points used", p - poff[s], "nan", int(np.isnan(B).any(-1).sum()))
np.save("B_steps.npy", np.array(out))
