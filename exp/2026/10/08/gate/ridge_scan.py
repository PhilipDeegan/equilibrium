"""10-08 gate: the trigger quantity (pass count, lowest inter-pass ridge per toroidal angle) on saved coil states."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from equilibrium.substrate.walk import to_complex

sys.path.insert(0, str(Path(__file__).parent))
from gate import scan  # noqa: E402

for sp in map(Path, sys.argv[1:]):
    nm = sp.parent.name
    P = int(nm.split("_P")[1].split("_")[0])
    side = P / np.sqrt(3)
    pos = np.load(sp.parent / "pos.npy")
    ck = np.load(sp)
    rho = np.sum(np.abs(np.concatenate([to_complex(ck["L"]), to_complex(ck["R"])], 1))**2, 1)
    rcz = np.hypot(pos[:, 0], pos[:, 1])
    cc = (rho < 0.25) & (rcz > 15)
    wt = (1 - rho)[cc]
    k = 2 * np.pi / side
    rc = np.sum(wt * rcz[cc]) / wt.sum()
    zc = np.angle(np.sum(wt * np.exp(1j * k * pos[cc, 2]))) / k
    dz = (pos[cc, 2] - zc + side / 2) % side - side / 2
    dm = np.hypot(rcz[cc] - rc, dz)
    rad = np.sum(wt * dm) / wt.sum()
    tree = cKDTree(pos)
    for rr in (rad, 0.8 * rad, 1.2 * rad):
        s = scan(tree, rho, rc, zc, rr)
        n = s[:, 1].astype(int)
        print(f"{nm[:40]} t {int(ck['t'])}: rc {rc:.1f} rad(mean) {rad:.2f} circle {rr:.2f}: passes per angle "
              f"{np.bincount(n).tolist()}; ridge min {np.nanmin(s[:, 2]):.3f} median {np.nanmedian(s[:, 2]):.3f} "
              f"max {np.nanmax(s[:, 2]):.3f}", flush=True)
