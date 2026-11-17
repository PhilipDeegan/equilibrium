"""10-08 gate: the turns of an (1, N) coil are separated TOROIDALLY (each meridional plane is crossed once): density
along toroidal circles at poloidal angle phi (radius rc + rad cos phi, z = zc + rad sin phi): passes (minima) and the
ridges between adjacent turns, per phi. Inner side (phi = pi) = closest turns."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from equilibrium.substrate.walk import to_complex

sys.path.insert(0, str(Path(__file__).parent))
from gate import passes  # noqa: E402

NS = 2048
TH = np.linspace(0, 2 * np.pi, NS, endpoint=False)
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
    dm = np.hypot(rcz[cc] - rc, (pos[cc, 2] - zc + side / 2) % side - side / 2)
    rad = np.sum(wt * dm) / wt.sum()
    tree = cKDTree(pos)
    print(f"{nm[:42]} t {int(ck['t'])}: rc {rc:.1f} rad {rad:.2f}")
    for phi in np.radians([0, 90, 180, 270]):
        R = rc + rad * np.cos(phi)
        X = np.stack([R * np.cos(TH), R * np.sin(TH), zc + rad * np.sin(phi) + 0 * TH], 1)
        d, i = tree.query(X, k=8)
        w_ = 1 / np.maximum(d, 1e-6)**2
        prof = np.sum(w_ * rho[i], 1) / w_.sum(1)
        lo, rg = passes(prof, rmax=0.6)
        print(f"   phi {np.degrees(phi):3.0f} (circle r {R:.1f}, turn spacing {2 * np.pi * R / max(len(lo), 1):.1f}): passes {len(lo)}, "
              f"pass rho {np.round(prof[lo], 2).tolist()}, ridges {np.round(rg, 3).tolist()}")
