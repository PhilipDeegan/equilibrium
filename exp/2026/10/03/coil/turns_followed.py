"""Turn count robust to an elongated torus: tube centre line C(th) = mean position of core cells within +-pi/4 in
toroidal angle (averages out the helix for N >= 4), then the pair-angle winding along C (the strand links C once per
poloidal turn). Also min density on C (C must stay off the cores)."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from equilibrium.substrate.walk import rest_state, to_complex

name, m = sys.argv[1], float(sys.argv[2])
d = Path(sys.argv[3] if sys.argv[3].startswith("/") else "/bk/dst/2026/10/03/coil") / name
if sys.argv[3].startswith("/"):
    sys.argv.pop(3)
pos = np.load(d / "pos.npy")
tree = cKDTree(pos)
x, y, z = pos.T
th = np.arctan2(y, x)
v, _ = rest_state(m)
inner = np.linalg.norm(pos, axis=1) < 0.75 * np.linalg.norm(pos, axis=1).max()
for tt in [int(a) for a in sys.argv[3:]]:
    s = np.load(d / f"state_t{tt:05d}.npz")
    rho = np.sum(s["L"] ** 2, 1) + np.sum(s["R"] ** 2, 1)
    amp = np.concatenate([to_complex(s["L"]), to_complex(s["R"])], 1) @ v.conj()
    c = (rho < 0.25) & inner
    P, tc = pos[c], th[c]
    ths = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    C = np.array([P[np.abs(np.angle(np.exp(1j * (tc - t0)))) < np.pi / 4].mean(0) for t0 in ths])
    seg = np.concatenate([np.linspace(C[i], C[(i + 1) % 720], 4, endpoint=False) for i in range(720)])
    idx = tree.query(seg)[1]
    a = amp[idx]
    w = np.sum(np.angle(a[np.r_[1:len(a), 0]] / a)) / (2 * np.pi)
    rr = np.hypot(C[:, 0], C[:, 1])
    ax2 = np.degrees(np.angle(np.sum(rr * np.exp(2j * ths)))) / 2
    print(f"t {tt}: turns along followed centre line {w:+.2f}; min rho on it {rho[idx].min():.2f}; centre-line radius "
          f"{rr.min():.1f}-{rr.max():.1f} (elongation {rr.max() / rr.min():.2f}), long axis {ax2:+.1f} deg", flush=True)
