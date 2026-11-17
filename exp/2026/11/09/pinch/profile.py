"""Radial u profile of pinch.py end states: shell means (width 0.5), core value, edge 90 % -> 10 % width, u / um.
usage: profile.py END.npz [END2 ...] --a=6"""
import sys
import numpy as np
opt = dict(a[2:].split("=") for a in sys.argv[1:] if a.startswith("--"))
a = float(opt.get("a", 6))
for f in [x for x in sys.argv[1:] if not x.startswith("--")]:
    z = np.load(f)
    u, r = z["u"], z["r"]
    um = float(f.split("_um")[1].rstrip(".npz")) if "_um" in f else np.inf
    edges = np.arange(0, 3 * a, 0.5)
    b = np.digitize(r, edges) - 1
    prof = np.array([u[b == i].mean() for i in range(len(edges) - 1)])
    rc = 0.5 * (edges[1:] + edges[:-1])
    core = prof[0]
    i90 = np.argmax(prof < 0.9 * core) if core > 0 else 0
    i10 = np.argmax(prof < 0.1 * core) if core > 0 else 0
    print(f"{f.split('/')[-1]:24s} core u {core:+.4f} (u/um {core / um:.2f})  r90 {rc[i90]:.2f}  r10 {rc[i10]:.2f}  "
          f"width {rc[i10] - rc[i90]:.2f}  min u {prof.min():+.4f} at r {rc[prof.argmin()]:.1f}")
    print("   r: " + " ".join(f"{x:5.1f}" for x in rc[::2]))
    print("   u: " + " ".join(f"{x:+.3f}"[:6] for x in prof[::2]))
