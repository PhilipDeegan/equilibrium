"""Pair vs solos: depletion d = kappa P along the x axis through both centres; superposition residual
d_pair - (d_m + d_p) with each run's own kappa (solo kappa used for the sum; pair either normalised or --kfix)."""
import sys

import numpy as np

D = "/bk/dst/2026/11/03/pair/"
pair = np.load(D + (sys.argv[1] if len(sys.argv) > 1 else "pair.npz"))
m, p = np.load(D + "solo_m.npz"), np.load(D + "solo_p.npz")
xs = pair["xs"]
print(f"kappa pair {float(pair['kappa']):.4e}  solo_m {float(m['kappa']):.4e}  solo_p {float(p['kappa']):.4e}")
for s in ("100", "200", "300"):
    dp, dm, dq = pair[f"d_{s}"], m[f"d_{s}"], p[f"d_{s}"]
    up, um, uq = pair[f"u2_{s}"], m[f"u2_{s}"], p[f"u2_{s}"]
    res = dp - (dm + dq)
    sel = lambda a, b: (xs >= a) & (xs <= b)  # noqa: E731
    print(f"t {s}:")
    for name, a, b in [("between |x|<=4", -4, 4), ("gap 4-10", 4, 10), ("ring +x 10-26", 10, 26), ("outside x 30-44", 30, 44)]:
        k = sel(a, b) | sel(-b, -a) if a >= 0 and name != "between |x|<=4" else sel(a, b)
        print(f"  {name:18s} d pair {dp[k].mean():.3f}  d solo sum {(dm + dq)[k].mean():.3f}  resid {res[k].mean():+.3f}"
              f"  (max-min clip 0.9)   u2 pair/solo-sum {up[k].sum() / max((um + uq)[k].sum(), 1e-300):.3f}")
    print("  d along x (every 4 cells):  x " + " ".join(f"{x:5.0f}" for x in xs[::4]))
    print("                          pair " + " ".join(f"{v:5.2f}" for v in dp[::4]))
    print("                          sum  " + " ".join(f"{v:5.2f}" for v in (dm + dq)[::4]))
