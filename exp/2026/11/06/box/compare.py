"""Pair vs solos along the x axis: d = kappa P, residual d_pair - (d_m + d_p), u^2 ratio, per window.
usage: compare.py PAIR SOLO_M SOLO_P CX   (names relative to /bk/dst; CX = half separation)"""
import sys

import numpy as np

B = "/bk/dst/2026/"
pair, m, p = (np.load(B + a) for a in sys.argv[1:4])
cx = float(sys.argv[4])
xs = pair["xs"]
print(f"kappa pair {float(pair['kappa']):.4e}  solo_m {float(m['kappa']):.4e}  solo_p {float(p['kappa']):.4e}")
inner = cx - 8 - 6
for s in ("100", "200", "300"):
    dp, dm, dq = pair[f"d_{s}"], m[f"d_{s}"], p[f"d_{s}"]
    up, um, uq = pair[f"u2_{s}"], m[f"u2_{s}"], p[f"u2_{s}"]
    res = dp - (dm + dq)
    ax = np.abs(xs)
    print(f"t {s}:")
    for name, k in [("between |x|<=4", ax <= 4), (f"gap 4-{inner:.0f}", (ax > 4) & (ax <= inner)),
                    ("rings", (ax > inner) & (ax <= cx + 8)), ("outside", ax > cx + 12)]:
        print(f"  {name:16s} d pair {dp[k].mean():.3f}  solo sum {(dm + dq)[k].mean():.3f}  resid {res[k].mean():+.3f}"
              f"  u2 pair/sum {up[k].sum() / max((um + uq)[k].sum(), 1e-300):.3g}")
