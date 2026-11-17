"""Fitted own z drift (least squares on true z = centre z + frame shift, the 500-tick 't N:' lines of the run log) with
slope standard error, for lone coils (box_gp P 180 / 220 / 340) and the CsCl pair (primary of q -1 / q +1, P 300);
windows 1k-5k (common with the stopped q +1) and 1k-end. Replaces the two-point drifts of the 10-07 pair table."""
import re
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).parent
logs = sys.argv[1:] or ["box_gp_N2_a10_r30_P180.log", "box_gp_N2_a10_r30_P220.log", "box_gp_N2_a10_r30_P340.log",
                        "box_rp_N2_a10_r30_P220.log", "pair_q1_N2_a10_r30_P300.log", "pair_q1_N2_a10_r30_P240_T8k.log", "pair_q1_N2_a10_r30_P300_T8k.log", "pair_q1_N2_a10_r30_P360_T8k.log", "pair_q-1_N2_a10_r30_P300.log"]
pat = re.compile(r"^\s+t (\d+): .*centre \(rho, z\) \(([-+\d.]+), ([-+\d.]+)\).*frame shift ([-+\d.]+)")


def fit(t, z):
    A = np.stack([t, np.ones_like(t)], 1)
    c, res, *_ = np.linalg.lstsq(A, z, rcond=None)
    r = z - A @ c
    se = np.sqrt(r @ r / max(len(t) - 2, 1) / np.sum((t - t.mean())**2))
    return c[0], se, r.std()


print(f"{'log':34s} {'1k-5k v':>9s} {'+-':>7s} {'1k-end v':>9s} {'+-':>7s} {'res sd':>6s}  T")
for lg in logs:
    rows = [(float(m[1]), float(m[3]) + float(m[4])) for m in map(pat.match, open(here / lg)) if m]
    t, z = np.array(rows).T
    a = (t >= 1000) & (t <= 5000)
    b = t >= 1000
    v1, s1, _ = fit(t[a], z[a])
    v2, s2, rs = fit(t[b], z[b])
    print(f"{lg:34s} {v1:+.5f} {s1:.5f} {v2:+.5f} {s2:.5f} {rs:6.2f}  {t[-1]:.0f}")
