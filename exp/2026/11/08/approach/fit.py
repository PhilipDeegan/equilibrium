"""Approach vs separation from pair / solo logs: toward(t) = pair centroid shift toward the partner minus solo shift,
averaged over both bodies; fit toward = A t^2 (+ B t) over t <= TMAX. usage: fit.py [TMAX]"""
import re
import sys
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[2]
RUNS = {  # label: (half sep, pair, solo_m, solo_p)
    "18 n112": (18, "03/pair/pair.log", "03/pair/solo_m.log", "03/pair/solo_p.log"),
    "24 n128": (24, "04/pair/pair24.log", "04/pair/solo_m24.log", "04/pair/solo_p24.log"),
    "24 n144": (24, "06/box/pair24.log", "06/box/solo_m24.log", "06/box/solo_p24.log"),
    "30 n144": (30, "04/pair/pair30.log", "04/pair/solo_m30.log", "04/pair/solo_p30.log"),
}
L = re.compile(r"^\s*t\s+(\S+)\s.*?body0 share \S+ cen (\S+) (\S+) \S+(?:\s+body1 share \S+ cen (\S+) (\S+))?")


def read(f):
    out = {}
    for line in open(E / f):
        m = L.match(line)
        if m:
            out[float(m[1])] = [float(v) if v else np.nan for v in m.groups()[1:]]
    return out


tmax = float(sys.argv[1]) if len(sys.argv) > 1 else 300
res = {}
for lab, (cx, fp, fm, fq) in RUNS.items():
    P, M, Q = read(fp), read(fm), read(fq)
    ts = np.array(sorted(t for t in P if t in M and t in Q and t <= tmax))
    tw = np.array([(P[t][0] - M[t][0] - (P[t][2] - Q[t][0])) / 2 for t in ts])
    sd = np.array([(P[t][1] - M[t][1] - (P[t][3] - Q[t][1])) / 2 for t in ts])
    A2 = np.linalg.lstsq(ts[:, None] ** 2, tw, rcond=None)
    a = A2[0][0]
    resid = tw - a * ts ** 2
    da = np.sqrt((resid ** 2).sum() / (len(ts) - 1) / (ts ** 4).sum())
    c = np.polyfit(ts, tw, 2)
    res[lab] = (cx, a, da)
    print(f"{lab}: n {len(ts)}  toward ~ A t^2: A {a:.3e} +- {da:.1e}  (free quad: {c[0]:.3e} t^2 {c[1]:+.2e} t {c[2]:+.2f})"
          f"  end avg(last 5) toward {tw[-5:].mean():+.2f} side {sd[-5:].mean():+.2f}  wobble rms {resid.std():.2f}")
for a_, b_ in (("24 n144", "30 n144"), ("24 n128", "30 n144"), ("18 n112", "24 n128"), ("18 n112", "30 n144")):
    (x1, A1, d1), (x2, A2_, d2) = res[a_], res[b_]
    p = np.log(A1 / A2_) / np.log(x2 / x1)
    dp = np.hypot(d1 / A1, d2 / A2_) / np.log(x2 / x1)
    print(f"{a_} vs {b_}: A ratio {A1 / A2_:.2f}  -> A ~ d^-{p:.1f} +- {dp:.1f}")
