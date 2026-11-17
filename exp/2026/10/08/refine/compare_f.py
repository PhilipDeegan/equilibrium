"""10-08 refine: N 4 at f 1 (07/slosh, xi 3.9, 20k / 40k) vs f sqrt2 (this dir, xi 5.515, 40k) in xi units and scaled
time (t / f^2): probe-line rad / xi, rc / xi, windings, k4 helix modulation / xi; Et_in / f (energy ~ strand length)."""
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).parent
s07 = here / "../../07/slosh"
sys.path.insert(0, str(s07))
from slosh_io import load  # noqa: E402

F = np.sqrt(2)


def probes(files):
    out = {}
    for f in files:
        for ln in open(f):
            if ln.startswith("P t"):
                t = int(ln.split()[2][:-1])
                rad = float(ln.split("rad ")[1].split()[0])
                rc = float(ln.split("c (")[1].split(",")[0])
                wind = ln.split("wind ")[1].split()[0]
                k4 = float(ln.split(" k4 z ")[1].split("@")[0]) if " k4 z " in ln else np.nan
                out[t] = (rad, rc, wind, k4)
    return out


for R, Rf in ((30, "42.43"), (40, "56.57")):
    p1 = probes(sorted(s07.glob(f"quant_gp_N4_a10_r{R}_P220_T20k.log")) + sorted(s07.glob(f"quant_gp_N4_a10_r{R}_P220_from20000.log")))
    p2 = probes([here / f"N4_r{Rf}_f1.41.log"])
    e1 = load(s07 / f"slosh_prof_pp_springbox_N4_S1_q1_a10_r{R}_xi3.9_m0.8_P220_z0_wxi_rc_par_gpL.log")
    e2 = load(here / f"slosh_prof_pp_springbox_N4_S1_q1_a14.14_r{Rf}_xi5.515_m0.8_P312_z0_wxi_rc_par_f141.log")
    print(f"\nr0 {R} (f 1, xi 3.9)  vs  r0 {Rf} (f sqrt2, xi 5.515); t = f-1 ticks (f sqrt2 at 2 t)")
    print("  t(k)   rad/xi  f1 | f141    rc/xi f1 | f141   k4/xi f1 | f141   wind f1 | f141   Et/f (2k bin) f1 | f141")
    T1 = max(p1)
    for tk in range(0, 21, 2):
        t1, t2 = tk * 1000, tk * 2000
        if t1 > T1 or t2 > max(p2):
            continue
        a = p1[min(p1, key=lambda v: abs(v - t1))]
        b = p2[min(p2, key=lambda v: abs(v - t2))]
        m1 = (e1["tE"] >= t1) & (e1["tE"] < t1 + 2000)
        m2 = (e2["tE"] >= t2) & (e2["tE"] < t2 + 4000)
        E1 = e1["Et"][m1].mean() if m1.any() else np.nan
        E2 = e2["Et"][m2].mean() / F if m2.any() else np.nan
        print(f"  {tk:3d}    {a[0] / 3.9:5.2f} | {b[0] / 5.515:5.2f}     {a[1] / 3.9:5.2f} | {b[1] / 5.515:5.2f}"
              f"    {a[3] / 3.9:5.2f} | {b[3] / 5.515:5.2f}    {a[2]:>4} | {b[2]:<4}   {E1:7.1f} | {E2:7.1f}")
