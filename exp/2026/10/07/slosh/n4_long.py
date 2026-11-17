"""N 4 a 10 P 220 to 20k at r0 30 / 40 (scan_N4_long.sh; r0 30 resumed to 40k, scan_N4_resume40k.sh): Et_in (within a + 28) in 2k bins + fitted slopes, D_in, rc;
the coil's rad / winding from the run log's probe lines."""
from pathlib import Path

import numpy as np

from slosh_io import load

here = Path(__file__).parent
for R in (30, 40):
    L = load(here / f"slosh_prof_pp_springbox_N4_S1_q1_a10_r{R}_xi3.9_m0.8_P220_z0_wxi_rc_par_gpL.log")
    tE, Et, t, D, rc = L["tE"], L["Et"], L["t"], L["D"], L["rc"]
    P = {int(ln.split()[2][:-1]): ln for f in sorted(here.glob(f"quant_gp_N4_a10_r{R}_P220_*.log")) for ln in open(f) if ln.startswith("P t")}
    T = int(t[-1])
    print(f"r0 {R}\n  t(k)   Et mean  sd     D_in    rc    rad(start of bin)  wind")
    for k in range(0, T // 1000, 2):
        m, n = (tE >= k * 1000) & (tE < (k + 2) * 1000), (t >= k * 1000) & (t < (k + 2) * 1000)
        p = P[min(P, key=lambda v: abs(v - k * 1000))]
        print(f"  {k:2d}-{k + 2:<2d} {Et[m].mean():7.1f} {Et[m].std():5.1f} {D[n].mean():7.0f} {rc[n].mean():5.1f}"
              f"   {p.split('rad ')[1].split()[0]:>6}   {p.split('wind ')[1].split()[0]}")
    for a, b in [(2000, 10000), (10000, 20000), (20000, 30000), (30000, 40000)][: T // 10000]:
        m = (tE >= a) & (tE <= b)
        print(f"  Et slope {a // 1000}k-{b // 1000}k: {np.polyfit(tE[m], Et[m], 1)[0] * 1000:+.2f} / 1k")
