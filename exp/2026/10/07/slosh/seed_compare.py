"""GP seed (scan_gp.sh, _gp) vs relaxed-profile seed (scan_relaxed_seed.sh, _rp), N 2 r0 30 a 10 P 220 8k: initial
excess deficit, D_in swing (first minimum, amplitude), Et_in level / sd, residual oscillation and rc breathing periods
(FFT after 1k, detrended). Ringing ~ seed mismatch; the ~2k / ~5k periods should not change if coil-local."""
from pathlib import Path

import numpy as np

from slosh_io import load

here = Path(__file__).parent


def top(t, v, k=3):
    v = v - np.polyval(np.polyfit(t, v, 1), t)
    f = np.fft.rfftfreq(len(v), t[1] - t[0])[1:]
    a = np.abs(np.fft.rfft(v))[1:]
    j = np.argsort(a)[::-1][:k]
    return ", ".join(f"{1 / f[i]:.0f} ({a[i] / len(v) * 2:.3g})" for i in j)


print("seed  D_in(0)  D_in mean>1k  first min (t, D)  swing max-min 0-2k  >2k sd   Et_in mean+-sd >1k  Et(0)   D periods (amp)        rc periods")
for tag in ("gp", "rp"):
    L = load(here / f"slosh_prof_pp_springbox_N2_S1_q1_a10_r30_xi3.9_m0.8_P220_z0_wxi_rc_par_{tag}.log")
    t, D, rc, tE, Et = L["t"], L["D"], L["rc"], L["tE"], L["Et"]
    e = t <= 3000
    i0 = np.argmin(D[e][5:]) + 5
    m, mE = t >= 1000, tE >= 1000
    print(f"{tag}   {D[0]:7.0f}  {D[m].mean():7.0f}       {t[e][i0]:5.0f} {D[e][i0]:6.0f}    {np.ptp(D[t <= 2000]):6.0f}"
          f"             {D[t > 2000].std():5.0f}    {Et[mE].mean():6.1f} +- {Et[mE].std():4.1f}      {Et[0]:6.1f}  "
          f"{top(t[m], D[m])}   {top(t[m], rc[m], 2)}")
