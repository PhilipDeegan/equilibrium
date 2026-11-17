"""q -1 pair (opposite sense, body centre, P 300): primary and partner speeds through the pass vs a lone coil.
True z every 20 ticks from the slosh log (zc unwrapped across the --recentre whole-lattice-vector shifts; partner =
zc + dz_rel); speeds = linear fits over a sliding window; minus the lone coil (box_gp P 220, same seed / breathing
phase, same sampling). Prints the speed excess vs time and vs the minimum-image z separation of the nearest partner copy."""
from pathlib import Path

import numpy as np

from slosh_io import load

here = Path(__file__).parent
A_C, side, W = 2 / np.sqrt(3), 300 / np.sqrt(3), 600


def true_z(t, zc):
    d = np.diff(zc)
    j = np.abs(d) > 1.0
    d[j] -= np.round(d[j] / A_C) * A_C
    return np.r_[zc[0], zc[0] + np.cumsum(d)]


def speed(t, z):
    c = np.arange(W // 2, t[-1] - W // 2 + 1, 200)
    return c, np.array([np.polyfit(t[(t >= a - W / 2) & (t <= a + W / 2)], z[(t >= a - W / 2) & (t <= a + W / 2)], 1)[0] for a in c])


P = load(here / "slosh_pairq-1_prof_pp_springbox_N2_S1_q1_a10_r30_xi3.9_m0.8_P300_z0_wxi_rc_par_gp.log")
L = load(here / "slosh_prof_pp_springbox_N2_S1_q1_a10_r30_xi3.9_m0.8_P220_z0_wxi_rc_par_gp.log")
zp = true_z(P["t"], P["zc"])
zq = zp + np.unwrap(P["dz_rel"] * 2 * np.pi / side) * side / (2 * np.pi)
zl = true_z(L["t"], L["zc"])
c, vp = speed(P["t"], zp)
_, vq = speed(P["t"], zq)
_, vl = speed(L["t"], zl)
sep = (np.interp(c, P["t"], zq - zp) + side / 2) % side - side / 2
print(f"window {W} ticks; lone mean {vl[c >= 1000].mean():.5f}")
print("    t   z-sep   v_prim    v_partner   lone     prim-lone  |partner|-lone")
for i in range(len(c)):
    print(f"{c[i]:5.0f} {sep[i]:+7.1f}  {vp[i]:.5f}  {vq[i]:+.5f}  {vl[i]:.5f}  {vp[i] - vl[i]:+.5f}   {-vq[i] - vl[i]:+.5f}")
