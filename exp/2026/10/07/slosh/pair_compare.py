"""CsCl-type pair (pair_box.py): q -1 (opposite-sense partner at the body centre) vs q +1 (same sense, BCC control).
Per run from the slosh_*.log: relative z drift of the partner (dz_rel, unwrapped) -> mean relative speed and the
residual from a straight line (interaction = curvature of dz_rel(t) / speed change at passes); primary / partner centre
radii; Et_in = E_in + g/2 D_in (primary, own Voronoi region incl. the partner tubes) mean / sd / drift."""
import argparse

import numpy as np

from slosh_io import load

p = argparse.ArgumentParser()
p.add_argument("--log", action="append", required=True)
p.add_argument("--P", type=int, default=300)
p.add_argument("--t_min", type=int, default=1000)
p.add_argument("--t_max", type=int, default=10**9, help="common window (q +1 control stopped at 5000)")
args = p.parse_args()
side = args.P / np.sqrt(3)
for lg in args.log:
    L = load(lg)
    m_ = L["t"] <= args.t_max
    for k in ("t", "D", "rho_mid", "rc", "zc", "D_tot"):
        L[k] = L[k][m_]
    for k in ("E_in", "E_tot", "Et"):
        L[k] = L[k][L["tE"] <= args.t_max]
    L["tE"] = L["tE"][L["tE"] <= args.t_max]
    for k in ("dz_rel", "rcp"):
        L[k] = L[k][L["tP"] <= args.t_max]
    L["tP"] = L["tP"][L["tP"] <= args.t_max]
    t, dz = L["tP"], np.unwrap(L["dz_rel"] * 2 * np.pi / side) * side / (2 * np.pi)
    ok = t >= args.t_min
    c = np.polyfit(t[ok], dz[ok], 2)
    lin = np.polyfit(t[ok], dz[ok], 1)
    res = dz[ok] - np.polyval(lin, t[ok])
    v_blocks = [np.polyfit(t[(t >= a) & (t < a + 1000)], dz[(t >= a) & (t < a + 1000)], 1)[0] for a in range(0, int(t[-1]), 1000)]
    tE, Et = L["tE"], L["Et"]
    okE = tE >= args.t_min
    print(f"{lg}\n  relative z speed {lin[0]:+.5f} / tick (quadratic coef {c[0]:+.2e}), residual sd {res.std():.2f} cells; "
          f"passes (dz_rel through 0 / side) {int(abs(dz[-1] - dz[0]) // side)}")
    print("  speed per 1k block: " + " ".join(f"{v:+.4f}" for v in v_blocks))
    print(f"  rc {L['rc'][t >= args.t_min].mean():.2f} +- {L['rc'][t >= args.t_min].std():.2f}, rcp {L['rcp'][ok].mean():.2f} +- "
          f"{L['rcp'][ok].std():.2f}; Et_in {Et[okE].mean():.1f} +- {Et[okE].std():.2f} (drift {Et[okE][-1] - Et[okE][0]:+.1f}); "
          f"D_in {L['D'][t >= args.t_min].mean():.0f}")
