"""D_in(t), rho_mid(t), Et_in(t) series + zero-padded spectrum of D_in (t >= t_min, detrended); marks the box sound
modes side / (|n| c_s), n = (1,0,0), (1,1,0), (1,1,1), (2,0,0)."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slosh_io import load

p = argparse.ArgumentParser()
p.add_argument("--run", action="append", required=True, help="run dir or slosh_*.log")
p.add_argument("--P", type=int, action="append", required=True)
p.add_argument("--t_min", type=int, default=1000)
p.add_argument("--m", type=float, default=0.8)
p.add_argument("--out", default="slosh_spectrum.png")
args = p.parse_args()
n = np.sqrt(1 - args.m**2)
fig, ax = plt.subplots(len(args.run), 2, figsize=(15, 3.6 * len(args.run)), squeeze=False)
for r, (run, P) in enumerate(zip(args.run, args.P)):
    L = load(run)  # run dir or slosh_*.log (slosh_io.py)
    g = L["g"]
    cs, side = np.sqrt(g * n / (3 * args.m)), P / np.sqrt(3)
    t, D, tE, Et = L["t"], L["D"], L["tE"], L["Et"]
    a = ax[r, 0]
    a.plot(t, D / 1e3, lw=0.8, label="D_in / 1e3")
    a2 = a.twinx()
    a2.plot(tE, Et, "k.", ms=3, label="Et_in = E_in + g/2 D_in")
    a.set_title(f"P {P} (side {side:.0f}): D_in (blue), Et_in (black, right)")
    a.set_xlabel("t")
    ok = t >= args.t_min
    v = D[ok] - np.polyval(np.polyfit(t[ok], D[ok], 1), t[ok])
    nf = 16 * len(v)
    F = np.abs(np.fft.rfft(v * np.hanning(len(v)), nf))
    f = np.fft.rfftfreq(nf, t[1] - t[0])
    b = ax[r, 1]
    b.plot(f[1:] * 1e3, F[1:] / F[1:].max())
    for nn, lbl in [(1, "100"), (np.sqrt(2), "110"), (np.sqrt(3), "111"), (2, "200")]:
        b.axvline(nn * cs / side * 1e3, color="r", ls=":", lw=1)
        b.text(nn * cs / side * 1e3, 0.95, lbl, color="r", fontsize=8)
    b.set_xlim(0, 3)
    b.set_xlabel("frequency (1 / 1000 ticks)")
    b.set_title("D_in spectrum (zero-padded x16), red: box sound modes side / (|n| c_s)")
    j = np.argsort(F[1:])[::-1] + 1
    pk = []
    for i in j:
        if all(abs(i - q) > 16 for q in pk) and 0 < i < len(F) - 1 and F[i] >= F[i - 1] and F[i] >= F[i + 1]:
            pk.append(i)
        if len(pk) == 3:
            break
    print(f"P {P} side {side:.1f} c_s {cs:.4f}: modes 100 {side / cs:.0f} 110 {side / cs / np.sqrt(2):.0f} 111 "
          f"{side / cs / np.sqrt(3):.0f} 200 {side / cs / 2:.0f}; D_in peaks " + ", ".join(f"{1 / f[i]:.0f} ({F[i] / F[pk[0]]:.2f})" for i in pk)
          + f"; D_in sd {v.std():.0f} of {D[ok].mean():.0f}; Et_in {Et[tE >= args.t_min].mean():.1f} sd {Et[tE >= args.t_min].std():.2f}")
fig.tight_layout()
fig.savefig(args.out, dpi=80)
