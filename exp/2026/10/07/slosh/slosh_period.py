"""Slosh time series from spring_box.py slosh_from0.npz: D_in (deficit within slosh_dt of the tube), rho_mid, E_in;
dominant period (FFT after the seed transient, detrended) vs the box sound-mode prediction side / c_s,
c_s = sqrt(g n / (3 m)) (walk Bogoliubov, omega = om0 + n k^2 / (6 m))."""
import argparse
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--run", action="append", required=True, help="run dir (with slosh_from0.npz, rec.npz)")
p.add_argument("--P", type=int, action="append", required=True)
p.add_argument("--t_min", type=int, default=1000)
p.add_argument("--m", type=float, default=0.8)
args = p.parse_args()
n = np.sqrt(1 - args.m**2)


def peaks(t, v, k=3):
    v = v - np.polyval(np.polyfit(t, v, 1), t)
    dt = t[1] - t[0]
    F = np.abs(np.fft.rfft(v * np.hanning(len(v))))
    f = np.fft.rfftfreq(len(v), dt)
    j = np.argsort(F[1:])[::-1][:k] + 1
    return [(1 / f[i], F[i] / F[1:].sum()) for i in j], np.std(v)


for run, P in zip(args.run, args.P):
    d = Path(run)
    S = np.load(d / "slosh_from0.npz", allow_pickle=True)["slosh"]
    g = float(np.load(d / "rec.npz")["g"]) if (d / "rec.npz").exists() else 0.00822
    cs = np.sqrt(g * n / (3 * args.m))
    side = P / np.sqrt(3)
    t = np.array([s["t"] for s in S], float)
    ok = t >= args.t_min
    D = np.array([s["D_in"] for s in S])
    rm = np.array([s["rho_mid"] for s in S])
    print(f"{d.name}: P {P} side {side:.1f} c_s {cs:.4f} -> side / c_s {side / cs:.0f}, side / (sqrt2 c_s) "
          f"{side / np.sqrt(2) / cs:.0f}, side / (sqrt3 c_s) {side / np.sqrt(3) / cs:.0f}; t {t[0]:.0f}-{t[-1]:.0f}")
    for lbl, v in [("D_in", D), ("rho_mid", rm)]:
        pk, sd = peaks(t[ok], v[ok])
        print(f"  {lbl:8s} mean {v[ok].mean():.5g} sd {sd:.4g}; periods " + ", ".join(f"{T:.0f} ({a:.2f})" for T, a in pk))
    tE = np.array([s["t"] for s in S if "E_in" in s], float)
    E = np.array([s["E_in"] for s in S if "E_in" in s])
    Et = np.array([s["E_tot"] for s in S if "E_in" in s])
    if len(E) > 8:
        okE = tE >= args.t_min
        pk, sd = peaks(tE[okE], E[okE])
        print(f"  E_in     mean {E[okE].mean():.1f} sd {sd:.2f}; periods " + ", ".join(f"{T:.0f} ({a:.2f})" for T, a in pk)
              + f"; E_tot {Et.min():.3f}..{Et.max():.3f}")
        Gt = E + 0.5 * g * np.interp(tE, t, D)  # grand-potential form: e + (g/2)(1 - rho), 2nd order for a uniform floor
        pk, sd = peaks(tE[okE], Gt[okE])
        print(f"  Et_in    mean {Gt[okE].mean():.1f} sd {sd:.2f} (E_in + g/2 D_in); periods "
              + ", ".join(f"{T:.0f} ({a:.2f})" for T, a in pk))
        print(f"  corr(D_in, E_in) at E samples {np.corrcoef(np.interp(tE[okE], t, D), E[okE])[0, 1]:+.2f}")
