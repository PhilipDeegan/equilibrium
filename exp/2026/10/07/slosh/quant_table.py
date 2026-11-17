"""GP-seed r0 / a scan (scan_quant.sh + the 8k P 220 r0 30 a 10 run), read from the tracked slosh_*.log (slosh_io.py; dump new runs first): per run Et_in = E_in + (g/2) D_in within a + 28
(t >= 1k mean, sd, and the 1k -> end drift), D_in, co-moving rc, windings intact; Et per strand length (helix
length L = sum over the seed polygon) -> a dip / step vs r0 or a at fixed N = quantization; smooth = continuous family."""
from pathlib import Path

import numpy as np

from slosh_io import load

here = Path(__file__).parent
rows = []
for lg in sorted(here.glob("slosh_prof_pp_springbox_N*_S1_q1_a*_r*_xi3.9_m0.8_P220_*gp*.log")):
    tok = lg.stem.split("_")
    N = int(next(v for v in tok if v.startswith("N"))[1:])
    a, r0 = float(next(v for v in tok if v.startswith("a"))[1:]), float(next(v for v in tok if v.startswith("r") and v[1:2].isdigit())[1:])
    L = load(lg)
    t, D, rc, tE, Et = L["t"], L["D"], L["rc"], L["tE"], L["Et"]
    ok, okD = tE >= 1000, t >= 1000
    s_ = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    hel = np.stack([(r0 + a * np.cos(N * s_)) * np.cos(s_), (r0 + a * np.cos(N * s_)) * np.sin(s_), a * np.sin(N * s_)], 1)
    Lh = np.sum(np.linalg.norm(np.roll(hel, -1, 0) - hel, axis=1))
    rl = sorted(here.glob(f"*_gp_N{N}_a{a:g}_r{r0:g}_P220*.log"), key=lambda p: p.stat().st_mtime)[::-1]
    w = [ln for ln in open(rl[0]) if ln.startswith("P t")] if rl else []
    wind = w[-1].split("wind ")[1].split()[0] if w and "wind " in w[-1] else "?"
    rows.append((N, r0, a, t[-1], Et[0], Et[ok].mean(), Et[ok].std(), Et[ok][-1] - Et[ok][0], D[okD].mean(), rc[okD].mean(), Lh, wind))
print(" N  r0   a   T     Et(0)  Et mean +- sd  drift   D_in    rc     L(helix)  Et/L    wind(end)")
for r in sorted(rows):
    print(f"{r[0]:2d} {r[1]:4.0f} {r[2]:4.0f} {r[3]:5.0f} {r[4]:7.1f} {r[5]:7.1f} +- {r[6]:4.1f} {r[7]:+6.1f} {r[8]:7.0f} {r[9]:6.1f} {r[10]:8.1f} {r[5] / r[10]:7.3f}   {r[11]}")
