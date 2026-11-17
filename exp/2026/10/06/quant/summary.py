"""table of scan budgets: structure E at t 0 / 2k / 8k, relative structure loss by 2k and 2k->8k, outside E share at
8k, census pieces (max over t) per budget_*.log."""
import re
import sys
from pathlib import Path

rows = []
for f in sorted(Path(".").glob("budget_*.log")):
    txt = f.read_text()
    m = re.match(r"budget_N(\d+)_a([\d.]+)_r([\d.]+)\.log", f.name)
    blocks = re.split(r"\n(?=springbox)", txt.strip())
    d = {}
    for b in blocks:
        t = int(re.search(r" t (\d+):", b).group(1))
        es = float(re.search(r"structure .*? E ([\d.]+)", b).group(1))
        et = float(re.search(r"   E .*\| total ([\d.]+)", b).group(1))
        pc = re.search(r"census .*?: (\d+) pieces", b)
        d[t] = (es, et, int(pc.group(1)) if pc else 0)
    if 0 not in d or 8000 not in d:
        continue
    e0, e2, e8 = d[0][0], d.get(2000, d[8000])[0], d[8000][0]
    rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), e0, e2, e8, (e0 - e2) / e0, (e2 - e8) / e2,
                 1 - d[8000][0] / d[8000][1], max(v[2] for v in d.values())))
print(" N    a    r0   E_str0   E_str2k  E_str8k  loss0-2k  drift2-8k  out_share8k  pieces")
for r in sorted(rows):
    print(f"{r[0]:2d} {r[1]:4g} {r[2]:5g} {r[3]:8.1f} {r[4]:8.1f} {r[5]:8.1f} {r[6]:9.3f} {r[7]:+10.3f} {r[8]:11.3f} {r[9]:6d}")
