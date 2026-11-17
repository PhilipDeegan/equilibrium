import re
import sys
from pathlib import Path

for f in sorted(Path(".").glob("R*_w*.log")):
    rows = []
    for ln in f.read_text().splitlines():
        m = re.match(r"\s+(\d+\.\d+)\s+\d+\s+\S+\s+\S+\s+\S+\s+(\S+)\s+\S+\s+\S+\s+\S+\s+(\S+)", ln)
        if not m or "rms2" not in ln:
            continue
        g = lambda k: float(re.search(k + r" (\S+)", ln).group(1))
        rows.append(dict(t=float(m[1]), z=float(m[2]), E=float(m[3]), rms=g("rms"), rms2=g("rms2"), G=g(r"sum\|G\|"),
                         Eu=g("Eu"), lk=float(re.search(r"linked (\S+)", ln).group(1))))
    if len(rows) < 2:
        print(f.name, "no data"); continue
    a, b = rows[0], rows[-1]
    mid = rows[len(rows) // 2]
    pc = lambda k, r: 100 * (r[k] / a[k] - 1)
    R = float(re.match(r"R(\d+)", f.name)[1])
    print(f"{f.stem:22s} t {b['t']:5.1f} travel/R {(b['z'] - a['z']) / R:+.4f}  rms {a['rms']:.3f} {pc('rms', mid):+6.2f}% {pc('rms', b):+6.2f}%"
          f"  rms2 {a['rms2']:.3f} {pc('rms2', mid):+6.2f}% {pc('rms2', b):+6.2f}%  sum|G| {pc('G', mid):+6.2f}% {pc('G', b):+6.2f}%"
          f"  Eu {mid['Eu']:.2e} {b['Eu']:.2e}  dE/E {b['E'] / a['E'] - 1:+.1e}  linked {b['lk']:+.6f}")
