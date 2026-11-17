import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = Path(sys.argv[1])
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for f in sorted(d.glob("*.txt")):
    rows = [[float(x) for x in re.findall(r"(?:t|A|B|gap|out|absorbed)\s+([-+\d.e]+)", ln)[:6]]
            for ln in f.read_text().splitlines() if ln.startswith("  t ")]
    h = np.array(rows)
    lab = f.stem
    ax[0].plot(h[:, 0], h[:, 1], label=lab)
    ax[1].plot(h[:, 0], h[:, 2], label=lab)
    ax[2].plot(h[:, 0], h[:, 3], label=lab)
for a, t in zip(ax, ["energy in A (starts holding ~0.82)", "energy in B", "gap snapshot energy (in flight between)"]):
    a.set_title(t)
    a.set_xlabel("t")
ax[2].set_yscale("log")
ax[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(d / "pair.png", dpi=120)
