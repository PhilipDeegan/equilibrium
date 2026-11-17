import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium.substrate.lattice import wave_speed

here = Path(__file__).parent
R = 12.0
cross = 2 * R / wave_speed()
pattern = sys.argv[1] if len(sys.argv) > 1 else "*"

groups = defaultdict(list)
for f in sorted(here.glob(f"rec_potential_{pattern}_s*.npz")):
    d = np.load(f)
    if d["rec"].shape[1] < 10:
        continue
    key = re.sub(r"_s\d+$", "", f.stem[len("rec_potential_"):])
    r = d["rec"]
    t = r[:, 0] / cross
    m = re.search(r"_amp([0-9.]+)_", f.stem)
    excess = (r[:, 6] - r[:, 7]) / (float(d["E0"]) / (float(m.group(1)) ** 2 if m else 1.0))
    tau = np.concatenate([[0.0], np.cumsum(0.5 * (r[1:, 9] + r[:-1, 9]) * np.diff(r[:, 0]))]) / cross
    groups[key].append((t, excess, tau))

fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
print(f"{'case':38s} n  excess@late(mean +- sem)  seeds                               tau_core/t")
for ax, (key, runs) in zip(axes[0], sorted(groups.items())):
    late = []
    for t, ex, tau in runs:
        ax.plot(t, ex, lw=0.8)
        late.append(ex[t > 0.7 * t[-1]].mean())
    late = np.array(late)
    tmin = min(len(t) for t, _, _ in runs)
    ax.plot(runs[0][0][:tmin], np.mean([ex[:tmin] for _, ex, _ in runs], axis=0), "k", lw=2)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(key, fontsize=8)
    ax.set_xlabel("crossings")
    ax.set_ylabel("(E_core - E_core,bath) / E0")
    sem = late.std(ddof=1) / np.sqrt(len(late)) if len(late) > 1 else np.nan
    ratio = np.mean([tau[-1] / t[-1] for t, _, tau in runs])
    print(f"{key:38s} {len(late)}  {late.mean():+.3f} +- {sem:.3f}         {np.array2string(late, precision=3)}  {ratio:.3f}")
fig.tight_layout()
fig.savefig(here / f"seeds_{pattern.replace('*', '')}.png", dpi=120)
