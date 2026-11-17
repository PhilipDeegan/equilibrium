"""10-24 DST spheromak (step() + Cartan, 10-22 logs) vs the continuum references (rung.py logs), common units:
time in turnovers a / amp (amp 0.3 both), content = DST face content E (sum phi^2) vs reference field energy EB (the DST
seed puts the Taylor field profile into phi). Panels: total content / initial, share inside r < a, shape change
|f - f0| / |f0|."""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H = Path(__file__).parent
D = H.parents[1] / "22" / "spheromak"
num = r"([-+0-9.eE]+)"


def dst(files, a):
    rows = {}
    for f in files:
        for ln in open(f):
            m = re.search(rf"t +{num} +E {num} .*\|dphi\|/\|phi0\| {num} .*in/shell/out {num}", ln)
            if m:
                rows[float(m[1])] = tuple(map(float, m.groups()[1:]))
    t = np.array(sorted(rows))
    E, d, s = np.array([rows[k] for k in t]).T
    return t / (a / 0.3), E / E[0], s, d


def ref(f, a):
    t, E, ov, s = [], [], [], []
    for ln in open(f):
        m = re.search(rf"t +{num} +EB/EB0 {num} .*overlap {num} +in/shell/out {num}", ln)
        if m:
            t.append(float(m[1])), E.append(float(m[2])), ov.append(float(m[3])), s.append(float(m[4]))
    t, E, ov, s = map(np.array, (t, E, ov, s))
    d = np.sqrt(np.maximum(E + 1 - 2 * ov * np.sqrt(E), 0))
    return t / (a / 0.3), E, s, d


series = [
    ("DST a 9 (step + Cartan)", dst([D / "a9.txt"], 9), dict(color="#c0392b", lw=2)),
    ("DST a 20 (step + Cartan)", dst([D / "a20.txt", D / "a20_ext.txt"], 20), dict(color="#e67e22", lw=2)),
    ("MHD eta 5e-4", ref(H / "mhd_e5e-4_n96.txt", 9), dict(color="#08306b", lw=2)),
    ("MHD eta 2e-3", ref(H / "mhd_e2e-3_n64.txt", 9), dict(color="#2171b5", lw=2)),
    ("MHD eta 1e-2", ref(H / "mhd_e1e-2_n64.txt", 9), dict(color="#6baed6", lw=2)),
    ("vacuum Maxwell (unheld)", ref(H / "vacuum.txt", 9), dict(color="0.45", lw=1.5, ls="--")),
    ("diffusion eta 5e-2", ref(H / "diffuse.txt", 9), dict(color="0.65", lw=1.5, ls=":")),
]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
for lab, (t, E, s, d), st in series:
    axs[0].plot(t, E, label=lab, **st)
    axs[1].plot(t, s, **st)
    axs[2].plot(t, d, **st)
for ax, yl in zip(axs, ["total content / initial", "share inside r < a", "shape change |f - f0| / |f0|"]):
    ax.set_xlabel("turnovers  t / (a / amp)")
    ax.set_ylabel(yl)
    ax.set_xlim(0, 12)
    ax.grid(alpha=0.25)
axs[0].set_ylim(0, 1.05)
axs[1].set_ylim(0, 1.0)
axs[2].set_ylim(0, 1.5)
axs[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Spheromak a 9 / a 20: DST step() + Cartan vs lattice-free references", fontsize=11)
fig.savefig(H / "compare.png", dpi=110)
print("wrote", H / "compare.png")
