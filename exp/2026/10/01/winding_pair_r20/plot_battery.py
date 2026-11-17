"""Fixed centred depletion battery (run.py --lam_shape S --lam0 L --tag _bat): structure / tube energy, winding
amplitudes and their phase (travelling vs frozen), grid-scale fraction, vs t and proper time. Colour = shape and lam0,
line style = case (solid helix control, dashed pair_opp)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).parent
runs = []
for fn in sorted(here.glob("*_bat_rec.npz")):
    d = np.load(fn)
    a = json.loads(str(d["args"]))
    runs.append((a["lam_shape"], a["lam0"], a["case"], d))
runs.sort(key=lambda r: (r[0], r[1], r[2]))
keys = sorted({(s, l) for s, l, _, _ in runs})
colour = {k: f"C{i}" for i, k in enumerate(keys)}
style = {"helix": "-", "pair_opp": "--", "pair_cnt": ":", "pair_same": "-."}

fig, ax = plt.subplots(2, 4, figsize=(20, 9))
figp, axp = plt.subplots(1, 2, figsize=(13, 4.8))
for s, l0, case, d in runs:
    r, modes = d["rec"], [tuple(m) for m in d["modes"]]
    t, Et, sig, Es = r[:, 0], r[:, 1], r[:, 2], r[:, 8] / r[0, 8]
    tau = np.concatenate([[0], np.cumsum(0.5 * (sig[1:] + sig[:-1]) * np.diff(t))])
    gf = d["grid_frac"]
    amp = np.abs(d["amps"])
    lab = f"{s} {l0:g} {case}"
    kw = dict(color=colour[(s, l0)], ls=style[case], label=lab)
    for row, xx, xl in ((0, t, "t"), (1, tau, "proper time (tube, E-weighted sigma)")):
        ax[row, 0].plot(xx, Es, **kw)
        ax[row, 1].plot(xx, Et, **kw)
        ax[row, 2].semilogy(xx, amp[:, modes.index((1, 1))], **kw)
        if case != "helix":
            ax[row, 2].semilogy(xx, amp[:, modes.index((1, -1))], color=colour[(s, l0)], ls=":", lw=1)
        ax[row, 3].semilogy(xx, gf, **kw)
        for c, yl in enumerate(["E_struct (r < r0 + 4w) / initial", "E_tube / E0",
                                "|a_(n,+1)| (dotted: |a_(n,-1)| of pairs)", "grid-scale fraction of u (tube)"]):
            ax[row, c].set_xlabel(xl)
            ax[row, c].set_ylabel(yl)
    for j, m in enumerate(((1, 1), (1, -1))):
        if case == "helix" and m == (1, -1):
            continue
        ph = np.unwrap(np.angle(d["amps"][:, modes.index(m)]))
        axp[j].plot(t, ph - ph[0], **kw)
        n = len(t) // 3
        late = np.polyfit(t[-n:], ph[-n:], 1)[0]
        early = np.polyfit(t[:n], ph[:n], 1)[0]
        print(f"{lab:24s} mode {m}: |a| {amp[0, modes.index(m)]:.4f} -> {amp[-1, modes.index(m)]:.4f}  "
              f"phase rate early {early:+.4f} late {late:+.4f}")
    print(f"{lab:24s} sigma_tube(0) {sig[0]:.3f}  E_struct end {Es[-1]:.3f}  E_tube end {Et[-1]:.3f}  "
          f"grid frac {gf[0]:.2e} -> {gf[-1]:.2e}  tau/t {tau[-1] / t[-1]:.3f}  E_sponge end {r[-1, 7]:.3f}")
for j, m in enumerate(((1, 1), (1, -1))):
    axp[j].set_xlabel("t")
    axp[j].set_ylabel(f"unwrapped phase of a_{m} - initial")
    axp[j].set_title("travelling: steady slope; frozen: flat")
axp[0].legend(fontsize=8)
ax[0, 0].legend(fontsize=8)
fig.suptitle("fixed centred depletion battery (linear step, sigma on both legs)")
fig.tight_layout()
fig.savefig(here / "battery.png", dpi=100)
figp.tight_layout()
figp.savefig(here / "battery_phase.png", dpi=100)
print(f"-> {here / 'battery.png'}  {here / 'battery_phase.png'}")
