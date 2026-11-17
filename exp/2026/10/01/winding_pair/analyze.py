"""Summarise run.py outputs: retention vs t and proper time, mode spectra (travelling / frozen), end split."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from equilibrium.substrate.lattice import _W, SubstrateLattice

here = Path(__file__).parent
CASES = ["ring", "helix", "pair_opp", "pair_cnt", "pair_same"]
alphas = [float(a) for a in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0", "2", "8", "32"])]
lat = SubstrateLattice(radius_cells=14.0)  # same cell set as DampedLattice(14)


def split(u, phi, core):
    """u / gradient / solenoidal energy fractions of the core."""
    i, k = np.nonzero(lat.nb >= 0)
    j = lat.nb[i, k]
    o = i < j
    fi, fk, fj = i[o], k[o], j[o]
    w = _W[fk]
    F = len(fi)
    G = sp.csr_matrix((np.concatenate([w, -w]), (np.tile(np.arange(F), 2), np.concatenate([fj, fi]))), shape=(F, lat.N))
    f = phi[fi, fk]
    fg = G @ lsqr(G, f, atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    fc = core[fi] & core[fj]
    eu, eg, es = 0.5 * np.sum(u[core] ** 2), 0.5 * np.sum(fg[fc] ** 2), 0.5 * np.sum((f - fg)[fc] ** 2)
    t = eu + eg + es
    return eu / t, eg / t, es / t


fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a in alphas:
    for cs in CASES:
        fn = here / f"{cs}_a{a:g}.npz"
        if not fn.exists():
            continue
        d = np.load(fn)
        r, A, modes = d["rec"], d["amps"], d["modes"]
        t, E, sig = r[:, 0], r[:, 1], r[:, 2]
        tau = np.concatenate([[0], np.cumsum(0.5 * (sig[1:] + sig[:-1]) * np.diff(t))])
        late = t > 0.6 * t[-1]
        dtr = t[1] - t[0]
        # spectrum of each mode over the late window: power fraction at |f| < 0.3 omega (frozen) and its peak frequency
        om = d["omegas"][0]
        spec = []
        for jm, m in enumerate(modes[:3]):
            x = A[late, jm] - 0.0
            P = np.abs(np.fft.fft(x * np.hanning(len(x)))) ** 2
            f = 2 * np.pi * np.fft.fftfreq(len(x), dtr)
            tot = P.sum()
            spec.append((tuple(m), np.sqrt(np.mean(np.abs(x) ** 2)), P[np.abs(f) < 0.3 * om].sum() / tot, f[np.argmax(P)]))
        pos = d["pos"]
        rho = np.hypot(pos[:, 0], pos[:, 1])
        core = np.hypot(rho - 5.0, pos[:, 2]) < 4.5
        su, sg, ss = split(d["u"], d["phi"], core)
        print(f"a={a:5g} {cs:9s} Etube/E0 late {E[late].mean():.4f} end {E[-1]:.4f}  sigma {sig[late].mean():.3f}  "
              f"tau_end {tau[-1]:.0f}/{t[-1]:.0f}  lam_max {r[-1, 5]:.2f}  z_c {r[-1, 3]:+.3f}  split u/g/s {su:.2f}/{sg:.2f}/{ss:.2f}")
        for m, rms, frz, fpk in spec:
            print(f"        mode {m}: rms|a| {rms:.4f}  frozen frac {frz:.2f}  peak omega {fpk:+.3f} (seed {om:.3f})")
        ls = {0.0: ":", 2.0: "-.", 8.0: "--", 32.0: "-"}.get(a, "-")
        col = f"C{CASES.index(cs)}"
        ax[0].semilogy(t, E, ls, color=col, label=f"{cs} a{a:g}")
        ax[1].semilogy(tau, E, ls, color=col)
ax[0].set_xlabel("t")
ax[1].set_xlabel("proper time (tube, E-weighted)")
ax[0].set_ylabel("E_tube / E0")
ax[0].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(here / "retention.png", dpi=110)
