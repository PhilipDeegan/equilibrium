"""
Overlays circ_rms/H_ratio for the domain-confinement confound check:
c2=0 (plain step(), no psi) vs c2=1e-3 (the coupling), same IC/domain/step
count. Parses the already-completed background-task logs directly, no
re-simulation needed.
"""

import re
import sys

sys.path.insert(0, ".")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equilibrium.plot import savefig

C2_ZERO_LOG = "/tmp/claude-1000/-home-p-git-eq/51895a28-537a-46d0-a25c-eca7d3c6f2ca/tasks/b19xpgv2s.output"
C2_COUPLED_LOG = "/tmp/claude-1000/-home-p-git-eq/51895a28-537a-46d0-a25c-eca7d3c6f2ca/tasks/bqcz2v27s.output"
OUT_DIR = "exp/2026/09/06/10_52_phi_psi_ordering_long_control"


def parse(path):
    t, circ, h = [], [], []
    pat = re.compile(r"t_phys=\s*([\d.]+)\s+circ_rms/circ_rms0=([\d.]+)\s+.*?H_ratio=([\d.eE+-]+)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                tp, cr, hv = m.groups()
                t.append(float(tp))
                circ.append(float(cr))
                h.append(float(hv))
    return np.array(t), np.array(circ), np.array(h)


def main():
    t0, circ0, h0 = parse(C2_ZERO_LOG)
    t1, circ1, h1 = parse(C2_COUPLED_LOG)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)

    ax = axes[0]
    ax.plot(t0, circ0, label="c2=0 (plain step(), no psi)", color="tab:blue")
    ax.plot(t1, circ1, label="c2=1e-3 (with (phi,psi) coupling)", color="tab:red")
    ax.set_ylabel("circ_rms / circ_rms0")
    ax.set_title("Domain-confinement confound check: same IC/domain/steps, coupling on vs off")
    ax.legend()
    ax.axhline(1.0, color="k", lw=0.5, ls=":")

    ax = axes[1]
    ax.plot(t0, h0, label="c2=0", color="tab:blue")
    ax.plot(t1, h1, label="c2=1e-3", color="tab:red")
    ax.set_ylabel("H_ratio")
    ax.set_xlabel("t_phys")
    ax.legend()
    ax.axhline(1.0, color="k", lw=0.5, ls=":")

    fig.tight_layout()
    savefig(f"{OUT_DIR}/confound_timeseries.png", fig=fig, dpi=140)


if __name__ == "__main__":
    main()
