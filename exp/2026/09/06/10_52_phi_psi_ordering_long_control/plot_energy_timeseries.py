"""
Turns the already-logged H_ratio/circ_rms/max|psi| time series from
probe_long_energy_bound.py (curl-free control) and probe_long_ring_bound.py
(real toroidal ring) into an actual plot, instead of leaving the total-
energy check buried in scrollback -- answers "should we have a total energy
check across the lattice from IC onwards" directly: yes, and here it is.
"""

import re
import sys

sys.path.insert(0, ".")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equilibrium.plot import savefig

CONTROL_LOG = "/tmp/claude-1000/-home-p-git-eq/51895a28-537a-46d0-a25c-eca7d3c6f2ca/tasks/bf76x25cr.output"
RING_LOG = "/tmp/claude-1000/-home-p-git-eq/51895a28-537a-46d0-a25c-eca7d3c6f2ca/tasks/bqcz2v27s.output"
OUT_DIR = "exp/2026/09/06/10_52_phi_psi_ordering_long_control"


def parse(path, extra_field=None):
    t, h, mp, extra = [], [], [], []
    N = h0 = None
    header_pat = re.compile(r"^N=(\d+).*?h0=([\d.eE+-]+)")
    pat = re.compile(
        r"t_phys=\s*([\d.]+)\s+"
        + (r"circ_rms/circ_rms0=([\d.]+)\s+" if extra_field else "")
        + r"max\|psi\|=([\d.eE+-]+)\s+H_ratio=([\d.eE+-]+)"
    )
    with open(path) as f:
        for line in f:
            if N is None:
                hm = header_pat.search(line)
                if hm:
                    N, h0 = int(hm.group(1)), float(hm.group(2))
            m = pat.search(line)
            if m:
                if extra_field:
                    tp, ex, mpv, hv = m.groups()
                    extra.append(float(ex))
                else:
                    tp, mpv, hv = m.groups()
                t.append(float(tp))
                mp.append(float(mpv))
                h.append(float(hv))
    return np.array(t), np.array(h), np.array(mp), np.array(extra), N, h0


def main():
    t_c, h_c, mp_c, _, N_c, h0_c = parse(CONTROL_LOG)
    t_r, h_r, mp_r, circ_r, N_r, h0_r = parse(RING_LOG, extra_field="circ")

    e_c = h_c * h0_c / N_c
    e_r = h_r * h0_r / N_r
    print(f"control: N={N_c} energy_per_cell(t=0)={e_c[0]:.6e}")
    print(f"ring:    N={N_r} energy_per_cell(t=0)={e_r[0]:.6e}")

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=False)

    ax = axes[0]
    ax.set_title("Curl-free control (Gaussian bump IC, R=22, N=%d, c2=1e-3)" % N_c)
    ax.plot(t_c, mp_c, color="tab:orange", label="max|psi|")
    ax.set_xlabel("t_phys")
    ax.set_ylabel("max|psi|", color="tab:orange")
    ax.tick_params(axis="y", labelcolor="tab:orange")
    ax2 = ax.twinx()
    ax2.plot(t_c, e_c, color="tab:blue", label="energy per cell (H/N)")
    ax2.set_ylabel("energy per cell, H/N", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.axhline(e_c[0], color="k", lw=0.5, ls=":")

    ax = axes[1]
    ax.set_title("Real toroidal ring (R=50, sph_frac=0.9, N=%d, c2=1e-3)" % N_r)
    ax.plot(t_r, circ_r, color="tab:green", label="circ_rms/circ_rms0")
    ax.plot(t_r, mp_r, color="tab:orange", label="max|psi|")
    ax.set_xlabel("t_phys")
    ax.set_ylabel("circ_rms/circ_rms0, max|psi|")
    ax.legend(loc="center left")
    ax2 = ax.twinx()
    ax2.plot(t_r, e_r, color="tab:blue", label="energy per cell (H/N)")
    ax2.set_ylabel("energy per cell, H/N", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.axhline(e_r[0], color="k", lw=0.5, ls=":")

    fig.suptitle("Total energy PER CELL stays bounded from IC onwards while circulation/psi oscillate")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(f"{OUT_DIR}/energy_timeseries.png", fig=fig, dpi=140)


if __name__ == "__main__":
    main()
