"""Coarse-grained view of the a 9 run: each phi dump smoothed by curl-curl diffusion (time tau, as the seed), then
content E_s / E_s(0), overlap with the smoothed seed, helicity sum A.curl A of the smoothed field / its t 0 value.
usage: coarse.py DUMPDIR [tau ...] [--P=96]"""
import sys
from pathlib import Path

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

C.spacing_of = lambda lat: 1.0
P_ = int(next((a[4:] for a in sys.argv if a.startswith("--P=")), 96))
sys.argv = [a for a in sys.argv if not a.startswith("--P=")]
D = Path(sys.argv[1])
taus = [float(v) for v in sys.argv[2:]] or [1.0, 4.0]
ac = C.AnchorCluster(cpp_box(P_, n_threads=4))
CH = np.array_split(np.arange(ac.N), max(1, ac.N // 65536))


def smooth(phi, tau, eps=0.02):
    for _ in range(int(round(tau / eps))):
        phi = phi - eps * ac.curl_curl_dec(phi)
    return phi


def hel(phi):
    rows = lambda j: phi[j]
    return sum(np.sum(C.reconstruct_A_at(ac, i, phi[i]) * C.curl_A_at(ac, i, rows)) for i in CH)


files = sorted(D.glob("phi_t*.npz"))
ref = {}
print("t      " + "  ".join(f"tau {t:g}: E_s/E_s0  overlap  H_s/H_s0  E_s/E_raw" for t in taus))
for f in files:
    t = float(f.stem[5:])
    phi = np.load(f)["phi"].astype(float)
    Er = np.sum(phi**2)
    out = []
    for tau in taus:
        ps = smooth(phi, tau)
        E, H = np.sum(ps**2), hel(ps)
        if tau not in ref:
            ref[tau] = (ps, E, H)
        p0, E0, H0 = ref[tau]
        out.append(f"{E / E0:8.4f}  {np.sum(ps * p0) / np.sqrt(E * E0):7.4f}  {H / H0:8.4f}  {E / Er:7.4f}")
    print(f"{t:6.1f}  " + "      ".join(out), flush=True)
