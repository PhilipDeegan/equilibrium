"""Where does the coarse helicity go? Per phi dump of the 11-08 cascade runs:
- LS helicity sum A.curl A (A = 3-DOF least-squares vector per cell) of raw phi, of the curl-curl smoothed phi_s
  (tau), of the fine rest phi_f = phi - phi_s, and the cross term; + positive / negative parts of the fine density.
- signed helicity spectrum of the reconstructed A (FFT on the P^3 grid, BCC sites filled, shells below the alias
  radius 0.35 P), summed in bands of n = |k| side / 2 pi.
- non-vector share (anchor_residual^2 / face content): the 4 of 7 face DOF the vector A (and so every helicity
  above) does not see.
usage: fine_hel.py DUMPDIR --P=96 [--tau=1] [--every=1]"""
import sys
from pathlib import Path

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

C.spacing_of = lambda lat: 1.0
opt = dict(a[2:].split("=") for a in sys.argv[1:] if a.startswith("--"))
P_, tau, every = int(opt.get("P", 96)), float(opt.get("tau", 1.0)), int(opt.get("every", 1))
D = Path(next(a for a in sys.argv[1:] if not a.startswith("--")))
ac = C.AnchorCluster(cpp_box(P_, n_threads=4))
CH = np.array_split(np.arange(ac.N), max(1, ac.N // 65536))
side = P_ / np.sqrt(3)
g = np.rint((ac.pos - ac.pos.min(0)) * np.sqrt(3)).astype(int) % P_
n1 = np.fft.fftfreq(P_, 1 / P_)
NX, NY, NZ = np.meshgrid(n1, n1, n1, indexing="ij")
NN = np.sqrt(NX**2 + NY**2 + NZ**2)
KV = [2 * np.pi / side * v for v in (NX, NY, NZ)]
BANDS = [(0, 3), (3, 6), (6, 10), (10, 16), (16, 0.35 * P_)]


def smooth(phi, eps=0.02):
    for _ in range(int(round(tau / eps))):
        phi = phi - eps * ac.curl_curl_dec(phi)
    return phi


def hel_dens(phi):
    rows = lambda j: phi[j]
    return np.concatenate([np.sum(C.reconstruct_A_at(ac, i, phi[i]) * C.curl_A_at(ac, i, rows), 1) for i in CH])


def spectrum(phi):
    A = np.concatenate([C.reconstruct_A_at(ac, i, phi[i]) for i in CH])
    Ah = []
    for d in range(3):
        G = np.zeros((P_,) * 3)
        G[g[:, 0], g[:, 1], g[:, 2]] = A[:, d]
        Ah.append(np.fft.fftn(G))
    cur = [1j * (KV[1] * Ah[2] - KV[2] * Ah[1]), 1j * (KV[2] * Ah[0] - KV[0] * Ah[2]), 1j * (KV[0] * Ah[1] - KV[1] * Ah[0])]
    h = sum(np.real(np.conj(Ah[d]) * cur[d]) for d in range(3))
    e = sum(np.abs(Ah[d]) ** 2 for d in range(3))
    return [(h[(NN >= lo) & (NN < hi)].sum(), e[(NN >= lo) & (NN < hi)].sum()) for lo, hi in BANDS]


files = sorted(D.glob("phi_t*.npz"))[::every]
ref = None
print(f"# {D}  P {P_}  tau {tau}  ratios to t0 raw LS helicity H0; bands n = |k| side / 2pi: {BANDS}")
print("t       Hraw    Hs      Hf      Hx      Hf+     Hf-    nonvec | spec H per band / H0 ... | spec E share per band")
for f in files:
    t = float(f.stem[5:])
    phi = np.load(f)["phi"].astype(float)
    ps = smooth(phi)
    pf = phi - ps
    dr, ds, df = hel_dens(phi), hel_dens(ps), hel_dens(pf)
    Hr, Hs, Hf = dr.sum(), ds.sum(), df.sum()
    sp = spectrum(phi)
    if ref is None:
        ref = (Hr, sum(h for h, _ in sp))
    H0, S0 = ref
    res = C.anchor_residual(ac, phi)
    nv = np.sum(res**2) / np.sum(np.where(ac.nb >= 0, phi / C.W[None, :], 0.0) ** 2)
    Etot = sum(e for _, e in sp)
    print(
        f"{t:6.1f}  {Hr / H0:6.3f}  {Hs / H0:6.3f}  {Hf / H0:6.3f}  {(Hr - Hs - Hf) / H0:6.3f}  "
        f"{df[df > 0].sum() / H0:6.3f}  {df[df < 0].sum() / H0:6.3f}  {nv:6.3f} | "
        + " ".join(f"{h / S0:7.3f}" for h, _ in sp)
        + " | "
        + " ".join(f"{e / Etot:5.3f}" for _, e in sp),
        flush=True,
    )
