"""Scattering of a plane wave packet off one unit size defect in the discrete rule (step() + face sigma = mean of s / S).

Defects: 'cell' = one cell contracted by one unit (s = S - 1, net depletion); 'pair' = one unit moved across a hex junction
(A S - 1, B S + 1; conserved, relational). Scattered energy = H(defect run - clean run) after the right-going half passed;
cross-section = scattered energy / (incident right-going energy per unit transverse area). Linear: expect ~ (1 / S)^2.
"""

import argparse
import time

import numpy as np

from bcc import Box, diff_energy

p = argparse.ArgumentParser()
p.add_argument("--nt", type=int, default=48)
p.add_argument("--nz", type=int, default=192)
p.add_argument("--k", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
p.add_argument("--S", type=int, nargs="+", default=[100, 50])
p.add_argument("--dt", type=float, default=0.05)
args = p.parse_args()

nt, nz = args.nt, args.nz
n = (nt, nt, nz)
z0, zd = nz // 2 - 40, nz // 2 + 8
c = (nt // 2, nt // 2, zd)
z = np.arange(float(nz))
print(f"box {nt}x{nt}x{nz} ({2 * nt * nt * nz} cells) packet z0 {z0} defect z {zd} dt {args.dt}", flush=True)

for k in args.k:
    w = max(6.0, 3.0 / k)
    f = lambda zz: np.exp(-((zz - z0) ** 2) / (2 * w * w)) * np.cos(k * zz)
    T = (zd - z0 + 5 * w) / 1.9
    nst = int(T / args.dt)
    runs = {"clean": None}
    for S in args.S:
        sA, sB = np.full(n, S), np.full(n, S)
        sA[c] = S - 1
        runs[f"cell S{S}"] = (sA.copy(), sB.copy(), S)
        sB[c] = S + 1
        runs[f"pair S{S}"] = (sA, sB, S)
    boxes = {}
    for name, sz in runs.items():
        b = Box(n) if sz is None else Box(n, *sz)
        b.uA[:] = f(z)
        b.uB[:] = f(z + 0.5)
        boxes[name] = b
    E0 = boxes["clean"].energy()
    t0 = time.time()
    for i in range(nst):
        for b in boxes.values():
            b.step(args.dt)
        if i == 20:
            print(f"  k {k}: {nst} steps, ETA {(time.time() - t0) / 21 * nst / 60:.1f} min", flush=True)
    I = 0.5 * E0 / (nt * nt)
    for name, b in boxes.items():
        if name == "clean":
            continue
        S = int(name.split("S")[1])
        dE = diff_energy(b, boxes["clean"])
        print(f"RESULT k {k:5.2f} w {w:4.1f} {name:10s} sigma_s {dE / I:.4e}  sigma_s * S^2 {dE / I * S * S:.4e}", flush=True)
