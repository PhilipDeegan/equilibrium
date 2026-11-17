"""10-06 shell logs (spacing/shell_profile.py --ref_mid) -> raw (rho0 1) cum D and cum e within dt_max, and the
grand-potential form Et = E + (g / 2) D (e + (g/2)(1 - rho): zero to 2nd order for a uniform floor). Raw from rel:
D = D_rel + N (1 - rho_ref), E = E_rel + N e_ref, N = cells (own) up to dt_max."""
import argparse
import re

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("log", nargs="+")
p.add_argument("--dt", type=float, default=38.0)
p.add_argument("--m", type=float, default=0.8)
p.add_argument("--xi", type=float, default=3.9)
args = p.parse_args()
g2 = 0.5 * np.sqrt(1 - args.m**2) / 3 / (2 * args.m * args.xi**2)
rows = {}
for fn in args.log:
    blk = None
    for ln in open(fn):
        m = re.search(r"P (\d+) side .* t (\d+):", ln)
        if m:
            blk = dict(P=int(m[1]), t=int(m[2]), rref=1.0, eref=0.0, N=0)
            continue
        m = re.search(r"rho_ref ([\d.]+), e_ref ([-+\d.e]+)", ln)
        if m:
            blk["rref"], blk["eref"] = float(m[1]), float(m[2])
            continue
        f = ln.split()
        if blk and len(f) == 6 and f[0].replace(".", "").isdigit():
            blk["N"] += int(f[5])
            if float(f[0]) == args.dt:
                D = float(f[3]) + blk["N"] * (1 - blk["rref"])
                E = float(f[4]) + blk["N"] * blk["eref"]
                rows.setdefault(blk["P"], {})[blk["t"]] = (D, E, E + g2 * D)
print(f"g/2 = {g2:.6f}, within dt {args.dt}")
for P, r in sorted(rows.items()):
    a = np.array([[t, *v] for t, v in sorted(r.items())])
    print(f"P {P}: t " + " ".join(f"{int(v):5d}" for v in a[:, 0]))
    for j, lbl in [(1, "D raw"), (2, "E raw"), (3, "Et   ")]:
        print(f"  {lbl} " + " ".join(f"{v / 1e3:5.1f}" if j == 1 else f"{v:5.0f}" for v in a[:, j])
              + f"   mean {a[:, j].mean():.4g} sd {a[:, j].std():.3g}"
              + (f"  corr(D,E) {np.corrcoef(a[:, 1], a[:, 2])[0, 1]:+.2f}" if j == 2 else ""))
