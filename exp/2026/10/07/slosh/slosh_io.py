"""slosh_from<t>.npz (gitignored; resumed segments merged) <-> slosh_<name>.log (text, tracked): series rows every probe (t, D_in, rho_mid, rc,
zc, D_tot, [E_in, E_tot, Et_in]) and, at the energy samples, the dt shells (deficit, energy, cells per SH_DR bin).
python3 slosh_io.py <run_dir> [...] writes slosh_<P..._tag>.log beside this file; load(path) reads either form."""
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).parent


def log_name(run_dir):
    p = Path(run_dir).parts
    i = max(j for j, s in enumerate(p) if s == "springbox")
    return here / ("slosh_" + "_".join(p[i - 3 if p[i - 3].startswith("pairq") else i - 2:]) + ".log")


def dump(run_dir):
    d = Path(run_dir)
    segs = sorted(d.glob("slosh_from*.npz"), key=lambda p: int(p.stem[len("slosh_from"):]))  # resumes append segments
    S = [s for p in segs for s in np.load(p, allow_pickle=True)["slosh"]]
    S = list({int(s["t"]): s for s in S}.values())  # a resume repeats its start tick
    g = float(np.load(d / "rec.npz")["g"]) if (d / "rec.npz").exists() else 0.0082185
    out = log_name(d)
    with open(out, "w") as f:
        f.write(f"# run {d}\n# g {g:.8g} SH_DR 2.0\n# series: t D_in rho_mid rc zc D_tot [E_in E_tot Et_in] [dz_rel rcp]\n")
        for s in S:
            ln = f"S {s['t']} {s['D_in']:.3f} {s['rho_mid']:.7f} {s['rc']:.4f} {s['zc']:.4f} {s['D_tot']:.3f}"
            if "E_in" in s:
                ln += f" {s['E_in']:.5f} {s['E_tot']:.5f} {s['E_in'] + 0.5 * g * s['D_in']:.5f}"
            if "dz_rel" in s:
                ln += f" dz {s['dz_rel']:.4f} {s['rcp']:.4f}"
            f.write(ln + "\n")
        for s in S:
            if "esh" in s:
                k = np.nonzero(s["nsh"])[0]
                f.write(f"H {s['t']} " + " ".join(f"{s['dsh'][j]:.2f}:{s['esh'][j]:.5f}:{s['nsh'][j]}" for j in k) + "\n")
    return out


def load(path):
    """-> dict(t, D, rho_mid, rc, zc, tE, E_in, E_tot, Et, g, shells {t: (dsh, esh, nsh)}) from a run dir or a log."""
    p = Path(path)
    if p.is_dir():
        p = log_name(p) if log_name(p).exists() else dump(p)
    g, rows, rowsE, sh, rowsP = None, [], [], {}, []
    for ln in open(p):
        f = ln.split()
        if ln.startswith("# g"):
            g = float(f[2])
        elif f and f[0] == "S":
            rows.append([float(v) for v in f[1:7]])
            if len(f) >= 10 and f[7] != "dz":
                rowsE.append([float(f[1])] + [float(v) for v in f[7:10]])
            if "dz" in f:
                j = f.index("dz")
                rowsP.append([float(f[1]), float(f[j + 1]), float(f[j + 2])])
        elif f and f[0] == "H":
            a = np.array([[float(x) for x in c.split(":")] for c in f[2:]])
            sh[int(f[1])] = (a[:, 0], a[:, 1], a[:, 2].astype(int))
    R, E = np.array(rows), np.array(rowsE)
    return dict(t=R[:, 0], D=R[:, 1], rho_mid=R[:, 2], rc=R[:, 3], zc=R[:, 4], D_tot=R[:, 5], tE=E[:, 0], E_in=E[:, 1],
                E_tot=E[:, 2], Et=E[:, 3], g=g, shells=sh,
                tP=np.array(rowsP)[:, 0] if rowsP else None, dz_rel=np.array(rowsP)[:, 1] if rowsP else None,
                rcp=np.array(rowsP)[:, 2] if rowsP else None)


if __name__ == "__main__":
    for r in sys.argv[1:]:
        print(dump(r))
