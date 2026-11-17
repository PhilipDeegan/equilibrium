import numpy as np

import tdisp as T_
from tdisp import DH, E

w = np.asarray(T_.C.LAP_W, float)
w = w / w.sum()
ME = np.zeros((28, 3)); MH = np.zeros((28, 3))
for k in range(14):
    for p in range(2):
        ME[2 * k + p] = np.sqrt(w[k]) * E[k, p]
        MH[2 * k + p] = np.sqrt(w[k]) * np.cross(DH[k], E[k, p])
QE = np.linalg.qr(ME)[0]; QH = np.linalg.qr(MH)[0]
PE, PH = QE @ QE.T, QH @ QH.T
print("E.H overlap", np.abs(QE.T @ QH).max())
R = np.eye(28) - PE - PH
M = np.hstack([ME, MH])
for sE in (1, -1):
    for sH in (1, -1):
        for sR in (1, -1):
            S = sE * PE + sH * PH + sR * R
            out = []
            for name, dv in (("100", np.array([1, 0, 0.])), ("111", np.ones(3) / np.sqrt(3)), ("110", np.array([1, 1, 0.]) / np.sqrt(2)), ("123", np.array([1, 2, 3.]) / np.sqrt(14))):
                ev, vec = np.linalg.eig(T_.T(0.02 * dv, S))
                om = np.angle(ev)
                mv = np.where((om > 1e-6) & (om < 0.5 * np.pi))[0]
                r = []
                for i in mv:
                    f = np.linalg.lstsq(M, vec[:, i], rcond=None)[0]
                    Ef, Hf = f[:3], f[3:]
                    nE = np.linalg.norm(Ef)
                    r.append(f"{om[i] / 0.02:.3f}{'L' if nE > 1e-9 and abs(Ef @ dv) / nE > 0.5 else 'T'}")
                out.append(f"{name}: {' '.join(sorted(r))}")
            print(f"sE {sE:+d} sH {sH:+d} sR {sR:+d} | " + " | ".join(out))
