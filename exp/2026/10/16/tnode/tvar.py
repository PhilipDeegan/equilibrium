import numpy as np

import tdisp as T_
from tdisp import D, DH, E, MIR

base_w = np.asarray(T_.C.LAP_W, float)


def S_var(cols, rhex):
    w = np.where(np.arange(14) < 8, rhex, 1.0) * (base_w > 0)
    w = w / w.sum()
    M = np.zeros((28, 6))
    for k in range(14):
        for p in range(2):
            M[2 * k + p, :3] = np.sqrt(w[k]) * E[k, p]
            M[2 * k + p, 3:] = np.sqrt(w[k]) * np.cross(DH[k], E[k, p])
    M = M[:, cols]
    Q, _ = np.linalg.qr(M)
    return 2 * Q @ Q.T - np.eye(28), M


def summary(S, M):
    res = {}
    for name, dv in (("100", np.array([1, 0, 0.])), ("111", np.ones(3) / np.sqrt(3)), ("110", np.array([1, 1, 0.]) / np.sqrt(2))):
        q = 0.02 * dv
        ev, vec = np.linalg.eig(T_.T(q, S))
        om = np.angle(ev)
        mv = np.where((om > 1e-6) & (om < 0.5 * np.pi))[0]
        r = []
        for i in mv:
            f = np.linalg.lstsq(M, vec[:, i], rcond=None)[0][:3] if M.shape[1] >= 3 else None
            lon = abs(f @ dv) / max(np.linalg.norm(f), 1e-30) if f is not None else np.nan
            r.append((om[i] / 0.02, lon))
        res[name] = sorted(r)
    return res


for lab, cols in (("E only", [0, 1, 2]), ("H only", [3, 4, 5]), ("E+H", list(range(6)))):
    for rh in (0.5, 1.0, 2.0, 4.0):
        S, M = S_var(cols, rh)
        r = summary(S, M)
        print(f"{lab:6s} hex:sq {rh:4.1f} | " + " | ".join(f"{k}: " + " ".join(f"{v:.3f}({l:.1f})" for v, l in r[k]) for k in r))
