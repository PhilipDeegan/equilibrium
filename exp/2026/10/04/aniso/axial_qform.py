"""quadratic form Q of the kinetic part, kin(k) ~ k.Q.k, fitted at |k| 0.02, for one-axis tick cycles."""
import numpy as np

from axial_ticks import Ca, seq, kin, X, Xp

e = np.eye(3)
pairs = [(i, j) for i in range(3) for j in range(i + 1, 3)]


def Q(f, t, kk=0.02):
    q = np.zeros((3, 3))
    for i in range(3):
        q[i, i] = kin(f(kk * e[i]), t) / kk**2
    for i, j in pairs:
        p = kin(f(kk * (e[i] + e[j]) / np.sqrt(2)), t) / kk**2
        mm = kin(f(kk * (e[i] - e[j]) / np.sqrt(2)), t) / kk**2
        q[i, j] = q[j, i] = (p - mm) / 2
    return q


x, y, z = (Ca(a) for a in range(3))
xp, yp, zp = (Ca(a, True) for a in range(3))
for name, s in {"XXp": [X, Xp], "xyz": [x, y, z], "xyzzyx": [x, y, z, z, y, x], "xyz xp yp zp": [x, y, z, xp, yp, zp],
                "xxp yyp zzp": [x, xp, y, yp, z, zp]}.items():
    print(name, "\n", np.round(Q(*seq(s)), 4))
