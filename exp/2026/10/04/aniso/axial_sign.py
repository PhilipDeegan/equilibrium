"""sign of the cubic k^4 term: om(k e_100) - om(k e_111) for XXp and the axial palindrome pair; required mix."""
import numpy as np

from axial_mix import X, Xp, Sc, om, m

o0 = np.arcsin(m)
e100, e110, e111 = np.array([1, 0, 0.]), np.array([1, 1, 0.]) / np.sqrt(2), np.ones(3) / np.sqrt(3)
D = lambda k: Xp(k) @ X(k)
A = lambda k: Sc(k, (2, 1, 0)) @ Sc(k, (0, 1, 2))
for kk in [0.05, 0.1, 0.2]:
    for name, f in (("XXp", D), ("CC'", A)):
        w = [om(f(kk * e), 2)[1] - o0 for e in (e100, e110, e111)]
        print(f"|k| {kk} {name}: kin(100) {w[0]:.4e} (100)-(111) {w[0] - w[2]:+.3e} rel {(w[0] - w[2]) / w[0]:+.3e} "
              f"(110)-(111) {w[1] - w[2]:+.3e}")
