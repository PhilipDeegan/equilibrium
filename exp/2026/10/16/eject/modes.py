import numpy as np

from disp import T, D, MIR, w

for y in (0.0, 1.0):
    q = np.array([0.1, 0.03, 0.0])
    ev, vec = np.linalg.eig(T(q, y))
    om = np.angle(ev)
    print(f"y {y}, q {q}")
    for i in np.argsort(np.abs(om)):
        v = vec[:, i]
        a = v[:14]
        V = np.sum(w * a) + y * v[14]
        J = np.sum((w * a)[:, None] * D, 0)
        Jl = abs(J @ q) / np.linalg.norm(q)
        Jt = np.linalg.norm(J - (J @ q) * q / (q @ q))
        sym = np.linalg.norm(a + a[MIR]) / np.linalg.norm(a)
        print(f"  omega {om[i]:+.3f}  |V| {abs(V):.3f}  |stub| {abs(v[14]):.3f}  |J_long| {Jl:.3f}  |J_trans| {Jt:.3f}  |a + a_mirror|/|a| {sym:.3f}")
