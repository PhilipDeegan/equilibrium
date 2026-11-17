"""parity-image compensation: second tick = B X(-k) B (B swaps the L / R halves, Dirac parity in the chiral basis),
and variants. Relative anisotropy = spread of the upper frequency over k directions / (mean - arcsin m)."""
import numpy as np
from compensate import S, om, R, m

I2, Z2 = np.eye(2), np.zeros((2, 2))
B = np.block([[Z2, I2], [I2, Z2]])
V = {
    "X": (lambda k: S(k), 1),
    "inv": (lambda k: S(-k) @ S(k), 2),
    "parity": (lambda k: B @ S(-k) @ B @ S(k), 2),
    "parity-": (lambda k: B @ S(-k, -1) @ B @ S(k), 2),
    "swapLR": (lambda k: B @ S(k) @ B @ S(k), 2),
    "swapLR-": (lambda k: B @ S(k, -1) @ B @ S(k), 2),
    "Y": (lambda k: S(k, -1), 1),
}
rng = np.random.default_rng(1)
dirs = rng.normal(size=(400, 3))
dirs /= np.linalg.norm(dirs, axis=1)[:, None]
o0 = np.arcsin(m)
for kk in [0.1, 0.2, 0.3, 0.5]:
    print(f"|k| = {kk}")
    for name, (f, t) in V.items():
        w = np.array([om(f(kk * d), t) for d in dirs])
        kin = w[:, 1].mean() - o0
        two = max(max(np.abs(om(f(kk * (R(a) @ d)), t) - om(f(kk * d), t)).max() for d in dirs[:100]) for a in "xyz")
        print(f"  {name:8s} kinetic {kin:.3e}  rel spread {np.ptp(w[:, 1]) / kin:.2e}  rel 90-deg asym {two / kin:.2e}")
