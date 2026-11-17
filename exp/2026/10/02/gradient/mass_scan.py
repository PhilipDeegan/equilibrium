"""Background stability vs mass m (plain turn, xi fixed -> g = c^2 / (2 m xi^2)): max departure growth per tick over q
(nq^3 grid + refinement). Also the same in units of the structure's own time (growth / g ~ per GP time unit)."""
import sys
import numpy as np
from scipy.optimize import minimize
src = open("variants_dense.py").read()
src = src[src.index("hu = HS"):src.index("ax_ = np.linspace")]
from equilibrium.substrate.walk import A_REAL, HS, J4, healing_g, real, rest_state
nq = int(sys.argv[1]) if len(sys.argv) > 1 else 32
xi = float(sys.argv[2]) if len(sys.argv) > 2 else 3.9
ax_ = np.linspace(-np.pi, np.pi, nq, endpoint=False) + np.pi / nq
Q = np.stack(np.meshgrid(ax_, ax_, ax_, indexing="ij"), -1).reshape(-1, 3)
rows = []
for m in [float(a) for a in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['0.02','0.05','0.1','0.15','0.2','0.25','0.3','0.35','0.4','0.45','0.5','0.55','0.6','0.65','0.7','0.75','0.8','0.85','0.9','0.95'])]:
    n = np.sqrt(1 - m * m)
    ns = dict(np=np, A_REAL=A_REAL, HS=HS, J4=J4, real=real, rest_state=rest_state, m=m, n=n)
    exec(src, ns)
    gg = healing_g(m, xi)
    a = np.concatenate([ns["growth"](Q[i:i + 20000], gg, "plain") for i in range(0, len(Q), 20000)])
    best = a.max()
    for i in np.argsort(a)[-8:]:
        r = minimize(lambda q: -ns["growth"](q[None], gg, "plain")[0], Q[i], method="Nelder-Mead",
                     options=dict(xatol=1e-9, fatol=1e-16, maxiter=2000))
        best = max(best, -r.fun)
    rows.append((m, gg, best))
    print(f"m {m:.2f}: g {gg:.4f}, max growth {best:.2e}/tick, growth/g {best / gg:.3f}, unstable grid fraction "
          f"{(a > 1e-12).mean():.1e}", flush=True)
np.save(f"mass_scan_xi{xi:g}_nq{nq}{'_' + sys.argv[3] if len(sys.argv) > 3 else ''}.npy", np.array(rows))
