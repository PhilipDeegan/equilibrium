import numpy as np
from equilibrium.substrate import continuous as C
ac = C.anchor_cluster(30.0, 1.0, n_threads=4)
B, kap = 0.05, 1.0
bg = C.project_div_free(ac, C.sample_A(ac, lambda x: 0.5 * B * np.stack([-x[:, 1], x[:, 0], 0 * x[:, 0]], 1)))
T = ac.cartan_transport(bg)
Tdf, psi = C.project_div_free(ac, T, return_psi=True)
print(f"|T| {np.linalg.norm(T):.3e}  div-free part {np.linalg.norm(Tdf):.3e}")
dt = C.stable_dt(ac)
for sgn in (0, 1, -1):
    ac.u, ac.phi = sgn * kap * psi, bg
    for _ in range(330):
        ac.alat.step_cartan(ac.i, dt, kap, 3)
    print(f"sign {sgn:+d}: phi drift {np.linalg.norm(ac.phi - bg) / np.linalg.norm(bg):.3e}  "
          f"u drift {np.linalg.norm(ac.u - sgn * kap * psi):.3e} (|psi| {np.linalg.norm(psi):.3e})", flush=True)
