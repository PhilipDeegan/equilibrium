import sys
from equilibrium import lattice as L
from equilibrium.substrate import continuous as C
import numpy as np
n_threads = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ac = C.anchor_cluster(16.0, 1.0, n_threads=n_threads)
rng = np.random.default_rng(1)
ac.u, ac.phi = rng.standard_normal(ac.N), C._antisym(ac, 0.3 * rng.standard_normal((ac.N, 14)))
dt = 0.3 * C.stable_dt(ac)
L.start_phlop()
for _ in range(50):
    ac.alat.step_cartan(ac.i, dt, 0.7, 3)
L.stop_phlop()
print("N", ac.N)
