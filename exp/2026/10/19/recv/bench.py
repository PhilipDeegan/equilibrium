import sys, time
import numpy as np
from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

P, nt = int(sys.argv[1]), int(sys.argv[2])
alat = cpp_box(P, n_threads=nt)
ac = C.AnchorCluster(alat)
rng = np.random.default_rng(0)
ac.u, ac.phi = np.zeros(ac.N), C._antisym(ac, 1e-3 * rng.standard_normal((ac.N, 14)))
dt = 0.4 / float(np.sum(C.W**2))
alat.step_cartan(0, dt, 1.0, 2)
t0 = time.time()
for _ in range(8):
    alat.step_cartan(0, dt, 1.0, 2)
print(f"P {P} N {ac.N} threads {nt}: {(time.time() - t0) / 8:.3f} s/tick", flush=True)
