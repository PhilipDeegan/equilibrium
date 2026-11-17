"""Growth of a random field under continuous.AE (no source) vs dt factor."""
import sys
import numpy as np
from equilibrium.substrate import continuous as C

C.set_weights("legacy")  # written for W = 3:1 on both legs (pre-2026-09-28; see continuous.WEIGHTS)
from equilibrium.substrate.lattice import SubstrateLattice

lat = SubstrateLattice(radius_cells=6.0, spacing=0.5)
rng = np.random.default_rng(0)
NSTEP = int(sys.argv[1])
for fac in [float(a) for a in sys.argv[2:]]:
    for clean in (False,):
        lat.u = np.zeros(lat.N)
        lat.phi = C.project_div_free(lat, C.sample_A(lat, lambda x: rng.standard_normal(x.shape)), tol=1e-10)
        ae = C.AE(lat, clean=clean)
        dt = C.stable_dt(lat) * fac
        n0 = np.sqrt((lat.phi**2).sum() + (ae.e**2).sum())
        out = []
        for i in range(1, NSTEP + 1):
            ae.step(dt)
            if i % (NSTEP // 6) == 0:
                out.append(np.sqrt((lat.phi**2).sum() + (ae.e**2).sum()) / n0)
        print(f"dt_fac={fac} clean={clean}: norm ratio at 6 checkpoints of {NSTEP} steps: " + " ".join(f"{v:.3e}" for v in out), flush=True)
