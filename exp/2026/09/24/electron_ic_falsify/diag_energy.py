"""Why did E grow 3x under the linear step for the t2 torus IC? Check phi
antisymmetry of the seeded IC and energy drift under neumann vs no BC."""
import sys
sys.path.insert(0, "/home/deegan/git/eq")
import numpy as np
from equilibrium import lattice as _lat
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0
from tst.substrate.adaptive.adaptive_flux_tube import _FACE_HOP, _MIRROR, leaf_nb_j

R, R0, A, m = 30.0, 15.0, 8.0, 4
def fields(x, y, z):
    rho = np.hypot(x, y); rp = np.hypot(rho - R0, z)
    g = np.where(rp < A, np.cos(0.5*np.pi*rp/A)**2, 0.0); th = np.arctan2(y, x)
    s = np.where(rho > 1e-9, rho, 1.0)
    return g, th, np.stack([-y/s, x/s, 0*x], -1)

for bc in ["none", "neumann"]:
    alat = _lat.AdaptiveLattice(); alat.add_cluster(); alat.set_n_threads(8); cl = alat[0]
    _lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, lam0=0.0); alat.rebuild(0)
    pos = np.array(cl.leaf_pos); N = cl.n_leaves; nbj = leaf_nb_j(cl)
    g, th, _ = fields(*pos.T); u = g*np.cos(m*th)
    phi = np.zeros((14, N))
    for k in range(14):
        gm, thm, t = fields(*(pos + _FACE_HOP[k]).T)
        phi[k] = gm*np.cos(m*thm)*(t @ _FACE_HOP[k])
    # antisymmetry residual and neighbour-geometry check
    res = []; geo = []
    for k in range(14):
        j = nbj[:, k]; ok = j >= 0
        res.append(np.abs(phi[k, ok] + phi[_MIRROR[k], j[ok]]).max())
        geo.append(np.abs(pos[j[ok]] - pos[ok] - 2*_FACE_HOP[k]).max())
    print(f"[{bc}] max|phi_ik + phi_jk'| = {max(res):.3e}  (max|phi|={np.abs(phi).max():.3e})"
          f"  max|dpos - 2*hop| = {max(geo):.3e}", flush=True)
    cl.set_leaf_u(u); cl.set_leaf_phi(phi)
    if bc == "neumann": cl.set_bc(_lat.BoundaryCondition.neumann())
    dt = _compute_dt_l0(cl, D1=1.0); ts = _lat.TimeStepper.by_level(dt)
    ev = _lat.AdaptiveLatticeEvolver(alat)
    E = lambda: float(np.sum(np.asarray(cl.leaf_u)**2) + 0.5*np.sum(np.asarray(cl.leaf_phi)**2))
    E0 = E(); out = []
    for s in range(1, 601):
        ev.step(ts)
        if s % 100 == 0: out.append(f"{E()/E0:.4f}")
    print(f"[{bc}] E/E0 every 100 steps: {' '.join(out)}", flush=True)
