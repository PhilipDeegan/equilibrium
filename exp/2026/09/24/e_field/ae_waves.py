import numpy as np, time, sys
from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate import continuous as C

C.set_weights("legacy")  # written for W = 3:1 on both legs (pre-2026-09-28; see continuous.WEIGHTS)
R,w,T,x0=8.0,1.2,0.6,-1.0
f=lambda s: np.exp(-0.5*((s-x0)/w)**2)
fp=lambda s: -(s-x0)/w**2*np.exp(-0.5*((s-x0)/w)**2)
c=C.C0
def run(h, kind, use_E, dtfac=1.0):
    lat=SubstrateLattice(radius_cells=R,spacing=h)
    dt=C.stable_dt(lat)*dtfac; n=int(np.ceil(T/dt)); dt=T/n
    if kind=="long":
        uf,Af=C.plane_wave([1,0,0],f); Ef=lambda x,t: np.zeros_like(x)
    else:
        uf=lambda x,t: np.zeros(len(x))
        Af=lambda x,t: np.stack([0*x[:,0], f(x[:,0]-c*t)/c, 0*x[:,0]],1)
        Ef=lambda x,t: np.stack([0*x[:,0], fp(x[:,0]-c*t), 0*x[:,0]],1)
    C.seed(lat,uf,Af,dt=dt)
    if use_E:
        ae=C.AE(lat, C.sample_A(lat, lambda x: Ef(x,0.0))); ae.advance(dt,n)
    else:
        C.advance(lat,dt,n,n_threads=8)
    r=np.linalg.norm(lat.pos,axis=1); m=r<R-C.C0*T-1.5
    Ae=Af(lat.pos,T+0.5*dt); Ar=C.reconstruct_A(lat)
    eA=np.sqrt(np.mean((Ar[m]-Ae[m])**2)/np.mean(Ae[m]**2))
    ue=uf(lat.pos,T)
    eu=np.sqrt(np.mean((lat.u[m]-ue[m])**2)/np.mean(ue[m]**2)) if kind=="long" else np.abs(lat.u[m]).max()
    return eA, eu
dtfac=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
for kind in ("trans","long"):
    for use_E in (False,True):
        for h in (0.5,0.25):
            t0=time.time()
            with np.errstate(all="raise"):
                try: eA,eu=run(h,kind,use_E,dtfac)
                except FloatingPointError: eA,eu=np.nan,np.nan
            print(f"{kind:5s} E={use_E!s:5s} h={h} errA={eA:.3e} u={eu:.3e} ({time.time()-t0:.0f}s)",flush=True)
