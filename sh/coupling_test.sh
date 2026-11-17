#!/usr/bin/env bash
# Test where phi and chi must interact — looking for the coupling point
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" && cd "$CWD"/..
set -e
export PYTHONPATH=$PWD

# --- Test 1: mid-run chi injection ---
echo "=== TEST 1: mid-run chi injection ==="
echo "Run half, inject massive chi blob at r=10, run other half"
python3 -Ou - <<'PYEOF'
import sys, numpy as np
sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from run.substrate.solar_sandbox import *

# reuse solar_sandbox functions with default args
args.n_steps = 40000
args.report  = 5000
args.save_prefix = ""

alat2 = _lat.AdaptiveLattice()
alat2.add_cluster()
cl2 = alat2[0]
_lat.build_bcc_sphere(cl2, args.R, d1=D1, u0=0.0, ell0=1.0, lam0=0)
alat2.rebuild(0)
alat2.set_n_threads(args.threads)
info2 = _lat.DSTInfo(); info2.alpha = args.alpha; info2.split_hi = 0.0; info2.split_lo = 0.0; info2.max_level = 0
cl2.info = info2
apply_chi(cl2); apply_phi(cl2)
cl2.set_bc(_lat.BoundaryCondition.neumann())
dt2 = compute_dt(cl2, D1=D1)

L0 = None
for s in range(1, args.n_steps + 1):
    # Inject a massive chi blob at step 20000
    if s == 20000:
        print("  >>> injecting chi blob at r=10, chi=0.4 <<<", flush=True)
        pos = np.array(cl2.leaf_pos)
        ell = np.array(cl2.leaf_ell)
        centre = np.array([10.0, 0.0, 0.0])
        dist   = np.linalg.norm(pos - centre, axis=1)
        bump   = 0.4 * np.exp(-0.5*(dist/2.0)**2)
        ell    = np.maximum(D1/16, ell - bump)
        cl2.set_leaf_ell(ell)

    alat2.step(0, dt2)

    if s % args.report == 0:
        r_xy  = np.linalg.norm(np.array(cl2.leaf_pos)[:,:2], axis=1)
        phi_a = np.array(cl2.leaf_phi)
        pos_a = np.array(cl2.leaf_pos)
        safe  = np.maximum(r_xy, 1e-9)
        tx    = -pos_a[:,1]/safe; ty = pos_a[:,0]/safe
        L = sum(float(np.sum(FACE_W[k]*phi_a[k]*(FACE_DIRS[k,0]*tx+FACE_DIRS[k,1]*ty))) for k in range(14))
        if L0 is None and abs(L)>1e-12: L0=abs(L)
        ratio = abs(L)/L0 if L0 else float('nan')
        p5  = float(np.mean([sum(FACE_W[k]*phi_a[k,m]*(FACE_DIRS[k,0]*tx[m]+FACE_DIRS[k,1]*ty[m]) for k in range(14)) for m in np.where(np.abs(r_xy-5)<2)[0]])) if np.any(np.abs(r_xy-5)<2) else float('nan')
        p10 = float(np.mean([sum(FACE_W[k]*phi_a[k,m]*(FACE_DIRS[k,0]*tx[m]+FACE_DIRS[k,1]*ty[m]) for k in range(14)) for m in np.where(np.abs(r_xy-10)<2)[0]])) if np.any(np.abs(r_xy-10)<2) else float('nan')
        print(f"  {s:6d}  L/L0={ratio:.4f}  L={L:.1f}  p5={p5:.4f}  p10={p10:.4f}", flush=True)
PYEOF

echo ""
echo "=== TEST 2: collision — two chi islands approaching ==="
python3 -Ou run/substrate/solar_sandbox.py \
    --R 40 --r-sun 2 --chi-sun 0.6 --chi-ism 0.005 \
    --planets "8:0.4,9:0.0001,10:0.0001,11:0.0001" \
    --v-circ 0.05 --alpha 0.005 \
    --n-steps 50000 --report 10000 --threads 6 \
    --save-prefix /tmp/collision 2>&1 \
    | grep -E "^\s+[0-9]+|early|planet|v_circ"

echo ""
echo "=== TEST 3: two separate solar systems, offset by 20 cells ==="
# Second sun at (20,0,0) — two suns in the same domain
python3 -Ou - <<'PYEOF'
import sys, numpy as np
sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt

D1=1.0; L_MIN=_lat.DiscreteCell.L_MIN
_R3I=0.5773502691896258
FACE_DIRS=np.array([[-_R3I,-_R3I,-_R3I],[-_R3I,-_R3I,+_R3I],[-_R3I,+_R3I,-_R3I],[-_R3I,+_R3I,+_R3I],[+_R3I,-_R3I,-_R3I],[+_R3I,-_R3I,+_R3I],[+_R3I,+_R3I,-_R3I],[+_R3I,+_R3I,+_R3I],[-1.,0.,0.],[+1.,0.,0.],[0.,-1.,0.],[0.,+1.,0.],[0.,0.,-1.],[0.,0.,+1.]])
FACE_W=np.array([3.,3.,3.,3.,3.,3.,3.,3.,1.,1.,1.,1.,1.,1.])

alat = _lat.AdaptiveLattice(); alat.add_cluster(); cl = alat[0]
_lat.build_bcc_sphere(cl, 40.0, d1=D1, u0=0.0, ell0=1.0, lam0=0)
alat.rebuild(0); alat.set_n_threads(6)
info=_lat.DSTInfo(); info.alpha=0.005; info.split_hi=0.0; info.split_lo=0.0; info.max_level=0
cl.info=info

pos=np.array(cl.leaf_pos)
# Sun A at origin, Sun B at (20,0,0)
r_A=np.linalg.norm(pos,axis=1)
r_B=np.linalg.norm(pos-np.array([20.,0.,0.]),axis=1)
chi  = 0.6*np.exp(-r_A/3) + 0.6*np.exp(-r_B/3)
chi  = np.clip(np.maximum(chi,0.005), 0.0, 1.0-L_MIN)
cl.set_leaf_ell(np.maximum(L_MIN, 1.0-chi))

# Rotation around origin only — one-sided rotation
r_xy=np.linalg.norm(pos[:,:2],axis=1); safe=np.maximum(r_xy,1e-9)
v=0.05*np.where(r_xy<=3,r_xy/3,np.sqrt(3/safe))
tx=-pos[:,1]/safe; ty=pos[:,0]/safe
phi=np.array(cl.leaf_phi)
for k in range(14):
    proj=FACE_DIRS[k,0]*tx+FACE_DIRS[k,1]*ty
    phi[k]+=v*proj
cl.set_leaf_phi(phi)
cl.set_bc(_lat.BoundaryCondition.neumann())

dt=compute_dt(cl,D1=D1); L0=None
print(f"two-sun system: N={cl.n_leaves} dt={dt:.3e}")
for s in range(1, 50001):
    alat.step(0,dt)
    if s%10000==0:
        phi_a=np.array(cl.leaf_phi); pos_a=np.array(cl.leaf_pos)
        r_xy2=np.linalg.norm(pos_a[:,:2],axis=1); safe2=np.maximum(r_xy2,1e-9)
        tx2=-pos_a[:,1]/safe2; ty2=pos_a[:,0]/safe2
        L=sum(float(np.sum(FACE_W[k]*phi_a[k]*(FACE_DIRS[k,0]*tx2+FACE_DIRS[k,1]*ty2))) for k in range(14))
        if L0 is None and abs(L)>1e-12: L0=abs(L)
        print(f"  {s:6d}  L/L0={abs(L)/L0:.4f}  L={L:.1f}", flush=True)
PYEOF

echo ""
echo "=== coupling tests done ==="
