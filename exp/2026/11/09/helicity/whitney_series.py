"""Whitney (all-DOF) helicity per dump: H_W / H_W(0); split vector part (edge a_vec = 1/2 (A_i + A_j) . d, A = LS
vector per cell) / rest (the hidden DOF) / cross; coarse (curl-curl smoothed, tau) / fine with + / - density parts;
instantaneous Cartan rate d ln H_W / dt = [H(phi, T) + H(T, phi)] / H, T = cartan_transport(phi) (kappa 1; step()'s
gradient part cannot change H_W: gauge-exact); finite-difference rate from consecutive dumps as a cross-check.
usage: whitney_series.py DUMPDIR --P=96 [--tau=1] [--every=1]"""
import sys
from pathlib import Path
import numpy as np
from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box
sys.path.insert(0, str(Path(__file__).parent))
import whitney as Wt

C.spacing_of = lambda lat: 1.0
opt = dict(a[2:].split("=") for a in sys.argv[1:] if a.startswith("--"))
P_, tau, every = int(opt.get("P", 96)), float(opt.get("tau", 1.0)), int(opt.get("every", 1))
Dd = Path(next(a for a in sys.argv[1:] if not a.startswith("--")))
ac = C.AnchorCluster(cpp_box(P_, n_threads=4))
w = Wt.Whitney(ac)
CH = np.array_split(np.arange(ac.N), max(1, ac.N // 65536))
nbs = np.where(ac.nb >= 0, ac.nb, 0)


def smooth(phi, eps=0.02):
    for _ in range(int(round(tau / eps))):
        phi = phi - eps * ac.curl_curl_dec(phi)
    return phi


def vec_part(phi):
    A = np.concatenate([C.reconstruct_A_at(ac, i, phi[i]) for i in CH])
    out = np.empty_like(phi)
    for k in range(14):
        out[:, k] = C.W[k] * 0.5 * np.einsum("nd,d->n", A + A[nbs[:, k]], C.D_HAT[k])
    return np.where(ac.nb >= 0, out, 0.0)


files = sorted(Dd.glob("phi_t*.npz"))[::every]
H0 = prev = None
print(f"# {Dd} P {P_} tau {tau}; all H as H_W / H_W(t0)")
print("t       H_W     Hvec    Hrest   Hcross  Hs      Hf      Hf+     Hf-    | rate_Cartan  rate_fd   (d ln H/dt)")
for f in files:
    t = float(f.stem[5:])
    phi = np.load(f)["phi"].astype(float)
    H = w.H(phi)
    H0 = H if H0 is None else H0
    pv = vec_part(phi)
    pr = phi - pv
    Hv, Hr = w.H(pv), w.H(pr)
    ps = smooth(phi)
    pf = phi - ps
    df = w.density(pf)
    T = ac.cartan_transport(phi)
    rate = (w.H(phi, T) + w.H(T, phi)) / H
    fd = "" if prev is None else f"{np.log(H / prev[1]) / (t - prev[0]):+.5f}"
    prev = (t, H)
    print(f"{t:6.1f}  {H / H0:6.3f}  {Hv / H0:6.3f}  {Hr / H0:6.3f}  {(H - Hv - Hr) / H0:6.3f}  {w.H(ps) / H0:6.3f}  "
          f"{df.sum() / H0:6.3f}  {df[df > 0].sum() / H0:6.3f}  {df[df < 0].sum() / H0:6.3f} | {rate:+.5f}  {fd}", flush=True)
