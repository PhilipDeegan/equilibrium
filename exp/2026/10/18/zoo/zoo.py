"""10-18 zoo: many candidate structures under one rule (step() + Cartan, kappa 1, C++ alat.step_cartan), one scorecard.
A structure = closed vortex curves, each with integer circulation n_i (whole turns): theta = sum n_i Omega_i / 2
(solid angle, periodic images), phi = W wrap(d theta), smoothed (curl curl diffusion, tau) + div-free projection.
Scorecard (generic, no ring geometry): |Gamma|-weighted centroid (travel), rms2 distance of |Gamma| to the initial
curves rigidly shifted by the centroid displacement (shape kept), leak = |Gamma| weight beyond rcut from them,
sum|Gamma| (total circulation magnitude), E_lf (energy)."""
import argparse
import time
from pathlib import Path

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box, solid_angle

C.spacing_of = lambda lat: 1.0
p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=72)
p.add_argument("--R", type=float, default=9.0, help="structure scale (cells)")
p.add_argument("--struct", default="ring")
p.add_argument("--n", type=int, default=1, help="circulation per curve (turns)")
p.add_argument("--T", type=float, default=20.0)
p.add_argument("--every", type=float, default=1.0)
p.add_argument("--smooth", type=float, default=1.0, help="diffusion time (scales as (R / 9)^2)")
p.add_argument("--eps", type=float, default=0.02)
p.add_argument("--kappa", type=float, default=1.0)
p.add_argument("--threads", type=int, default=1)
p.add_argument("--rcut", type=float, default=3.0, help="leak radius in units of R / 9")
p.add_argument("--seed", default="")
p.add_argument("--save", default="")
p.add_argument("--ckpt", default="", help="resume / checkpoint file (default: <seed>_ckpt.npz when --seed is given)")
p.add_argument("--tilt", type=float, default=0.0, help="face throughput sigma = 1 + tilt sin(2 pi x / side) (gradient across the structure)")
p.add_argument("--dip", type=float, nargs=3, default=None, metavar=("DEPTH", "DIST", "WIDTH"),
               help="receiver stand-in: sigma -= DEPTH exp(-|x - xr|^2 / 2 WIDTH^2), xr at DIST along +x from the structure")
p.add_argument("--csig", action="store_true", help="sigma also scales the Cartan transport (per triangle, mean edge sigma)")
p.add_argument("--vcap", type=float, default=0.0, help="smooth carrier bound in the Cartan transport (0 = off)")
p.add_argument("--scap", type=float, default=0.0, help="cell holding capacity S_c as a fraction of the seed's max S_j = 1/4 sum_k phi^2 (0 = off)")
p.add_argument("--cap_p", type=int, default=1, help="capacity exponent: G' = 1 + (S / S_c)^p")
p.add_argument("--relax", type=float, default=0.0, help="rest relaxation under the store before the run: step() only (kappa 0), u damped at rate 1 / --relax_tau, for this long; then u = 0")
p.add_argument("--relax_tau", type=float, default=1.0)
p.add_argument("--save_relaxed", default="")
p.add_argument("--Sc", type=float, default=0.0, help="absolute cell capacity S_c (overrides --scap)")
p.add_argument("--pulse", type=float, nargs=3, default=None, metavar=("AMP", "DIST", "WIDTH"),
               help="after relax: plane u slab AMP exp(-(x - x_r - DIST)^2 / 2 WIDTH^2) across the box (splits into +-x waves)")
p.add_argument("--ediv", action="store_true", help="print the store energy of the div-free (circulation) part of phi and core S")
p.add_argument("--smax", type=float, default=0.0, help="expected peak S / S_c for the dt bound (default: seed's)")
p.add_argument("--R2", type=float, default=0.0, help="side: second ring radius (default R)")
p.add_argument("--sep", type=float, default=0.0, help="side: centre distance along x (default side / 2)")
p.add_argument("--ang", type=float, default=45.0, help="side_tilt: each ring's normal tilted toward the other by this (deg)")
p.add_argument("--gap", type=float, default=0.25, help="side_tilt: gap between the near rims along x, in units of R")
p.add_argument("--ckpt_every", type=int, default=5, help="prints between checkpoints")
args = p.parse_args()
side = args.P / np.sqrt(3)
D, W = C.D_HAT, C.W
TWO_PI = 2 * np.pi
wrap = lambda a: (a + np.pi) % TWO_PI - np.pi
wrapx = lambda x: (x + side / 2) % side - side / 2
OFF = np.array([0.13, 0.07, 0.11])
R = args.R
s_ = np.linspace(0, TWO_PI, 400, endpoint=False)


def ring(r, z=0.0, sense=1, ax=2, c=(0, 0, 0)):
    q = np.stack([r * np.cos(sense * s_), r * np.sin(sense * s_), z + 0 * s_], 1)
    return np.roll(q, (ax - 2) % 3, 1) + np.array(c)


def tilted(r, side_sign):
    a = np.radians(args.ang) * -side_sign
    q = ring(r)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    x0 = r * np.cos(a) + 0.5 * args.gap * r
    return q @ rot.T + np.array([side_sign * x0, 0, 0])


def torus_knot(p_, q_, r1, r2):
    t = s_
    rr = r1 + r2 * np.cos(q_ * t)
    return np.stack([rr * np.cos(p_ * t), rr * np.sin(p_ * t), r2 * np.sin(q_ * t)], 1)


STRUCTS = {
    "ring": lambda: [(ring(R), 1)],
    "pair_co": lambda: [(ring(R, -0.6 * R), 1), (ring(R, 0.6 * R), 1)],
    "pair_counter": lambda: [(ring(R, -0.6 * R), 1), (ring(R, 0.6 * R, -1), 1)],
    "concentric": lambda: [(ring(R), 1), (ring(0.5 * R, sense=-1), 1)],
    "hopf_link": lambda: [(ring(R, c=(-0.5 * R, 0, 0)), 1), (ring(R, ax=1, c=(0.5 * R, 0, 0)), 1)],
    "trefoil": lambda: [(torus_knot(2, 3, R, 0.45 * R), 1)],
    "coil_1_3": lambda: [(torus_knot(1, 3, R, 0.35 * R), 1)],
    "coil_1_5": lambda: [(torus_knot(1, 5, R, 0.35 * R), 1)],
    "coil_3_2": lambda: [(torus_knot(3, 2, R, 0.35 * R), 1)],
    "side": lambda: [(ring(R, c=(-SEP / 2, 0, 0)), 1), (ring(args.R2 or R, c=(SEP / 2, 0, 0)), 1)],
    "side_opp": lambda: [(ring(R, c=(-SEP / 2, 0, 0)), 1), (ring(args.R2 or R, sense=-1, c=(SEP / 2, 0, 0)), 1)],
    "side_tilt": lambda: [(tilted(R, -1), 1), (tilted(R, 1), 1)],
    "side_a": lambda: [(ring(R, c=(-SEP / 2, 0, 0)), 1)],
    "side_b": lambda: [(ring(args.R2 or R, c=(SEP / 2, 0, 0)), 1)],
}
SEP = args.sep or side / 2
curves = [(c + OFF, k * args.n) for c, k in STRUCTS[args.struct]()]

alat = cpp_box(args.P, n_threads=args.threads)
ac = C.AnchorCluster(alat)
pos, nb, N = ac.pos, ac.nb, ac.N
shifts = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float) * side
phase = lambda X: sum(0.5 * k * solid_angle(c, X, shifts=shifts, n_threads=args.threads) for c, k in curves)
if args.seed and Path(args.seed).exists():
    phi = np.load(args.seed)["phi"]
else:
    rng = np.random.default_rng(0)
    jumps = []
    for ax in range(3):
        smp = (rng.random((400, 3)) - 0.5) * side
        smp[:, ax] = -side / 2
        hi = smp.copy()
        hi[:, ax] += side
        jumps.append(np.angle(np.mean(np.exp(1j * (phase(hi) - phase(smp))))))
    th = (phase(pos) - sum(jumps[a] * pos[:, a] / side for a in range(3))) % TWO_PI
    phi = W[None, :] * wrap(th[nb] - th[:, None])
    for it in range(int(round(args.smooth / args.eps))):
        phi = phi - args.eps * ac.curl_curl_dec(phi)
    phi = C.project_div_free(ac, phi)
    if args.seed:
        Path(args.seed).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.seed, phi=phi)

tri = []
for ka in range(8):
    for kb in range(8):
        for ks in range(8, 14):
            if np.allclose(D[kb] - D[ka], D[ks]):
                tri.append((ka, kb, ks))
x = wrapx(pos)
cen = np.stack([x + (D[ka] + D[kb]) / 3 for ka, kb, _ in tri], 1).reshape(-1, 3)
CP = np.concatenate([c[::4] for c, _ in curves])
kk = TWO_PI / side


def Gam(ph):
    a = ph / W
    return np.stack([a[:, ka] + a[nb[:, ka], ks] - a[:, kb] for ka, kb, ks in tri], 1).ravel() / TWO_PI


def centroid(w):
    return np.array([np.angle(np.sum(w * np.exp(1j * kk * cen[:, j]))) / kk for j in range(3)])


def dist_to_curves(shift):
    from scipy.spatial import cKDTree
    t = cKDTree(wrapx(CP + shift) % side, boxsize=side)
    return t.query(cen % side, workers=args.threads)[0]


rc = args.rcut * R / 9
w0 = np.abs(Gam(phi))
c0 = centroid(w0)
REF = np.concatenate([c for c, _ in curves]) - c0


def rigid_rms(w, dd, cnow, it_max=60):
    from scipy.spatial import cKDTree
    t_ = cKDTree(REF)
    m = (dd < rc) & (w > 1e-3 * w.max())
    X, ww = wrapx(cen[m] - cnow), w[m] ** 2
    Rm, tv = np.eye(3), np.zeros(3)
    for _ in range(it_max):
        Q = REF[t_.query((X - tv) @ Rm, workers=args.threads)[1]]
        mq, mx = ww @ Q / ww.sum(), ww @ X / ww.sum()
        U, _, Vt = np.linalg.svd((Q - mq).T @ ((X - mx) * ww[:, None]))
        Rn = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        tn = mx - Rn @ mq
        done = np.abs(Rn - Rm).max() < 1e-9 and np.abs(tn - tv).max() < 1e-9
        Rm, tv = Rn, tn
        if done:
            break
    d = t_.query((wrapx(cen - cnow) - tv) @ Rm, workers=args.threads)[0]
    ang = np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1)))
    return np.sqrt(np.sum(w**2 * d**2) / np.sum(w**2)), ang


rr0 = rigid_rms(w0, dist_to_curves(np.zeros(3)), c0)[0]
rr = None
ac.u, ac.phi = np.zeros(N), phi
if args.tilt or args.dip:
    fm = wrapx(pos[:, None, :] + 0.5 * D[None, :, :])
    dsig = args.tilt * np.sin(TWO_PI * fm[..., 0] / side)
    if args.dip:
        xr = np.array(OFF) + np.array([args.dip[1], 0, 0])
        dsig = dsig - args.dip[0] * np.exp(-np.sum(wrapx(fm - xr)**2, -1) / (2 * args.dip[2]**2))
    alat.set_face_medium(0, np.ascontiguousarray(dsig.T), np.zeros((14, N)))
    alat.set_cartan_sigma(args.csig)
    print(f"medium: sigma range {1 + dsig.min():.4f} .. {1 + dsig.max():.4f}", flush=True)
alat.set_cartan_vcap(args.vcap)
S_c = args.Sc or (args.scap * 0.25 * np.max(np.sum(phi**2, 1)) if args.scap else 0.0)
alat.set_cell_capacity(S_c, args.cap_p)


def gS(ph):
    S = 0.25 * np.sum(ph**2, 1)
    return 1 + (S / S_c) ** args.cap_p if S_c else np.ones(len(S))


def Gs(S):
    return S + (S_c / (args.cap_p + 1) * (S / S_c) ** (args.cap_p + 1) if S_c else 0.0)


SH = np.array([0.5, 1.0, 1.5, 2.0]) * R
HALF = args.struct.startswith("side")
if HALF:
    inA = wrapx(pos[:, 0] - OFF[0]) < 0
    inA_t = wrapx(cen[:, 0] - OFF[0]) < 0
dt = 0.4 / float(np.sum(W**2))
if S_c:
    xmax = args.smax or (1 / args.scap if args.scap and not args.Sc else 0.25 * np.max(np.sum(phi**2, 1)) / S_c)
    stiff = 1 + (2 * args.cap_p + 1) * max(xmax, 1.0) ** args.cap_p
    dt /= np.sqrt(stiff)
    print(f"capacity S_c {S_c:.4e} (scap {args.scap}, p {args.cap_p}), max stiffness {stiff:.2f} -> dt {dt:.4e}", flush=True)
nst, ev = int(round(args.T / dt)), max(1, int(round(args.every / dt)))
print(f"struct {args.struct} n {args.n} P {args.P} R {R} N {N} curves {len(curves)} smooth {args.smooth} ticks {nst}", flush=True)
if args.relax:
    nr, t_r = int(round(args.relax / dt)), time.time()
    for i in range(nr + 1):
        if i % max(1, nr // 10) == 0:
            Fp = ac.phi * 0.5 * (gS(ac.phi)[:, None] + gS(ac.phi)[nb])
            dv = np.sum(W[None, :] * Fp, 1)
            print(f"  relax t {i * dt:6.2f}  Eu {np.sum(ac.u**2):.4e}  |div F'| rms {np.sqrt(np.mean(dv**2)):.4e}  ({time.time() - t_r:.0f} s)", flush=True)
        if i == nr:
            break
        alat.step(0, dt)
        ac.u = ac.u * (1 - dt / args.relax_tau)
    ac.u = np.zeros(N)
    if args.save_relaxed:
        np.savez(args.save_relaxed, phi=ac.phi)
if args.pulse:
    xr = wrapx(pos[:, 0] - c0[0] - args.pulse[1])
    ac.u = ac.u + args.pulse[0] * np.exp(-xr**2 / (2 * args.pulse[2] ** 2))
    print(f"pulse: Eu {np.sum(ac.u**2):.4e}", flush=True)
psi_df = None
hist = []
ph_prev, t_wall = ac.phi, time.time()
disp = np.zeros(3)
cprev = c0
s0 = 0
ck = args.ckpt or (args.seed.replace(".npz", "_ckpt.npz") if args.seed else "")
if ck and Path(ck).exists():
    z = np.load(ck)
    ac.phi, ac.u, ph_prev, s0 = z["phi"], z["u"], z["ph_prev"], int(z["s"])
    disp, cprev, hist = z["disp"], z["cprev"], [tuple(r) for r in z["hist"]]
    print(f"resumed {ck} at t {s0 * dt:.2f}", flush=True)
nprint = 0
for s in range(s0, nst + 1):
    if s % ev == 0:
        ph = ac.phi
        w = np.abs(Gam(ph))
        c = centroid(w)
        disp += wrapx(c - cprev)
        cprev = c
        dd = dist_to_curves(disp)
        rms2 = np.sqrt(np.sum(w**2 * dd**2) / np.sum(w**2))
        leak = w[dd > rc].sum() / w.sum()
        rr, ang = rigid_rms(w, dd, c0 + disp)
        ec = ac.u**2 + 2 * Gs(0.25 * np.sum(ph_prev * ph, 1))
        E = ec.sum()
        rc_ = np.linalg.norm(wrapx(pos - (c0 + disp)), axis=1)
        shells = np.bincount(np.searchsorted(SH, rc_), weights=ec, minlength=len(SH) + 1)
        if not (s == s0 and s0 > 0):
            hist.append((s * dt, *disp, rms2, leak, w.sum(), E))
        wt = (time.time() - t_wall) / max(1, s - s0)
        print(f"  t {s * dt:6.2f}  disp {disp[0]:+7.3f} {disp[1]:+7.3f} {disp[2]:+7.3f}  rms2 {rms2:.4f}  rms2r {rr:.4f} rot {ang:5.1f}  leak {leak:.4e}  "
              f"sum|G| {w.sum():.5e}  E {E:.8e}  Eu {np.sum(ac.u**2):.4e}  shells {' '.join(f'{q:.4e}' for q in shells)}  ({wt:.2f} s/tick, ETA {wt * (nst - s) / 60:.1f} min)", flush=True)
        if args.struct == "side_tilt":
            from scipy.sparse.csgraph import connected_components
            from scipy.sparse import coo_matrix
            from scipy.spatial import cKDTree
            xm = wrapx(cen[:, 0] - OFF[0])
            cs = []
            for fr in (0.3, 0.5):
                strong = w > fr * w.max()
                Xs = cen[strong] % side
                pr = cKDTree(Xs, boxsize=side).query_pairs(1.0, output_type="ndarray")
                nc, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(Xs),) * 2), directed=False)
                big = np.sort(np.bincount(lab))[::-1]
                cs.append(f"{fr}: {nc} [{' '.join(str(b) for b in big[:3])}]")
            L, Rt = xm < 0, xm >= 0
            print(f"    bridge |x| < R / 4: {w[np.abs(xm) < R / 4].sum() / w.sum():.4e}  |x| < R / 8: {w[np.abs(xm) < R / 8].sum() / w.sum():.4e}  "
                  f"x halves {w[L] @ xm[L] / w[L].sum():+.3f} {w[Rt] @ xm[Rt] / w[Rt].sum():+.3f}  comps {'  '.join(cs)}", flush=True)
        if HALF:
            print(f"    halves E {ec[inA].sum():.6e} {ec[~inA].sum():.6e}  Eu {np.sum(ac.u[inA]**2):.4e} {np.sum(ac.u[~inA]**2):.4e}  "
                  f"sum|G| {w[inA_t].sum():.5e} {w[~inA_t].sum():.5e}", flush=True)
        if args.ediv:
            df, psi_df = C.project_div_free(ac, ph, tol=1e-9, x0=psi_df, return_psi=True)
            Sd, Sf = 0.25 * np.sum(df**2, 1), 0.25 * np.sum(ph**2, 1)
            Sref = S_c or Sf.max()
            print(f"    div-free store {2 * np.sum(Gs(Sd)):.6e}  core max S / S_c {Sf.max() / Sref:.3f}  n > S_c {(Sf > Sref).sum()}  "
                  f"div-free max S / S_c {Sd.max() / Sref:.3f}", flush=True)
        nprint += 1
        if ck and s > s0 and s < nst and nprint % args.ckpt_every == 0:
            np.savez(ck + ".tmp.npz", phi=ph, u=ac.u, ph_prev=ph_prev, s=s, disp=disp, cprev=cprev, hist=np.array(hist))
            Path(ck + ".tmp.npz").replace(ck)
    if s == nst:
        break
    ph_prev = ac.phi
    alat.step_cartan(0, dt, args.kappa, 2)
h = np.array(hist)
v = np.polyfit(h[:, 0], h[:, 1:4], 1)[0]
print(f"RESULT {args.struct} n {args.n} R {R} P {args.P}  |v| {np.linalg.norm(v):.4f} v {v[0]:+.4f} {v[1]:+.4f} {v[2]:+.4f}  "
      f"rms2 {h[0, 4]:.3f}->{h[-1, 4]:.3f} ({100 * (h[-1, 4] / h[0, 4] - 1):+.2f} %)  leak {h[0, 5]:.3e}->{h[-1, 5]:.3e}  "
      f"rms2r {rr0:.3f}->{rr:.3f} ({100 * (rr / rr0 - 1):+.2f} %)  sum|G| {100 * (h[-1, 6] / h[0, 6] - 1):+.2f} %  dE/E {h[-1, 7] / h[0, 7] - 1:+.1e}")
if args.save:
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.save, phi=ac.phi, hist=h, args=str(vars(args)))
