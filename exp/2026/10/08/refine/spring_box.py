"""10-07 slosh: 10-06/branch/spring_box.py + --tail (blend a --profile to rho0 between two distances; core/gp_vortex.py
writes the exact GP vortex profile, the analytic clean seed) and an in-run slosh probe (--slosh_every): deficit
sum(rho0 - rho) per 2-cell shell of dt (distance to the co-moving tube centre circle, own Voronoi part vs the 26 images)
every few ticks, and the walk's energy e = e_q + e_nl - background (shed/quasi_energy.py local split) within --slosh_dt
every --slosh_e_every ticks -> slosh_from<t>.npz. Snapshots under ~/dst/2026/10/07/slosh (no /bk on this machine).

10-06 shed: quant/spring_box.py + the coil/spring_cpp.py 10-06 in-run probes (--probe_every; coil-only centre,
offset-circle windings, axis ring, census), z distances taken periodically.

Periodic-box version of coil/spring_cpp.py (exp/2026/10/03/pbox/status.md): build_bcc_box_periodic of period P
(even, grid units h = 1/sqrt3; side P h = neighbour spacing), no sponge, no ghosts; seed phase = q/2 x solid angle
summed over the 27 nearest images (the 26 outer ones from a coarse polygon), minus the uniform term that removes
the mean phase jump across each face; recentre = exact periodic shift along z.

C++ walk version of spring.py (AdaptiveLattice::step_walk; same seed, scores, plots), plus --z0 (start height) and
--sponge. Spring coil (Philip 10-02: a coil is a spring, each turn advances toroidally -- the 10-02 meridional loops were
stellarator-like, no horizontal element): one continuous vortex strand along the torus helix
    theta_tor = s, theta_pol = N s + 2 pi j / S,  s in [0, 2 pi)
(N poloidal turns per toroidal circuit, S parallel strands = multi-start), minor radius a, centre circle r0.
Same rule as coil.py / gp.py (real-form Dirac walk + local density turn, uniform background, sponge to background).

Seed: phase = q Omega / 2, Omega = solid angle the closed strand(s) subtend (Van Oosterom-Strackee, cone to a
generic apex; Omega is defined mod 4 pi, the phase mod 2 pi), so the winding around the strand is q everywhere along it;
envelope d / sqrt(d^2 + xi^2), d = distance to the nearest strand.
Scores (co-moving: torus centre (rc, zc) = core-weighted mean (rho, z)): core cells (rho < 0.25 rho0); phase winding
along the centre circle (= N S while every turn is intact) and along a meridional circle of radius a + 3 xi around the
tube (= S, the toroidal circuit); mean minor radius of the core; helix phase arg sum w exp(i (theta_pol - N
theta_tor)) (spring advance / rotation). Full state (L, R) at plot times in /bk/dst/2026/10/02/coil/<name>/.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from equilibrium.substrate.walk import CppWalk, cpp_box, healing_g, solid_angle, to_complex

here = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--m", type=float, default=0.5)
p.add_argument("--xi", type=float, default=3.9)
p.add_argument("--rho0", type=float, default=1.0)
p.add_argument("--N", type=int, default=8, help="poloidal turns per toroidal circuit")
p.add_argument("--S", type=int, default=1, help="parallel strands")
p.add_argument("--q", type=int, default=1)
p.add_argument("--a", type=float, default=10.0)
p.add_argument("--r0", type=float, default=30.0)
p.add_argument("--P", type=int, default=346, help="box period in grid units (even); side = P / sqrt3")
p.add_argument("--img_seg", type=int, default=12, help="polygon segments per turn for the 26 outer images")
p.add_argument("--T", type=int, default=1200)
p.add_argument("--every", type=int, default=20)
p.add_argument("--n_plot", type=int, default=4)
p.add_argument("--damp", type=float, default=0.05)
p.add_argument("--seg_per_turn", type=int, default=96)
p.add_argument("--z0", type=float, default=0.0)
p.add_argument("--theta0", type=float, default=0.0, help="rotate the seed strand about z by this many degrees")
p.add_argument("--walk_xi", action="store_true", help="xi = true core width (walk dispersion n/(6m)), not the c^2 = 1/3 label")
p.add_argument("--threads", type=int, default=8)
p.add_argument("--recentre", type=int, default=0, help="every this many ticks shift the state by whole lattice vectors along z so zc ~ 0 (0 = off)")
p.add_argument("--parity", action="store_true", help="alternate with the parity-image tick (walk.py parity; aniso/status.md)")
p.add_argument("--tag", default="")
p.add_argument("--profile", default="", help="npz (d, rho) from core_profile.py: seed amplitude a0 sqrt(rho(d)) instead of d / sqrt(d^2 + xi^2)")
p.add_argument("--compensate", action="store_true", help="rescale the seed so total norm = rho0 x cells (D_tot = 0; background raised to absorb the deficit)")
p.add_argument("--project_plus", action="store_true", help="project the seed onto the + quasi-energy branch (shed/quasi_energy.py branch_split)")
p.add_argument("--resume", default="", help="state_t*.npz to continue from (its t, z_shift); seed skipped")
p.add_argument("--snap_root", default="/bk/dst/2026/10/08/refine" if Path("/bk/dst").is_dir() else str(Path.home() / "tmp/dst/2026/10/08/refine"),
               help="/bk/dst on the main machine, ~/tmp/dst on the fedora box (no /bk)")
p.add_argument("--tail", default="", help="d1,d2: blend the --profile rho linearly to rho0 between d1 and d2")
p.add_argument("--slosh_every", type=int, default=20, help="deficit shells every this many ticks (0 = off)")
p.add_argument("--slosh_e_every", type=int, default=200, help="energy within --slosh_dt every this many ticks (0 = off)")
p.add_argument("--slosh_dt", type=float, default=38.0)
p.add_argument("--probe_every", type=int, default=40, help="in-run coil / axis-ring probes every this many ticks (0 = off)")
args = p.parse_args()
c2 = 1.0 / 3.0
xi = args.xi
g = healing_g(args.m, xi, args.rho0, walk=args.walk_xi)
name = ("comp_" if args.compensate else "") + ("prof_" if args.profile else "") + ("pp_" if args.project_plus else "") + f"springbox_N{args.N}_S{args.S}_q{args.q}_a{args.a:g}_r{args.r0:g}_xi{xi:g}_m{args.m:g}_P{args.P}_z{args.z0:g}{'_wxi' if args.walk_xi else ''}{'_rc' if args.recentre else ''}{f'_th{args.theta0:g}' if args.theta0 else ''}{'_par' if args.parity else ''}{args.tag}"

run_dir = here / Path(*name.split("_"))
run_dir.mkdir(parents=True, exist_ok=True)
t0 = time.time()
alat = cpp_box(args.P, n_threads=args.threads)
side = args.P / np.sqrt(3)
pos = np.array(alat[0].leaf_pos)
x, y, z = pos.T
rho_c = np.hypot(x, y)
th_c = np.arctan2(y, x)
r_all = np.linalg.norm(pos, axis=1)
a0 = np.sqrt(args.rho0)


def strand(j, seg=None):
    s = np.linspace(0, 2 * np.pi, args.N * (seg or args.seg_per_turn), endpoint=False)
    tp = args.N * s + 2 * np.pi * j / args.S
    rr = args.r0 + args.a * np.cos(tp)
    s0 = s + np.radians(args.theta0)
    return np.stack([rr * np.cos(s0), rr * np.sin(s0), args.z0 + args.a * np.sin(tp)], 1)


strands = [strand(j) for j in range(args.S)]
r_inner = np.inf
w = CppWalk(alat, 0, m=args.m, g=g, rho0=args.rho0, parity=args.parity)
ix = np.rint(pos * np.sqrt(3)).astype(np.int64)
H = args.P // 2
lut = np.full((args.P,) * 3, -1, np.int64)
lut[ix[:, 0] + H, ix[:, 1] + H, ix[:, 2] + H] = np.arange(len(pos))
sys.argv, argv_ = ["qe", "--m", str(args.m)], sys.argv
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "10/06/shed"))
import quasi_energy as qe  # noqa: E402

sys.argv = argv_
t_start, z_shift = 0, 0.0  # total lattice shift applied (true z = z + z_shift)
if args.resume:
    ck = np.load(args.resume)
    t_start, z_shift = int(ck["t"]), float(ck["z_shift"])
    w.set_state(ck["L"], ck["R"], t=t_start)
    del ck
else:
    coarse = [strand(j, args.img_seg) for j in range(args.S)]
    shifts = [np.array([a, b, c]) * side for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if a or b or c]

    def phase(X):
        return 0.5 * args.q * (sum(solid_angle(P, X, n_threads=args.threads) for P in strands)
                               + sum(solid_angle(P, X, shifts=np.array(shifts), n_threads=args.threads) for P in coarse))

    rng_ = np.random.default_rng(0)
    jumps = []
    for ax in range(3):  # mean jump across the face normal to ax -> uniform term
        smp = (rng_.random((400, 3)) - 0.5) * side
        smp[:, ax] = -side / 2
        hi = smp.copy()
        hi[:, ax] += side
        jump = np.angle(np.mean(np.exp(1j * (phase(hi) - phase(smp)))))
        print(f"  seed: mean phase jump across face {'xyz'[ax]} {jump:+.4f}", flush=True)
        jumps.append(jump)
    Om = 2 * (phase(pos) - sum(jumps[a] * pos[:, a] / side for a in range(3))) / args.q
    d = cKDTree(np.concatenate(strands)).query(pos)[0]
    if args.profile:
        pf = np.load(args.profile)
        rr = np.interp(d, pf["d"], pf["rho"], right=args.rho0)
        if args.tail:
            d1, d2 = map(float, args.tail.split(","))
            bl = np.clip((d - d1) / (d2 - d1), 0, 1)
            rr = (1 - bl) * rr + bl * args.rho0
        w.seed(a0 * np.sqrt(rr) * np.exp(1j * args.q * Om / 2))
    else:
        w.seed(a0 * d / np.sqrt(d**2 + xi**2) * np.exp(1j * args.q * Om / 2))
    del Om, d
    if args.project_plus:
        from equilibrium.substrate.walk import to_real
        L0, R0 = w.state()
        psi = np.concatenate([to_complex(L0), to_complex(R0)], 1)
        n0 = np.sum(np.abs(psi)**2)
        psi = psi - qe.branch_split(psi, (ix + args.P // 2) % args.P, args.P, want_psi=True)
        n1 = np.sum(np.abs(psi)**2)
        print(f"  seed projected onto + branch: norm {n0:.3f} -> {n1:.3f} (branch- {n0 - n1:.3f})", flush=True)
        w.set_state(to_real(psi[:, :2]), to_real(psi[:, 2:]), t=0)
        del psi, L0, R0
    if args.compensate:
        L0, R0 = w.state()
        n0 = np.sum(L0**2) + np.sum(R0**2)
        sc = np.sqrt(args.rho0 * len(pos) / n0)
        print(f"  seed compensated: norm {n0:.3f} -> {args.rho0 * len(pos):.3f} (x {sc**2:.6f}; D_tot 0)", flush=True)
        w.set_state(L0 * sc, R0 * sc, t=0)
        del L0, R0
inner = np.ones(len(pos), bool)
assert max(np.abs(P).max() for P in strands) + 3 * xi < side / 2
tree = cKDTree(pos)
ps = np.linspace(0, 2 * np.pi, 256, endpoint=False)
rm = args.a + 3 * xi


def probes(rc, zc):
    """centre circle and meridional circle (y = 0, x > 0) of the torus centred at (rho, z) = (rc, zc)."""
    pc = tree.query(np.stack([rc * np.cos(ps), rc * np.sin(ps), zc + 0 * ps], 1))[1]
    pm = tree.query(np.stack([rc + rm * np.cos(ps), 0 * ps, zc + rm * np.sin(ps)], 1))[1]
    return pc, pm


snap_dir = Path(args.snap_root) / name
snap_dir.mkdir(parents=True, exist_ok=True)
np.save(snap_dir / "pos.npy", pos)


print(f"{name}: N_cells={len(pos)} build+seed {time.time() - t0:.0f}s; g {g:.4f} xi {xi:.2f}; strand pitch "
      f"(toroidal advance per turn) {2 * np.pi * args.r0 / args.N:.1f} cells at r0, inner-side turn spacing "
      f"{2 * np.pi * (args.r0 - args.a) / (args.N * args.S):.1f}", flush=True)


def amp():
    return w.amp()


def winding(a_, idx):
    return np.sum(np.angle(a_[idx][np.r_[1 : len(idx), 0]] / a_[idx])) / (2 * np.pi)


A_C = 2 / np.sqrt(3)  # conventional cube edge = lattice vector along z (hex distance 1)


def recentre(zc):
    """shift the state down by n whole lattice vectors along z (n = round(zc / A_C)): cell j takes the state of the
    cell at pos_j + n A_C z^, wrapped (periodic box). Exact on the lattice."""
    global z_shift
    nz = int(round(zc / A_C))
    if nz == 0:
        return 0
    src = lut[ix[:, 0] + H, ix[:, 1] + H, (ix[:, 2] + 2 * nz + H) % args.P]
    L, R = w.state()
    w.set_state(L[src], R[src])
    z_shift += nz * A_C
    return nz


def pdz(dz):
    return (dz + side / 2) % side - side / 2


def zmean(zz, wt):
    """weighted periodic mean of z (circular)."""
    k = 2 * np.pi / side
    return np.angle(np.sum(wt * np.exp(1j * k * zz))) / k


def state(rho_, a_):
    """co-moving: the torus centre (rc, zc) = core-weighted mean (rho, z) of the coil-only core (rho_cyl > r0 / 2);
    probes and minor radius relative to it. Also the deficit sum(rho0 - rho) within a + 4 xi (D_near) and a + 1.5 xi
    (D_core) of the centre circle."""
    core = (rho_ < 0.25 * args.rho0) & (rho_c > 0.5 * args.r0)
    n = core.sum()
    if not n:
        return [n] + [np.nan] * 8 + [z_shift]
    wt = (args.rho0 - rho_)[core]
    rc, zc = np.sum(wt * rho_c[core]) / wt.sum(), zmean(z[core], wt)
    pc, pm = probes(rc, zc)
    wc, wm = winding(a_, pc), winding(a_, pm)
    dzc = pdz(z[core] - zc)
    dm = np.hypot(rho_c[core] - rc, dzc)
    thp = np.arctan2(dzc, rho_c[core] - rc)
    hel = np.angle(np.sum(wt * np.exp(1j * (thp - args.N * th_c[core]))))
    dt_ = np.hypot(rho_c - rc, pdz(z - zc))
    Dn = np.sum((args.rho0 - rho_)[dt_ < args.a + 4 * xi])
    Dc = np.sum((args.rho0 - rho_)[dt_ < args.a + 1.5 * xi])
    return [n, np.sum(wt * dm) / wt.sum(), hel, wc, wm, rc, zc, Dn, Dc, z_shift]


OFFS = [(0.0, 0.0)] + [(r * np.cos(t), r * np.sin(t)) for r in (2.0, 4.0) for t in np.arange(7) * 2 * np.pi / 7]
KS = sorted({2, 4, 8, args.N})
qbox = np.mod(pos + side / 2, side)
qbox[qbox >= side] = 0.0


def probe(rho_, a_):
    """as coil/spring_cpp.py probe(), periodic in z: coil-only (rho_cyl > r0 / 2) and axis-ring (rho_cyl < r0 / 2)
    pieces of the core; census of off-structure pieces (rho < 0.6, dt >= a + 4 xi, >= 10 cells, periodic adjacency)."""
    core = rho_ < 0.25 * args.rho0
    cc = core & (rho_c > 0.5 * args.r0)
    ca = core & (rho_c <= 0.5 * args.r0)
    out = {"t": w.t, "z_shift": z_shift, "n_coil": int(cc.sum()), "n_axis": int(ca.sum())}
    if cc.sum():
        wt = (args.rho0 - rho_)[cc]
        rc, zc = np.sum(wt * rho_c[cc]) / wt.sum(), zmean(z[cc], wt)
        wn, mr = [], np.inf
        for dr, dz in OFFS:
            idx = tree.query(np.stack([(rc + dr) * np.cos(ps), (rc + dr) * np.sin(ps), zc + dz + 0 * ps], 1))[1]
            wn.append(int(round(winding(a_, idx))))
            mr = min(mr, rho_[idx].min())
        dR, dZ = rho_c[cc] - rc, pdz(z[cc] - zc)
        C = np.cov(np.stack([dR, dZ]), aweights=wt)
        ev, evec = np.linalg.eigh(C)
        out.update(rc=rc, zc=zc, wind=np.array(wn), wind_min_rho=mr, aspect=np.sqrt(ev[1] / max(ev[0], 1e-12)),
                   tilt=np.degrees(np.arctan2(evec[1, 1], evec[0, 1])) % 180, rad=np.sqrt(np.trace(C)))
        th = th_c[cc]
        for k in KS:
            e = np.exp(-1j * k * th)
            out[f"z{k}"], out[f"r{k}"] = 2 * np.sum(wt * dZ * e) / wt.sum(), 2 * np.sum(wt * dR * e) / wt.sum()
        off = np.nonzero((rho_ < 0.6) & (np.hypot(rho_c - rc, pdz(z - zc)) >= args.a + 4 * xi))[0]
        pcs = []
        if len(off):
            po = cKDTree(qbox[off], boxsize=side).query_pairs(1.0 + 1e-6, output_type="ndarray")
            nc, lab = connected_components(coo_matrix((np.ones(len(po)), (po[:, 0], po[:, 1])),
                                                      shape=(len(off), len(off))), directed=False)
            for c in range(nc):
                idx = off[lab == c]
                if len(idx) >= 10:
                    pcs.append((len(idx), rho_[idx].min(), np.sum(args.rho0 - rho_[idx]), rho_c[idx].mean(),
                                np.degrees(np.arctan2(y[idx].mean(), x[idx].mean())),
                                pdz(zmean(z[idx], np.ones(len(idx))) - zc)))
        out["pieces"] = np.array(sorted(pcs, key=lambda c_: -c_[2]))
    if ca.sum() >= 20:
        wt = (args.rho0 - rho_)[ca]
        ra = np.sum(wt * rho_c[ca]) / wt.sum()
        za = zmean(z[ca], wt)
        idx = tree.query(np.stack([ra + 4 * np.cos(ps), 0 * ps, pdz(za + 4 * np.sin(ps))], 1))[1]
        out.update(ra=ra, za=za, wind_axis=winding(a_, idx))
        if "zc" in out:
            out["za_rel"] = pdz(za - out["zc"])
    return out


SH_DR = 2.0
_geo = {}
_H2 = []


def local_energy(psi):
    """per-cell e_q = Re psi^dag (H2 psi) / 2 (shed/quasi_energy.py local=True) with H2(k) cached per kx slab."""
    from equilibrium.substrate.walk_energy import symbols as wsym

    P = args.P
    iq = (ix + P // 2) % P
    F = np.zeros((4, P, P, P), complex)
    for c in range(4):
        F[c][iq[:, 0], iq[:, 1], iq[:, 2]] = psi[:, c]
    F = np.fft.fftn(F, axes=(1, 2, 3))
    k = 2 * np.pi * np.fft.fftfreq(P)
    if not _H2:
        KY, KZ = np.meshgrid(k, k, indexing="ij")
        for kx in k:
            S, Sp = wsym(kx, KY, KZ, args.m)
            U = Sp @ S
            Om = np.arccos(np.clip(np.real(np.trace(U, axis1=-2, axis2=-1)) / 4, -1, 1))
            _H2.append((Om / np.sin(Om))[..., None, None] * 0.5j * (U - np.conj(np.swapaxes(U, -1, -2))))
    for i in range(P):
        F[:, i] = np.einsum("...ij,...j->...i", _H2[i], F[:, i].transpose(1, 2, 0)).transpose(2, 0, 1)
    F = np.fft.ifftn(F, axes=(1, 2, 3))
    return 0.5 * np.real(sum(np.conj(psi[:, c]) * F[c][iq[:, 0], iq[:, 1], iq[:, 2]] for c in range(4)))


def geometry(rc, zc):
    """dt (distance to the tube centre circle (rc, zc)) and shell index, fresh; own Voronoi mask vs the 26 images and
    the midplane mask cached until the centre moves by > 2 cells (zc in the lab frame incl. z_shift; only the far
    boundary moves)."""
    zz = pdz(z - zc)
    dt_ = np.hypot(rho_c - rc, zz)
    key = (rc, zc + z_shift)
    if not (_geo and abs(_geo["key"][0] - key[0]) < 2 and abs(pdz(_geo["key"][1] - key[1])) < 2):
        dimg = np.full_like(dt_, np.inf)
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                for sz in (-1, 0, 1):
                    if sx or sy or sz:
                        dimg = np.minimum(dimg, np.hypot(np.hypot(x + sx * side, y + sy * side) - rc, zz + sz * side))
        _geo.update(key=key, own=dt_ < dimg, mid=np.abs(dt_ - dimg) < 2.0)
    return dt_, _geo["own"], np.minimum((dt_ / SH_DR).astype(np.int64), int(side / SH_DR))


def slosh(rho_, want_e):
    """deficit per dt shell (own Voronoi part), D within slosh_dt; e within slosh_dt if want_e."""
    cc = (rho_ < 0.25 * args.rho0) & (rho_c > 0.5 * args.r0)
    if not cc.sum():
        return None
    wt = (args.rho0 - rho_)[cc]
    rc, zc = np.sum(wt * rho_c[cc]) / wt.sum(), zmean(z[cc], wt)
    dt_, own, k = geometry(rc, zc)
    mid = _geo["mid"]
    dsh = np.bincount(k[own], (args.rho0 - rho_)[own], minlength=int(side / SH_DR) + 1)
    nsh = np.bincount(k[own], minlength=int(side / SH_DR) + 1)
    ins = own & (dt_ < args.slosh_dt)
    out = {"t": w.t, "rc": rc, "zc": zc, "D_in": np.sum((args.rho0 - rho_)[ins]), "dsh": dsh, "nsh": nsh,
           "rho_mid": np.mean(rho_[mid]), "D_tot": np.sum(args.rho0 - rho_)}
    if want_e:
        L_, R_ = w.state()
        psi = np.concatenate([to_complex(L_), to_complex(R_)], 1)
        del L_, R_
        e = local_energy(psi)
        del psi
        e = e - qe.om0 * rho_ + 0.5 * g * rho_ * (rho_ - args.rho0)
        out.update(E_in=np.sum(e[ins]), E_tot=np.sum(e), esh=np.bincount(k[own], e[own], minlength=len(dsh)))
    return out


def probe_line(o):
    s_ = f"P t {o['t']}: coil {o['n_coil']}"
    if "rc" in o:
        u = np.unique(o["wind"])
        s_ += (f" c ({o['rc']:.1f},{o['zc']:+.1f}) wind {'/'.join(str(v) for v in u)} minrho {o['wind_min_rho']:.2f}"
               f" asp {o['aspect']:.2f} tilt {o['tilt']:.0f} rad {o['rad']:.2f} |"
               + "".join(f" k{k} z {abs(o[f'z{k}']):.2f}@{np.degrees(np.angle(o[f'z{k}'])) / k:+.1f}"
                         f" r {abs(o[f'r{k}']):.2f}@{np.degrees(np.angle(o[f'r{k}'])) / k:+.1f}" for k in KS))
    s_ += f" | axis {o['n_axis']}"
    pc_ = o.get("pieces", np.zeros((0, 6)))
    if len(pc_):
        s_ += f" | pieces {len(pc_)} D {pc_[:, 2].sum():.0f}: " + " ".join(
            f"[{int(c_[0])} {c_[1]:.2f} r{c_[3]:.0f} th{c_[4]:+.0f} z{c_[5]:+.0f}]" for c_ in pc_[:4])
    if "ra" in o:
        s_ += f" r {o['ra']:.1f} z {o['za']:+.1f} zrel {o.get('za_rel', np.nan):+.1f} wind {o['wind_axis']:+.2f}"
    return s_


def plot(tag, rc=None, zc=0.0):
    rc = args.r0 if rc is None or not np.isfinite(rc) else rc
    zc = 0.0 if not np.isfinite(zc) else zc
    from scipy.interpolate import griddata

    rho_ = w.density()
    lim = side / 2
    gg = np.linspace(-lim, lim, 401)
    G1, G2 = np.meshgrid(gg, gg)
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    zp = z[np.argmin(np.abs(z - zc))]
    m_ = (np.abs(z - zp) < 0.05) & (np.abs(x) < lim + 2) & (np.abs(y) < lim + 2)
    im = ax[0].imshow(griddata((x[m_], y[m_]), rho_[m_], (G1, G2), method="linear"), origin="lower",
                      extent=(-lim, lim, -lim, lim), cmap="viridis")
    ax[0].set_title(f"density, xy lattice plane z = {zp:+.2f} (nearest zc)")
    fig.colorbar(im, ax=ax[0], shrink=0.8)
    core = (rho_ < 0.25 * args.rho0) & inner
    sc = ax[1].scatter(x[core], y[core], c=z[core] - zc, s=1, cmap="coolwarm", vmin=-args.a - xi,
                    vmax=args.a + xi)
    for P in strands:
        ax[1].plot(*P[:, :2].T, "k-", lw=0.3, alpha=0.4)
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)
    ax[1].set_aspect("equal")
    ax[1].set_title(f"core cells (rho < 0.25), colour z - zc (zc {zc:+.1f}); grey = seed strand")
    fig.colorbar(sc, ax=ax[1], shrink=0.8)
    lim2 = args.a + 4 * xi
    m2 = (np.abs(y) < 0.3) & (x > 0) & (np.abs(x - rc) < lim2 + 2) & (np.abs(z - zc) < lim2 + 2)
    g2 = np.linspace(-lim2, lim2, 201)
    H1, H2 = np.meshgrid(rc + g2, zc + g2)
    im = ax[2].imshow(griddata((x[m2], z[m2]), rho_[m2], (H1, H2), method="linear"), origin="lower",
                      extent=(rc - lim2, rc + lim2, zc - lim2, zc + lim2), cmap="viridis")
    ax[2].set_title("density, meridional plane y = 0, centred on (rc, zc)")
    fig.colorbar(im, ax=ax[2], shrink=0.8)
    fig.suptitle(f"{name}  t = {w.t}  (frame shifted by {z_shift:+.1f} in z)")
    fig.tight_layout()
    fig.savefig(run_dir / f"{tag}.png", dpi=80)
    plt.close(fig)


plot(f"t{t_start:05d}" if t_start else "t0000")
rec, prb, slo = [], [], []
slosh_path = run_dir / f"slosh_from{t_start}.npz"
probe_path = run_dir / f"probes_from{t_start}.npz"
t0 = time.time()
plot_every = max(args.every, (args.T // args.n_plot) // args.every * args.every)
for t in range(t_start, args.T + 1, args.every):
    rho_, a_ = w.density(), amp()
    st = state(rho_, a_)
    rec.append([t] + st)
    if args.probe_every and t % args.probe_every < args.every:
        prb.append(probe(rho_, a_))
        print(probe_line(prb[-1]), flush=True)
    if args.slosh_every and t % args.slosh_every < args.every:
        so = slosh(rho_, args.slosh_e_every and t % args.slosh_e_every < args.every)
        if so is not None:
            slo.append(so)
            if "E_in" in so:
                print(f"S t {t}: D_in {so['D_in']:.0f} E_in {so['E_in']:.2f} E_tot {so['E_tot']:.3f} "
                      f"rho_mid {so['rho_mid']:.5f}", flush=True)
    del rho_, a_
    if t % (plot_every // 2) < args.every:
        print(f"  t {t}: core cells {st[0]}, minor radius {st[1]:.1f}, helix phase {st[2]:+.3f}, winding centre "
              f"circle {st[3]:+.2f}, meridional {st[4]:+.2f}, centre (rho, z) ({st[5]:.1f}, {st[6]:+.1f}), "
              f"D_near {st[7]:.0f} D_core {st[8]:.0f}, frame shift {st[9]:+.1f}", flush=True)
    if t % plot_every < args.every and (t > t_start or not args.resume):
        Ls, Rs = w.state()
        np.savez(snap_dir / f"state_t{t:05d}.npz", t=t, L=Ls, R=Rs, z_shift=z_shift)
        del Ls, Rs
    if t and t % plot_every < args.every:
        plot(f"t{t:05d}", st[5], st[6])
        if prb:
            np.savez(probe_path, probes=np.array(prb, dtype=object))
        if slo:
            np.savez(slosh_path, slosh=np.array(slo, dtype=object))
    if args.recentre and t > t_start and t % args.recentre < args.every and np.isfinite(st[6]):
        if recentre(st[6]):
            _geo.clear()
    if t < args.T:
        w.step(args.every)
rec = np.array(rec, float)
print(f"  {args.T} ticks in {time.time() - t0:.0f}s", flush=True)
if prb:
    np.savez(probe_path, probes=np.array(prb, dtype=object))
if slo:
    np.savez(slosh_path, slosh=np.array(slo, dtype=object))
np.savez(run_dir / (f"rec_from{t_start}.npz" if t_start else "rec.npz"), rec=rec, args=str(vars(args)), xi=xi, g=g)
if slo:
    from slosh_io import dump

    print(f"  slosh series -> {dump(run_dir)}", flush=True)
