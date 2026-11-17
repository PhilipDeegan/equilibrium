"""
probe_bell_canal.py — Stage 1 of the on-lattice Bell/canal probe:
the SEPARABLE (classical, factorizable) baseline.

Two cells A and B at real lattice separation D on a cylindrical domain each
receive an independent transverse-phi "polarization" imprint at t=0 (the
hidden variable: A at angle alpha, B at alpha+pi/2 — singlet convention,
matching res/zip/ham/.../bell.py's classical_lhv).  The imprints then evolve
under the genuine substrate update rule for T ticks, optionally against a
ReciprocalBC.thermalize() background bath, before a local Malus-law readout
at each end.

Because the (u, phi) step is exactly linear, the full hidden-variable
ensemble is reconstructed from FOUR basis evolutions (A-x, A-y, B-x, B-y
unit imprints) plus K noise-only evolutions: the state for any alpha is a
cos/sin superposition of the recorded responses, and the Monte Carlo over
alpha runs offline at numpy speed instead of one lattice run per trial.

Expected stage-1 results (acceptance criteria, Feynman benchmark):
  - CHSH S ~= sqrt(2)  (this specific stochastic-Malus LHV model's value)
  - cascade P_agree(30 deg) ~= 5/8 = 0.625, safely under Feynman's
    deterministic-LHV ceiling of 2/3 — and far from the QM 3/4.
  - No static u structure along the A-B line (separable imprints must not
    "gravitate": transverse phi patches are divergence-free per cell).
A joint-canal construction (stage 2) must beat 2/3 and reach 3/4 to count.

Readout is angle-only (Malus on the local transverse-phi direction);
intensity decay shows up only through its signal-to-noise ratio against the
bath, not by itself — flagged in the report as |v_A|.
"""

import argparse
from pathlib import Path

import numpy as np

from equilibrium.substrate.lattice import DampedLattice
from tst.substrate.test_saha import ReciprocalBC

# sq faces are 8..13, mirror pairs (8,9), (10,11), (12,13); the face->axis
# assignment is detected empirically from the built lattice (see
# detect_transverse_faces), not assumed from any documented convention.
_MIRROR = np.array([7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 11, 10, 13, 12])

CHSH_ANGLES = dict(a=0.0, a_p=np.pi / 4, b=np.pi / 8, b_p=3 * np.pi / 8)
FEYNMAN_ANGLE = np.pi / 6  # 30 deg


def detect_transverse_faces(lat):
    """Return (kx_plus, kx_minus, ky_plus, ky_minus): the sq faces whose
    neighbour offsets point along +x, -x, +y, -y."""
    c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
    faces = {}
    for k in range(8, 14):
        j = int(lat.nb[c0, k])
        assert j >= 0, f"centre cell missing sq neighbour {k}"
        d = lat.pos[j] - lat.pos[c0]
        axis = int(np.argmax(np.abs(d)))
        faces[(axis, int(np.sign(d[axis])))] = k
    return faces[(0, 1)], faces[(0, -1)], faces[(1, 1)], faces[(1, -1)]


def patch_indices(lat, centre, radius: float) -> np.ndarray:
    return np.where(np.linalg.norm(lat.pos - np.asarray(centre), axis=1) <= radius)[0]


def imprint_transverse(lat, faces, patch: np.ndarray, vx: float, vy: float):
    """Uniform transverse phi vector (vx, vy) on the patch, antisymmetry-
    consistent: interior faces pair up (phi[i,+x]=+v / phi[j,-x]=-v), faces
    leaving the patch get the mirror entry on the outside neighbour."""
    kxp, kxm, kyp, kym = faces
    patch_set = set(int(i) for i in patch)
    nb = lat.nb
    dv = {kxp: vx, kxm: -vx, kyp: vy, kym: -vy}
    for i in patch:
        i = int(i)
        for k, v in dv.items():
            lat.phi[i, k] += v
            j = int(nb[i, k])
            if j >= 0 and j not in patch_set:
                lat.phi[j, int(_MIRROR[k])] -= v


def read_transverse(lat, faces, patch: np.ndarray) -> np.ndarray:
    kxp, kxm, kyp, kym = faces
    vx = float(np.mean((lat.phi[patch, kxp] - lat.phi[patch, kxm]) / 2.0))
    vy = float(np.mean((lat.phi[patch, kyp] - lat.phi[patch, kym]) / 2.0))
    return np.array([vx, vy])


def _reset(lat):
    lat.u[:] = 0.0
    lat.phi[:] = 0.0


def evolve_record(
    lat,
    faces,
    dt,
    checkpoints,
    patches,
    n_threads,
    bc=None,
    noise=0.0,
    noise_mode="boundary",
    rng=None,
    u_profile_bins=None,
):
    """Step to each checkpoint (in ticks), recording the transverse vector at
    every patch; optionally record a |u| z-profile.  Returns
    (responses[name][ick] -> vec2, profiles[ick] -> (bin_centres, u_rms))."""
    responses = {name: [] for name in patches}
    profiles = []
    z = lat.pos[:, 2]

    def snapshot():
        for name, p in patches.items():
            responses[name].append(read_transverse(lat, faces, p))
        if u_profile_bins is not None:
            idx = np.digitize(z, u_profile_bins)
            rms = np.array(
                [
                    np.sqrt(np.mean(lat.u[idx == b] ** 2)) if np.any(idx == b) else 0.0
                    for b in range(1, len(u_profile_bins))
                ]
            )
            profiles.append(rms)

    step = 0
    for ck in checkpoints:
        while step < ck:
            lat.step(dt, n_threads=n_threads)
            if noise > 0.0:
                if noise_mode == "boundary":
                    bc.thermalize(lat, noise)
                else:  # bulk: zero-mean essence noise everywhere (ambient bath)
                    eta = noise * rng.standard_normal(len(lat.u))
                    lat.u += eta - eta.mean()
            step += 1
        snapshot()
    return {k: np.array(v) for k, v in responses.items()}, np.array(profiles)


def sample_statistics(RA_c, RA_s, RB_c, RB_s, noiseA, noiseB, rng, n):
    """Monte Carlo over the hidden variable alpha ~ U(0, pi) using recorded
    basis responses: v_A(alpha) = cos(a)*RA_c + sin(a)*RA_s (likewise B).
    Returns dict with CHSH S, per-setting correlations, and cascade
    P_agree(30 deg)."""
    al = rng.uniform(0.0, np.pi, n)
    ca, sa = np.cos(al)[:, None], np.sin(al)[:, None]
    vA = ca * RA_c + sa * RA_s
    vB = ca * RB_c + sa * RB_s
    if noiseA is not None and len(noiseA):
        k = rng.integers(0, len(noiseA), n)
        vA = vA + noiseA[k]
        vB = vB + noiseB[k]

    def outcomes(v, theta):
        ang = np.arctan2(v[:, 1], v[:, 0])
        p_plus = np.cos(theta - ang) ** 2
        return np.where(rng.random(n) < p_plus, 1.0, -1.0)

    def C(ta, tb):
        return float(np.mean(outcomes(vA, ta) * outcomes(vB, tb)))

    a, a_p, b, b_p = (CHSH_ANGLES[k] for k in ("a", "a_p", "b", "b_p"))
    corr = {
        ("a", "b"): C(a, b),
        ("a", "b_p"): C(a, b_p),
        ("a_p", "b"): C(a_p, b),
        ("a_p", "b_p"): C(a_p, b_p),
    }
    S = abs(
        corr[("a", "b")]
        - corr[("a", "b_p")]
        + corr[("a_p", "b")]
        + corr[("a_p", "b_p")]
    )
    C30 = C(0.0, FEYNMAN_ANGLE)
    return dict(S=S, correlations=corr, C30=C30, p_agree_30=(1.0 - C30) / 2.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--radius", type=float, default=6.0)
    ap.add_argument("--half-length", type=float, default=24.0)
    ap.add_argument("--sep", type=float, default=14.0, help="A-B separation D")
    ap.add_argument("--patch-r", type=float, default=1.6)
    ap.add_argument("--checkpoints", type=str, default="0,25,50,100,200")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--noise-mode", choices=("boundary", "bulk"), default="boundary")
    ap.add_argument("--noise-k", type=int, default=16, help="noise-only evolutions")
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    checkpoints = [int(s) for s in args.checkpoints.split(",")]

    lat = DampedLattice.cylinder(args.radius, args.half_length)
    faces = detect_transverse_faces(lat)
    dt = lat.stable_dt()
    bc = ReciprocalBC(lat, rng=rng)
    zA, zB = -args.sep / 2.0, +args.sep / 2.0
    patches = dict(
        A=patch_indices(lat, (0, 0, zA), args.patch_r),
        B=patch_indices(lat, (0, 0, zB), args.patch_r),
    )
    print(
        f"cells={len(lat.pos)}  dt={dt:.4f}  |A patch|={len(patches['A'])}  "
        f"|B patch|={len(patches['B'])}  D={args.sep}  noise={args.noise} "
        f"({args.noise_mode})"
    )

    # --- four single-site basis evolutions -------------------------------
    bins = np.linspace(-args.half_length, args.half_length, 25)
    basis, prof = {}, None
    for name, (site, vx, vy) in dict(
        ax=("A", 1.0, 0.0),
        ay=("A", 0.0, 1.0),
        bx=("B", 1.0, 0.0),
        by=("B", 0.0, 1.0),
    ).items():
        _reset(lat)
        imprint_transverse(lat, faces, patches[site], vx, vy)
        resp, p = evolve_record(
            lat,
            faces,
            dt,
            checkpoints,
            patches,
            args.threads,
            u_profile_bins=(bins if name == "ax" else None),
        )
        basis[name] = resp
        if name == "ax":
            prof = p

    # --- K noise-only evolutions (linearity: additive to any signal) -----
    noiseA = noiseB = None
    if args.noise > 0.0:
        nA, nB = [], []
        for _ in range(args.noise_k):
            _reset(lat)
            resp, _ = evolve_record(
                lat,
                faces,
                dt,
                checkpoints,
                patches,
                args.threads,
                bc=bc,
                noise=args.noise,
                noise_mode=args.noise_mode,
                rng=rng,
            )
            nA.append(resp["A"])
            nB.append(resp["B"])
        noiseA = np.array(nA)  # (K, n_ck, 2)
        noiseB = np.array(nB)

    # --- compose singlet responses and sample ----------------------------
    print(
        f"\n{'ticks':>6} {'t':>8} {'|v_A|':>9} {'xtalk':>9} "
        f"{'S':>7} {'P_agr(30)':>10}   [LHV cap 2/3=0.667, QM 3/4]"
    )
    results = []
    for ick, ck in enumerate(checkpoints):
        RA_c = basis["ax"]["A"][ick] + basis["by"]["A"][ick]
        RA_s = basis["ay"]["A"][ick] - basis["bx"]["A"][ick]
        RB_c = basis["ax"]["B"][ick] + basis["by"]["B"][ick]
        RB_s = basis["ay"]["B"][ick] - basis["bx"]["B"][ick]
        stats = sample_statistics(
            RA_c,
            RA_s,
            RB_c,
            RB_s,
            None if noiseA is None else noiseA[:, ick],
            None if noiseB is None else noiseB[:, ick],
            rng,
            args.samples,
        )
        amp = float(np.linalg.norm(basis["ax"]["A"][ick]))
        xtalk = float(np.linalg.norm(basis["ax"]["B"][ick]))  # A's wave seen at B
        print(
            f"{ck:>6} {ck * dt:>8.2f} {amp:>9.2e} {xtalk:>9.2e} "
            f"{stats['S']:>7.3f} {stats['p_agree_30']:>10.4f}"
        )
        results.append(dict(ticks=ck, t=ck * dt, amp=amp, xtalk=xtalk, **stats))

    # --- gravity diagnostic: static u structure along the A-B line? ------
    mid = (np.abs(lat.pos[:, 2]) < args.sep / 2 - args.patch_r) & (
        np.linalg.norm(lat.pos[:, :2], axis=1) < 2.0
    )
    print(
        f"\nu-profile (A-x imprint run): final u_rms on A-B axis segment "
        f"({int(mid.sum())} cells) = {np.sqrt(np.mean(lat.u[mid]**2)):.3e}"
    )
    print("z-binned u_rms at final checkpoint:")
    centres = (bins[:-1] + bins[1:]) / 2
    for c, v in zip(centres, prof[-1]):
        print(
            f"  z={c:+7.2f}  u_rms={v:.3e}"
            + ("  <-- A" if abs(c - zA) < 1 else "  <-- B" if abs(c - zB) < 1 else "")
        )

    if args.outdir:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / "bell_canal_stage1.npz",
            checkpoints=np.array(checkpoints),
            dt=dt,
            results=np.array(
                [
                    (r["ticks"], r["t"], r["amp"], r["xtalk"], r["S"], r["p_agree_30"])
                    for r in results
                ]
            ),
            u_profile_bins=bins,
            u_profiles=prof,
            **{f"basis_{k}_{s}": v[s] for k, v in basis.items() for s in ("A", "B")},
        )
        print(f"\nsaved -> {out / 'bell_canal_stage1.npz'}")


if __name__ == "__main__":
    main()
