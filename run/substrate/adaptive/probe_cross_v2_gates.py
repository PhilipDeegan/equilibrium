"""
step_cross v2 gate comparison — D1 decision probe (doc/step_cross_v2.md §10).

Two configurations, same knot scale, same domain:
  electron — double-ring (poloidal Taylor component), lam = 0, zero-net-force
             IC.  Requirement R1: must NOT erode at rest.
  neutron  — full Taylor spheromak (toroidal + poloidal, Hopf-linked), lam
             frozen at lam_target.  Requirement R2: MUST keep unwinding.

Arms: gate v1 (reference), shear, product, opposed, curl — the latter four
swept over the saturation reference `ref` of g(x) = x/(x+ref). `opposed` is
the >90 deg-only variant of shear (o = max(0, -v_i.v_nb)): exactly zero on any
static loop family whose per-hop rotation stays below 90 deg, targeting the
shear gate's predicted failure mode (firing on the electron's own ring
curvature, not just on genuine linkage). `curl` is the holonomy arm: phi read
as a discrete connection, B_i built from the three Cartesian-plaquette Wilson
loops (cross_v2.CrossV2.curl_vectors), gated on the same shear-style
misalignment but applied to B_i instead of v_i. Unlike shear/product/opposed
(each a function of one cell's own 14 phi values compared to one neighbour's),
B_i genuinely mixes three cells' phi per component and is EXACTLY zero (not
asymptotically small) on any spatially uniform coherent-transit pattern —
the empirical test of whether reading phi as a relational/topological
quantity (field lines, not field values) succeeds where the one-ring-out
vector comparison was shown to structurally fail on R2 (see
doc/mems/project_step_cross_v2.md's cross_scale sweep follow-up).

Audit phase first: each IC evolves cross-free for --n-audit steps while the
raw gate variables (shear m, product q, opposed o) are sampled over engaged faces
(inward phi above threshold).  With --refs auto the scan grid is the
neutron's {p50, p90, p99} of the final audit sample — anchored to what must
fire, so the electron's survival at each ref directly measures the
separation window (or its absence, which is also a result).

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_cross_v2_gates.py
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_cross_v2_gates.py \
      --audit-only
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_cross_v2_gates.py \
      --R 20 --n-steps 8000 --gates shear --refs 1e-4,1e-3
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from run.substrate.adaptive.cross_v2 import CrossV2
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    face_flux_geometry,
    init_electron_dipole_ic,
    init_spheromak_ic,
    phi_poloidal_scan,
    phi_toroidal_scan,
)

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=15.0)
parser.add_argument("--sph-frac", type=float, default=0.2, help="knot scale = R*frac")
parser.add_argument("--B0", type=float, default=1.0)
parser.add_argument(
    "--lam-target", type=float, default=1.0, help="neutron lam (frozen)"
)
parser.add_argument("--u-max", type=float, default=1.0, help="neutron u amplitude")
parser.add_argument("--cross-scale", type=float, default=0.01)
parser.add_argument("--ics", type=str, default="electron,neutron")
parser.add_argument("--gates", type=str, default="v1,shear,product,opposed,curl")
parser.add_argument(
    "--refs",
    type=str,
    default="auto",
    help="comma list of gate refs, or 'auto' = neutron audit {p50,p90,p99}",
)
parser.add_argument("--n-steps", type=int, default=4000)
parser.add_argument("--n-audit", type=int, default=600)
parser.add_argument("--audit-report", type=int, default=200)
parser.add_argument("--audit-only", action="store_true")
parser.add_argument("--report", type=int, default=250)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=10)
parser.add_argument(
    "--eng-frac",
    type=float,
    default=1e-3,
    help="engaged-face threshold: inward > frac * max|phi|",
)
parser.add_argument("--plot-out", type=str, default="probe_cross_v2_gates.png")
parser.add_argument("--no-plot", action="store_true")
parser.add_argument("--data-out", type=str, default=None)

S_SCAN_FRAC = (0.5, 1.0, 1.5)  # survival shells, units of knot scale

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def build_lattice(args):
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_inner = float(r_all.max()) * (1.0 - args.sponge)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0, damp_max=args.damp_max, r_all=r_all, r_inner=r_inner
        )
    )
    return alat, cl


def seed_ic(cl, ic, args, scale):
    """Returns the survival diagnostic closure for this configuration."""
    s_scan = tuple(f * scale for f in S_SCAN_FRAC)
    if ic == "electron":
        _, r_axis = init_electron_dipole_ic(
            cl,
            R_sph=1.0,
            scale=scale,
            B0=args.B0,
            lam_target=0.0,
            u_bg=0.0,
            amplitude=1.0,
        )

        def winding(centre=np.zeros(3)):
            mid, n_hat, phi_f = face_flux_geometry(cl, centre)
            pol = phi_poloidal_scan(mid, n_hat, phi_f, r_axis, s_scan)
            return float(np.abs(pol).max())

    elif ic == "neutron":
        init_spheromak_ic(
            cl,
            R_sph=1.0,
            scale=scale,
            B0=args.B0,
            lam_target=args.lam_target,
            u_max=args.u_max,
            amplitude=1.0,
        )

        def winding(centre=np.zeros(3)):
            mid, n_hat, phi_f = face_flux_geometry(cl, centre)
            tor = phi_toroidal_scan(mid, n_hat, phi_f, s_scan)
            return float(np.abs(tor).max())

    else:
        raise ValueError(ic)
    return winding


# ---------------------------------------------------------------------------
# Audit — raw gate-variable distributions, cross off
# ---------------------------------------------------------------------------


def engaged_percentiles(op, phi, eng_frac, raw):
    phiv = np.where(op.valid, phi, 0.0)
    inward = np.maximum(-phiv, 0.0)
    mask = inward > eng_frac * np.abs(phiv).max()
    vals = raw[mask]
    if vals.size == 0:
        return dict(p50=0.0, p90=0.0, p99=0.0, max=0.0, n=0)
    return dict(
        p50=float(np.percentile(vals, 50)),
        p90=float(np.percentile(vals, 90)),
        p99=float(np.percentile(vals, 99)),
        max=float(vals.max()),
        n=int(vals.size),
    )


def audit(ic, args, scale):
    alat, cl = build_lattice(args)
    seed_ic(cl, ic, args, scale)
    dt = compute_dt(cl)
    shear_op = CrossV2(cl, gate="shear")
    prod_op = CrossV2(cl, gate="product")
    opp_op = CrossV2(cl, gate="opposed")
    curl_op = CrossV2(cl, gate="curl")

    print(f"\n--- audit {ic}  (cross off, {args.n_audit} steps, dt={dt:.4e}) ---")
    print(
        f"{'step':>6} {'var':>8} {'p50':>10} {'p90':>10} {'p99':>10} {'max':>10} {'n':>7}"
    )
    out = {}
    step = 0
    while True:
        phi = np.array(cl.leaf_phi).T
        phiv = np.where(shear_op.valid, phi, 0.0)
        for name, op, raw in (
            ("shear", shear_op, shear_op.raw_shear(phiv)),
            ("product", prod_op, prod_op.raw_product(phiv)),
            ("opposed", opp_op, opp_op.raw_opposed(phiv)),
            ("curl", curl_op, curl_op.raw_curl(phiv)),
        ):
            st = engaged_percentiles(op, phi, args.eng_frac, raw)
            print(
                f"{step:6d} {name:>8} {st['p50']:10.3e} {st['p90']:10.3e}"
                f" {st['p99']:10.3e} {st['max']:10.3e} {st['n']:7d}",
                flush=True,
            )
            out[name] = st  # keep last sample
        if step >= args.n_audit:
            break
        for _ in range(args.audit_report):
            alat.step(0, dt)
            alat.step_sponge(0)
        step += args.audit_report
    return out


# ---------------------------------------------------------------------------
# Decay run — one arm
# ---------------------------------------------------------------------------


def run_arm(ic, gate, ref, args, scale, save_state=False):
    alat, cl = build_lattice(args)
    winding = seed_ic(cl, ic, args, scale)
    dt = compute_dt(cl)
    if gate == "barrier":
        op = CrossV2(cl, gate=gate, m0=ref if ref is not None else 1.0)
    else:
        op = CrossV2(cl, gate=gate, ref=ref if ref is not None else 1.0)

    w0 = winding()
    phi_rms0 = float(np.sqrt(np.mean(np.array(cl.leaf_phi) ** 2)))
    hist = [(0.0, 1.0, 1.0, _engagement(op, args))]
    if gate == "v1":
        label = gate
    elif gate == "barrier":
        label = f"barrier/m0={ref:.2e}"
    else:
        label = f"{gate}/ref={ref:.2e}"
    print(f"\n--- {ic}  {label}  cs={args.cross_scale}  dt={dt:.4e} ---")
    print(f"{'step':>6} {'surv':>8} {'rms':>8} {'<G>eng':>8}")
    print(f"{0:6d} {1.0:8.4f} {1.0:8.4f} {hist[0][3]:8.4f}", flush=True)

    for step in range(1, args.n_steps + 1):
        alat.step(0, dt)
        alat.step_sponge(0)
        if args.cross_scale > 0.0:
            op.step(dt, args.cross_scale)
        if step % args.report == 0:
            surv = winding() / max(w0, 1e-30)
            rms = float(np.sqrt(np.mean(np.array(cl.leaf_phi) ** 2))) / max(
                phi_rms0, 1e-30
            )
            eng = _engagement(op, args)
            hist.append((step * dt, surv, rms, eng))
            print(f"{step:6d} {surv:8.4f} {rms:8.4f} {eng:8.4f}", flush=True)
    result = dict(ic=ic, gate=gate, ref=ref, label=label, dt=dt, hist=np.array(hist))
    if save_state:
        result["final_phi"] = np.array(cl.leaf_phi)
        result["final_pos"] = np.array(cl.leaf_pos)
        result["final_lam"] = np.array(cl.leaf_lam)
        result["final_u"] = np.array(cl.leaf_u)
    return result


def _engagement(op, args):
    """Mean gate value over engaged (inward) faces — how much the arm fires."""
    phi = np.array(op.cl.leaf_phi).T
    G, _ = op.gate_field(phi)
    phiv = np.where(op.valid, phi, 0.0)
    inward = np.maximum(-phiv, 0.0)
    mask = inward > args.eng_frac * max(np.abs(phiv).max(), 1e-30)
    return float(G[mask].mean()) if mask.any() else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parser.parse_args()
    ics = [s.strip() for s in args.ics.split(",")]
    gates = [s.strip() for s in args.gates.split(",")]
    scale = args.R * args.sph_frac
    print(
        f"R={args.R}  scale={scale:.1f}  cs={args.cross_scale}  ics={ics}  gates={gates}"
    )

    audits = {ic: audit(ic, args, scale) for ic in ics}
    if args.audit_only:
        return

    if args.refs == "auto":
        # p50 fires on nearly everything (≈ v1); the interesting window is at
        # and above the neutron's engaged tail
        src = audits.get("neutron") or audits[ics[0]]
        refs = {"barrier": [1.6e-1, 3.0e-1, 2.49]}
        refs.update(
            {
                g: [src[g]["p90"], src[g]["p99"], 3.0 * src[g]["p99"]]
                for g in ("shear", "product", "opposed", "curl")
            }
        )
    else:
        lst = [float(v) for v in args.refs.split(",")]
        refs = {
            "shear": lst,
            "product": lst,
            "opposed": lst,
            "curl": lst,
            "barrier": lst,
        }
    for g in ("shear", "product", "opposed", "curl", "barrier"):
        if g in gates:
            print(f"refs[{g}] = {['%.3e' % r for r in refs[g]]}")

    arms = []
    for g in gates:
        arms += [(g, None)] if g == "v1" else [(g, r) for r in refs[g]]

    results = [run_arm(ic, g, r, args, scale) for ic in ics for g, r in arms]

    # summary — survival at end + separation per (gate, ref)
    print(f"\n{'gate':>22} {'surv_e':>8} {'surv_n':>8} {'separation e/n':>15}")
    by_label = {}
    for r in results:
        by_label.setdefault(r["label"], {})[r["ic"]] = r["hist"][-1][1]
    for label, d in by_label.items():
        se, sn = d.get("electron", np.nan), d.get("neutron", np.nan)
        sep = se / sn if sn and not np.isnan(sn) else np.nan
        print(f"{label:>22} {se:8.4f} {sn:8.4f} {sep:15.2f}")

    if args.data_out:
        np.savez(
            args.data_out,
            **{
                f"{r['ic']}__{r['label'].replace('/', '_')}": r["hist"] for r in results
            },
        )
        print(f"data saved → {args.data_out}")

    if args.no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(ics), figsize=(7 * len(ics), 5.5), squeeze=False)
    fig.suptitle(
        f"step_cross v2 gate comparison  R={args.R} scale={scale:.1f}"
        f" cs={args.cross_scale}",
        fontsize=10,
    )
    cmap = plt.cm.tab10
    labels = sorted({r["label"] for r in results})
    color = {lb: cmap(i % 10) for i, lb in enumerate(labels)}
    for ax, ic in zip(axes[0], ics):
        for r in [r for r in results if r["ic"] == ic]:
            h = r["hist"]
            ax.semilogy(
                h[:, 0],
                np.maximum(h[:, 1], 1e-6),
                lw=1.5,
                color=color[r["label"]],
                label=r["label"],
            )
        ax.set_title(f"{ic} — winding survival")
        ax.set_xlabel("time")
        ax.set_ylabel("survival (log)")
        ax.axhline(1.0, color="k", lw=0.5, ls=":")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=130, bbox_inches="tight")
    print(f"plot saved → {args.plot_out}")


if __name__ == "__main__":
    main()
