"""
PHARE hybrid mirror instability — 2D homogeneous anisotropic plasma.

Reference case for comparing against DST's occupation-field (occ) anisotropy
model (equilibrium/substrate/ab.py kernels, exercised in
vis/substrate/plot_brazil_instabilities.py). Both sides use the same
stability thresholds (CGL/kinetic-MHD, normalised units mu0=1):

    beta_para = 2*p_para / B^2
    R         = p_perp / p_para   (= T_perp/T_para at fixed n)
    Mirror:    R > 1 + 1/beta_para
    Firehose:  R < 1 - 1/beta_para   (beta_para > 1 only)

IC: uniform B = B0 x_hat, uniform density n0, bi-Maxwellian protons with
vthx (parallel) != vthy = vthz (perpendicular) set from (beta_para, aniso)
so the plasma starts exactly at the requested (beta_para, R) point. No
explicit perturbation is injected — the instability grows out of PIC
particle noise, which is the physically correct seed for mirror/firehose.

Defaults (beta_para=3, aniso=2.5) sit at R=2.5 vs R_mirror=1+1/3=1.33,
comfortably mirror-unstable.

Usage:
    PYTHONPATH=/home/p/git/phare/master/pyphare:/home/p/git/phare/master/build:$PWD \\
        python3 run/substrate/mirror_instability_phare.py

    PYTHONPATH=... python3 run/substrate/mirror_instability_phare.py \\
        --beta-para 5 --aniso 0.4 --final-time 150   # firehose-unstable instead

Extract IC / early / late snapshots to npz after the run:
    PYTHONPATH=... python3 run/substrate/mirror_instability_phare.py --extract
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

DIAG_DIR = "phare_outputs/mirror_instability"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--beta-para",
        type=float,
        default=3.0,
        help="parallel plasma beta = 2*p_para/B0^2 (default 3.0)",
    )
    p.add_argument(
        "--aniso",
        type=float,
        default=2.5,
        help="R = T_perp/T_para (default 2.5, mirror-unstable; "
        "use <1 with beta-para>1 for firehose)",
    )
    p.add_argument("--B0", type=float, default=1.0)
    p.add_argument("--n0", type=float, default=1.0)
    p.add_argument("--Lx", type=float, default=100.0, help="box x width (d_i)")
    p.add_argument("--Ly", type=float, default=100.0, help="box y width (d_i)")
    p.add_argument("--dl", type=float, default=0.4, help="cell size (d_i)")
    p.add_argument("--dt", type=float, default=0.005, help="time step (1/Omega_ci)")
    p.add_argument("--final-time", type=float, default=150.0)
    p.add_argument("--ppc", type=int, default=100, help="particles per cell")
    p.add_argument(
        "--interp-order",
        type=int,
        default=2,
        help="must match the compiled pyphare module (dim=2,interp_order=2)",
    )
    p.add_argument(
        "--n-diag",
        type=int,
        default=30,
        help="number of diagnostic dumps over [0, final_time]",
    )
    p.add_argument(
        "--t-early",
        type=float,
        default=None,
        help="'a few steps' checkpoint time (default: 2 diag intervals in)",
    )
    p.add_argument("--seed", type=int, default=13579)
    p.add_argument(
        "--extract",
        action="store_true",
        help="skip the run, just extract saved diagnostics to npz",
    )
    p.add_argument("--out", default="mirror_instability_phare.npz")
    return p


# ---------------------------------------------------------------------------
# Simulation config
# ---------------------------------------------------------------------------


def build_config(args):
    import pyphare.pharein as ph

    ph.NO_GUI()

    Nx = int(round(args.Lx / args.dl))
    Ny = int(round(args.Ly / args.dl))

    n_diag = max(args.n_diag, 2)
    timestamps = np.linspace(0.0, args.final_time, n_diag + 1)

    sim = ph.Simulation(
        time_step=args.dt,
        final_time=args.final_time,
        cells=(Nx, Ny),
        dl=(args.dl, args.dl),
        interp_order=args.interp_order,
        diag_options={
            "format": "phareh5",
            "options": {"dir": DIAG_DIR, "mode": "overwrite"},
        },
    )

    T_para = args.beta_para * args.B0**2 / (2.0 * args.n0)
    T_perp = args.aniso * T_para
    vth_para = float(np.sqrt(T_para))
    vth_perp = float(np.sqrt(T_perp))

    def density(x, y):
        return args.n0

    def bx(x, y):
        return args.B0

    def by(x, y):
        return 0.0

    def bz(x, y):
        return 0.0

    def v0(x, y):
        return 0.0

    def vthx(x, y):
        return vth_para

    def vthy(x, y):
        return vth_perp

    def vthz(x, y):
        return vth_perp

    ph.MaxwellianFluidModel(
        bx=bx,
        by=by,
        bz=bz,
        protons={
            "charge": 1,
            "mass": 1,
            "density": density,
            "vbulkx": v0,
            "vbulky": v0,
            "vbulkz": v0,
            "vthx": vthx,
            "vthy": vthy,
            "vthz": vthz,
            "nbr_part_per_cell": args.ppc,
            "init": {"seed": args.seed},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.0)

    for qty in ["E", "B"]:
        ph.ElectromagDiagnostics(quantity=qty, write_timestamps=timestamps)
    for qty in ["density", "pressure_tensor"]:
        ph.FluidDiagnostics(
            quantity=qty, write_timestamps=timestamps, population_name="protons"
        )

    print(
        f"beta_para={args.beta_para}  R={args.aniso}  "
        f"vth_para={vth_para:.4f}  vth_perp={vth_perp:.4f}  "
        f"R_mirror={1+1/args.beta_para:.3f}  "
        f"R_firehose={(1-1/args.beta_para) if args.beta_para>1 else float('nan'):.3f}"
    )
    print(
        f"domain: {Nx}x{Ny} cells, dl={args.dl}  ppc={args.ppc}  "
        f"dt={args.dt}  final_time={args.final_time}"
    )
    print(f"timestamps: {timestamps}")

    return sim, timestamps


# ---------------------------------------------------------------------------
# Extraction: PHARE diagnostics -> npz snapshots at IC / early / late
# ---------------------------------------------------------------------------


def _assemble_2d(hier, time, key):
    """Assemble all level-0 patches for field `key` at `time` into one 2D array."""
    level = hier.level(0, time)
    patches = level.patches

    full_shape = np.zeros(2, dtype=int)
    for patch in patches:
        pd = patch.patch_datas[key]
        g = pd.ghosts_nbr
        sl = tuple(slice(int(gi), -int(gi)) if gi > 0 else slice(None) for gi in g)
        trimmed_shape = np.array(pd.dataset[sl].shape)
        full_shape = np.maximum(full_shape, pd.box.lower + trimmed_shape)

    full = np.zeros(tuple(full_shape), dtype=np.float32)
    for patch in patches:
        pd = patch.patch_datas[key]
        g = pd.ghosts_nbr
        sl = tuple(slice(int(gi), -int(gi)) if gi > 0 else slice(None) for gi in g)
        trimmed = pd.dataset[sl]
        lo = pd.box.lower
        s = trimmed.shape
        full[lo[0] : lo[0] + s[0], lo[1] : lo[1] + s[1]] = trimmed

    return full


def _nearest_time(available, target):
    available = sorted(available)
    return min(available, key=lambda t: abs(t - target))


def extract(args):
    from pyphare.pharesee.run import Run

    run = Run(DIAG_DIR)
    all_t = run.all_times()
    print("Available diagnostics:", list(all_t.keys()))

    b_times = sorted(all_t["EM_B"])
    final_time = b_times[-1]
    t_early = args.t_early
    if t_early is None:
        dt_diag = b_times[1] - b_times[0] if len(b_times) > 1 else final_time
        t_early = b_times[0] + 2 * dt_diag

    checkpoints = {
        "ic": _nearest_time(b_times, 0.0),
        "early": _nearest_time(b_times, t_early),
        "late": _nearest_time(b_times, final_time),
    }
    print(f"Checkpoints: {checkpoints}")

    out = {}
    for label, t in checkpoints.items():
        fields = {}

        b_hier = run.GetB(t, all_primal=False)
        bkeys = sorted(b_hier.level(0, t).patches[0].patch_datas.keys())
        for c, key in zip("xyz", bkeys):
            fields[f"b{c}"] = _assemble_2d(b_hier, t, key)

        n_hier = run.GetN(t, pop_name="protons")
        n_key = next(iter(n_hier.level(0, t).patches[0].patch_datas))
        fields["density"] = _assemble_2d(n_hier, t, n_key)

        # Run.GetPressure's internal M/Flux/N merge mismatches arrays for
        # this anisotropic bi-Maxwellian IC (protons_Pxx comes out negative,
        # which is impossible: P = M - F^2*mass/N is mass*n*Var(v) >= 0
        # algebraically). Bypass it: pull the raw momentum_tensor/flux/density
        # hierarchies (the same ones GetPressure uses internally) and compute
        # P = M - F^2*mass/N ourselves. Verified against this IC: gives
        # Pxx ~ [0.99, 2.04] (mean 1.50), matching the analytic n*T_para=1.5.
        mass = 1.0
        M = run._get_hier_for(t, "ions_pop_protons_momentum_tensor")
        F = run._get_hier_for(t, "ions_pop_protons_flux")
        m_patch_keys = M.level(0, t).patches[0].patch_datas.keys()
        f_comps = {c: _assemble_2d(F, t, f"protons_F{c}") for c in "xyz"}
        density = fields["density"]
        for a, b in [
            ("x", "x"),
            ("x", "y"),
            ("x", "z"),
            ("y", "y"),
            ("y", "z"),
            ("z", "z"),
        ]:
            comp = f"{a}{b}"
            m_key = (
                f"protons_M{comp}"
                if f"protons_M{comp}" in m_patch_keys
                else f"protons_M{comp[::-1]}"
            )
            Mab = _assemble_2d(M, t, m_key)
            fields[f"P{comp}"] = Mab - f_comps[a] * f_comps[b] * mass / density

        # bx/by/bz live on staggered (Yee) grids, offset by 1 cell from each
        # other and from density/pressure (primal). Crop everything to the
        # common minimum shape -- fine for this qualitative comparison, not
        # meant to be sub-cell accurate.
        common_shape = np.min([a.shape for a in fields.values()], axis=0)
        nx, ny = int(common_shape[0]), int(common_shape[1])
        for key, arr in fields.items():
            out[f"{label}_{key}"] = arr[:nx, :ny]

        out[f"{label}_time"] = np.float32(t)

    dx = args.dl
    nx, ny = out["ic_density"].shape
    out["x"] = np.arange(nx) * dx
    out["y"] = np.arange(ny) * dx
    out["beta_para_ic"] = np.float32(args.beta_para)
    out["aniso_ic"] = np.float32(args.aniso)
    out["B0"] = np.float32(args.B0)

    np.savez_compressed(args.out, **out)
    print(f"Saved -> {args.out}")
    for label in checkpoints:
        bmag = np.sqrt(
            out[f"{label}_bx"] ** 2 + out[f"{label}_by"] ** 2 + out[f"{label}_bz"] ** 2
        )
        print(
            f"  {label:5s} t={out[f'{label}_time']:.3f}  "
            f"|B| range [{bmag.min():.4f},{bmag.max():.4f}]"
        )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parser().parse_args()

    if args.extract:
        extract(args)
    else:
        from pyphare.simulator.simulator import Simulator, startMPI

        startMPI()
        sim, timestamps = build_config(args)
        Simulator(sim).run()
        print("Done. Re-run with --extract to save IC/early/late snapshots to npz.")
