"""
PHARE spheromak — tiny-step diagnostic run for DST lam calibration.

Runs the Taylor spheromak IC (identical to plasmoses.py) for a small number
of steps and saves proton density + B at every step.  The resulting density
snapshots are extracted to a .npz file so explore_taylor.py can seed lam
from the PHARE density profile.

Mapping:
    PHARE proton density n_i  <->  DST lam
    PHARE B field              <->  DST phi face fluxes

Run (from eq repo root, with pyphare on PYTHONPATH):
    PYTHONPATH=/home/deegan/git/phare/master/pyphare:$PWD \\
        python3 run/substrate/phare_spheromak_tiny.py

Extract after run:
    PYTHONPATH=/home/deegan/git/phare/master/pyphare:$PWD \\
        python3 run/substrate/phare_spheromak_tiny.py --extract
"""

import argparse

import numpy as np
from scipy.special import spherical_jn

try:
    import pyphare.pharein as ph
    from pyphare.simulator.simulator import Simulator
except Exception:
    print("PHARE not found not found or some other related error, skipping")
    ph = None
    Simulator = None


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

DIAG_DIR = "phare_outputs/sph_tiny"

cells = (101, 101, 101)
dl = (0.1, 0.1, 0.1)

time_step = 0.001
n_steps = 250  # tiny run — increase to taste
final_time = time_step * n_steps
ppc = 500
timestamps = np.round(np.arange(0, final_time + time_step * 0.5, time_step), 10)


# ---------------------------------------------------------------------------
# Taylor spheromak field (identical to plasmoses.py)
# ---------------------------------------------------------------------------


def get_taylor_spheromak(x, y, z, cx, cy, cz, R=1.0, B0=1.0):
    dx, dy, dz = x - cx, y - cy, z - cz
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    mask = (r <= R) & (r > 0)

    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)

    k = 4.493 / R
    rho = k * r[mask]
    j1r = spherical_jn(1, rho)
    cf = B0 * j1r / r[mask]

    bx[mask] = cf * (dz[mask] * dx[mask] / r[mask] - dy[mask])
    by[mask] = cf * (dz[mask] * dy[mask] / r[mask] + dx[mask])
    bz[mask] = cf * (-(dx[mask] ** 2 + dy[mask] ** 2) / r[mask])

    return bx, by, bz


# ---------------------------------------------------------------------------
# PHARE config
# ---------------------------------------------------------------------------


def config():
    sim = ph.Simulation(
        time_step=time_step,
        final_time=final_time,
        dl=dl,
        cells=cells,
        refinement="tagging",
        max_nbr_levels=1,
        hyper_resistivity=0.001,
        resistivity=0.001,
        diag_options={
            "format": "pharevtkhdf",
            "options": {"dir": DIAG_DIR},
        },
        strict=False,
    )

    L = sim.simulation_domain()
    mid = np.array(L) / 2
    cx, cy, cz = mid[0], mid[1], mid[2]

    def density(x, y, z):
        return 0.5

    def bx(x, y, z):
        return get_taylor_spheromak(x, y, z, cx, cy, cz)[0]

    def by(x, y, z):
        return get_taylor_spheromak(x, y, z, cx, cy, cz)[1]

    def bz(x, y, z):
        return get_taylor_spheromak(x, y, z, cx, cy, cz)[2]

    def vxyz(x, y, z):
        return 0.0

    def vthxyz(x, y, z):
        return 0.00001

    C = "xyz"
    vvv = {
        **{f"vbulk{c}": vxyz for c in C},
        **{f"vth{c}": vthxyz for c in C},
        "nbr_part_per_cell": ppc,
    }
    ph.MaxwellianFluidModel(
        bx=bx,
        by=by,
        bz=bz,
        protons={
            "charge": 1,
            "mass": 1,
            "density": density,
            **vvv,
            "init": {"seed": 12334},
        },
    )
    ph.ElectronModel(closure="isothermal", Te=0.0)

    ph.FluidDiagnostics(
        quantity="density",
        write_timestamps=timestamps,
        population_name="protons",
    )
    for qty in ["E", "B"]:
        ph.ElectromagDiagnostics(quantity=qty, write_timestamps=timestamps)

    return sim


# ---------------------------------------------------------------------------
# Post-run extraction: density + B → DST lam npz
# ---------------------------------------------------------------------------


def _assemble(hier, time, key):
    """
    Assemble all level-0 patches for field `key` at `time` into a single array.
    Uses pd.box.lower + actual trimmed shape for placement — handles both primal
    (N+1 points) and dual (N points) fields without assuming box.upper alignment.
    """
    level = hier.level(0, time)
    patches = level.patches

    # first pass: determine full array shape from actual trimmed extents
    full_shape = np.zeros(3, dtype=int)
    for patch in patches:
        pd = patch.patch_datas[key]
        g = pd.ghosts_nbr
        sl = tuple(slice(int(gi), -int(gi)) if gi > 0 else slice(None) for gi in g)
        trimmed_shape = np.array(pd.dataset[sl].shape)
        full_shape = np.maximum(full_shape, pd.box.lower + trimmed_shape)

    full = np.zeros(tuple(full_shape), dtype=np.float32)

    # second pass: place each patch
    for patch in patches:
        pd = patch.patch_datas[key]
        g = pd.ghosts_nbr
        sl = tuple(slice(int(gi), -int(gi)) if gi > 0 else slice(None) for gi in g)
        trimmed = pd.dataset[sl]
        lo = pd.box.lower
        s = trimmed.shape
        full[lo[0] : lo[0] + s[0], lo[1] : lo[1] + s[1], lo[2] : lo[2] + s[2]] = trimmed

    return full


def extract(diag_dir=DIAG_DIR, out_npz="sph_tiny_density.npz"):
    """
    Read PHARE diagnostics and save to npz:

        times          : (T,)             — saved timestamps
        density        : (T, Nx, Ny, Nz) — proton n_i (cell-centred)
        x / y / z      : (N,)            — cell-centre coordinates for density
        delta_n        : (Nx, Ny, Nz)    — n_i(t_final) - n_i(t=0)

        bx / by / bz   : (T, *, *, *)    — B components on their staggered grids
                                            (shapes differ from density by ±1 per axis)

    delta_n is the key output for DST lam seeding.
    """
    from pyphare.pharesee.run import Run

    run = Run(diag_dir)
    all_t = run.all_times()
    print("Available diagnostics:", list(all_t.keys()))

    times = sorted(all_t.get("protons_density", all_t.get("EM_B", [])))
    if not times:
        print("No recognised timestamps found.")
        return
    print(f"Extracting {len(times)} timestamps: {times}")

    densities = []
    b_snaps = {c: [] for c in "xyz"}
    n_shape = None

    for t in times:
        # --- density: assemble all patches ---
        ni_hier = run.GetN(t, pop_name="protons")
        ni_key = next(iter(ni_hier.level(0, t).patches[0].patch_datas))
        ni = _assemble(ni_hier, t, ni_key)
        if n_shape is None:
            n_shape = ni.shape
        densities.append(ni)

        # --- B: assemble each component separately (different staggered shapes) ---
        b_hier = run.GetB(t, all_primal=False)
        bkeys = sorted(b_hier.level(0, t).patches[0].patch_datas.keys())
        for c, key in zip("xyz", bkeys):
            b_snaps[c].append(_assemble(b_hier, t, key))

    density = np.stack(densities)
    delta_n = density[-1] - density[0]

    # coordinates: primal points go 0, dx, 2dx, ... N*dx  (N+1 values for N cells)
    dx = dl[0]
    x = np.arange(n_shape[0]) * dx
    y = np.arange(n_shape[1]) * dx
    z = np.arange(n_shape[2]) * dx

    arrays = dict(
        times=np.array(times),
        density=density,
        x=x,
        y=y,
        z=z,
        delta_n=delta_n,
        bx=np.stack(b_snaps["x"]),
        by=np.stack(b_snaps["y"]),
        bz=np.stack(b_snaps["z"]),
    )

    np.savez_compressed(out_npz, **arrays)
    print(f"Saved → {out_npz}")
    print(
        f"  density shape : {density.shape}  range [{density.min():.4f}, {density.max():.4f}]"
    )
    print(f"  delta_n range : [{delta_n.min():.4e}, {delta_n.max():.4e}]")
    for c in "xyz":
        print(f"  b{c} shape    : {arrays[f'b{c}'].shape}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if ph is None:
        raise SystemExit("PHARE not found -- cannot run (import failed above)")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extract",
        action="store_true",
        help="skip simulation, just extract saved diagnostics to npz",
    )
    parser.add_argument("--out", default="sph_tiny_density.npz")
    args = parser.parse_args()

    if args.extract:
        extract(out_npz=args.out)
    else:
        print(f"Running Taylor spheromak: {n_steps} steps, dt={time_step}")
        print(f"  domain: {cells}  dl={dl}")
        print(f"  timestamps: {timestamps}")
        print(f"  output: {DIAG_DIR}/")
        Simulator(config()).run()
        print("Done.  Re-run with --extract to save density snapshots to npz.")
