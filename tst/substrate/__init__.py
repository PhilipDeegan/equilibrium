import numpy as np

from equilibrium.substrate.lattice import SubstrateLattice


def build_substrate_sphere(R: float, u_origin: float = 1.0):
    """Build a SubstrateLattice BCC sphere with u=u_origin at the cell nearest origin."""
    sl = SubstrateLattice(radius_cells=R, spacing=1.0)
    r2 = (sl.pos**2).sum(axis=1)
    origin_idx = int(np.argmin(r2))
    sl.u[origin_idx] = u_origin
    return sl, origin_idx


def radial_profile(pos, values, centre, r_edges):
    """Mean of values in each radial shell defined by r_edges."""
    r = np.linalg.norm(pos - centre, axis=1)
    out = []
    for r0, r1 in zip(r_edges[:-1], r_edges[1:]):
        m = (r >= r0) & (r < r1)
        out.append(float(values[m].mean()) if m.any() else 0.0)
    return np.array(out)


def sph_bin(x, r_all, r_bins):
    """Per-shell mean of scalar field x (NaN for empty shells)."""
    out = np.empty(len(r_bins) - 1)
    for i in range(len(out)):
        m = (r_all >= r_bins[i]) & (r_all < r_bins[i + 1])
        out[i] = x[m].mean() if m.any() else np.nan
    return out


def lam_com(pos, lam):
    """Lam-weighted centre of winding."""
    total = lam.sum()
    if total < 1e-30:
        return np.zeros(3)
    return (lam[:, None] * pos).sum(axis=0) / total


def knot_mask(pos, centre, radius=2.5):
    """Boolean mask of cells within radius of centre."""
    return np.linalg.norm(pos - centre, axis=1) < radius


def u_slab(pos, u, z_lo, z_hi):
    """Mean u in a z-slab."""
    m = (pos[:, 2] >= z_lo) & (pos[:, 2] < z_hi)
    return float(u[m].mean()) if m.any() else 0.0


def axis_sel(pos, r_cut=1.5):
    """Return (mask, sort_order) for cells near the z-axis, sorted by z."""
    near = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2) < r_cut
    order = np.argsort(pos[near, 2])
    return near, order


def mpl_agg():
    """Lazy matplotlib import with Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
