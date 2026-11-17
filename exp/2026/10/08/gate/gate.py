"""10-08 gate (Philip: impose a classical threshold for when a structure may disassemble; the continuum walk lacks the
discrete geometry that would set it). Imposed calibration slot: rho_c.

Trigger: around the tube centre circle (rc, zc) at toroidal angle th, sample the density on the meridional circle of the
coil's own minor radius (8-nearest inverse-distance weights); the strand passes are the minima, and the density maximum
on the arc between two adjacent passes is their separating ridge. Passes merge (trigger) when a ridge < rho_c.
Release: reconnect the two passes there. Parallel strands a0 -> a1, b0 -> b1 (same toroidal sense) become a0 -> b1,
b0 -> a1; the change is the closed bow-tie a1 -> a0 -> b1 -> b0 -> a1, imprinted as psi *= exp(i q Omega_bowtie / 2)
(Omega mod 4 pi -> continuous for |q| = 1) and the amplitude zeroed on the two new bridges by d / sqrt(d^2 + xi^2);
the old cores on the cancelled chords are left for the walk to fill."""
import numpy as np

from equilibrium.substrate.walk import solid_angle, to_complex, to_real

NS = 256
PH = np.linspace(0, 2 * np.pi, NS, endpoint=False)


def ring(tree, rho, rc, zc, rad, th):
    """density on the meridional circle (radius rad about the tube centre) at toroidal angle th, and its points."""
    R = rc + rad * np.cos(PH)
    X = np.stack([R * np.cos(th), R * np.sin(th), zc + rad * np.sin(PH)], 1)
    d, i = tree.query(X, k=8)
    wt = 1 / np.maximum(d, 1e-6)**2
    return np.sum(wt * rho[i], 1) / wt.sum(1), X


def passes(prof, rmax=0.5):
    """indices of the pass minima (local minima below rmax, circular, one per dip) and the ridge max between neighbours."""
    s = np.convolve(np.r_[prof[-3:], prof, prof[:3]], np.ones(7) / 7, "valid")
    lo = np.nonzero((s < np.roll(s, 1)) & (s <= np.roll(s, -1)) & (s < rmax))[0]
    if len(lo) < 2:
        return lo, np.array([])
    ridge = []
    for j in range(len(lo)):
        a, b = lo[j], lo[(j + 1) % len(lo)]
        arc = np.r_[a:b] if b > a else np.r_[a:NS, 0:b]
        ridge.append(prof[arc].max())
    return lo, np.array(ridge)


def scan(tree, rho, rc, zc, rad, nth=64):
    """per toroidal angle: number of passes and the lowest ridge; returns (th, n, ridge_min, k of that ridge) arrays."""
    out = []
    for th in np.linspace(0, 2 * np.pi, nth, endpoint=False):
        prof, _ = ring(tree, rho, rc, zc, rad, th)
        lo, rg = passes(prof)
        out.append((th, len(lo), rg.min() if len(rg) else np.nan, int(np.argmin(rg)) if len(rg) else -1))
    return np.array(out)


def pass_points(tree, rho, rc, zc, rad, th, near):
    """the two pass positions at toroidal angle th whose poloidal angles are closest to near (2,) [rad]."""
    prof, X = ring(tree, rho, rc, zc, rad, th)
    lo, _ = passes(prof)
    out = []
    for ang in near:
        j = lo[np.argmin(np.abs(np.angle(np.exp(1j * (PH[lo] - ang)))))]
        out.append(X[j])
    return out


def reconnect(w, pos, tree, rho, rc, zc, rad, th, k, xi, q=1, dth=None, n_threads=8):
    """reconnect passes k, k+1 of the ring at th (as found by scan); returns the bow-tie corners."""
    prof, X = ring(tree, rho, rc, zc, rad, th)
    lo, _ = passes(prof)
    pa, pb = PH[lo[k]], PH[lo[(k + 1) % len(lo)]]
    dth = dth if dth is not None else 2 * xi / rc
    a0, b0 = pass_points(tree, rho, rc, zc, rad, th - dth, (pa, pb))
    a1, b1 = pass_points(tree, rho, rc, zc, rad, th + dth, (pa, pb))
    bow = np.array([a1, a0, b1, b0])
    Om = solid_angle(bow, pos, n_threads=n_threads)
    L, R = w.state()
    psi = np.concatenate([to_complex(L), to_complex(R)], 1)
    psi *= np.exp(0.5j * q * Om)[:, None]
    env = np.ones(len(pos))
    for P0, P1 in ((a0, b1), (b0, a1)):
        u = P1 - P0
        tt = np.clip((pos - P0) @ u / (u @ u), 0, 1)
        d = np.linalg.norm(pos - (P0 + tt[:, None] * u), axis=1)
        env *= d / np.sqrt(d**2 + xi**2)
    psi *= env[:, None]
    w.set_state(to_real(psi[:, :2]), to_real(psi[:, 2:]))
    return bow
