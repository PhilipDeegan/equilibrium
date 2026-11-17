"""
Casimir energy between two parallel plates on the Kelvin cell substrate.

Method: exact lattice mode sum.

The BCC lattice (conventional cube side a) stacked along z is a sequence of
square-lattice layers (spacing a) at layer spacing h = a/2, alternately offset
by (h, h). For each transverse momentum (Kx, Ky) the layers form a chain:

    omega^2 u_j = D0 u_j - t1 (u_{j+1} + u_{j-1}) - t2 (u_{j+2} + u_{j-2})
    D0 = 8 W_HEX^2 + 2 W_SQ^2 + W_SQ^2 (4 - 2 cos(aKx) - 2 cos(aKy))
    t1 = 4 W_HEX^2 cos(hKx) cos(hKy)       (hex faces, next layer)
    t2 = W_SQ^2                            (square faces, two layers on)

which reproduces the bulk dispersion
    omega^2 = 8 W_HEX^2 (1 - cos(hKx)cos(hKy)cos(hKz))
            + W_SQ^2 (6 - 2(cos(aKx) + cos(aKy) + cos(aKz))).

Plates are u = 0 layers at j = 0 and j = N, two layers thick (the square
faces reach two layers, so a single fixed layer is hopped over); d = N h.
E(N) = 1/2 sum omega per unit area, averaged over the transverse zone
[-pi/a, pi/a]^2. Every mode is gapped by the plates, so the transverse
midpoint sum converges exponentially once the transverse period nxy a is
well above d (nxy >~ 3 N_max: nxy = 96 with N <= 32 gives the continuum
ratio to 2e-4; nxy = 64 with N up to 40 is off by 2e-3).

E(N) = alpha N + beta + gamma / N^3 + g4 / N^4 + g5 / N^5 + ...
alpha is the bulk vacuum energy, beta the surface term, g4 the effective
plate position, g5 the leading lattice correction. gamma h^3, the 1/d^3
coefficient, is the Casimir term: it equals the scalar Dirichlet
continuum value -pi^2 c / (1440 d^3), c = sqrt(W_HEX^2 + W_SQ^2) a, for any
weights (exp/2026/09/29/casimir/status.md) -- it is set by the long-wavelength
end alone and carries no lattice-scale information.
"""

import numpy as np

from equilibrium.substrate.lattice import W_HEX, W_SQ  # noqa: E402


def _dispersion(Kx, Ky, Kz, a: float) -> np.ndarray:
    """Exact Kelvin lattice dispersion omega(Kx, Ky, Kz)."""
    h = a / 2.0
    w2 = 8.0 * W_HEX**2 * (
        1.0 - np.cos(h * Kx) * np.cos(h * Ky) * np.cos(h * Kz)
    ) + W_SQ**2 * (6.0 - 2.0 * (np.cos(a * Kx) + np.cos(a * Ky) + np.cos(a * Kz)))
    return np.sqrt(np.maximum(w2, 0.0))


def wave_speed(a: float = 1.0, w_hex: float = W_HEX, w_sq: float = W_SQ) -> float:
    return float(np.sqrt(w_hex**2 + w_sq**2) * a)


def casimir_continuum(d: float, c: float) -> float:
    """Scalar field, Dirichlet plates: E/A = -pi^2 c / (1440 d^3)."""
    return -(np.pi**2) * c / (1440.0 * d**3)


def plate_energy(
    n_layers: int,
    a: float = 1.0,
    nxy: int = 96,
    w_hex: float = W_HEX,
    w_sq: float = W_SQ,
    chunk: int = 4096,
) -> float:
    """Zero-point energy per unit area, 1/2 sum omega, between plates d = n_layers a/2."""
    h = a / 2.0
    dk = 2.0 * np.pi / (nxy * a)
    k = np.arange(nxy) * dk - np.pi / a + dk / 2.0
    Kx, Ky = np.meshgrid(k, k, indexing="ij")
    Kx, Ky = Kx.ravel(), Ky.ravel()
    D0 = 8 * w_hex**2 + 2 * w_sq**2 + w_sq**2 * (4 - 2 * np.cos(a * Kx) - 2 * np.cos(a * Ky))
    t1 = 4 * w_hex**2 * np.cos(h * Kx) * np.cos(h * Ky)

    m = n_layers - 1
    i = np.arange(m)
    total = 0.0
    for s in range(0, len(Kx), chunk):
        sl = slice(s, s + chunk)
        M = np.zeros((len(Kx[sl]), m, m))
        M[:, i, i] = D0[sl, None]
        M[:, i[:-1], i[:-1] + 1] = -t1[sl, None]
        M[:, i[:-1] + 1, i[:-1]] = -t1[sl, None]
        M[:, i[:-2], i[:-2] + 2] = -(w_sq**2)
        M[:, i[:-2] + 2, i[:-2]] = -(w_sq**2)
        total += np.sqrt(np.maximum(np.linalg.eigvalsh(M), 0.0)).sum()
    return 0.5 * total / len(Kx) / a**2


def casimir_coefficient(
    ns=(10, 12, 16, 20, 24, 32),
    a: float = 1.0,
    nxy: int = 96,
    w_hex: float = W_HEX,
    w_sq: float = W_SQ,
    powers=(3, 4, 5),
) -> dict:
    """
    Fit E(N) = alpha N + beta + sum_p g_p / N^p over plate separations and
    return the 1/d^3 coefficient against the continuum.

    Keys: alpha (bulk, per layer), beta (surface), gamma (1/N^3 coefficient),
    casimir (gamma h^3: the 1/d^3 coefficient), continuum (-pi^2 c / 1440),
    ratio, c, coef (all fitted coefficients), E, ns.
    """
    ns = np.asarray(ns)
    E = np.array([plate_energy(int(n), a, nxy, w_hex, w_sq) for n in ns])
    A = np.stack(
        [ns, np.ones_like(ns)] + [1.0 / ns.astype(float) ** p for p in powers], axis=1
    ).astype(float)
    coef = np.linalg.lstsq(A, E, rcond=None)[0]
    h = a / 2.0
    c = wave_speed(a, w_hex, w_sq)
    gamma = float(coef[2])
    casimir = gamma * h**3
    continuum = casimir_continuum(1.0, c)
    return {
        "alpha": float(coef[0]),
        "beta": float(coef[1]),
        "gamma": gamma,
        "casimir": casimir,
        "continuum": continuum,
        "ratio": casimir / continuum,
        "c": c,
        "coef": coef,
        "E": E,
        "ns": ns,
    }
