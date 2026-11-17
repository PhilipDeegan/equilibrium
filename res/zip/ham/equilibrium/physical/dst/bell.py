"""
DST Bell inequality — photon polarization correlations.

Three models:

  classical_lhv
  ─────────────
  Pre-set polarizations (φ, φ+π/2).  A and B measure independently.
  C(Δθ) = −½ cos(2Δθ)   CHSH S = √2 ≈ 1.41  (no violation)
  Limitation: treats the canal as a factorisable classical field.

  dst_canal  (cascade approximation)
  ────────────────────────────────────
  A measures, re-emits canal at θ_A; B measures the re-projected canal.
  C(Δθ) = −cos(2Δθ)     CHSH S = 2√2 ≈ 2.83  (Bell violated)
  Limitation: requires timelike ordering A→B; breaks for spacelike separation.

  dst_joint_canal  (correct spacelike model)
  ───────────────────────────────────────────
  Both A and B sample the pre-existing non-factorisable joint canal state
  simultaneously.  No cascade, no ordering, no FTL required.
  The canal was established causally during entanglement; both measurements
  read it from the substrate that is already there.
  C(Δθ) = −cos(2Δθ)     CHSH S = 2√2 ≈ 2.83  (Bell violated)

DST is outside Bell's theorem — not a new way to violate it.

Bell's factorisation P(A,B|θ_A,θ_B,λ) = P(A|θ_A,λ)·P(B|θ_B,λ) requires λ
to be assignable separately to each particle.  In DST there are no particles —
both patterns live in the same substrate field u(x).  The hidden variable is the
global substrate state, which cannot be partitioned into one piece belonging to A
and another belonging to B.  The canal IS the substrate between them; it belongs
to neither pattern alone.  Bell's theorem simply does not apply: its premise
(independent particles with separable hidden variables) is false in DST.

The classical_lhv model correctly captures Bell's assumed world (independent
particles, separable λ).  dst_joint_canal is outside that world entirely.

Standard CHSH angles: a=0, a'=π/4, b=π/8, b'=3π/8.
"""

import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Core measurement primitive
# ---------------------------------------------------------------------------


def _malus(phi: np.ndarray, theta: float, rng) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project canal polarization φ onto detector axis θ.

    P(+1) = cos²(θ − φ)  — Malus's law.
    After measurement: canal re-emitted at θ (+1) or θ+π/2 (−1).
    Returns (outcomes ±1, new canal angle).
    """
    p_plus = np.cos(theta - phi) ** 2
    outcomes = np.where(rng.random(len(phi)) < p_plus, +1.0, -1.0)
    phi_out = np.where(outcomes > 0, theta, theta + np.pi / 2)
    return outcomes, phi_out


# ---------------------------------------------------------------------------
# Two correlation models
# ---------------------------------------------------------------------------


def classical_lhv(
    theta_a: float, theta_b: float, n: int = 200_000, seed: int = 42
) -> float:
    """
    Classical local hidden variable model.

    Source emits pre-set polarizations (φ, φ+π/2).
    A and B measure independently — no canal communication.
    C(Δθ) = −½ cos(2Δθ).  CHSH S ≤ √2 < 2 — no Bell violation.
    """
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, np.pi, n)  # hidden variable: source angle

    res_a, _ = _malus(phi, theta_a, rng)  # A measures φ
    res_b, _ = _malus(phi + np.pi / 2, theta_b, rng)  # B measures φ+π/2 independently

    return float(np.mean(res_a * res_b))


def dst_canal(
    theta_a: float, theta_b: float, n: int = 200_000, seed: int = 42
) -> float:
    """
    DST canal model.

    Source emits correlated pair (φ, φ+π/2).
    A measures, re-emits canal at θ_A along the substrate.
    B receives the re-projected canal (rotated π/2 for singlet) and measures.
    C(Δθ) = −cos(2Δθ).  CHSH S = 2√2 ≈ 2.83 — Bell violated.
    """
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, np.pi, n)

    res_a, phi_a_out = _malus(phi, theta_a, rng)

    # Canal propagated to B: singlet → B receives orthogonal re-emission
    phi_b_in = phi_a_out + np.pi / 2
    res_b, _ = _malus(phi_b_in, theta_b, rng)

    return float(np.mean(res_a * res_b))


def dst_joint_canal(
    theta_a: float, theta_b: float, n: int = 200_000, seed: int = 42
) -> float:
    """
    DST joint canal model — correct for spacelike separation.

    Both detectors sample the pre-existing non-factorisable canal state
    simultaneously.  The canal was set up causally when the particles
    interacted; no measurement-time signal is required.

    The joint canal state for a singlet pair encodes:
        P(A=+1, B=+1) = P(A=−1, B=−1) = ½ sin²(Δθ)
        P(A=+1, B=−1) = P(A=−1, B=+1) = ½ cos²(Δθ)

    This is the Born-rule statistics of a non-factorisable Planck-scale
    quantum field — it cannot be decomposed into independent local
    contributions, hence the Bell violation without ordering or FTL.

    C(Δθ) = −cos(2Δθ).  CHSH S = 2√2 — identical to dst_canal but valid
    for all separations including spacelike.
    """
    rng = np.random.default_rng(seed)
    delta = theta_b - theta_a

    # Joint probabilities from the non-factorisable canal state
    p_same = 0.5 * np.sin(delta) ** 2  # P(+,+) = P(−,−)
    p_diff = 0.5 * np.cos(delta) ** 2  # P(+,−) = P(−,+)

    r = rng.random(n)
    res_a = np.empty(n)
    res_b = np.empty(n)

    m_pp = r < p_same
    m_pm = (r >= p_same) & (r < p_same + p_diff)
    m_mp = (r >= p_same + p_diff) & (r < p_same + 2 * p_diff)
    m_mm = r >= p_same + 2 * p_diff

    res_a[m_pp] = +1.0
    res_b[m_pp] = +1.0
    res_a[m_pm] = +1.0
    res_b[m_pm] = -1.0
    res_a[m_mp] = -1.0
    res_b[m_mp] = +1.0
    res_a[m_mm] = -1.0
    res_b[m_mm] = -1.0

    return float(np.mean(res_a * res_b))


# ---------------------------------------------------------------------------
# CHSH parameter
# ---------------------------------------------------------------------------


def chsh(
    model,
    a: float = 0.0,
    a_p: float = np.pi / 4,
    b: float = np.pi / 8,
    b_p: float = 3 * np.pi / 8,
    n: int = 200_000,
) -> dict:
    """
    Compute CHSH parameter S = |C(a,b) − C(a,b') + C(a',b) + C(a',b')|.

    Classical bound: S ≤ 2.
    QM (Tsirelson) bound: S ≤ 2√2 ≈ 2.828.
    Standard angles give S = 2√2 for dst_canal, S = √2 for classical_lhv.
    """
    C = {(x, y): model(x, y, n) for x, y in [(a, b), (a, b_p), (a_p, b), (a_p, b_p)]}
    S = abs(C[(a, b)] - C[(a, b_p)] + C[(a_p, b)] + C[(a_p, b_p)])
    return dict(
        correlations=C,
        S=S,
        S_classical_max=2.0,
        S_qm=float(2 * np.sqrt(2)),
        violated=bool(S > 2.0),
    )


# ---------------------------------------------------------------------------
# Correlation curve  C(Δθ)
# ---------------------------------------------------------------------------


def correlation_curve(
    n_angles: int = 37, n: int = 100_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample C(Δθ) for both models over [0, π].

    Returns (deltas, lhv_corrs, canal_corrs).
    canal_corrs ≈ −cos(2Δθ), lhv_corrs ≈ −½ cos(2Δθ).
    """
    deltas = np.linspace(0, np.pi, n_angles)
    lhv_c = np.array([classical_lhv(0, d, n) for d in deltas])
    canal_c = np.array([dst_joint_canal(0, d, n) for d in deltas])
    return deltas, lhv_c, canal_c
