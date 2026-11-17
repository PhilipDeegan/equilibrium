# Entries for 2026-09-16 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-09-16, LATE — electron/positron IC reviewed for falsifiability, a real phi-convention + inherited-eigenvalue gap found blocking any magnetic-moment measurement, then pivoted (Philip) to the free-electron standing-wave/soliton question directly; self-sourced lam (Kerr-effect-style nonlinearity) shows a first real, monotonic self-confinement signal

Wide-ranging session: started from Dirac spinor requirements / topological
solitons / Pauli exclusion (conceptual synthesis,
[[project_dirac_spinor_exclusion_synthesis]]), then reviewed the electron/
positron double-ring dipole IC (`init_electron_dipole_ic`,
`tst/substrate/adaptive/adaptive_flux_tube.py`) for how to invalidate it —
ranked falsification checklist in [[project_electron_ic_falsification_review]]
(e/p inertia ratio still the most decisive untested item).

**Magnetic-moment measurement attempt found a real, two-part gap, not a
number.** Built `dipole_moment_fit()` (general far-field point-dipole
least-squares fit on `equilibrium.substrate.curl.b_vec_vectorized`) and
found it reads back exactly zero on the real electron IC. Traced this to
(1) `init_electron_dipole_ic`'s phi is a B.n_hat (flux) sample, but the
validated curl(phi)=B machinery needs an A.dl (circulation) sample —
confirmed by feeding it a known classical loop both ways. Built
`init_electron_dipole_ic_vecpot` (genuine A.dl seeding, no continuum
derivatives anywhere per the discrete-curl philosophy) which fixed the
on-axis field (was identically zero everywhere before — an exact algebraic
degeneracy, not just a convention mismatch) but revealed (2) a deeper
issue: `k_=4.493/R_sph` (j1's first zero) is the self-confined force-free
(Chandrasekhar-Kendall/Taylor-state) eigenvalue — inherited wholesale from
the neutron/spheromak lineage, never independently justified for a
charged, field-radiating electron (doc/dst.md: proton=OPEN knot,
neutron=CLOSED). At that k, the field is exactly zero outside R_sph*scale
by construction — no external field is possible AT ALL regardless of the
phi-seeding fix. Both scripts annotated with KNOWN ISSUES notes pointing
here. Full detail: [[project_dipole_moment_phi_convention_gap]]. Not yet
resolved: relax k (allow a real surface discontinuity/exterior tail) vs.
add an explicit exterior matching field vs. abandon this construction
family for the electron entirely.

**Pivoted to the free-electron standing-wave/soliton question directly**
(deliberately no proton, to avoid confounding with the bound/Bohr case) —
tests the idea that a free electron's fixed "size" comes from de Broglie
closure (`2*pi*r=n*lambda`) combined with its own field energy H fixing a
frequency via E=hbar*omega (mass=field-energy, no separate postulate
needed), NOT from an external confining force.

Built `exp/2026/09/16/20_50_free_electron_ring_closure/` (three script
versions). v1 (phi=0 IC): found a clean, r-INDEPENDENT omega(n) signature
(omega_meas/omega_pred ratio ~1.50 for n=2, ~1.33 for n=6, constant across
every tested r) but 45-99% energy "decay" from the ring region at every
config. v2 (genuine WKB companion-phi for a curved path, generalizing
`equilibrium.substrate.traveling_wave.companion_phi` from a fixed axis to
a position-dependent local tangential wavevector): confirmed the
construction is genuinely self-consistent (ratio->0.9985 when built with
the measured, not naive, omega) but did NOT reduce the "decay" — because
it isn't real dissipation at all: **verified total (unmasked) field energy
is conserved to ~7e-7 over the same run** — the "decay" is just the
envelope diffracting out of a fixed measurement window, ordinary behavior
for ANY localized solution of a linear wave equation (corrected mid-session
from a wrong "needs infinite plane wave" framing — Philip caught this — to
the precise, finite, no-infinities argument: Derrick's theorem, a static
finite-energy lump built from only quadratic terms is generically unstable
under scale transformations; the azimuthal winding number IS topologically
protected and never leaked, only the unprotected radial/vertical envelope
diffracted).

**Second correction (Philip): v1/v2 both ran in literal VOID (lam=0
everywhere, `SubstrateLattice`'s bare default, never overridden) — invalid,
since this project defines mass AS substrate depletion
([[project_lam_dual_role]]), so an electron's own presence requires
nonzero SELF-sourced lam, not an external field.** Built v3: lam sourced
fresh every step from the winding's own local field energy density
(`lam = alpha*(u^2+0.5*sum(phi^2))`, no continuum derivatives, no `_dirty`
flag needed since the neighbour cache is topology-only) — structurally the
Kerr effect from nonlinear optics (intensity-dependent propagation speed via
sigma=1/(1+lam) -> self-focusing balancing diffraction), the "self-trapping"
idea flagged and never attempted in [[project_dst_electron_topology]]'s
2026-07-12 section. **Result: a real, clean, MONOTONIC trend** — spread
growth over 6 periods goes from +31.6% (alpha=0, dispersing) through +34.0%,
+33.5%, +12.0%, down to -1.6% (crossover, alpha~1) and -3.2% (alpha=2,
lam_max~11.2, genuinely self-focusing); energy retention in the fixed
window also jumps sharply at alpha=2 (decay 49.9%->13.6%). First positive
signal that self-sourced lam provides genuine nonlinear self-confinement,
not an artifact. Caveats: single (n,r) config, only 6 alpha points, alpha=2
is an extreme/unvalidated regime. Full detail + exact numbers:
[[project_free_electron_soliton_needs_nonlinearity]].

**Queued next, not yet done**: fine alpha scan near the crossover (~0.5-1.5)
looking for a genuinely STABLE fixed point (spread growth ~0, not just a
sign crossing), replicated across other (n,r) pairs — not just more alpha
sweeps at the one tested config. Separately and independently: the
magnetic-moment k-eigenvalue question above (relax k vs. exterior tail vs.
abandon this IC family) is still open and not blocked on the soliton
thread.

---
