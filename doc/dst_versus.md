# DST versus established physics — where standard theory invalidates, not guides

Stance (Philip, 2026-11-03): modern theory is a set of signposts as to what is true, but solid as to what is
ruled out. This file lists what DST (a deterministic, local, discrete substrate with no independent particles)
must survive. It separates no-go results (theorems + experiments) from quantitative bars. Numbers are
order-of-magnitude bounds as commonly quoted; check the primary source before relying on a digit.

Severity: **A** = collides with a theorem backed by experiment (survive or be invalidated);
**B** = precision bound on a lattice / substrate theory; **C** = observed fact with no no-go theorem.

---

## Not in conflict

- **No independent particles.** QFT has none either: particles are field excitations, and identical-particle
  indistinguishability follows from that. DST's "structures are depressions of one substrate" sits closer to QFT
  than classical point particles do.
- **Contextuality.** Kochen–Specker excludes *noncontextual* hidden variables. DST is contextual by construction
  ("no noise, only context"); KS does not bite.
- **Exact charge conservation, bounded propagation speed, massless waves at c**: compatible.
- **Weinberg–Witten** (no emergent massless spin-2 from a Lorentz-invariant QFT with a covariant conserved
  stress tensor) assumes Lorentz invariance; a lattice substrate is outside it. The price is section B1.

---

## A1. Bell / CHSH — the decisive one

Loophole-free tests (Delft, Vienna, NIST, 2015) exclude every model that is **locally causal** with **measurement
independence** (settings uncorrelated with the state).

Bell's λ is the complete state of whatever exists in the relevant past light cones; it may be a single global
field. Factorisation P(A,B|a,b,λ) = P(A|a,λ) P(B|b,λ) follows from local causality (Bell 1976, "La nouvelle
cuisine" 1990), not from the system being made of separable particles. A deterministic nearest-neighbour rule with
bounded speed (step() + Cartan transport) is locally causal in exactly this sense: conditioned on the full field
at the source time, A's outcome depends only on a and the field inside A's past cone.

=> **doc/dst.md "DST is outside Bell's theorem" (§ Quantum correlations) does not hold as written.** "No particles"
does not remove the factorisation; a field-only local model is still a local hidden-variable model. A canal that is
built at the source and propagates at <= c is also covered. The bell.py "joint canal" C = -cos 2Δθ is put in by
hand; it is not shown to come from a local rule.

Ways out (each has a cost):
1. **Non-spatial adjacency.** The canal is a substrate connection that does not run through the emergent geometry:
   local on the graph, non-local in space. Consistent, but the rule must then forbid signalling (outcome
   marginals at A independent of b). That is an extra exact constraint the rule must meet.
   Precedent: Bohmian mechanics is explicitly non-local (instantaneous influence in a preferred frame) and still
   cannot signal. Superluminal influence only creates causal paradoxes without a preferred frame. DST has one (the
   substrate rest frame), so a superluminal canal update is consistent, but it is a definite claim needing a
   mechanism (memory project_dst_bohm_stepping_stone).

Constraint (Philip, 2026-11-03): no probability spaces, no abstract spaces (Bohm's guiding wave lives in 3N-dim
configuration space -> aphysical). Only the Bohm *lesson* carries over (preferred frame makes superluminal influence
consistent), not its mechanism. Bell still applies without probability spaces: outcomes are deterministic
functions A(a, λ), B(b, λ); "probabilities" are frequencies over runs; the spread of λ is the physical variety of
source / detector states from run to run.
=> the non-local influence must itself be a physical substrate structure.

Candidate (idea, untested): **the canal as a faster-than-ambient channel in 3-space.** Propagation speed is set by
cell size / depletion; ordinary vacuum is low but non-zero lam (lam = 0 unreachable), so measured c is the speed
*in that vacuum*, not the substrate maximum. A thread of cells less depleted than the ambient would carry influence
faster than ambient light *globally*: locally the speed is always c (a cell's own rate and size set local units);
measured in ambient units, a path through lower lam has higher throughput, and a front entering falling lam speeds
up progressively (Philip, 2026-11-03). The Bell-relevant quantity is the global one: arrival time at B in the
ambient frame's units via the canal vs via ambient light. If the thread is far below a wavelength across (photon ~5e37 cells), photons do not
resolve it (10-25 defect scattering: sub-wavelength features are transparent, only the mean size survives), so
it is not seen as a faster light path. Bars it must clear: no signalling (A's frequencies independent of b);
Tsirelson, S reaches 2√2 and no further (experiments match QM, not stronger correlations); cos² statistics from
the physical variety of states, not assumed.
No probability amplitudes either (Philip, 2026-11-03: reification; amplitudes are Hilbert-space coordinates). A DST
"amplitude" is only a physical substrate quantity (u / phi are the current, provisional degrees of freedom;
the real set is still being worked towards). Outcome frequencies (cos², 2√2) must follow from physical content
alone, e.g. the share of face content a polariser passes x how often a detector's threshold offload fires, across
the physical spread of runs.

Scope: Bell does **not** rule out hidden variables in general (Bohm survives). It rules out a class of *rules*:
local causality + measurement independence, whatever the variables are. Survivors drop one of these: non-local
dynamics (Bohm), correlated settings (superdeterminism), retrocausality, or no single outcome (many-worlds-type).
DST's variables are not the problem; its current rule (nearest-neighbour, bounded speed) is in the excluded class.

Note on terms: Bell's "locality" is local causality (no influence outside the light cone), not independence of
particles. λ may be shared and non-factorisable (a common cause); that is what gives classical S <= 2. The
loophole-free tests' spacelike separation of settings and detections only makes sense because the assumption tested
is "nothing faster than c".
2. **Superdeterminism.** Settings are correlated with the substrate state. Logically allowed, widely judged very
   costly; it conflicts with "measurement choices are still free" in dst.md.
3. **Settings act on the joint state.** A measurement reaches the shared canal faster than light. Same as 1, and
   it must still forbid signalling.

Prior work: run/substrate/probe_bell_canal.py, stage 1 (2026-07-16): the separable baseline gives S = √2 as
expected; stage 2 (a joint canal from the rule itself) was never built.

## A2. Single-photon anticorrelation (g²(0) < 1)

Grangier–Roger–Aspect 1986: a single photon at a beam splitter fires one detector, never both (g²(0) < 1).
Classical fields cannot do this (for any classical intensity, g²(0) >= 1); a wave splits and both sides get half.
Semiclassical theory (quantised detectors, classical field) explains the photoelectric effect and thresholds,
but not this anticorrelation.

DST's detectors are structures with an offload step (h), and that is not enough: energy conservation means half
the wave reaches each detector. Getting one click means the whole quantum is collected through one arm. Not necessarily
non-local (2026-11-03 correction): the bound assumes each detector responds to its own arm's field independently;
a collection that withdraws the other arm's content at <= c before that detector fires is local. It becomes
the single-particle form of A1 only if the two detections are spacelike separated (check which experiments do this).
The same source interferes in a Mach-Zehnder (Grangier et al. 1986), so the content does go both ways: a
beam splitter that routes the whole quantum into one arm (by its own microstate) is excluded.

## A3. Thermal radiation / Planck law

A classical field with many modes in thermal equilibrium equipartitions. A cutoff at the cell (~1e-44 m) gives
Rayleigh–Jeans up to absurd frequencies, not Planck. The discrete floor alone does not fix this; the offload
quantum (h as the discrete step of exchange vs "temperature") has to produce the exponential suppression
of hν >> kT modes. Already flagged (memory project_dst_blackbody_uv_catastrophe, project_h_offload_quantum).

---

## B1. Lorentz invariance, isotropy, preferred frame

- Isotropy (Michelson–Morley / Hughes–Drever-type): ~1e-17 or better. The BCC anisotropy seen in runs is coarse-
  mesh; at real scale it is suppressed as (ell/λ)^2 — fine at tree level.
- Photon dispersion (GRB 090510 and similar): a linear Planck-suppressed term is excluded at E_QG ≳ E_Planck.
  A 1e-44 m cell is far below that. Vacuum birefringence bounds (GRB polarisation) are also tight.
- Low-energy leakage (Collins, Perez, Sudarsky 2004): in interacting QFTs, lattice-scale Lorentz violation tends
  to feed into low-energy physics at O(1) unless protected. DST is not a loop-level QFT, so the argument does not
  transfer directly. The equivalent question is whether lattice-scale structures pass their anisotropy on to
  everything else.
- **Preferred frame**: a substrate with a rest frame is a preferred-frame gravity theory. PPN α1 ≲ 1e-4,
  α2 ≲ 1e-7 (solar spin axis). Motion through the substrate frame (CMB dipole ~370 km/s) sets the scale.

## B2. Gravity

- **Universality of free fall**: MICROSCOPE η ~1e-15. Current DST evidence: ±50 % (jigsaw v10).
- **Light bending**: Cassini γ − 1 ~1e-5. Pure scalar gravity gives zero light bending, so the factor 2 must come
  from both clock rate and space (cell size) changing. Eddington 2x vs 4x still pending.
- **Gravitational-wave speed**: GW170817, |c_gw − c|/c ≲ 1e-15.
- **Pressure-mediated gravity**: if gravity is pushing by a medium, the Le Sage objections apply (heating from
  absorption, drag, aberration). A depression gradient escapes them only if no energy is absorbed.
- **Perihelion / Shapiro / frame dragging**: β, γ to 1e-4–1e-5; Lense–Thirring (LAGEOS, GP-B) to ~1e-2–1e-1.

## B3. QED precision

The electron's g-factor matches QED to ~1e-12 relative; also the Lamb shift, hydrogen 1S–2S (~1e-15) and the
running couplings. None is a no-go, but whatever replaces loop corrections (memory
project_dst_virtual_photon_bookkeeping_idea) has to reproduce them to that precision.

## B4. Electromagnetism exactness

Coulomb's inverse-square law to ~1e-16 in the exponent; photon mass < ~1e-18 eV; charge neutrality of atoms
~1e-21 e. Linear Maxwell is well tested.

---

## C1. Spin ½, 4π, exclusion

4π rotation measured directly (Rauch–Werner neutron interferometry); Pauli exclusion is observed everywhere
(chemistry, white dwarfs). Classical topological solitons carry spin ½ only once quantised
(Finkelstein–Rubinstein). A deterministic substrate with fermionic statistics would be new, but no theorem known
here forbids it. Memories project_spinor_open_questions, project_pauli_exclusion_smoothness_idea.

## C2. Stable atoms and discrete spectra

A classical accelerated charge radiates; atoms do not. Non-radiating extended sources exist (anapoles,
Goedecke / Schott configurations), so this is not a no-go, but the ground state must be non-radiating and the
spectrum discrete. Bohr-orbit work so far: memory project_h0_study, project_bohr_orbit_amr_validation.

---

## Tests (to design)

Placeholder: one row per item; to be filled in discussion with Philip (2026-11-03).

| Item | What would invalidate | Candidate DST test |
|------|-----------------------|--------------------|
| A1 Bell | S <= 2 from the rule itself, or signalling | detector, relay, joint CHSH (below); canal speed >= ~1e4 ambient c |
| A2 g²(0) | both arms register a split quantum | A1 detector + relay across a beam splitter; coincidences / singles² < 1 |
| A3 Planck | equipartition in a thermalised box | linear box (no exchange), then receivers with continuous vs quantum offload |
| B1 isotropy / frame | direction or frame dependence surviving coarse-graining | |
| B2 free fall, γ | η > 1e-15 extrapolated; γ ≠ 1 | |
| B3 QED | g − 2 not reproduced | |
| C1 spin ½ | 2π-periodic structures only | |

### A1 design (2026-11-03, proposed)

Timing bar (from the theorem itself): in any bounded-speed rule, S <= 2 whenever A's setting-to-detection and B's
detection lie outside each other's fastest-signal cone. In DST the fastest signal is the canal, speed v_c. So S > 2
needs D < v_c Δt, and the experiments' lower bounds on the "speed of influence" (Salart et al. 2008, Yin et al.
2013: ~1e4 c in frames moving slowly relative to Earth; check the primary sources, and the CMB frame) set
v_c / c_ambient >~ 1e4. In today's step() the ratio is the sigma ratio, (1 + lam_bg) / (1 + lam_canal) <= 1 + lam_bg
(canal.py: 2x at lam_bg 1, needs >= ~7 cells across for the full ratio), so the sigma law as written needs
lam_bg >~ 1e4 in vacuum. The sigma law is provisional (ell not wired); the bar is on the ratio.

Shape: relay. The only known way a substrate-local mechanism reproduces QM exactly without signalling. In the substrate
rest frame one side measures first; its firing re-orients the shared canal content to a (fired) or a⊥ (not fired);
that reaches B through the canal; B reads it with Malus. Over the spread of runs A fires half the time, so B's
received state is a or a⊥ equally often and B's own frequencies do not depend on a (no signalling). A faithful
relay gives exactly cos² and S = 2√2, never more (Tsirelson automatic); a partial relay of share f lands between
the separable √2 and 2√2 (report f). Which side was first is unobservable (no signalling), consistent with a
preferred frame.

Steps, each a pass/fail on its own:
1. **Detector** (local, no pair): a threshold-offload structure whose firing frequency over the physical spread of
   its own state equals the share of content its polariser passes. Fail: frequency not linear in the passed share
   (no cos², S off). Also the A2 detector.
   2026-11-03: step() has no transverse vector branch (16 dof per BCC unit cell: 12 static, 4 = two scalar u
   branches; exp/2026/11/03/planck/branches.py). That does NOT block the test (Philip): observed polarisation is
   angle-dependent transmission through an oriented filter, cos² of the relative angle, period 180°. A coherent
   scalar packet can carry an orientation in its cross-section: a two-lobe transverse profile at angle a is
   cos(a) x-lobe + sin(a) y-lobe, so a filter passing only lobes at b passes cos²(a - b) of it (overlap).
   Test: two-lobe packet, slot / strip filter at b, passed share vs b - a; lattice splitting of the two lobe
   orientations' speeds (BCC birefringence) and coherence over distance are the ways it can fail.
2. **Relay**: A's firing changes the canal's content near B toward a / a⊥, ahead of ambient light. 2a with an
   imprinted canal (scaffolding only), 2b with a canal formed by the rule at pair creation (counts). Measure: B's
   received angle vs a, share f, arrival vs D in ambient units.
3. **Joint**: CHSH S and P_agree(30°) (target 3/4, LHV ceiling 2/3) over the spread of source / detector states;
   B's marginal vs a flat to the run's resolution. probe_bell_canal.py stage 1 stays the separable control (√2, 5/8).

### A3 design (2026-11-03, proposed)

0. **Linear box** (cheap, expected to fail): today's step() at fixed lam is linear and conservative, so each lattice
   mode keeps its energy; a box seeded with a spread of modes never thermalises. Confirms that equilibration needs
   exchanging bodies (emitter = receiver, finite capacity), not the field alone.
   Result 2026-11-03 (exp/2026/11/03/planck/): confirmed. Uniform box keeps its k and frequency spectra; fixed rough
   cell sizes scatter content in k (34 % above the seed band) at unchanged frequencies, nowhere near equipartition.
   With Cartan transport (exp/2026/11/03/planck/cartan_box.py): the closed box DOES thermalise, to equal energy per
   mode at the grid scale (shell by shell vs a white-noise reference), rate ~ amplitude = Rayleigh-Jeans in the
   current rule, the same failure as the 10-24 spheromak cascade. Planck needs a stop: the offload quantum.
1. **Receivers**: the box plus M finite-capacity bodies exchanging with the field. Continuous exchange should reach
   equal energy per mode up to the lattice cutoff (Rayleigh–Jeans = invalidation of continuous offload); exchange in
   steps of the offload quantum, with the bodies' content spread as "temperature", should give occupancy
   1 / (e^{hν/kT} - 1). Compare mode energy vs ν to both, counted on the lattice's own mode density (BCC
   dispersion), not ν². Needs a body with a quantised offload step, which the model does not have yet.
