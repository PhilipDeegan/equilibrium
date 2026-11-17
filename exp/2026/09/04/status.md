# Entries for 2026-09-04 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION CONTINUED (2026-09-04) — queued item 1 done: gate="barrier" re-test does NOT rescue the lockstep/free-energy finding

- See doc/dst/failures/gate-barrier-no-m0-discriminates.txt.
- This makes queued item 2 (question step_cross's ontology itself) more load-bearing — see next entry.

## SESSION CONTINUED (2026-09-04) — queued item 2 answered: step_cross is the WRONG mechanism family for decay

- See doc/dst/failures/step-cross-wrong-mechanism-for-decay.txt.
- Positive redirect: the open-vs-closed-knot radiation probe (plain step() only, no step_cross; proton candidate loses core energy ~4-5x faster than the neutron candidate) is the one decay-relevant result in this project that's actually theory-consistent — treat as the load-bearing decay evidence going forward.
- Concrete next step, not yet built: a genuinely derived decay probe placing a winding at a specific λ/resonance distance (not just "has toroidal component or doesn't") and measuring plain-step() radiation rate vs. that distance, per doc/dst.md's own falsifiable target (~line 544: resonant λ spectrum / lifetimes scaling as distance from nearest eigenstate).

## SESSION CONTINUED (2026-09-04, later) — resonance-detuning decay probe built and run; clean NULL on the specific knob tested, but the knob was probably wrong

- See doc/dst/failures/resonance-detuning-wrong-knob-null.txt.
- Real, separate finding worth keeping: circulation stays exactly frozen under boundary-closure detuning at every δ tested (0-10%) — a robust discrete-topological invariant.
- Also raised, not yet built: (1) does a genuine lam profile stabilize the proton dipole against its documented rapid domain-energy loss; (2) [[project_curl_full_dipole_spheromak]]'s circulation-quantization energy barrier (H·r=const) may mean real decay has to be a discrete topological-unraveling EVENT ejecting a separate object, not gradual leakage — untested; (3) relativistic fore-shock/motion-induced lam asymmetry — parked, separate from decay.

## CAVEAT ON ALL 2026-09-04 WORK ABOVE — every probe used R_SPH=8, below the smallest resolution point this project's own convergence study flagged as unreliable

- See doc/dst/resolution_requirements.txt, REQUIREMENT 1.
- Not yet done: cheap intermediate check at cells_across=55 or 110 to see if today's qualitative conclusions survive before trusting them or committing to the expensive cells_across≈220 rerun.

## SESSION PIVOT (2026-09-04, still later) — does a bare fixed-potential difference in VACUUM drive a real current toward equilibrium? Two findings, one open

- Finding 1 (confirmed, real): two fixed-u sources are NOT independently radiating — genuine interaction confirmed via a linearity/superposition test (matches to ~1e-7 through step 1000, then diverges — 0.7% at step 2000 growing to 1.4% by step 6000). Likely mechanism: a fixed-u cell must actively absorb/reflect an incoming disturbance to stay pinned, a genuine cross-source effect once the other source's wave arrives. Not yet fully characterized (reflection vs. mutual equilibration).
- Finding 2 (open): H and I_mid have not saturated by 20000 steps (I_mid still growing ~linearly, H ~quadratically) even though local u_rms near the sources has plateaued. Two live hypotheses: (a) sponge too weak / needs a longer run; (b) point sources carry a genuine self-energy-like divergence so "steady state" means constant dH/dt, not bounded H — needs checking whether dH/dt itself plateaus.
- Quantified against classical: fit u≈A/r_a+B/r_b, R²=0.999921, RMS residual 1.33% of mean|u|, residual grows with distance (0.75%→4.37%). Consistent with the already-known 2026-08-27 result that DST's static single-source potential falls off steeper than classical 1/r under sponge BC.
- Follow-up: solved the EXACT discrete Poisson equation on the same lattice graph (CG solve, 347,939 unknowns, converged) instead of fitting continuum 1/r. Result: RMS residual 1.29% vs 1.33% — barely moved. Critically, discrete-Poisson vs smooth continuum itself already shows 1.44% RMS residual — as large as DST's own deviation. **Conclusion: essentially all the ~1.3-1.4% systematic deviation from classical Coulomb is discrete-Laplacian truncation error inherent to the 14-neighbor BCC discretization, not DST's dynamics failing to equilibrate** — a reassuring, quantified result.
- Note: the real graph Laplacian uses W_HEX=3/W_SQ=1 face weights, which does NOT match doc/dst.md's stated w1=2/3, w2=1/3 — a real doc/code discrepancy worth checking separately (not chased).
- Loose end, not resolved: far bin (r∈[12,14)) got worse relative to discrete-Poisson (7.70%) than continuum (4.37%); three live candidates (incomplete settling at larger radius, CG precision floor, real anisotropy per [[project_bcc_dispersion_anisotropy]]) — not distinguished.
- See doc/dst/failures/category-errors-borrowed-macroscopic-analogies.txt for the primal/dual-mesh and lam-chirality corrections raised in this thread.
- Standing reframing kept: DST may have an unused geometric slot (cell edges, already real via the 12-triangle plaquettes) that could carry a missing FIELD/EFFECT — the Ampère/Lorentz-force term, parked mid-derivation. Concrete next step if pursued, not started: re-derive the coupling against the existing plaquette-edge geometry (not mhd.py) and check if it's naturally expressible as a genuine edge-based (1-form-like) quantity.
- Correction (Philip): φ already behaves correctly like the vector potential A topologically — curl(φ)=B matches classical wire/loop shapes exactly, and φ's macroscopic circulation is exactly conserved under step(), matching AB-effect holonomy. Does not contradict the resolved scalar-energy-flux finding (φ-as-A topologically, yes; local vector Poynting reconstruction, no — already settled separately).
- DST is a discrete quantum Gauss's law at this scale (Q_i=Σ_j A_ij·∂u/∂r over the Kelvin cell's 14-face surface); the Ampère counterpart should use the natural closed-loop analogue DST already has (12 plaquettes/cell), not MHD-flavored mechanisms.
- See doc/dst/failures/sg-raw-signal-swamped-by-background.txt — re-diagnosis of stern_gerlach.py's mechanism explains why the 2026-09-01 null was a diagnostic artifact. **DONE, reverses to a POSITIVE result**: isolated ratio 0.99-1.00 antisymmetric response while winding circulation was still intact. Full derivation in [[project_dst_em_field_convention_crisis]].
- stern_gerlach.py rebuilt (2026-09-04): background-only control run now automatic, isolated delta computed/plotted automatically, verdict evaluated at the last step where the background's tot_circ retains ≥ threshold (default 0.5) instead of a fixed late step. Smoke test gave isolated ratio -0.9986, "OPPOSITE DEFLECTION."
- Correction 4 (Philip): step_cross doesn't interact with photons/massless things AT ALL — exclusively for particle aggregates that can't translate without through-cell crossing. Photons/EM waves are pure face-hopping. The Poynting-dipole scripts driving their dipole via step_cross(gate="barrier") are a category error; probe_poynting_step_only.py (plain step()) is the only category-correct one of the four.
- Not yet done: (1) check dH/dt for a plateau directly; (2) sponge/damp_max sweep to see if I_mid saturates and to what value; (3) characterize the superposition-breakdown mechanism directly by watching the field arrive at the other source's location.

# Current status — read this first

Long-term AMR design knowledge (deep-telescope H-growth mechanism, elimination
history, hyper-resistivity mitigation design + calibration methodology,
CellSlab memory-layout plan) has moved to `doc/dst_amr.md` — it doesn't decay
the way a session log does, so it belongs there, not here. Full
session-by-session investigation history (every "first I thought X, then
corrected to Y" turn) lives in `doc/mems/project_odd27_amr_junction_leak.md`
and `project_hop_telescope_resolved_spheromak.md`. This file only tracks work
actually in flight, compressed to current conclusions — not a replay of how
we got here. (Compressed 2026-08-14 — see git history for the pre-compression
version of the two entries below if the blow-by-blow is needed.)
