# Entries for 2026-09-02 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION END (2026-09-02, final) — proton/neutron ontology converged (tentatively) to "two-flux-tube-at-c" + "folded/unfolded"

- Where this landed: proton = poloidal+toroidal "two flux tubes, both at c, cross product ~c²" structure (Philip's E=mc² framing, [[project_curl_full_dipole_spheromak]]), LOCKED together (nonzero curl, but stable, no free energy to shed). Neutron = the SAME structure "folded up" carrying an ADDITIONAL circulation component that IS genuine free energy (matches the real ~1.3 MeV/0.78 MeV proton-neutron-decay mass gap, [[project_neutron_decay_energetics]]), which step_cross is supposed to release as the neutron "unfolds" into a proton. The already-built dst_spheromak_to_dipole.py/[[project_spheromak_decay]] experiment tests exactly this, just not previously framed this way.
- Tonight's one dynamical test (lockstep/swing probe) came back not-confirmed but inconclusive, not a refutation — it used step_cross(gate="v1"), already flagged by [[project_diffusion_vs_gate_mismatch]] and [[project_step_cross_v2]] as probably the wrong tool (doc/dst.md describes crossing as a discrete threshold gate, not v1's continuous diffusion; v1 previously found "PROVEN inverted" on unrelated tests).
- Queued for next time, per Philip — likely BOTH: (1) re-test the same lockstep/swing diagnostic under gate="barrier" (theory-consistent threshold gate, requires calibrating m0, see doc/dst_next.md §8). (2) Question step_cross's ontology itself, independent of gate — per [[project_derive_dont_pick_parameters]]'s discipline, don't assume step_cross-in-some-gate is definitely the right mechanism family at all.
- (Both queued items were executed the same day — see the 2026-09-04 entries above: gate="barrier" did not rescue it, and step_cross was determined to be the wrong mechanism family for decay entirely.)

## DROPPED (2026-09-02) — lam-blob knot translation IC deleted (nonsensical, not just under-diagnosed)

See doc/dst/failures/lam-blob-knot-ic-nonsensical.txt — do not rebuild `probe_proton_translation.py`/`probe_moat_translation.py`'s lam-only IC (lam set with phi=0 everywhere, never reachable via step_lam). [[project_dst_proton_translation]] (2026-07-11 result) shares the same root problem and should now be read as failed, not just weakened.

## CORRECTION (2026-09-02): closed curl-full dipole/spheromak IC is a NEUTRON candidate, not proton

Do not test proton-translation hypotheses against the closed spheromak IC (`curl_full_dipole_ic.py::seed_curl_full_dipole_ic`) — see doc/dst/failures/proton-ic-mislabeling-current-loop-and-closed-spheromak.txt for the full mislabeling history.
Queued next step (per [[project_proton_earth_dipole_unification]]): extend the validated curl(phi)=B reconstruction from a straight wire to a bare toroidal ring current (`flux_tube.py::init_toroidal_winding`/`find_toroidal_ring`) and check whether it reproduces the classical magnetic-dipole B-field pattern outside the loop — done in the next entry below.
Standing caveat: every hand-built IC here is preliminary — DST has no independent ground truth yet for what a real proton structure is.

## CORRECTION (2026-09-02, later still): curl-loop result is NOT the proton test either

See doc/dst/failures/proton-ic-mislabeling-current-loop-and-closed-spheromak.txt. Real proton candidate: `dipole_mode_ic.py::seed_dipole_ic` (poloidal-only, `phi=grad(U)`, curl-free by construction, already confirmed exactly-zero macro circulation — [[project_curl_full_dipole_spheromak]]/[[project_dst_discrete_curl]]). Not yet tested against the "open knot" criterion (nonzero net outward u-flux through the domain boundary, contrast with the neutron's confined zero-net-flux). The curl-loop finding itself still stands as a general validation of DST's curl(phi)=B/Ampère mechanism — possibly relevant to a future proton "magnetic moment" property distinct from its charge.

## SESSION CONTINUED (2026-09-02, later) — curl(phi)=B mechanism validated against a classical current loop

Confirmed: DST's discrete curl(phi)=B reconstruction matches the classical circular-loop B field (elliptic-integral exact), independent of the earlier straight-wire check (2026-09-01). `LOOP_A=8`, `R=30`, N=147,047. Script: `exp/2026/09/02/20_00_proton_loop_dipole/probe_curl_loop_dipole.py`; promoted to `tst/substrate/test_curl_loop.py` (3 tests, passing).
- B_phi leakage (should be exactly 0 for an axisymmetric loop): ~4.7% of `|B|_rms` — small, discrete-geometry noise, smaller than other same-project discrete-curl artifacts.
- On-axis Bz(z) matches classical dipole falloff at a constant ratio: mean=1.3308, std=0.0354, n=58 bins, across the full ±29-cell range.
- Equatorial Bz(rho) matches, including the near-field sign reversal at rho≈9-10 (real finite-loop physics): mean=1.3548, std=0.0181, n=17 bins.
- Calibration constant matches the straight-wire test's: both ratios sit near `(2/√3)²=4/3≈1.3333` — ONE universal geometric calibration constant across two completely different source geometries — strong evidence the mechanism itself, not a coincidence of setup, is what's validated.
Caveat ([[feedback_stability_not_reality]]): only validates the static geometric relationship (hand-set analytic phi reconstructed via discrete curl), not real `step()` dynamics — see the correction above for why this is NOT itself the proton test.

## SESSION CONTINUED (2026-09-02, still later) — open/closed knot distinction confirmed DYNAMICALLY

Poloidal-only ("proton" candidate, curl-free) vs poloidal+toroidal ("neutron" candidate), both built on a flat `DampedLattice` (R=30, r_sph=8, N=147,047), stepped under plain `step()` only for 8000 steps. Script: `exp/2026/09/02/22_00_open_vs_closed_knot/probe_open_closed_radiation.py`.
- Core energy retention (`H_core/H0_core`): proton candidate plateaus ~0.12-0.13; neutron candidate plateaus ~0.56-0.67 — roughly 4-5x more retention for the closed structure.
- Total domain energy (`H/H0`): proton ends at 0.187 (81% gone by step 8000); neutron ends at 0.594 (41% gone).
- Real, consistent relative difference (~2x total / ~4-5x core) in the direction the open/closed naming predicts ([[project_dst_neutron]]) — not an absolute "one radiates, one doesn't" binary; both lose real energy fast at first since neither IC is an exact eigenstate of this finite masked domain.
Diagnostic gotcha found in the same probe: see doc/dst/failures/bc-flux-not-an-energy-diagnostic.txt — use `H(t)`/`H_core(t)` for any future radiation/confinement measurement, never `bc_flux`/`rad_cum` on a dipole/multipole-symmetric field.

## SESSION CONTINUED (2026-09-02, latest) — u_rms "drift" was actually periodic oscillation, not secular drift

Corrects [[project_curl_full_dipole_spheromak]]'s original 60-step lockstep probe. Over 6000 steps (curl-full dipole on flat `DampedLattice`, R=30, r_sph=8, N=147,047; `exp/2026/09/02/23_00_lockstep_free_energy/probe_lockstep_free_energy.py`) under plain `step()`: `u_rms` oscillates periodically between ~0.39x and ~1.49x its initial value (the old 60-step probe just caught the first quarter-cycle and mistook it for secular drift). Macro toroidal circulation stays exactly frozen to 5-6 decimal places for the entire run under plain `step()` — a THIRD independent confirmation this session of exact circulation conservation (after the ring-lean-drift and circulation-quantization-barrier findings below/above).
"Free energy consumed by decay → locked proton" hypothesis: NOT confirmed under `step_cross` — see doc/dst/failures/free-energy-lockstep-hypothesis-not-confirmed.txt. Hypothesis itself is not ruled out, only this mechanism/diagnostic (gate="v1").
Not yet tried: a diagnostic separating real secular erosion from the u-oscillation's own periodicity (cycle-averaged u_rms vs. circulation decay rate); checking whether eroded circulation reappears as u/poloidal content.

## SESSION END (2026-09-02) — circulation-quantization energy barrier derived (H·r = const, exact)

Solid, reproducible, machine-exact. `find_toroidal_ring(mode="square", hops=N)`: same winding number (w=+1.0000, confirmed every hops) at 8 radii (hops=1..10, r≈1.15-11.5), fixed total circulation Γ. `H·r = 591.2067`, exactly constant across the full 10x radius range (H=512 at hops=1 down to H=51.2 at hops=10, tracking 1/hops exactly). Script: `exp/2026/09/02/18_28_circulation_quantization/probe_circulation_energy_barrier.py`.
Combined with the independently-confirmed exact circulation conservation under `step()` (2026-09-01 night, below): a real derived barrier against collapse to a point (H~1/r for fixed Γ) — falls out of DST's own bookkeeping, not borrowed from classical vortex theory ([[feedback_reproduce_dont_reify]]).
Caveat ([[feedback_stability_not_reality]]): only the static/algebraic half is shown. Next step if pursued: apply a genuine local inward force to a real ring under free dynamics (avoid the boundary-kick confound — push directly at the ring, not via a distant shell) and check whether the ring's radius actually resists shrinking, or whether `step_cross` erosion lets it cheat the barrier by leaking circulation.
