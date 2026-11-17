# Proton-knot translation probe — first positive collective-coherence result

Companion to the step_cross v2 gate work (`exp/2026/07/11/17_43_step_cross_v2_gate_sweep/`,
`exp/2026/07/11/19_10_step_cross_v2_opposed_cross_scale_sweep/`), but testing a lam-carrying
"open knot" (proton-like mass concentration) instead of the lam-free electron double-ring
winding, on the uniform lattice (`equilibrium.substrate.lattice.SubstrateLattice`/
`DampedLattice`) rather than the AMR adaptive cluster.

## Motivating question

A proton is a collective entity — it moves and resists as one while it exists. Does that
"moves as one" relationship already emerge from the existing state (u, phi, lam, sigma(lam))
and existing operators (step, step_cross, step_lam_force), or is an explicit relational
quantity missing? Electron double-ring result (`exp/2026/07/10/21_15_electron_translation_gate/`):
a lam-free circulation-type winding only erodes under an external pressure tilt, never
translates coherently. Is the same true for a lam-carrying, non-circulating (curl(phi)=0)
knot, or is that a different topological class?

## IC construction (three iterations before a valid one was found)

1. **First attempt**: bare point u-spike (u_central at origin, 1/r background elsewhere,
   lam via `set_winding`), nothing pinned. Result: catastrophic — a huge whole-domain
   breathing mode (u swinging from +5 to -4.65 and back, phi_rms oscillating 0.02-0.2 with
   no monotonic settling over 20k steps) that dwarfs any external stimulus by orders of
   magnitude. Also found the pre-seeded 1/r background's local gradient (~2.5 between
   adjacent cells near the core) swamped the external kick tilt (~5e-4/cell) by ~4 orders
   of magnitude even before the breathing-mode problem was diagnosed.
2. **Second attempt**: pin the origin's u permanently (`fixed_u=True`) to force a stable
   reference, mirroring `probe_proton.py`/`project_lam_gravity`'s approach. Result: worse —
   `phi_rms` grew monotonically and unboundedly (0.038→0.467 over 10k steps, no turnover).
   A permanently pinned source is an inexhaustible battery: it can never equilibrate with
   its neighbours, so it injects flux every tick with nothing capping the rate — the same
   "imposed stress has to go somewhere" pathology as `project_lam_gravity`'s unsolved
   lam-sink blowup, just via a different channel.
3. **Working IC**: uniform u=0 + phi=0 everywhere + a lam(r) profile from `set_winding`
   (chi_0=0.9, profile="1/r2"). Verified numerically to be an EXACT fixed point of the full
   operator set (step + step_cross + step_lam_force) for any lam(r) profile — max|u| and
   max|phi| stay at literal machine 0 over 5000 steps with no kick. This follows directly
   from the update rule: Δu ∝ Σφ (zero since φ=0), Δφ ∝ (u_j-u_i) (zero since u is
   uniform), step_lam_force's pull ∝ max(u_nb-u_i,0) (zero since u is uniform). sigma(lam)
   only sets the RATE of departure from equilibrium, not whether zero-flux/uniform-u is an
   equilibrium — so no warmup and no point-spike seeding is needed at all. This mirrors
   exactly why the electron probe's IC worked: an exact fixed point (uniform u +
   divergence-free phi) rather than a shocked state.

## Protocol

Same base/kick/coast structure as `probe_electron_translation.py`: n_base=1000 (confirms
the fixed point holds), external u tilt (gradient × position along kick axis) applied and
pinned at a battery ring just inside the sponge for n_kick=6000, released for n_coast=4000.
R=20 (N=43599), dt=5.128e-3. Two drive levels: baseline (cross_scale=0.01, force_scale=1e-2,
matching the electron probe's scale) and strong (cross_scale=0.3, force_scale=0.3, 30×).

Tracked (the J=curl(B) three-channel accounting — movement/heat/light as coherence
patterns of the same u/phi/lam state, not separate physical quantities): lam-weighted COM
along the kick axis (movement), lam-weighted spread around its own COM and total Σlam
(heat/erosion proxy — shape and mass conservation), cumulative `bc_flux()` through the
sponge boundary (light/radiated proxy).

## Results

| drive (cross_scale=force_scale) | axis | v_kick | v_coast | mobility v/g | sign match? | mass_frac | spread_frac |
|---|---|---|---|---|---|---|---|
| 0.01 (baseline, matches electron probe) | z | +2.7e-7 | -8.8e-7 | +5.4e-4 | no (noise floor) | 1.0000 | 1.0000 |
| 0.03 | z | -1.34e-6 | -2.95e-6 | -2.7e-3 | **yes** | 1.0000 | 1.0001 |
| 0.1 | z | -1.96e-5 | -5.86e-6 | -3.9e-2 | **yes** | 1.0000 | 1.0001 |
| 0.3 | z | -1.15e-4 | -2.57e-5 | -2.31e-1 | **yes** | 1.0000 | 1.0001 |
| 0.3 | x | -1.19e-4 | -2.00e-5 | -2.38e-1 | **yes** | 1.0000 | 1.0001 |

At every drive level from 0.03 up, `com` moves monotonically in one direction through the
entire kick phase and *continues in the same direction* after the tilt is removed —
sign-consistent kick/coast velocity, the same criterion that separated genuine transport
from erosion in the electron probe, satisfied here across a 10× range of drive strength,
not just at one lucky setting. Mobility scales steeply (super-linearly, roughly ×10-15 per
×3.3 step in drive) but continuously — **this is not a discontinuous gate/threshold**, it
is a smooth (if steep) mobility curve that was simply too weak to distinguish from the
noise floor at 0.01, the electron probe's drive level. Meanwhile Σlam is conserved to 4
decimal places and the spread changes by ≤0.01% at every drive level — the knot is not
leaving a decaying trail, the whole lam distribution shifts together.

**Isotropy**: x-axis and z-axis kicks at the strong drive (0.3) agree closely (mobility
-0.238 vs -0.231, within ~3%) — rules out the BCC-lattice dispersion anisotropy
(`project_bcc_dispersion_anisotropy` memory) as an explanation; the response is a genuine
isotropic monopole-type reaction, not a lattice-direction artifact.

## Interpretation

This is the first clean positive result in the "does collective coherence already emerge"
line of inquiry: a lam-carrying, non-circulating knot — properly started from an exact
equilibrium rather than a shocked IC — shows genuine rigid collective translation (COM
moves, shape and mass don't) using only the pre-existing operator set, once the drive
crosses some threshold between 0.01 and 0.3. No new field was needed for this topological
class. This contrasts sharply with the electron (circulation-type, curl≠0 globally), which
only erodes under the same kind of external tilt at comparable drive levels — consistent
with the working hypothesis that the electron's failure mode is topological (a
circulation/winding has an internal degree of freedom for the imposed stress to leak into
as erosion) rather than a universal absence of a translation-cost mechanism.

The mobility scaling and isotropy were confirmed in a follow-up scan (see Results table
above, both now folded in) — continuous super-linear scaling across 0.03/0.1/0.3, and
x/z agreement within ~3% at the 0.3 drive level.

## Caveats — not yet confirmed

- **Resolution/scale independence unchecked.** Only R=20 tested; whether the mobility
  curve and the coherent-translation character hold at a different R/N is unverified.
- **Mobility scaling law not characterized precisely.** Only 4 drive points on a steep
  curve — a proper power-law fit (and separating the cross_scale vs force_scale
  contributions, currently varied together) would clarify the mechanism.
- **Shape check is aggregate-moment only.** `mass`/`spread` are Σlam and its second moment
  around the COM, not a full profile-shape comparison — a translating-but-slowly-deforming
  knot could in principle pass this check. Worth a direct profile overlay (lam(r-com) at
  start vs end) if the effect survives the above checks.

## Artefacts

- `probe_proton_translation.py` — script copy at run time (the exact-fixed-point IC version)
- `run_output.txt` — full stdout, both drive levels
- `proton_translate_{baseline,strong}.npz` — per-run (t, phase, com, mass, spread, phi_rms,
  radiated_cum) histories
