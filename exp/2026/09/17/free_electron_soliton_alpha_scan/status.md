# Free-electron self-sourced-lam alpha scan — status (2026-09-17)

## Context

Follow-up to `exp/2026/09/16/20_50_free_electron_ring_closure/`
(`ring_closure_probe_v3.py`), which found a first, single-config (n=2, r=10)
"positive signal" that self-sourced `lam = alpha*(u^2 + 0.5*sum(phi^2))`
(mass-as-depletion feeding back into `sigma=1/(1+lam)`, Kerr-effect-style
nonlinearity) could stop a free-standing ring winding from spreading, with
`spread_growth` crossing zero near alpha~1.0. Queued next: a finer alpha
grid, a genuine stability diagnostic (not just start/end comparison), and
replication across other (n,r).

Also raised mid-session, before any more scans: is a SINGLE ring/flux-tube
even the right structure to test at all? See below — this turned out to be
the more important question.

## Scripts

- `alpha_scan_v4.py` — fine alpha scan (9 values, 0.0-2.0 focused on
  0.7-1.3) x 3 configs ((2,6), (2,10) [v3 baseline], (6,14)), on the
  TRAVELING companion-phi IC (single circulating component, same
  construction as v2/v3). Adds a `slope_frac_per_period` diagnostic: linear
  fit over the LATTER HALF of the run, not just start-vs-end, to catch a
  transient zero-crossing vs a genuine flat plateau.
- `alpha_scan_v5_standing.py` — identical alpha grid/configs/diagnostics,
  but on the STANDING (phi=0 at t=0) IC instead (v1-style). Built to test
  the "two counter-propagating components" idea below; turned out to
  reduce to an already-known-bad IC (see Interpretation).

Both self-contained, `N_WORKERS=6` multiprocessing (each run is
single-threaded, ~6min/run measured, 27 runs each -> ~30min-2h depending on
decay-window compute). Results saved to `alpha_scan_v4_results.npz` /
`alpha_scan_v5_standing_results.npz`.

## Results

### v4 (traveling, one-component IC)

| config | alpha=0 grow% | alpha* (grow=0) | slope range across alpha |
|---|---|---|---|
| n=2, r=6  | -6.25% (already shrinking) | never crosses (nan) | -7.1% to -2.5%, extrapolates to 0 around alpha~3 |
| n=2, r=10 | +31.16% | 0.899 | **+1.3% to +2.3%, ALWAYS POSITIVE** — never flat |
| n=6, r=14 | -16.66% | never crosses (nan) | grow/slope disagree in SIGN at alpha=0 |

**Verdict: the v3 "crossover" at (2,10) is not a genuine fixed point.** The
slope diagnostic (latter-half linear fit) stays positive at every alpha
tested — the ring is still net-growing throughout, even right at the
endpoint-growth zero-crossing; that crossing is a transient wobble
cancelling at the two sampled instants, not stability. alpha* also isn't
universal — (2,6) needs ~3x more alpha than (2,10) to reach the same
balance point. (6,14) is the least trustworthy config in EITHER script
(see below).

### v5 (standing, phi=0 IC)

| config | alpha=0 decay% (energy left window) | alpha=0 grow% | alpha* |
|---|---|---|---|
| n=2, r=6  | 99.99% | +55.39% | 1.562 (likely noise — see below) |
| n=2, r=10 | 66.73% | +34.96% | never crosses (nan), non-monotonic in alpha |
| n=6, r=14 | 92.99% | -8.15% | 0.394 (wildly non-monotonic in alpha) |

**Verdict: worse than v4 across the board, not better.** Decay is far more
severe at every config/alpha (up to 99.99% of windowed energy gone at
(2,6) — at that level the reported spread numbers are almost certainly
measuring noise in a near-empty remnant). The slope-vs-endpoint-growth
disagreement from v4 recurs and is worse. (2,10)'s growth is non-monotonic
in alpha (rises at alpha=0.7 before falling), unlike v4's clean monotonic
trend for the same config.

## Interpretation

**Why v5 was built**: same-day discussion (see `now` and
`project_chirality_mixing_two_component_mass` memory) established, from
three independent directions — DST's own double-ring dipole electron model
(required for "omnibalance"/zero-net-force), the zitterbewegung/chirality
framing already in this project, and the Dirac equation's own structure
(the mass term literally couples left/right Weyl chirality components,
zitterbewegung being the beat between them) — that a real electron-like
structure plausibly needs (at least) two coupled, counter-something
components, not one. v4's whole scan used a single ring.

**Why v5 doesn't settle that question**: worked out by hand before running
anything that summing an `n_prop=+n` and `n_prop=-n` companion-phi wave
(v2/v3/v4's construction) at equal amplitude/matched time cancels phi
EXACTLY (`Phi_k_amp` flips sign under `K_local -> -K_local`, `sin` is odd)
— this is algebraically identical to v1's original phi=0 "shocked" IC, not
a new construction. Running it confirmed it performs worse (faster
dispersal out of the measurement window), consistent with v1's original
finding that this IC is a poor, broadband/shocked state. **This is a
negative result about one specific (degenerate) way of building "two
components," not evidence against the underlying QM argument.**

**(6,14)'s anomaly, reinterpreted**: originally guessed (after v4) to be a
WKB-companion-phi validity breakdown (largest K=n/r of the three configs).
v5 uses NO companion-phi/WKB machinery at all and STILL shows the same
flavor of anomaly (grow/slope sign disagreement, wild non-monotonicity in
alpha) — this weakens the WKB explanation. A more mundane, better-fitting
explanation: (6,14) sits closest to the domain/sponge edge of the three
configs (~19% radial clearance vs 34-48% for the other two) at the shared
`R_domain=34.0, width=2.0` setup. Boundary proximity, not WKB or n-specific
physics, is now the leading suspect.

**Also still unresolved from earlier in the session** (not re-tested by
v4/v5, still open): the ring envelope width (`width=2.0` cells, only ~2
lattice cells per Gaussian sigma against the lattice's own unit-1.0 hop
spacing) has never been resolution-tested — a real, independent risk that
under-resolution (not physics) is behind some of the observed "spreading"/
noisy alpha-dependence in ALL of v1-v5. Not yet disentangled from anything
above.

## Next ideas (not started)

1. **Resolution check, should come FIRST before any more alpha scans**:
   rerun the (2,10) baseline (and ideally (6,14)) at 2-3x `width` with a
   matched larger domain; see whether the v4 growth/slope numbers actually
   converge or keep shifting. If they shift a lot, everything in v3/v4/v5
   needs to be re-run at proper resolution before any alpha* claim means
   anything.
2. **A genuinely non-degenerate two-component IC** — either (a) sample the
   +n/-n companion-wave sum away from their common zero-velocity turning
   point (a temporal/phase offset, so phi doesn't cancel), or (b) port the
   ACTUAL double-ring dipole IC (`init_electron_dipole_ic`,
   `tst/substrate/adaptive/adaptive_flux_tube.py:681`) from the AMR
   cluster API onto the plain `SubstrateLattice` used here — a spatially
   (not temporally) paired kind of "two-ness" that does NOT reduce to v1.
   Note: `init_electron_dipole_ic` as constructed is already an EXACT
   static fixed point under the plain reversible step (uniform u +
   divergence-clean phi) — testing self-sourced lam on it meaningfully
   would require perturbing/kicking it first and checking whether lam helps
   it relax back, a genuine confinement/restoring-force test rather than an
   anti-dispersion one.
3. **Re-run (6,14) at larger domain clearance** specifically to test the
   boundary-proximity hypothesis before trusting any future (6,14)-shaped
   finding in this probe family.
4. Separately and independently: the magnetic-moment k-eigenvalue question
   (`project_dipole_moment_phi_convention_gap`) is still open, not blocked
   on any of the above.
