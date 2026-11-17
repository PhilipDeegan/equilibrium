# Entries for 2026-08-21 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## 2026-08-21 (compressed): step_cross `barrier` gate at derived m0~3 is the first gate to get R1/R2 discrimination in the right direction

`probe_cross_v2_gates.py` gained `--gates barrier` wiring. R1/R2 at R=20, sph_frac=0.2, cross_scale=0.01, 3000 steps:

| gate/m0 | surv_e | surv_n | separation |
|---|---|---|---|
| opposed | ~1.0 | ~1.0 | freezes both, R2 always fails (structural) |
| barrier, m0=0.16 | 0.38 | 0.72 | 0.53 — backwards |
| barrier, m0=0.30 | 0.47 | 0.72 | 0.65 — backwards |
| barrier, m0=2.49 (neutron-derived) | 0.88 | 0.73 | 1.20 — right direction |
| **barrier, m0=3.0** | **0.955** | **0.752** | **1.27 — best so far** |
| barrier, m0=5.0 | 1.00 | 0.876 | 1.15 — both freezing |
| barrier, m0=10.0 | 1.00 | 1.00 | 1.00 — fully frozen |

Clean unimodal trend: m0~3 is the sweet spot between "backwards" (low m0) and "freezes everything" (high m0). First gate in the project's history to show both correct-direction separation and real neutron decay simultaneously — not fully passing, electron still loses 4.5% at the best point. `cross_scale=0.01` held fixed (never derived) — resolution-robustness of m0=2.49/3.0 checked 2026-08-24 (see above, at cs=0.1); a real 2D m0 x cross_scale grid is still not done.

**Electron construction reframed away from the double-ring** — real electron length scales vs. mass ratio contradiction; see `doc/dst/failures/electron-double-ring-topology-mass-ratio-mismatch.txt`.

**Discrete-exact ring circulation — dramatically validated (zero internal motion, a real limitation).** `seed_constant_current` (weighted flat current, Kirchhoff-consistent) + uniform u: exact fixed point (`du/dt=dphi/dt=0`, symbolic+numeric), held circulation to within 0.06% over 8000 steps under a full Saha background — vs. the continuum-sampled double-ring's ~70% erosion under the same protocol. Real evidence genuine topological protection (not continuum-approximated) is what survives a realistic environment. But it's a steady-DC-current analogue (real circulation, zero internal dynamics), not the "something traveling at c" ingredient E=mc^2 needs. A genuine quadrature (cos+sin, one-directional) companion attempt is documented as unfinished in `doc/dst/failures/toroidal-winding-cos-ic-bulk-leakage.txt` — the safer path is reusing the validated traveling-wave-reproducer formula (`exp/2026/08/19/18_30_traveling_wave_reproducer/`) adapted to ring geometry.

`init_toroidal_winding`'s cos/cos IC dominated by bulk leakage — see `doc/dst/failures/toroidal-winding-cos-ic-bulk-leakage.txt`. Reusable tooling: `probe_ring_bump_relaxation.py`, `plot_ring_bump_relaxation.py`, `probe_ring_standing_vs_traveling.py`.

**Conceptual corrections locked in this session, apply going forward:**
- No "microstate"/"macrostate"/"observer" — say "global state"/"local state"; "measurement" is an ordinary coupling like any other.
- Planck length is NOT the fundamental unit — `L_MIN=1` is. Planck length is the emergent, locally-varying void-cell size (`inc/eq/core/data/lattice/cell/discrete_cell.hpp`), tied to local `c`/`lambda` — void ~ `N_SAGA~2880` floor units (Sgr A*/S2-anchored), a different calibration from the ~10^30 l_P extragalactic figure in `doc/dst.md` — don't conflate. `lattice_step_minimum_from_gps_gravitational_data()` derives a quantization step (delta_l_P~8.42e-45m, from real GPS 45us/day data) — minimum resolvable granularity of cell-size variation, not an absolute cell size.
- cos/sin ICs impose continuum wave mechanics — legitimate only as a resolution/calibration yardstick (predicted wavelength vs cells-per-wavelength, same family as hops>=3-4/anisotropy-band checks), never as a claim about the discrete substrate's fundamental behavior. Shapeless/single-bump ICs are more honest for discovering real relaxation behavior.
- Mass (amount of flux, structural/static) vs. inertia (cost of translation, dynamical) may be genuinely separable, analogous to gravitational vs. inertial mass — `doc/dst.md`'s current text conflates them (defines mass AS crossing-cost). Not reconciled.
- CME/reconnection framing: a single pinch-off event can simultaneously open one part of a structure (proton, no closed loop) and seal another part into a new closed loop (electron) — resolves the "how can decay produce a new topologically-closed object" worry. Mechanism-level idea only, nothing built.
- Positronium para (2gamma, spin-singlet, fast) vs. ortho (3gamma, spin-triplet, slow) flagged as a good future target for DST's chirality/spin structure — ties to the still-open 720-vs-360-degree periodicity (spinor) question, unresolved.
