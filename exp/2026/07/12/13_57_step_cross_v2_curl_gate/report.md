# step_cross v2 — curl/holonomy gate (D1 follow-on)

## Motivation

Conversational thread: classical reconnection is driven by ∇·P_e (generalized
Ohm's law), not resistivity — a genuinely relational/multi-point quantity, not
a per-cell one. Reframed via the "field lines are relational, not literal
objects" discussion (cf. [[project_dst_proton_translation]]'s open question
about an explicit relational quantity vs. deriving relation from existing
per-cell state) and [[project_dst_magnetic_field]]'s flagged option of reading
`phi` as a discrete vector potential/connection.

Implemented `curl` as a fifth `step_cross_v2` gate arm: `phi[i,k]` read as a
connection A_k(i) on the outward edge (i, nb_k(i)); B_i built from the three
Cartesian (square-face) Wilson-loop plaquettes
(`CrossV2._plaquette`/`curl_vectors`); gated via the same shear-style
misalignment test (`|B_i||B_nb|-B_i·B_nb`) applied to B_i instead of the
transit vector v_i.

Unlike shear/product/opposed (each a function of one cell's own 14 phi values
compared against one neighbour's), B_i genuinely mixes three cells' phi per
component, and is provably (and numerically confirmed) EXACTLY zero — not
just asymptotically small like shear's ~1/R² — on any spatially uniform
coherent-transit pattern. Self-test extended in `cross_v2.py` accordingly;
all conservation/antisymmetry checks pass for the new gate
(`max|B| = 5.6e-17` on the uniform pattern).

## Run

`probe_cross_v2_gates.py` defaults: R=15, scale=3.0 (sph_frac=0.2), cs=0.01,
4000 steps, n_audit=600, gates=v1,shear,product,opposed,curl, refs=auto
(neutron audit p90/p99/3·p99). Full output: `output.log`. Data/plot:
`gate_sweep.npz` / `gate_sweep.png`.

## Result

`curl` behaves like shear/product, NOT like opposed:

| gate | surv_e | surv_n | separation e/n |
|---|---|---|---|
| v1 | 0.2738 | 0.6394 | 0.43 |
| shear/ref=1.28e-02 | 0.3644 | 0.6458 | 0.56 |
| shear/ref=5.93e-01 | 0.6969 | 0.7147 | 0.98 |
| shear/ref=1.78e+00 | 0.8074 | 0.7647 | 1.06 |
| product/ref=2.63e-02 | 0.3945 | 0.6437 | 0.61 |
| product/ref=1.68e-01 | 0.5421 | 0.6629 | 0.82 |
| product/ref=5.03e-01 | 0.6515 | 0.6960 | 0.94 |
| opposed/ref=0.00e+00 | 1.0146 | 0.9995 | 1.02 |
| opposed/ref=2.14e-02 | 1.0141 | 0.9995 | 1.01 |
| opposed/ref=6.41e-02 | 1.0133 | 0.9995 | 1.01 |
| curl/ref=7.70e-10 | 0.2955 | 0.6394 | 0.46 |
| curl/ref=6.19e-03 | 0.6780 | 0.6768 | 1.00 |
| curl/ref=1.86e-02 | 0.7419 | 0.7122 | 1.04 |

Raising `ref` moves surv_e and surv_n up TOGETHER (0.30→0.68→0.74 vs.
0.64→0.68→0.71) — the same "suppress everything, don't discriminate" collapse
as shear/product, not genuine linkage detection. Only `opposed` still nails
R1 outright (surv_e≈1.01, growing) — at the total cost of R2 (surv_n≈0.9995,
no decay at all). The exact-zero-on-uniform-transit theoretical property
(confirmed in the self-test) did not translate into empirical linkage
sensitivity.

## Caveat raised after the run (important — read before drawing conclusions)

Per [[project_bcc_dispersion_anisotropy]]: the BCC/Kelvin-cell dispersion
relation is anisotropic (diamond/cross artifact) at feature wavelengths
~1.5-8 cells, confirmed negligible only above ~3.6 cells. The electron/neutron
knots here were sized at scale=3.0 — near the edge of "safe" for their OWN
size. But the `curl` plaquette is a SINGLE-HOP loop (one cell edge per side)
regardless of knot size — it measures curl at ~1-cell wavelength, squarely in
the worst part of the anisotropic regime, independent of how large the overall
winding is. **This run's negative result may therefore be confounded by
lattice-direction noise from an undersized measurement loop, not a genuine
verdict on whether a holonomy-style gate can see linkage at all.** Not
re-tested this session.

## Next steps

1. Rerun with a LARGER threaded loop (multi-cell radius, e.g. ≥4 cells per
   side, matching or exceeding the spheromak's confirmed-safe scale) before
   concluding holonomy-style gates are structurally blind to R2 — the current
   single-plaquette result does not yet settle this.
2. If a larger loop still fails to separate: the "field lines are relational"
   idea likely needs a loop literally threading BOTH windings' interpenetration
   region (scaled to the crossing geometry), not just any bigger fixed-radius
   loop — a qualitatively different construction from a Cartesian plaquette.
3. Alternative already flagged in [[project_step_cross_v2]]: build the linkage
   detector directly from face-adjacent phi patterns per
   [[project_dst_phi_diffusion]] (Sweet-Parker phi-diffusion — already known
   to reconnect); converge toward that mechanism instead of a novel gate.

## Artefacts

- `cross_v2.py`, `probe_cross_v2_gates.py` — code at time of run (curl gate
  added)
- `output.log` — full run output (audit + all 13 arms × 2 ICs)
- `gate_sweep.png`, `gate_sweep.npz` — plot / raw survival histories
- `run.sh` — copy of `sh/run.sh` at time of run
