# DST Caveats — open issues flagged during review

---

## Theory (doc/dst.md)

### Notation / presentation

- ~~**φ absent from vocabulary table.**~~ **Fixed 2026-07-16** — φ now has a Vocabulary
  table entry.

- ~~**"No vectors" vs. φ.**~~ **Fixed 2026-07-16** — abstract and Vocabulary section now
  both say "no *imposed* vector fields; φ is 14 scalars per cell."

- ~~**Abstract omits σ.**~~ **Fixed 2026-07-16** — abstract equation now includes σ(λ_i),
  matching the body (line 82).

- **φ, $\phi$, "phi" used interchangeably.** Standardise to LaTeX throughout for a theory doc.

- **"Sintered macro-cell" (BH section).** First and only use; not in vocabulary table.

### Weak or unsupported quantitative claims

- **"72 triangular loops" (neutron topology).** Stated without derivation. Easy to derive from
  BCC geometry — show it or cite it.

- **"~10^58 sub-cells" for the proton.** Key number connecting cell-crossing inertia to
  observed proton mass. Needs derivation from (L_vacuum, L_proton, lattice geometry), or
  flagged explicitly as an estimate.

- **L_vacuum = 10^30 l_P used as a given.** The vacuum hierarchy table and time-dilation section
  use it as a fixed input, but open question 1 asks "What fixes L_vacuum?" — it is not yet
  derived. Mark those quantitative rows as provisional.

- **E = mc² from crossing rate.** The argument ("at v=c every tile crosses every tick; the cost
  is the rest mass") is an intuition sketch, not a derivation. Flag it as a motivation, not a result.

- **Magic numbers 28+ derivation.** The claim "1,4,10,20 → ×2 chirality → 2,8,20" doesn't show
  what sets those closure counts on the Kelvin lattice. The falsifiable prediction (lines 332–338)
  is the strongest part of the section; the setup needs to be tighter to earn it.

### Physics gaps

- ~~**CHSH values.**~~ **Fixed 2026-07-16** — dst.md now derives both S values
  after the table (at the standard angles all four terms are ±√2/2, adding
  coherently to S = 2√2·A, so amplitude A=½ gives √2 and A=1 gives 2√2) and
  states explicitly that √2 is the stochastic-Malus model's value, not the LHV
  bound (deterministic instruction sets reach S=2; 2√2 is Tsirelson).
  Verified numerically against bell.py the same day (S=1.417 / 2.828).

- **Wick rotation to Schrödinger.** The continuum limit gives the real diffusion equation. The
  document applies Wick rotation and asserts the Schrödinger equation emerges, but gives no
  account of how real-valued u becomes a complex wavefunction. What is the complex amplitude in
  DST terms? This is either an open derivation gap or the key step — it needs to be one or the
  other, not a breezy assertion.

- **ℏ/m ↔ 2KσL² identification.** σ and L both depend on position, so ℏ/m would vary with
  local cell size. How does a universal ℏ emerge? Unaddressed.

- **Beta decay / electron identity.** "The moat re-expands to the Bohr orbit (emitted as an
  electron)" — if the electron *is* the moat topology, "emitted as" is misleading. A clean
  one-sentence statement of what the electron is in DST terms would fix this.

- **Antineutrino as "topological accounting entry."** Too vague to be falsifiable. What DST
  object is it — a φ-wave pulse? A winding deficit? Needs a concrete description.

- **Quantisation "from topology" is not shown (2026-10).** Under step() + Cartan transport,
  circulation is a continuous label: off-quantum loops are held exactly, response is linear,
  with no threshold (`exp/2026/10/{13,14}`). The "Quantised charge = integer winding" row in
  *Theoretical status* is a target, not a result. Quantized offload is expected to be an
  aggregate or contextual property, not a local valve (Philip 10-21).

- **Structures cascade instead of decaying (2026-10).** A spheromak under step() + Cartan holds
  ~3.4 turnovers, then its content runs down to the smallest loops available, in place, with no
  decay; same onset in turnovers at a 9 and a 20 (`exp/2026/10/{22,24}`). MHD reference: held,
  slow self-similar Taylor decay, helicity outlasting energy (`equilibrium/reference/spectral.py`).
  The rule lacks a stop at structure scale and real offload channels. A local hand-off store
  holds a 6 but weakens as $(\ell/a)^6$; a travelling partner only carries the near field.

- **Closed structures do not interact.** Closed spheromaks have zero content beyond ~1.5 a, so
  pairs (n / n-bar analogue) see nothing of each other without contact, and DST has no
  independent push. How two bodies approach is open.

- **No electron candidate.** Every pre-09-24 electron/dipole IC is neutral (magnetic content
  only); every Taylor k=4.493 IC is closed; hop-built rings never become circular
  (`exp/2026/09/24/electron_ic_falsify`). An $|\varphi|^2$-magnitude drag cannot make charge by
  symmetry. Charge = div φ = −u_t in step(); a static Coulomb exterior exists only with the core
  held; a seeded divergence without a structure has no carrier.

### Weak or unsupported quantitative claims (code-level)

- **Casimir–Polder $1/r^3$.** `casimir_polder.py` integrates half the BCC zone and states a
  $1/r^3$ attraction; real Casimir–Polder is $1/r^6$–$1/r^7$ (09-29).
- **Legacy-weights numbers.** Everything before 2026-09-28 that rests on $W_\text{HEX}=3$
  (9:1 star, $c=\sqrt{10}$) is a legacy value: the curl calibration 4/3 in `test_curl.py`,
  `psi_pair.py`'s `C2_CALIBRATED=0.7`, traveling-wave $v_\text{phase}=3.62$, the 1.3 % Coulomb
  deviation. ~20 pre-09-28 scripts are pinned to the legacy ratio.

---

## Code (inc/eq/substrate.hpp)

### Correctness concern

- **`fixed_phi` not consulted in phase 1 of `substrate_step_cross`.** When a face is marked
  fixed and has inward phi, phase 1 still computes `dphi` for its adjacent faces. Phase 2 then
  applies that dphi to the adj faces but discards the drain term for the fixed face itself
  (because `fp_i[k]` blocks the apply). Result: phi is injected into adjacent faces without
  the corresponding drain on the source face — conservation is broken for cells that contain
  fixed faces. If fixed faces are *intended* as driven flux sources this is deliberate; if they
  are meant to freeze a face value while staying passive, this is a bug. Needs a decision and
  a comment either way.

### Latent hazards (2026-10 review)

- Seeders without $W_k$ (`seed_dipole_ic`, `curl_full_dipole_ic`) do not sample a single A:
  37 % error, no convergence under φ≈A. Fix or retire. `traveling_wave.py` is fine.
- `step_lam_force` (`equilibrium/substrate/lattice.py`) has no φ gate; only opt-in probes call it.
- `mhd.py` `SQ_PLAQUETTES["y"]` sign bug still unpatched (failures/mhd-py-curl-y-sign-bug.txt).
- `test_injecting_bc.py::test_lam_on_ring_kills_circulation` compares circulations at
  fp-noise level (cos-pattern IC).
- `lattice.py` `_KELVIN_VOLUME = 2.177` vs BCC cell volume 0.770 (affects `step_lam_sink` cost).
- `inc/eq/nanobind.hpp` capsule deleter frees `AlignedAllocator` memory with `delete[]`.

### Waste / overhead

- **`du` array is dead weight.** `du[i]` is written in pass1 and used only to compute
  `u_new[i]` on the same line. The vector (N doubles) is never read again. Replace:
  ```cpp
  du[i]    = -dt * si * acc;
  u_new[i] = fixed_u[i] ? u[i] : u[i] + du[i];
  ```
  with:
  ```cpp
  u_new[i] = fixed_u[i] ? u[i] : u[i] + (-dt * si * acc);
  ```
  and drop the `du` declaration.

- **Per-call heap allocation.** `u_new(N)`, `du(N)`, and `dphi_buf(N*14)` are allocated on
  every step call. For a hot loop over many steps these are repeated malloc/free pairs. Consider
  passing pre-allocated scratch buffers in, or holding them in a wrapper struct.

### Consistency

- **`n_actual` vs `n_avail`.** `substrate_step` uses `n_actual` for the effective thread count;
  `substrate_step_lam` and `substrate_step_cross` call the same variable `n_avail`. Pick one name.

### Design question (worth documenting)

- **Arithmetic mean for face conductance.** Pass 2 uses `sface = 0.5*(si+sj)`. For two
  conductances in series (which is what a face between two cells is) the harmonic mean
  `2*si*sj/(si+sj)` is the physically natural choice. The arithmetic mean is correct for
  parallel conductances. The choice may be deliberate (simpler, small difference for
  moderate lam gradients) but deserves a comment explaining why.

- **Phase 1 of `substrate_step_cross` is single-threaded.** The comment correctly notes scatter
  writes require it, but for large N this pass is O(N) on one core. At the scale where threading
  matters this will be the bottleneck. Flagging now so it is not a surprise later.
