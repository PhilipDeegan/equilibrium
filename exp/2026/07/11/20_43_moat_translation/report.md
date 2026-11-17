# Electron-as-moat translation — three-way comparison (proton knot / phi-ring / lam-deficit moat)

Tests an alternative electron model against the two already tested this session
(`exp/2026/07/11/19_45_proton_knot_translation/`, `exp/2026/07/11/20_20_electron_translation_uniform/`):
CLAUDE.md's canonical spec describes the electron as "a moat — a reciprocal deficit of u
surrounding the proton knot... analogous to a doping hole in a semiconductor," not a
phi-circulation (winding) topology. `project_dst_electron_topology.md` already flags free-
electron topology as unresolved; the toroidal-ring probes (both this session's uniform-
lattice version and the original adaptive-cluster one) tested one candidate, not a settled
model. This probe tests the moat candidate directly, on the same lattice/protocol as both
prior probes.

## IC

Same proton knot as `probe_proton_translation.py` (`set_winding`, chi_0=0.9, profile=
"1/r2"), with a LOCAL DEFICIT carved into lam at a shell radius (r_moat=5, width=1.0,
depth=0.6 of the local value): `lam_final = max(0, lam_full - depth * exp(-((r-r_moat)/w)^2)
* lam_full)`. u=0 and phi=0 everywhere, as for the plain knot — this is still an exact
fixed point of the full operator set regardless of the SHAPE of lam(r) (verified for the
monotonic knot profile; the argument doesn't reference the shape, only that phi=0 and u is
uniform, so Δu=0 and Δφ=0 trivially, and step_lam_force's pull is zero since u is uniform).
The moat itself is tracked via the DEFICIT field (lam_full − lam_actual), isolating the
shell from the underlying knot for the COM/mass/spread diagnostics.

## Results — full three-way table (R=20, N=43599, same protocol throughout)

| object | drive | v_kick | v_coast | sign match? | mass_frac | spread_frac |
|---|---|---|---|---|---|---|
| proton knot | 0.01 | +2.7e-7 | -8.8e-7 | no | 1.0000 | 1.0000 |
| proton knot | 0.03 | -1.3e-6 | -3.0e-6 | yes | 1.0000 | 1.0001 |
| proton knot | 0.1 | -2.0e-5 | -5.9e-6 | yes | 1.0000 | 1.0001 |
| proton knot | 0.3 | -1.2e-4 | -2.6e-5 | yes | 1.0000 | 1.0001 |
| electron ring | 0.01 | +1.5e-3 | +3.0e-3 | yes | 33.00 | 10.11 |
| electron ring | 0.03 | +4.6e-3 | -4.9e-3 | **no** | 20.76 | 10.15 |
| electron ring | 0.1 | +1.5e-2 | -3.0e-3 | **no** | 7.01 | 10.47 |
| electron ring | 0.3 | +3.2e-2 | +5.3e-2 | yes | 2.37 | 10.92 |
| **moat** | 0.01 | -6.4e-5 | +1.9e-5 | no | 1.015 | 0.995 |
| **moat** | 0.03 | +6.4e-5 | +3.0e-4 | yes | 1.018 | 1.000 |
| **moat** | 0.1 | +2.1e-3 | +1.4e-3 | yes | 1.014 | 1.048 |
| **moat** | 0.3 | +1.3e-2 | +6.6e-3 | yes | 1.083 | 1.206 |

## Findings

1. **The moat translates coherently, like the proton knot and unlike the ring** —
   sign-consistent kick/coast velocity from drive 0.03 up (same pattern as the proton;
   the ring's sign is erratic across levels). It is a genuine third character, though,
   not just a faster version of the proton: mass is NOT machine-conserved (creeps up
   1.4%–8.3% across the sweep, vs the knot's exact 1.0000 at every level) and spread
   visibly grows with drive (+20.6% at 0.3, vs the knot's flat ~0%) — the moat has real
   internal give, "fluid" rather than rigid, while remaining nowhere near the ring's
   uncontrolled 2×-33× mass blowup and ~10× spread inflation.
2. **The moat is ~100× more mobile than the proton's own core** at matched drive levels
   (e.g. at 0.1: moat v_kick=2.1e-3 vs knot v_kick=-2.0e-5). This is the qualitatively
   right electron/proton mass asymmetry (though this toy setup isn't calibrated to the
   real 1836× ratio) — and it falls out naturally from the moat sitting at r=5, away
   from the knot's extremely lam-throttled core, in a region of much higher σ(λ).
3. **Isotropy: exact (not just close) between x and z kicks** — bit-for-bit identical
   summary statistics, because the moat construction is purely radial (a function of r
   only, no preferred axis at all) and the underlying BCC lattice/sponge/gradient
   machinery is exactly symmetric under coordinate-axis permutation. This is a stronger
   confirmation than the proton knot's own isotropy check, which showed a small ~3%
   x/z difference (worth a closer look eventually, though it doesn't change the
   headline finding either way — likely a difference in construction symmetry between
   the two probes, not a real physical anisotropy).

## Interpretation

This reframes the electron/proton asymmetry found earlier this session. It is NOT
"protons can translate coherently and electrons categorically cannot" — that would be
physically wrong (real electrons plainly do move coherently). It is "u/lam-family
structures (knot, moat) translate coherently; phi-circulation structures (the ring)
erode" — and the moat, being topologically a u/lam-deficit object rather than a
circulation, belongs to the mobile-and-coherent class. The earlier "electron erodes"
finding was a statement about the WRONG topological candidate for what an electron is,
not a real physical conclusion about electron inertia.

## Open questions / next steps

- **Ionization/capture**: can a bare proton knot (no moat) capture a free-floating moat
  into a bound state, and can a bound moat be stripped off to leave an ion? This is a
  genuinely different (two-body) experiment from single-object translation, queued next.
- **Free-electron quantization**: does an unbound moat show any banding/level structure
  from propagating through the periodic Kelvin-cell substrate (a DST-native analogue of
  Bloch band structure, distinct from atomic Bohr-orbit quantization), building on the
  already-confirmed BCC dispersion anisotropy? Queued, no experiment run yet.
- Resolution/scale independence and a proper power-law mobility fit remain open for all
  three objects (shared caveat with the earlier two reports).
- The proton knot's own ~3% x/z isotropy discrepancy (vs the moat's exact match) is
  unexplained and worth a quick look.

## Artefacts

- `probe_moat_translation.py` — script copy at run time
- `run_output.txt` — full stdout, all four drive levels + x-axis isotropy check
- `moat_s{0.01,0.03,0.1,0.3}.npz`, `moat_strong_x.npz` — per-run histories
