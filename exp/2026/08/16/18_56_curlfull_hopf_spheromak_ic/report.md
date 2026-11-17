# Curl-full Hopf-linked-pair spheromak IC — R180L0 / R60L1, t=0 build + plot

## Why

Follow-on from `exp/2026/08/16/14_50_rebuild_r180l0_r60l1_post_pass1_fix`'s
side investigation (`project_dst_discrete_curl` memory): DST's dipole-mode
spheromak IC (`equilibrium.substrate.dipole_mode_ic`) is curl-free by
construction (`phi=grad(U)`), so it can never show PHARE-style flux-tube
merging regardless of step count. Validated a genuinely curl-full,
coordinate-free alternative (a Hopf-linked ring pair with CONSTANT phi
around each ring, not a scalar potential) at test scale (R_domain=90,
hops=41): both rings' circulation exactly conserved under pure
`evolver.step()`, eroded under `step_cross` — confirming `step_cross` is the
DST-native resistive/reconnection mechanism. Moved to its own experiment
directory (was briefly built inside the dipole-IC rebuild's directory by
mistake — user caught this, moved out to avoid interfering with that
directory's own plots).

## What this run did

Built the SAME two production domain configs the dipole-IC rebuild used
(`cells_across=220`, uniform R180 / L0=60+L1@46.667), but seeded with the
curl-full Hopf-linked pair (`equilibrium.substrate.curl_full_spheromak_ic.
seed_curl_full_pair`, `hops=41`, `amplitude_A=amplitude_B=1.0`) instead of
the dipole IC. `hops=41` chosen so Ring A's physical radius sits at ~half of
each config's `R_SPH*SCALE` envelope (empirically radius≈1.333*hops*cell_size
— same hops value works for both configs since the SCALE/cell-size ratio
cancels between them). lam seeded the same convention as the dipole runs
(`lam = lam_scale*(u^2+0.5*sum(phi^2))`, `lam_scale=1.0`).

**Important physical difference from the dipole IC**: `u` is never seeded
here — ALL field content lives on the ~332 ring/plaquette leaves' `phi`
faces, not spread through a smooth spherical volume. This is a genuine
current-loop construction, not a volume-filling spheromak field — flagged
explicitly, not a bug.

Both builds succeeded cleanly:
- `curlfull_uniform_r180`: 31.7M leaves, ring found in 10.4s, H0=332.0
- `curlfull_l1_wider_margin`: 15.5M leaves (L0=622634/L1=14919795), ring
  found in 4.9s, H0=332.0 (identical to uniform — same construction, same
  energy content regardless of mesh resolution, as expected)

Both configs found the SAME ring pair sizes (A=328 tiles, B=4 tiles) and
identical circulation (circA=328.0, circB=4.0 exactly — matches
`amplitude*N_ring`, confirming clean construction on both meshes).

Plotted ring geometry only (3D scatter, `plot_curlfull_ics.py` — a u-based
heatmap plot would show nothing, since u≡0 everywhere in this IC): both
configs show the expected octagonal Ring A loop with the small Ring B
plaquette Hopf-linked at one edge, no stray nonzero-phi cells elsewhere.
See `plots/curlfull_uniform_r180_ring_t0.png`,
`plots/curlfull_l1_wider_margin_ring_t0.png`.

## Next steps

- **Not yet done, deliberately deferred by user** ("we will step later"):
  stepping either config with `step_cross` to look for real merging/
  reconnection behavior at production scale. The mechanism (circulation
  erosion under step_cross) was already validated at test scale
  (project_dst_discrete_curl memory) — this run only confirms the ICs
  themselves are built correctly at production scale.
- Ring identity (`ring_A_js`/`ring_B_js`) is NOT stored in the `.h5`
  checkpoint — any follow-up script re-derives it deterministically via
  `find_square_ring_on_leaves`/`find_linked_loop_anchored` on the loaded
  lattice (same build, same result, ~5-11s at this scale).
- Ring B is still a small (4-tile) test-loop-scale poloidal partner, not a
  fully-scaled poloidal envelope wrapping the whole tube — a real
  simplification inherited from `adaptive_flux_tube.find_linked_loop`'s
  original design (see `curl_full_spheromak_ic.py`'s module docstring). A
  more realistic spheromak would need a properly-scaled poloidal winding,
  not attempted here.
