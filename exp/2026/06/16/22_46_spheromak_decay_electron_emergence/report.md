# Spheromak decay → spontaneous electron moat emergence

**Date:** 16 Jun 2026  
**Script:** `tst/substrate/test_spheromak_decay.py`  
**Domain:** R=40, sponge=0.15, damp_max=0.003, absorbing BC (u_bg=0)

---

## Parameters

| Parameter | Value |
|---|---|
| R | 40 cells |
| rings | both (Hopf-linked, ring A + ring B) |
| lam_rings | 1.0 (on ring tiles) |
| cross_scale | 0.005 |
| lam_scale | 0.05 |
| lam_mode | dynamic |
| BC | absorbing sponge (u_bg=0) |

## Snapshot chain

| Snapshot | Steps | Notes |
|---|---|---|
| `proton_candidate_R40.npz` | 20000 | Initial Hopf IC |
| `proton_candidate_R40_decay2.npz` | +20000 | circ_B ~90% unwound, circ_A 38% |
| `proton_candidate_R40_decay3.npz` | +20000 | circ_B ≈ 0, circ_A = +0.007, pure dipole |
| `proton_candidate_R40_decay4.npz` | +20000 | circ_A continuing to decay |
| `proton_candidate_R40_decay5.npz` | +50000 | electron moat visible |

Commands:
```sh
PYTHONPATH=$PWD python3 tst/substrate/test_spheromak_decay.py \
    --R 40 --rings both --lam-rings 1.0 \
    --cross-scale 0.005 --lam-scale 0.05 \
    --n-steps 20000 --report 2000 \
    --save-snapshot proton_candidate_R40.npz \
    --plot-out tmp/spheromak_R40.png

# ... repeated with --load / --save-snapshot for each decay stage
```

Comparison plot generated with:
```sh
PYTHONPATH=$PWD python3 tst/substrate/probe_decay_compare.py \
    --R 40 \
    proton_candidate_R40.npz \
    proton_candidate_R40_decay2.npz \
    proton_candidate_R40_decay3.npz \
    proton_candidate_R40_decay4.npz \
    proton_candidate_R40_decay5.npz \
    --plot-dir tmp
```

---

## Key findings

### Spontaneous electron moat

In the final snapshot (decay5), the u field in all three midplane cross-sections (XY, XZ, YZ) shows:

- A bright central knot at r≈0 — the residual proton winding (high u, stable lam)
- A clear depletion ring surrounding it — the electron moat, emerging without being explicitly placed

This is the first simulation in this codebase where the electron appears spontaneously as a consequence of decay dynamics rather than being initialised by hand. The moat is visible consistently in all three planes, confirming it is approximately spherically symmetric.

The lam field remains stable at the core knot throughout all snapshots — the topology has not fully unwound, leaving a residual proton knot that then sculpts the surrounding u field into the moat geometry.

### Decay sequence

- Ring B (the linking loop, associated with neutron topology) unwinds first and rapidly
- Ring A (the toroidal ring, the proton remnant) decays more slowly and leaves a residual circulation
- The outer shells (r=4–5) develop positive circulation as the inner topology opens — consistent with field leaking outward to form a dipole

### Interpretation

The moat appears because the residual proton knot continuously depletes local u (substrate) faster than diffusion can replenish it in the absorbing-BC setup. The depletion radius is set by the balance between the knot's suction rate and the diffusion length from the surrounding substrate.

---

## Artefacts

- `decay_compare_XY.png` — XY midplane: |phi|, u, lam across all 5 snapshots
- `decay_compare_XZ.png` — XZ midplane
- `decay_compare_YZ.png` — YZ midplane
- `proton_candidate_R40_decay5.npz` — final snapshot with electron moat

---

## Next steps

- Measure moat radius vs Bohr orbit expectation for R=40
- Test whether moat radius is BC-dependent (absorbing vs reservoir u_bg)
- Corrected lam IC (poles not ring tiles) under investigation in `test_spheromak_decay_lam_poles.py`
- u_bg BC (reservoir) added to DampedLattice — needs calibration (1.0 too large, floods domain)
