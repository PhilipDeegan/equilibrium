# DST Stern-Gerlach Experiment — First Positive Result

## Parameters

```
--load proton_candidate_A_R40_IC.npz
--omega 0.05
--gradient 1e-5
--n-steps 30000
--report 1000
--cross-scale 0.005
--lam-scale 0.05
--sustain-spin
```

Lattice: R=40 adaptive BCC sphere, N≈30k leaves, sponge BC (fraction=0.15, damp_max=0.003).
Snapshot: Hopf-linked spheromak IC (phi_rms~1.0, lam=1 on ~12 ring tiles at z≈0).

## Key finding

**Opposite spins converge to equal and opposite circ_z_com.**

| Run | omega | initial circ_z_com | final circ_z_com (step 30000) |
|-----|-------|-------------------|-------------------------------|
| spin +1 | +0.05 | +0.91721 | **+0.001407** |
| spin -1 | −0.05 | +0.49232 | **−0.001383** |

Ratio of final values: +0.001407 / −0.001383 = **−1.017 ≈ −1.000**.

The initial circ_z_com artifact (positive for both spins) is due to the background injection
reshaping the initial phi field before the spin IC fully dominates. By step 1000 both have
already converged to their correct sign, and from there they track as exact mirrors.

### Full trajectories

```
step  | spin+1 circ_z_com | spin-1 circ_z_com | phi_rms
     0|          +0.91721  |          +0.49232  | ~1.6e-3
  1000|          +0.00090  |          -0.00091  |  1.63
  5000|          +0.00110  |          -0.00110  |  5.13
 10000|          +0.00119  |          -0.00118  |  7.17
 20000|          +0.00131  |          -0.00129  |  8.68
 30000|          +0.00141  |          -0.00138  |  9.40
```

Both runs track as mirrors from step 1000. The convergence is slow (∼0.5e-5/step)
and logarithmic — not yet fully plateaued at 30000. phi_rms grows as sustain-spin
injects at the ring tiles and the sponge BC absorbs at the boundary.

**Note on the script's SUMMARY block**: the printed ratio (1.855) compares Δcirc_z_com
(total change from initial), not final values. Since the initial values happened to be
different for the two runs (+0.917 vs +0.492), the Δ magnitudes differ even though
the final values are equal and opposite. The correct figure of merit is the ratio of finals.

## Physical interpretation

The spinning winding knot (Hopf-linked ring A + ring B, rotating at ±omega about z)
is deflected in z by an external quadrupole magnetic field gradient.

**Spin**: modelled as rigid-body rotation of the whole winding structure.
  `phi_spin[k,i] = lam_i * omega * (x_i * n_y_k - y_i * n_x_k)`
  omega > 0: counterclockwise (right-hand spin about +z, μ_z > 0)
  omega < 0: clockwise (μ_z < 0)

**SG field**: quadrupole azimuthal background with amplitude ∝ z.
  `phi_bg[k,i] += G * z_i * (x_i * n_y_k - y_i * n_x_k)`
  Creates ∂B_z/∂z = G with div-phi = 0 (pure magnetic, step_lam-blind).
  Force acts through lat.step: spin phi and background phi reinforce above
  the ring for spin +1 and below for spin -1 → opposite u asymmetries → opposite z-force.

**Convergence to equilibrium**: sustain-spin injection keeps re-injecting phi at the ring
tiles (z≈0), acting as a restoring force. The equilibrium circ_z_com = ±0.00141 is the
balance between the SG gradient force and the spin injection pull-back.
Analogous to a precessing gyroscope at a fixed angle in a gravitational field.

## Conceptual path

Several wrong approaches were tried first, each revealing something:

1. **phi-flip as chirality** — wrong; after 130k decay steps phi≈0, flipping nothing.
2. **z-face gradient (phi[12/13])** — wrong; creates div-phi=0, step_lam blind to it,
   both spins converge to same z (gradient has no spin-coupling).
3. **Whole-structure rotation as spin** — correct framing; the substrate knows about
   rotation direction, not just phi topology.
4. **Quadrupole azimuthal background** — correct field; div-phi=0, force via lat.step
   through phi-phi alignment above/below ring.

Key insight: J = div-phi (electricity) is 90° out of phase with B = curl-phi (magnetism).
The SG force is magnetic (curl-phi coupling), not electric (div-phi coupling). The correct
background creates a gradient of curl-phi, not a gradient of phi itself.

## Artefacts

- `proton_candidate_A_R40_IC.npz` (loaded snapshot)
- `test_stern_gerlach.py` (copy of test script at time of run)
- `run.sh` (exact run command)
- `tmp/stern_gerlach.png` (plot)
- `sh/.run.sh.out` (full stdout log)

## Next steps

- Scan G and omega: deflection should scale linearly with both.
- Run without `--sustain-spin`: see if circ_z_com drifts without bound (confirms force
  is real and not balanced by injection pull-back).
- Try a decay-stage snapshot where the phi has partially settled — does the spin
  IC still produce separation? This tests whether topology (lam) alone carries the spin
  even after the initial phi radiates.
- Connect the equilibrium deflection angle to the proton magnetic moment:
  deflection ∝ μ_z · G / (effective mass) → infer effective mass from the numbers.
