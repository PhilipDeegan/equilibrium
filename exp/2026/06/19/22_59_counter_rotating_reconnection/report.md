# Counter-rotating spheromak reconnection — omega=0.1, lam-target=1.0

**Date:** 19 Jun 2026 22:59 CEST  
**Script:** `tst/substrate/adaptive/test_counter_rotating_spheromak_reconnection.py`

## Parameters

| Parameter | Value |
|---|---|
| R | 60 |
| scale | 12 |
| z-sep | 16 (z0 = ±8) |
| omega | 0.1 |
| lam-target | 1.0 |
| n-steps | 20000 |
| report | 500 |
| cross-scale | 0.005 |
| lam-scale | 0.05 |
| N cells | 1,175,219 |
| sigma | 0.7 |

## Output (columns: step, J_mid, phi_up, phi_mid, phi_dn, lam_up, lam_dn, czc, phi_rms)

See `run.out` for full output.

Key values at selected steps:

| step | J_mid | phi_mid | lam_up | lam_dn |
|---|---|---|---|---|
| 0 | ~0 | — | 0.0047 | 0.0047 |
| 1000 | 1.08e-02 | 2.93e-02 | 0.0047 | 0.0047 |
| 2500 | 1.41e-02 | 1.43e-02 | 0.0047 | 0.0047 |
| 5000 | 4.90e-03 | 3.16e-03 | 0.0047 | 0.0047 |
| 10000 | 2.12e-03 | 1.52e-03 | 0.0047 | 0.0047 |
| 20000 | 1.45e-04 | 3.25e-04 | 0.0047 | 0.0047 |

## Findings

**Phase 1 confirmed: X-point current sheet signature present.**

- J_mid rises from ~0 to peak ~1.4e-2 at step 2000–2500, then decays monotonically to 1.4e-4 by step 20000.
- phi_mid peaks at step 1000 at 2.9e-2 (~13× phi_rms) — field energy concentrating at the X-point — then decays in parallel with J_mid.
- phi_up ≈ phi_dn throughout — the two spheromaks remain symmetric.
- lam_up = lam_dn = 0.0047, **unchanged for all 20000 steps** — no topological reorganization. The winding topology is not responding to the phi dynamics.

**Phase 2 (lam redistribution across midplane) not observed.**

Root cause: phi amplitude relative to lam is too small. Only ~0.8% of cells have lam > 0, so the phi dissipates before it can drive topological change via step_lam.

## Next steps

Stronger driving: **omega=0.3, lam-target=5.0** — larger phi wake, larger lam scale, better phi/lam ratio.  
Run queued in `sh/run.sh`.
