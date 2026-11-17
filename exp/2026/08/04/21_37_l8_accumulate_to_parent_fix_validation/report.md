# L8 `accumulate_to_parent` fix validation — no effect on the base-update blowup

Follow-on to today's `accumulate_to_parent` catastrophic-cancellation fix (`doc/dst_algo.md`,
found via `refined_poynting_ring.py --max-level 4`, not the spheromak/neutron-decay harness at
all). That fix was flagged as "may be a real contributor" to the long-standing, still-open L8
base-update blowup (`project_odd27_amr_junction_leak` memory, Parts 23-25) since deeper AMR
levels subcycle more, giving the bug more chances to fire. This experiment tests that directly.

## Method

Exact reproduction of `exp/2026/08/04/14_29_l8_etah1e-13_scaleoverride_r20`'s run (the
properly-calibrated L8 repro that first matched `project_odd27_amr_junction_leak` Part 2's
H/H0=130.8 finding) — same flags, same eta_h ladder, same everything — rerun against the
freshly rebuilt `mod_lattice.so` containing the `accumulate_to_parent` fix. Only the underlying
`.so` differs from the historical baseline.

```
python3 -u run/substrate/adaptive/refined_neutron_decay.py \
    --R 20 --max-level 8 --region-r0 5.0 \
    --scale-override 1.3606e-3 --scale-ref-level 8 \
    --lam-target 1.0 --lam-scale 0.05 --lam-mode dynamic \
    --cross-scales 1e-3 --damp-max 0 \
    --eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 1e-13 \
    --n-steps 1 --report 1 --n-threads 8
```

## Result

```
H=4.393687e+05  H/H0=134.0626
ABORT: H exceeds 100.0x H0 -- diverged at step 1
```

**`H/H0=134.0626`** — bit-identical to Part 25's isolated-harness table entry for
`eta_h[8]=1e-13` (`134.0626`), and matching the original `14_29` repro's `134.1` to 4
significant figures. The `accumulate_to_parent` fix changes **nothing measurable** about the
L8 blowup.

## Conclusion

The `accumulate_to_parent` catastrophic-cancellation bug (real, fixed, verified via
`refined_poynting_ring.py` and 3 new permanent C++ regression tests — see `doc/dst_algo.md`)
is **not a contributing cause of the L8 base-update blowup**. The two issues are unrelated:
the cancellation bug was found via a completely different harness (poynting ring, not
spheromak/neutron-decay) and manifested as a hard crash (a debug-assertion conservation
violation) at a much shallower level (3-4), not the smooth H-growth signature seen at L8. This
closes that open question from `doc/dst_algo.md` cleanly: no further need to treat the fix as a
candidate explanation for the L8 story. The base-update L8 blowup remains fully open, with an
even longer ruled-out list now (eta_h, gate, domain size, multiplicity/coupling-magnitude,
cross_scale/lam dynamics, and now `accumulate_to_parent`'s cancellation bug).

Next live lead per `project_odd27_amr_junction_leak` memory (CLAWD, Part 23): the `n_same=4`
boundary-cell category combined with the cross-level ancestor-substitution `u_src` mechanism,
not yet re-examined specifically at those cells.
