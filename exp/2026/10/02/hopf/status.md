# Hopf seed on the C++ walk (`hopf.py`, m 0.8, true xi 3.9, R80 = 2.8M cells, a 15, 2000 ticks ~1 min at 8 threads)

Philip 10-02: tight coiling ~60 windings, "like a Hopf fibration" -- illustrates the type of windings (nested, all
linked); the core is the toroidal loop, windings wrap around it.
Seed: the k = 0 rest space is 2-fold degenerate; diag(U, U) v is an exact rest state for any SU(2) U (1e-16), so
L, R = U(x) v_L, U(x) v_R, U from (alpha^q, beta^p) / norm, alpha = cos f + i sin f z / r, beta = sin f (x + i y) / r,
f = pi / (1 + (r / a)^4). Density uniform at t 0. alpha = 0 on the ring r = a (core loop); beta fills it and turns p
times along the loop. Read-out n_z = (|L0|^2 - |L1|^2) / |L|^2 (+1 = background).
- p 1 (`hopf_p1_*_T2000`; 100-tick check `hopf_p1_*_R80`): density gathers on the axis (max 2.74 at t 100, then
  disperses); the flipped ring travels -z and shrinks 15 -> 6 (t 300-650); rho min 0.005 at t 550; flip gone by
  t 850 (unwound); a dissolving dip leaves -z, outgoing density waves; box ~background by t 1800.
- p 8, p 30 (`hopf_p8_*`, `hopf_p30_*`): flip gone by t ~250 / ~100 (cells n_z<0: 5724 -> 48; 2412 -> 17); no
  shrink; core empties -> plain ring, rho min 0.002, radius ~10-12, travelling -z ~0.025/tick, present at t 2000.
Reading: twist along the loop stops the shrink; nothing holds the windings on the loop; the bare ring persists.
Confound: beta^p narrows the core fill (~ sin^p f): p and fill width mixed. Next: beta = sin f e^{i p phi}.
No states saved.
