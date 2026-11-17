# Embedded spheromak (10-24 / 10-25)

dst_embed.py (step() + Cartan, a 6 P 96) and mhd_embed.py (spectral resistive MHD, eta 2e-3) put the spheromak inside a
box-scale force-free ambient B0 (0, sin kx, cos kx) of strength frac x body peak, orientation aligned / opposed / tilt45
relative to the body axis. T 150 = 7.5 turnovers. Logs: dst_<frac>_<orient>.txt, mhd_<frac>_<orient>.txt; DST states
/bk/dst/2026/10/24/embed/. Runners: run_dst.sh (10-24), run_rest.sh (10-25: frac 0.3 DST + all MHD).

Result table and reading: repo-root `now`, SESSION 2026-10-25 item 1 (copied to exp/2026/10/25/now.md on the next tidy).
Short: the ambient does not stop the DST cascade (loops x30-61, worse with frac, aligned / tilt worst), the DST body is
carried +z by the ambient; in MHD a frac 0.3 aligned / tilted body dissolves into the ambient, opposed persists.
