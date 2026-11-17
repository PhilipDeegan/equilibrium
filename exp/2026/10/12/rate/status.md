# 10-12 rate: per-cell / per-face rate of change from the local cell size (Philip: rate set by cell size -> gravity, dilation)

1. tickL.py (10-08 tick ring, sigma_i = 1 + g sin(2 pi x / side) on both legs, P 72 R0 9, T 30; tickL_*_10-12.log):
   g +-0.2 and 0 identical: x 0.000, 408 defects, one-off relaxation then static. A frozen structure has no rate of
   change to scale -> a rate gradient cannot act on it. Gravity test needs a structure that is changing (cycling).
2. Smooth self-propelled ring (../villain/vill.py --wt 3 --g, sigma_f at faces via set_face_medium + same factor on the
   commutator; smooth_*_10-12.log), t 0 -> 8, z +0.37 in all:
     both legs g +0.2: x +0.028 / +0.103 / +0.211 / +0.285 (t 2/4/6/8), g -0.2 exact mirror, g 0: 0. E drift 5e-4.
     -> towards LARGER sigma (faster, larger cells) = opposite of gravity. But:
     step() leg only: x +0.0007 / +0.018 / +0.042 / +0.023, E exact (1e-7) -> no clear deflection.
     commutator only: x +0.028 / +0.082 / +0.171 / +0.261, E drift 7e-4.
   The drift is from multiplying the commutator by sigma_f, which breaks its energy orthogonality (sum phi . sigma lie
   != 0) -- an artefact of how the rate was imposed, not a result.
Open: energy-exact way to put the rate into the commutator (sigma in the Hamiltonian, v = dH/da, so both step() and the
commutator derive from one H); smooth ring links ~0 net circulation (see ../villain/status.md); step-only needs a
longer / bigger run to resolve any deflection (|x| < 0.05 at t 8).
Philip: gravity will not show at electron scales until levels are refined near discrete -> rate-gradient test parked; uniform rate for electron-scale work.
