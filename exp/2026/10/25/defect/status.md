# Unit size defects in the discrete rule: scattering (2026-10-25)

Philip 10-25: discrete cell size changes are discrete (units of the min cell size), relational (a junction event between
two cells), a result of substrate depletion; only relative differences can be modelled. Optical-clock redshift is
continuous to ~1e-19 over mm (JILA 2022), so with a unit of 1 / S (~5e-10 if floor / void ~ 1e-44 / 1.6e-35) the units
cannot sit in planes (terraces would be ~1e6 m apart at Earth's gradient): they must be sparse scattered defects whose
density sets the mean size (dithering). Question: how much does a wave scatter off one unit defect?

bcc.py: periodic BCC box, step() with per-cell sigma = s / S (integer s), face sigma = mean of the pair; energy exact
(symplectic Euler shadow oscillation ~1 %); group speed checked against the dispersion (206.8 vs 206.8).
scatter.py: Gaussian plane packet along z (u = f, phi = 0 -> right half used), one defect; scattered energy =
H(defect run - clean run); cross-section = scattered / (incident right-going energy per unit transverse area, cubic
spacing 1). Defects: 'cell' = one cell s = S - 1 (net depletion); 'pair' = one unit moved across a hex junction (S - 1 /
S + 1, conserved).

sigma_s * S^2 (nt 24, nz 400, S 100; scatter_lowk.txt; S 50 and nt 48 identical to 3 digits, scatter_nt*.txt):

| k    | cell     | pair     |
|------|----------|----------|
| 0.125| 1.61e-05 | 1.33e-06 |
| 0.25 | 1.26e-04 | 5.89e-06 |
| 0.5  | 2.11e-03 | 3.46e-04 |
| 1.0  | 2.50e-02 | 1.00e-02 |
| 2.0  | 3.6e-01  | 2.2e-01  |   (scatter_nt48.txt, nz 160)

- Exactly ~ (1 / S)^2 (linear perturbation), independent of transverse box (images negligible).
- Rayleigh: cell ~ k^4 (net-depletion monopole), pair ~ k^6 (conserved move = dipole), steepening toward the zone
  edge. The k 0.125 / 0.25 points (kw = 3) carry a ~1.7x spectral-width factor (<k^4> over the packet).
- nz 160 k 0.25 values in scatter_nt*.txt are contaminated (left half wraps onto the defect) -- use scatter_lowk.txt.

Reading: dithered unit defects scatter as k^4 / S^2 (k^6 for conserved pairs). Attenuation per cell length
alpha ~ C eps k^4 / S (defect fraction eps S for mean depletion eps). Visible light at real scale: k ~ 2 pi 1e-44 / 5e-7
~ 1e-37 -> k^4 ~ 1e-148: the dithered substrate is transparent; the mean size (delay, GR) is all that survives
coarse-graining at long wavelength. Strongly chromatic -> not an achromatic tired-light channel. Consistent with
clocks + light; no offload mechanism here.
