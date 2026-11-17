# 10-04 item C: residual cubic anisotropy of the parity walk -- symbol only (m 0.8)

Target: cancel the cubic k^4 term of X Xp (rel spread 1.48e-3 at |k| 0.1, 5.9e-3 at 0.2, ~k^2) by mixing in hops on the
6 axial (square-face, (+-2, 0, 0) grid) neighbours. Scalar-stencil motivation: the cubic k^4 / (k^2 coeff)^2 is -1/3 for
the 8 diagonals, +1/2 for the 6 axials -> a mix cancels.

- `axial_mix.py`, `axial_orders.py`, `axial_sign.py`: split-step tick W = W_x W_y W_z (W_a = exp(-i k_a' sigma_a)) is NOT
  axial -- the product expands into the 8 diagonal hops of length 2: the xyz / zyx palindrome pair == X Xp at 2k exactly
  (same cubic sign; a dilated copy of the BCC walk). Single xyz order or parity-reversed: 90-deg asymmetry ~ k.
- `axial_ticks.py`, `axial_qform.py`: genuinely axial = one axis per Dirac tick (mass coupling between). Fails at k^2:
  the spin doublet splits and the kinetic coefficient is direction dependent down to ~0 (xyz: lower branch 0.0002 -
  0.104, upper 0.119 - 0.316 at |k| 0.02; xyzzyx: 0.0004 - 0.118; x xp y yp z zp: 0.021 - 0.112). X Xp: doublet
  degenerate, 0.1250 flat at 0.02.
-> none of the simple axial constructions is usable; consistent with D'Ariano-Perinotti (isotropic two-component Weyl
walks in 3D only on BCC). Remaining options (not tried): axial hops inside the mass (coupling) term (Wilson-like,
unitarity to be checked); 4-component walks; or simply resolution -- the residual is ~ k^2 relative, so 1 / xi^2 (xi
3.9 -> 5.2 halves it). Parked.
