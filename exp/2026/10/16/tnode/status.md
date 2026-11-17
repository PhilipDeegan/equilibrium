# 2026-10-16 tnode -- transverse junction (discrete model, constant L, no stub)

tdisp.py: 2 amplitudes per face (polarisations shared by k / mirror k), b = (2P - 1) a, P = projector onto node
quantities (E-like: e_kp; H-like: d_k x e_kp, weights sqrt w_k). Exactly orthogonal + symmetric (energy exact).
tvar.py: variants. Speeds at q 0.02 (cells / tick), (L) = longitudinal, (T) = transverse:
- E + H, hex:sq 2 (LAP_W): [100] T 0.391 0.391 L 0.492 | [111] T 0.416 0.416 L 0.449 | [110] T 0.391 0.428 L 0.461
- E only: T 0.69 / 0.667 (anisotropic), L propagates (0.44-0.50)
- H only: nothing propagates (all static / flip)
- no hex:sq ratio (0.5-4) makes T isotropic or stops L.
Two transverse polarisations propagate, but always with a propagating longitudinal wave and ~6 % anisotropy: an
elastic-solid (Cauchy aether) junction, not Maxwell. 8 static + 8 flip branches remain.
Diagnosis: a one-cell projector onto cell fields gives grad div + curl curl; Maxwell needs only curl curl (longitudinal
E static = Coulomb). Next: junction where ports couple through loop (circulation) structure across cells (DEC: E on
face hops, B on Delaunay triangles; cf the 6 static transverse circulations of the scalar node), or a symmetric
condensed node built from face pairs (TLM SCN analogue) -- check dispersion first.
