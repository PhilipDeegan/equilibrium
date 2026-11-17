# Periodic box for the walk (items 3c / 3d; Philip 10-03: "research the periodic box")

Lattice: the walk only hops along the 8 hex diagonals (+-1, +-1, +-1) on the integer grid of same-parity points, so a
box of EVEN period P (grid units, h = 1/sqrt3 hex distances) per axis wraps exactly onto the same sublattice -- the
Harris-sheet failure (project_harris_periodic_bc: square faces at D1 = 1 incommensurate with the box) does not arise
(no square-face hops). numpy roll version already exists (exp/2026/10/02/gradient/periodic_check.py; tst
_periodic_bcc). build_bcc_box is symmetric (-I..I, odd count) so it CANNOT be wrapped exactly -> needs a new
builder: cells ix in [-P/2, P/2 - 1], nb wrapped mod P (all 14 faces; square faces also exact for even P), then
alat.rebuild -> anchors nbj periodic; step_walk unchanged (no ghosts, no sponge). Rebuild of the .so must wait
for running jobs (they map the module).

Seed (`seed_periodic.py`, ring a 8 in P 48): phase = q/2 sum of image solid angles. Jump across faces (mod 2 pi):
x, y faces 0 by symmetry; z faces mean -0.182 (uniform: the image lattice's mean flow) + residual rms 0.063 (1
image), 0.003 (27), 0.001 (125). -> 27 images + subtract the linear term (mean jump / P) z gives a seam < 0.01 rad.
Cost: full-resolution solid angle x 27 is ~2.6 h for 10M cells in numpy; far images can use a coarse polygon
(~32 segments) or the dipole form -> ~1-2x the single seed.

Physics notes: box side = neighbour spacing (copies at P in x, y, z: a cubic lattice of structures -> the cubic
lattice of copies itself imposes 4-fold symmetry, keep in mind with the walk's residual cubic term). No sponge:
seed transients cannot leave, they stay as waves in the box (closed system, energy conserved). Size: R125 parity
structure reaches rho_cyl ~92 -> sides ~200-260 hex units (10-23M cells).
Plan: (1) builder + binding + tests (plane wave on box == numpy roll, background stationary); (2) seed with images;
(3) N 8 parity at box sides 200 / 230 / 260: does the outer-leg expansion stop at a size set by the spacing.

## Prepared 10-03 (not compiled yet: waits for R150 to finish)
- bcc_geometry.hpp: build_bcc_box_periodic(cluster, Px, Py, Pz) (even P, all 14 faces wrapped); lattice.cpp binding,
  exported in continuous. walk.py: cpp_box(P); face_of_hops now uses the most central full cell (box edges wrap).
- tst/substrate/test_walk.py TestWalkBox: cell count, all faces wired, C++ plane wave on the box == symbol
  (plain + parity), background stationary with g.
- spring_box.py: spring_cpp.py on the box (--P, --img_seg), seed = 27-image solid angle (outer 26 coarse) minus the
  mean face jump (uniform term), exact periodic z recentre, snapshots /bk/dst/2026/10/03/pbox.

## First box run: N 8 --parity, P 346 (side 200), 20k (springbox_N8_P346_par.log, box_pieces.py / box_pieces_P346.log)
Seed face jumps: x, y 0, z -0.62 (removed as a uniform term). Background rho sd ~0.11 throughout (closed box: seed
transients stay). Turns 8 -> 4 (~4.5k) -> 0 (~8k); 9k "clover" (one piece); 12k flat square ring (24.7k cells, sd
39 x 39 x 4.8) + 4 like-|winding| rings (~4.4k) at the box faces (rho_cyl 105, z +44), exact C4 symmetry. The rings
TURN AROUND at the faces (12-13k), move back in (rho_cyl 103, 81, 64, 51, 39, 33 at 13-18k) while rising faster than
the main ring, wrap through z, enter the square ring's hole and MERGE (19k: main 21.8k -> 35.5k cells, only ~120-cell
fragments left); 20k one flat square ring, 21.2k cells. 13k: neutral (winding 0) 8.7k piece at the box corner where 4
copies meet (ring collision product?), gone by 14k. Philip's 10-03 "rings can reattach" -- here they do.
Winding signs in box_pieces.py are orientation-arbitrary (SVD normal); only |winding| is meaningful.
