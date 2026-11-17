## 2026-09-22 -- scoping a discrete (p,q) torus-knot helical winding

Scoping exercise per Philip's tokamak/stellarator framing (see
`status.md` for the full session record this follows on from): a purely
toroidal field doesn't confine (curvature/grad-B drift), real confinement
needs *rotational transform* -- field lines helically coupling the major
(toroidal) and minor (poloidal) circulation directions into ONE path, not
two superposed fields (`curl_full_dipole_ic.py`, already shown in this
session's gallery to still carry real circulation and still be closed --
superposition, not a helical path). Goal here: scope a genuinely discrete,
lattice-native (p,q) torus-knot winding -- built from actual BCC face-hops,
not a continuum field formula.

### Continuum target

Standard (p,q) torus-knot parametrization, major radius R, minor (tube)
radius r, major angle advancing p times and minor angle q times over one
full traversal:

```
x = (R + r*cos(q*t)) * cos(p*t)
y = (R + r*cos(q*t)) * sin(p*t)
z = r*sin(q*t)
```

The physically load-bearing feature (why this, and not a flat ring)
is that `x,y` (radial distance from the z-axis) genuinely varies with
`t`, not just `z` -- a real torus-knot path visits both the inboard
(small rho, favourable curvature) and outboard (large rho, unfavourable
curvature) sides of the tube as it advances, which is specifically what
cancels the net drift in real rotational-transform confinement.

### What's feasible on this BCC lattice, checked empirically

The lattice's 14 face directions are FIXED in the global frame (8 hex
diagonals + 6 axis-aligned sq faces: `_FACE_HOP`/`_FACE_NORM` in
`adaptive_flux_tube.py`) -- there's no locally-rotating "radial" direction
to hop along. `find_toroidal_ring`'s existing square/octagon modes already
solve the analogous in-plane problem (faking a smooth circle from 4 cardinal
directions, or 8 via chamfered diagonal staircases) -- the same
staircase-chamfer trick generalizes to the meridional (rho,z) plane, since
sq-hops are isotropic in length (2/sqrt(3)) across all 6 directions
including z.

**Built and tested** (`prototype_torus_knot_winding.py`): the simplest
non-planar case -- p=1 (one major lap, the existing 8-tile
`_TOROIDAL_PATTERNS['z']` pattern), z-only excursion (NOT yet real
rho+z coupling -- see caveat below) of height m inserted symmetrically
(up over one side of the loop, back down over the other) so the path still
closes exactly at z=0. Result: **closes cleanly on the real lattice at
every m tested (1, 2, 3, 5), `winding_number` reads exactly 1.0000 in every
case** (robust, not resolution-sensitive at these sizes). So the basic
mechanics of interleaving z-hops into an in-plane loop and getting a valid,
self-consistent closed path is NOT the hard part -- it works.

**What was NOT built**: genuine coupled rho+z motion (the real torus-knot
target above). What's built is a flat ring that detours out of its own
plane and back -- a "corrugated ring," not a path that visits both
inboard and outboard radii. This matters physically, not just
geometrically: a fixed-rho excursion doesn't sample the curvature
asymmetry that actual rotational transform cancels, so it isn't yet a
faithful discrete analogue of the confinement mechanism motivating this
whole thread. Building genuine rho+z coupling needs the meridional
extension of the octagon's chamfer trick (interleaving a radial hop
direction, not just z, at each step) -- straightforward to describe, not
yet implemented or tested; real design/engineering work, not a quick
follow-up.

### The more important finding, not anticipated going in

Seeded the closed z-wobble loop with a simple constant-amplitude
circulating current (same style `init_toroidal_winding` uses, just
amplitude=1 instead of `cos(theta)`) and measured `phi_ring_circulation`:
**13.55 (m=1) down to 12.90 (m=5) -- large, real, and only weakly
sensitive to how non-planar the loop is.** This is the crux: *any* closed
loop that genuinely carries current will show nonzero macro circulation by
construction -- that's what "current flows around a topologically closed
path" means, full stop. Making the path helical instead of flat doesn't
make the circulation vanish; nothing about non-planarity changes that.

This means the "circulation-free" criterion this session established
against the spheromak (`doc/mems/project_electron_ic_topology_wrong.md`'s
2026-09-22 update) needs a real correction, not just a footnote:
**"topologically closed winding carrying real current" and "zero net
circulation" are in tension by definition, not just in practice.**
`doc/dst.md`'s own mass mechanism explicitly requires a real closed
current loop (the literal "minimal toroidal winding spans 8 tiles") --
demanding it also carry zero circulation would rule out the doc's own
mechanism, not just the spheromak-equivalent construction found earlier
today.

Re-examining `doc/mems/project_spheromak_decay.md`'s own numbers with this
in mind: `circ_A` (the surviving piece) does NOT measure zero either --
it settles around +0.007 to +0.032, small but genuinely nonzero, same
sign convention throughout. What actually distinguished the two rings
wasn't "has circulation vs. doesn't" -- both did -- it was that `circ_B`
eroded ~90%+ under the tested dissipative dynamics (`step_cross` at a
given `cross_scale`) while `circ_A` eroded much more slowly. **The real
discriminator between "stable winding" and "decaying winding" is
dynamical -- does it erode under real stepping -- not a static
circulation reading at t=0.** Today's circulation measurements (this
session's gallery: vecpot dipole vs. spheromak, both ~500-630) are still a
meaningful red flag (same order of magnitude as a construction independently
known to decay), but "zero circulation" was never the right bar to clear,
and no construction that's an actual current-carrying winding (as opposed
to a curl-free gradient field like `dipole_mode_ic.py`) was ever going to
clear it.

### Recommendation

1. **Don't build the full rho+z-coupled (p,q) torus knot next.** It's a
   real, larger engineering task (meridional chamfering, closure isn't
   guaranteed the way the z-only case turned out to be, needs its own
   validation), and building it before answering the more basic question
   below risks a lot of design effort on the wrong axis.
2. **The load-bearing next step is dynamical, not geometric**: take the
   already-working closed winding (even the bare, un-helical 8-tile ring
   is enough for a first pass -- the z-wobble variant built here is a fine
   second case) and step it forward under the SAME dissipative operators
   that eroded the spheromak's `circ_B` (`step_cross` at a comparable
   `cross_scale`), tracking `phi_ring_circulation` (or `winding_number`,
   or both) over time. Does it erode like `circ_B`, or hold like `circ_A`?
   That's the real test of "is this winding stable enough to be an
   electron," and it's answerable on constructions that already exist,
   without waiting on the harder rho+z-coupled geometry.
3. If (and only if) a bare/simple winding turns out to be dynamically
   stable, THEN the coupled rho+z torus-knot becomes worth building --
   as a refinement toward the real confinement mechanism, not as the
   first thing tested.

### Files

`prototype_torus_knot_winding.py` (closure + circulation test, R_DOMAIN=40,
~348k leaves, runs in well under a minute for all four m values) -- not a
production script, kept here as the empirical basis for the above, not
meant to be built on directly without the rho+z extension noted above.

## CORRECTION 2026-09-22, LATER -- the prototype above used the wrong ring
regime; redone on `mode="octagon"`, same qualitative conclusion holds

Philip caught this directly: the prototype above spliced its z-excursion
into the raw minimal 8-tile SQUARE pattern (`_TOROIDAL_PATTERNS['z']`), not
`find_toroidal_ring(mode="octagon", ...)`. That's exactly the regime this
project's own code already warns against --
`init_electron_dipole_ic`'s own docstring: "scale below ~3 cells approaches
the 8-tile regime where the BCC k*L dispersion anisotropy (diamond
artifact) dominates" -- and is the specific reason `mode="octagon"`
(chamfered corners via diagonal hex-face hops, a rounder/more isotropic
loop) was built in the first place (2026-09-08, after `mode="square"`'s
literal-square shape was found to structurally prevent a smooth
axisymmetric near field at any hops/resolution).

**Redone properly** (`prototype_torus_knot_winding_octagon.py`): built the
real `find_toroidal_ring(axis="z", mode="octagon", hops=4)` base ring first
(64 tiles, `winding_number`=1.0000, confirmed closing on its own before any
modification), then spliced the SAME up-then-down z-excursion idea into the
octagon's own face sequence -- inserted at the leg diametrically opposite
the start/close point (bracketing the far straight leg with `[+z]*m` before
it and `[-z]*m` after), rather than at an arbitrary point. R_DOMAIN bumped
to 60 (1,175,219 leaves) to keep the larger ring comfortably inside the
domain with margin.

**Result: closes cleanly at every m tested (1, 3, 6), `winding_number`
exactly 1.0000 in every case** -- confirms the splice mechanics generalize
from the square case to a properly-resolved octagon ring, not just a
lucky/degenerate square-specific accident. Circulation with the same naive
constant-amplitude seeding: **107.79 (m=1) -> 107.64 (m=3) -> 107.22
(m=6)** -- real, large, weakly decreasing with hump height, same
qualitative shape as the square prototype's 13.55->12.90 (different
absolute scale, expected: this ring has 64+2m tiles vs. 8+2m, more
face-segments summing into the same circulation integral). **The core
finding from the square prototype is unchanged by fixing the ring
regime**: a topologically closed loop carrying a real current has real
macro circulation by construction, regardless of whether the base ring is
a small anisotropic square or a properly-resolved octagon -- this was never
an artifact of the square ring's poor resolution, so the correction this
memory and `doc/mems/project_electron_ic_topology_wrong.md` already made
(circulation-free is the wrong bar; dynamical erosion rate is the real
discriminator) stands as originally written, now on solid geometric
footing rather than a small-ring caveat hanging over it.

**What's still NOT built, same as before**: genuine coupled rho+z motion
(the real torus-knot target, visiting both inboard/outboard radii) --
this octagon version is still a "corrugated ring" (flat ring detouring out
of its own plane, visible directly in the meridional (rho,z) panels of
`torus_knot_prototype_path_octagon.png`: two vertical straight segments at
two closely-spaced rho values, not a smoothly varying rho as the path
progresses). The recommendation from the original scoping is unchanged:
don't build the full rho+z-coupled knot next; the dynamical erosion-rate
test (does an existing closed winding hold up under real dissipative
stepping) is still the load-bearing next step, answerable on constructions
that already exist.

**Files**: `prototype_torus_knot_winding_octagon.py`,
`torus_knot_prototype_path_octagon.png` (3D view + meridional projection,
m=1 and m=3 shown) -- kept alongside, not replacing, the original
square-ring prototype/figure (both remain on disk; the square one is now
annotated here as using the wrong regime, not deleted, so the correction
is visible in context).

## BUILT 2026-09-22, LATER -- a genuinely (p=1,q=1) rho+z-coupled closed
path (real construction, not another prototype scope), with an honest
caveat on the method used

Per Philip's stellarator framing, built `build_helical_torus_knot.py`: a
single closed path whose own geometry couples the major (toroidal) and
minor (poloidal) directions, unlike every earlier prototype here (flat
ring + z-only detour, zero rho variation).

**Method, and why it's not a literal continuum torus-knot parametrization**:
first attempt used each of the major square's 4 legs' own LOCAL "outward"
(away-from-z-axis) direction for the minor-loop radial hops (a rotating
frame, matching the continuum formula's spirit). Algebraic check (summing
`_FACE_3D_DIR` over the full face sequence before touching the lattice --
same practice as every prior prototype) caught this does NOT close: net
y-displacement -2R, because two of the four legs' local-outward directions
both happened to be -y (a real bug in reasoning about "outward," not a
lattice limitation) -- caught before wasting a lattice build on it, not
after. **Fixed** by using a FIXED global (x,z) plane for the minor diamond
loop instead (R hops tracing (+R,0)->(0,+R)->(-R,0)->(0,-R)->close,
independent of which major leg is current) -- this closes by the same
4-fold symmetry argument as the major square itself (verified algebraically
AND against the real lattice). Honest characterization: this is NOT a
literal (p,q) torus-knot parametrization -- the minor loop's own plane does
NOT co-rotate with the major tangent the way the continuum formula's does.
It still produces real coupled rho+z variation, just via a superposed
fixed-plane loop rather than a genuinely rotating local frame.

**Built and closed** (L=10 travel hops/leg, R=3 radial+z hops/leg,
R_DOMAIN=50, 680,211 leaves): closes cleanly, `winding_number`=1.0000
(independent confirmation beyond "the last hop returns to start" -- this
is the same arctan2-sum diagnostic used throughout today), `n_tiles`=64.
**Real rho+z coupling confirmed**: rho ranges (0.000, 17.166), z ranges
(-3.464, +3.464), and they're correlated (low-rho tiles cluster at negative
z, high-rho tiles at positive z) -- genuinely different from the flat
z-wobble prototypes (which had zero rho variation by construction).

**But the shape achieved is crude, not smooth** -- visible directly in
`helical_torus_knot_p1q1.png`'s meridional (rho,z) panel: two rectangular
"staples" side by side (one spanning rho~0-9 at z<0, another spanning
rho~11.5-17 at z>0), not a smooth diamond/circle. This is a direct
consequence of superposing a small FIXED-plane loop onto a coarse 4-leg
square (each major leg is a straight run, not a smooth arc, so combining it
with a fixed secondary loop produces angular, uneven rho excursions rather
than the smooth curve a true co-rotating frame would give) -- an honest
first result, not the polished stellarator-like geometry Philip is picturing.

`init_toroidal_winding` (unmodified -- it's generic, computes its own
plane/theta from position variance) seeded cleanly on this path: returned
`omega=0.098135` (matches the existing formula `2*sin(pi/64)` for a
64-tile ring, so the function's resonance estimate is self-consistent even
on this non-planar path), and `phi_ring_circulation=-0.0000` -- the
`cos(theta)` zero-circulation property held here too, on a genuinely
rho+z-varying path, not just a flat ring.

**Honest assessment of difficulty**: getting a genuinely CLOSING coupled
path took one real design iteration (the rotating-frame attempt failed
algebraically, caught before a lattice build) -- closure itself is not
fragile once the right invariant (net displacement per fixed direction pair
sums to zero) is respected. What's hard is getting a SMOOTH result: the
crude "two rectangular staples" shape shows the fixed-plane-diamond method
doesn't scale to something that actually looks like a wound coil without
real further design work -- likely needs the minor loop's radial direction
to genuinely rotate through more than 2 discrete orientations (e.g. an
octagon-style chamfer in the minor direction too, or interleaving the
radial/z hops evenly through each leg instead of blocking them at the leg's
end, both untried here) before extending toward higher q / multiple
parallel windings ("solid block" density) is worth attempting -- doing that
now would compound an already-crude base shape.

**Files**: `build_helical_torus_knot.py`, `helical_torus_knot_p1q1.png`.
