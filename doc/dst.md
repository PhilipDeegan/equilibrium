# Discrete Substrate Theory (DST)

A first-principles emergent model of physics built on a Planck-scale Kelvin cell lattice.
All physics — mass, gravity, time dilation, charge, nuclear structure — emerges from one
scalar update rule on one fixed topology. No Hamiltonians, no imposed forces, no vectors.


  Discrete Substrate Theory: A Planck-Scale Lattice Framework for Emergent Gravity and Quantum Phenomena

  We present Discrete Substrate Theory (DST), a candidate unified framework in which both general relativity and quantum mechanics emerge as large-scale approximations of a single discrete substrate rule. Space is modelled as a
  tessellation of Kelvin cells — the Wigner–Seitz cells of a body-centred cubic lattice — at the Planck scale. Each cell carries a scalar local potential $u$ and a scalar face flux $\varphi$ on each of its fourteen faces — not an
  imposed vector field, but 14 independent scalars whose directed differences reproduce vector-like behaviour. The complete dynamics follows from a single discrete flux rule,

  $$\frac{\Delta u_i}{\Delta t} = \sum_{j=1}^{14} \sigma(\lambda_i) \left[A_{ij} \frac{u_j - u_i}{d_{ij}}\right],$$

  with no imposed Hamiltonian, gauge symmetry, or force law.

  Gravitational attraction emerges from substrate depletion gradients: massive objects are tension knots — localised regions of high $u$ — that create a surrounding deficit in the substrate potential. Neighbouring cells advect along this
  gradient, recovering the $1/r^2$ force law without a separate field equation. The speed of light and gravitational time dilation follow from the same substrate stiffness and lattice geometry.

  Quantum structure emerges from the discrete topology of the lattice. The electron is not an independent particle but a reciprocal moat — a deficit of $u$ surrounding the proton tension knot — analogous to a hole in a doped semiconductor.
  Particle creation and annihilation correspond to knot formation and dissolution within the substrate flux field. Quantisation arises from the topological constraints of the Kelvin cell network rather than from an imposed Hilbert space.

  The framework offers a natural reconciliation of the two regimes: gravity and quantum phenomena are not governed by separate laws but are complementary limiting descriptions of the same substrate dynamics. We present numerical results
  from BCC Kelvin lattice simulations confirming stable topological winding configurations, spontaneous moat formation at a characteristic radius, and long-range $u$-gradients consistent with the Coulomb field. These results motivate DST
  as a falsifiable substrate model from which both spacetime geometry and quantum discreteness emerge without additional postulates.



---

## Vocabulary

DST uses its own terms rather than borrowing from frameworks it supersedes.

| Conventional term | DST term | Symbol |
|---|---|---|
| Mass | Substrate Impedance | $\mathcal{Z}$ |
| Gravity | Resolution Gradient | $\nabla\lambda$ |
| Inertia | Handoff Latency | — |
| Energy | Differential Tension | $\Delta u$ |
| Electric charge | Dipole polarity | $\pm$ (knot vs moat) |
| Spin | Lattice chirality | — |
| Wavefunction | Resonant mode of $u$ | — |
| Eigenvector | Stable tension mode | — |
| Face flux | Directed scalar per face | $\varphi$ |

There are no *imposed* vector fields in the fundamental theory. $\varphi$ is 14 independent
scalars per cell, one per face — not a vector. Displacement, force, and field are all derived
descriptions built from these scalar $u$-gradients and per-face $\varphi$ values, not primitives
of the theory.

---

## The Lattice

Space is a tessellation of **Kelvin cells** (truncated octahedra, the Wigner–Seitz cell
of the BCC lattice). The **discrete floor** — the smallest possible cell, the irreducible
unit of the substrate — is normalised to 1. This is fundamental; $\ell_P$ is not. The
locally-measured Planck length $\ell_P$ is the void cell size in metres — an emergent
scale, $N_\text{SAGA}$ floor units across ($N_\text{SAGA} \approx 2880$, anchored to the
S2/Sgr A* clock deficit, still an open calibration). Each cell has **14 face-sharing neighbours**:

- 8 through **hexagonal faces** (nearest, $d_1 = 1$, weight $w_1 = 2/3$)
- 6 through **square faces** (next-nearest, $d_2 = 2/\sqrt{3}$, weight $w_2 = 1/3$)

The 8+6 asymmetry is not incidental — it is the geometric source of what conventional
physics calls spin-orbit coupling. At nuclear resolution (λ=1), this anisotropy splits
degenerate tension modes the same way the spin-orbit term does in the shell model, without
being imposed by hand.

---

## Cell Variables

| Variable | Type | Role |
|---|---|---|
| $u_i$ | scalar | Local potential / "essence" — the internal pressure of cell $i$ |
| $A_{ij}$ | scalar | Face area — acts as bandwidth; changes under deformation |
| $K$ | constant | Substrate stiffness / coupling: $K = \ell_P / (t_P \cdot Z_0)$ |
| $\lambda$ | discrete | Resolution index: 0 = vacuum, 1 = proton/electron knot, max = black hole |
| $L$ | scalar | Cell size in Planck lengths — encodes local time rate |

---

## The Update Rule (one tick)

$$\frac{\Delta u_i}{\Delta t} = \sum_{j=1}^{14} \sigma(\lambda_i) \left[ A_{ij} \frac{u_j - u_i}{d_{ij}} \right]$$

This is a discrete flux/diffusion rule. No Hamiltonian. No imposed potential.

### Gauge freedom — u₀ is not a parameter

The update rule couples only to differences $(u_j - u_i)$. Adding any constant to every
cell leaves the dynamics unchanged. The background tension $u_0$ is therefore pure gauge
— unobservable and unnecessary to specify. All physical content is in the gradients.

This holds exactly because $\lambda$ is determined by winding topology, not by absolute
$u$. No initial $u_0$ assumption is needed.

### Throughput function $\sigma(\lambda)$

$$\sigma(\lambda) = \frac{L}{1 + \lambda}$$

- Vacuum ($\lambda=0$): $\sigma = L$ — full rate, one Planck-time tick per Planck-length cell.
- Proton knot ($\lambda=1$): $\sigma = L/2$ — half rate. Local time runs at half-speed.
- Black hole ($\lambda = \lambda_\text{max}$): $\sigma = L/(1+\lambda_\text{max}) > 0$ — minimum rate.

**There is no zero.** A black hole is not frozen — it is ticking at the substrate's minimum
rate. The apparent freezing is a time-dilation effect observed from outside, not a property
of the interior. $\lambda_\text{max}$ is finite; the denominator is always finite.

### σ is relational — it sits at the face

Throughput belongs to the junction, not the cell: the face value $\sigma_f$ (from both
adjacent cells) multiplies both legs of the update ($u$ from face flux, $\varphi$ from the
$u$ difference). With face $\sigma$ the rule conserves $\sum u$ and
$H = \tfrac12\sum u^2 + \tfrac12\sum\varphi^2$ for any $\lambda$ distribution
(`tst/substrate/test_kernel_invariants.py`); cell $\sigma$ on one leg loses this. With
state-dependent $\sigma$ the explicit (symplectic Euler) step has an $O(dt)$ energy error;
implicit midpoint removes it (`equilibrium/substrate/self_lam.py`).

### A uniform λ is unobservable

A uniform additive floor $\varepsilon$ on $\lambda$ is an exact relabelling of time:
$(\varepsilon, \alpha, dt) \equiv (0, \alpha/(1+\varepsilon), dt/(1+\varepsilon))$ to round-off
(`exp/2026/09/30/electron_survival/`). Only $\lambda$ differences act — the $\lambda$ analogue of
the $u_0$ gauge freedom. Where the background is uniform at a structure's scale (empty space),
it cannot hold the structure; persistence there must be intrinsic.

### Circulation transport (Cartan form)

The $u/\varphi$ update alone freezes the divergence-free part of $\varphi$ in shape and place
for any fixed $\sigma$ (`exp/2026/10/01/poloidal_pulse`): every closed-loop sum of $\varphi$ is
time-invariant, only $u$ and the gradient part propagate. Self-transport of circulation is a
separate relation, carried by loop sums only: per Delaunay triangle
(12 per cell), $-i_v(da)$ moves edge content by the triangle's own circulation, no vectors
(`continuous.cartan_transport`, C++ `step_cartan`, tests `TestCartanTransport`). It is
antisymmetric per triangle (energy exact), zero on gradients (creates no circulation in
irrotational flow), Kelvin-exact, with $\kappa = 1$ (Euler's coefficient), and converges under
refinement. Consequence: under step() + Cartan circulation is a continuous label with no
preferred value or threshold, and in the continuum limit topology is frozen (Lie–Poisson) —
quantization, offload and reconnection must come from a further relation
(`exp/2026/10/{13,14,21}`).

---

## Gauss's Theorem on the Kelvin Cell

The update rule is the discrete divergence theorem.  The right-hand side sums
the tension gradient across all 14 Wigner–Seitz faces, weighted by face area
and divided by centre-to-centre distance:

$$\frac{\Delta u_i}{\Delta t} = \sum_{j=1}^{14} A_{ij} \frac{u_j - u_i}{d_{ij}}$$

In vector calculus this is $\nabla^2 u$.  At the substrate level it is more
primitive: a direct accounting of tension flow across the natural boundaries of
the cell, using the actual face areas of the 14-face Kelvin cell.

### Steady-state form — discrete Poisson equation

At equilibrium ($\Delta u_i / \Delta t = 0$) every free cell has zero net
tension flow across its 14 faces.  At a fixed-source cell (a knot), the
boundary condition is nonzero.  This is the discrete Poisson equation.  Its
far-field solution is $u(r) \propto 1/r$ — Coulomb's law, derived, not imposed.

### Charge = net outward flux across 14 faces

$$Q_i \equiv \sum_{j=1}^{14} A_{ij} \frac{\partial u}{\partial r}\bigg|_j$$

| Topology | $Q_i$ | Observable |
|---|---|---|
| Open knot (proton) | $\neq 0$ | Electric charge |
| Closed knot (neutron) | $= 0$ | Charge neutral |
| Free cell | $0$ (by equilibrium) | Vacuum |

Charge quantisation follows from topology: a knot either closes on itself or
it does not.  There is no continuous interpolation between zero and nonzero net
flux — the winding is an integer.

### The Kelvin cell as the natural Gaussian surface

In standard Gauss's law one integrates $\mathbf{E} \cdot d\mathbf{A}$ over any
closed surface enclosing the source.  In DST the natural closed surface is the
14-face Wigner–Seitz cell itself — it tiles space with no gaps and no overlaps,
so the discrete sum over 14 faces IS the Gaussian integral for that cell.

There is no concept of choosing a sphere or any other shape.  The surface is
fixed by the lattice geometry.  The tension replaces flux; the 14 faces replace
the continuous integral.

### Why 14 faces and these specific weights?

The weights are not free parameters: the Kelvin cell fixes them.  The flux
through a face goes as its area over the link length (the Hodge star of
discrete exterior calculus / finite volumes), which for the Kelvin cell is
exactly $3:1$ (hexagonal : square).  The update rule applies the per-face
weight $W$ on both legs (the divergence feeding $u$ and the gradient feeding
$\phi$), so $W_{hex} : W_{sq} = \sqrt3 : 1$ and the Laplacian coefficients are
$W^2 = 3:1$.  (Until 2026-09-28 the code had $W = 3:1$ on both legs, i.e. $9:1$.)

A different ratio, $w_1 = 2/3$, $w_2 = 1/3$ ($2:1$), is the unique
**maximally isotropic** one: its leading-order anisotropy error is
rotation-invariant, in both the scalar and the transverse (curl curl) sector.
The fundamental lattice does not take it -- the lattice is the physics, and
its residual grid-scale anisotropy at $3:1$ is a prediction, not an error.
The continuous model does (its per-leg $W = \sqrt2 : 1$): it only
approximates the lattice, and its Kelvin cells are there to avoid a preferred
axis.  See `exp/2026/09/28/hodge_star/` and `tst/substrate/test_hodge_star.py`.

The 8+6 split also encodes the two face types that appear at every scale — the
asymmetry is the geometric origin of what conventional physics calls spin-orbit
coupling (see Nuclear structure section).

---

## Emergent Physics

### Structures, reception and offload (working model, 2026-10)

Working statements, not settled results:

- A structure is a substrate depression: cells contract near it and stretch only where
  unforced. A closed structure with nothing outside its core is unobservable and cannot
  interact without contact; bodies interact where their disturbances overlap. Every real body
  sits inside larger-scale disturbances (Earth, Sun, galaxy), so an empty box is a limiting case.
- Energy tends to diffuse; a structure persists because its content blocks that (floor,
  topology, inverse cascade). Forming one needs activation energy, supplied by context.
  Single-cell events do not occur — the background suppresses them; quantized units are collective.
- Any emitter is a receiver; raw vacuum is not. Offload happens because a body's holding
  capacity is finite (lowest-energy preference); it can cycle between bodies, and the space
  between can hold up to twice the content. Radiation is contextual (Purcell,
  Wheeler–Feynman); offload is quantized and transactional, and the valve is not local — it is
  a property of aggregates.
- $\lambda$ is the rate-side effect of a structure; capacitance is also structure-caused but
  distinct, and reaches outward (Coulomb).
- Decay success criterion: after a decay event the state should show the products as separate
  identifiable structures that close the ledger, with nothing left as unattributed diffuse loss.
  Decay needs real offload channels (radiative, resistive, Landau damping) — never an added
  friction term; the reference ladder in `equilibrium/reference/spectral.py` is what DST's own
  dynamics must reproduce.

### Dipole structure — universal at every scale

Every stable configuration produces a knot/moat pair: a high-$u$ region (small $L$, high $\lambda$)
surrounded by a compensating low-$u$ moat (larger $L$, lower $\lambda$).
This structure is a product of the update rule, not an imposed symmetry.

| Scale | Knot | Moat |
|---|---|---|
| Nuclear | Proton ($\lambda=1$, high $u$) | Electron (deficit shell, $u<0$) |
| Atomic | Nucleus | Electron cloud |
| Stellar | Star | Surrounding space |
| Galactic | Galactic disk (compressed cells) | Surrounding void |
| Cluster | Galaxy cluster | Cosmic void |

This structure is scale-invariant. The same update rule, the same 14-neighbour topology,
different accounting ($L$, $\lambda$) per level.

### Particle topology — no quarks required

DST has two fundamental stable topologies for matter at $\lambda=1$.

Both the proton and the neutron are **collective resonant patterns** spanning an enormous
number of Planck cells (~$10^{58}$ at Planck scale). A single simulation cell is a
coarse-grained representative of that collective; any profile computed at simulation
scale is the envelope of the pattern, not individual lattice steps. One or two cells
can only ever be an approximation.

**Proton** — open knot. A high-$u$ region whose flux radiates outward through all
14 faces. The long-range $1/r$ field this creates is what is observed as electric charge.
The surrounding low-$u$ moat stabilises at the Bohr orbit. No internal quark structure is
needed or posited — the knot is the fundamental mode.

**Neutron** — toroidal knot (spheromak). A closed loop of circulating $u$-flux using the
minimal BCC triangular plaquette: **2 hexagonal faces + 1 square face**. There are 72
such triangular loops on the Kelvin lattice. The flux closes on itself, so no external
field escapes — the neutron is charge-neutral by topology, not by cancellation.

The neutron is heavier than the proton by 0.782 MeV/c² — the $\Delta u$ cost of maintaining
the closed winding against the tendency of the substrate to relax to the open (radiant) state.

**Beta decay** = the toroidal winding unravels. The moat re-expands to the Bohr orbit
(emitted as an electron). The antineutrino is the topological accounting entry for the
change in winding — it carries the $\Delta u$ needed to close the ledger.

**Bound neutron stability**: surrounding nucleons provide substrate pressure that holds the
winding shut. Remove that pressure (free neutron) and it unwinds in ~15 minutes.

**Winding chirality**: the circulation around the triangular plaquette can be clockwise or
counterclockwise. This two-state handedness is what is observed as the neutron's
"magnetic moment" (μ_n = −1.913 nuclear magnetons) — the circulating $u$-flux IS the
moment. No separate magnetic degree of freedom is needed.

### Mass / Substrate Impedance

Tension knots — aggregates of cells with high $u$ and small $L$. The proton is a knot of
~$10^{58}$ Planck cells, modelled in simulation as one or a few large cells with
correspondingly high $u$ and $\lambda=1$.

### Inertia and the Face/Cell Gate

**Two propagation modes exist on the Kelvin lattice, with different costs:**

1. **Face-hopping** — $\phi$ propagates along cell boundaries, transferring between
   adjacent cells through shared faces.  This is the natural, low-cost mode.  EM waves,
   gravitational waves, and $u$-field gradients all use this mode.  Speed: $c$.

2. **Through-cell crossing** — to translate a topological winding (a closed $\phi$ loop)
   one cell along an axis, the $\phi$ pattern must cross the interior of the leading cell:
   the coupling between the entry face and the directly opposite exit face does not exist
   in the DST update rule.  This is harder than face-hopping by construction.

This asymmetry — faces are natural couplings, cell interiors are not — is the physical
origin of inertia.  There is no stored transit field.  The crossing either happens or it
doesn't: it is a **gate condition** on the instantaneous $\phi$ state.

**Collective gate.**  A winding is a closed topological loop across multiple tiles.
Translating it one cell requires every tile in the loop to simultaneously complete its
cell crossing.  A partial crossing (some tiles but not all) breaks the loop topology and
is not a valid state — the winding either hops as a whole or stays put.

This makes translation a collective threshold event:

- **Below threshold** — the external $u$-gradient breaks omnibalance at each ring tile
  but the net $\phi_z$ is insufficient for a simultaneous collective crossing.  $\phi$
  oscillates, the winding deforms elastically, and springs back when the gradient weakens.
  No net translation.

- **Above threshold** — all tiles simultaneously meet the crossing condition.  The
  winding hops exactly one lattice step.  Translation is discrete, not continuous.

**Inertial mass from topology.**  The collective threshold scales with the number of
tiles in the winding:

$$m \propto N_\text{cells} \times (\text{per-cell crossing resistance})$$

The proton is heavy because ~$10^{58}$ sub-cells must all cross together.
The electron is light because its minimal toroidal winding spans 8 tiles.

**$E = mc^2$ from crossing rate.**  At velocity $v$ the winding hops $v/c$ cells per
tick.  At $v = c$ every tile is crossing every tick — the maximum rate.  The energy cost
of maintaining this rate across all $N_\text{cells}$ is the rest-mass energy.  Relativistic
mass increase at high $v$ follows from the same per-cell geometry, with no imposed postulate.

**No auxiliary field.**  Inertia is an emergent property of the winding topology and the
face/cell asymmetry in the DST update rule.  No $V_i$, no $J_\text{lam}$, no coupling
constant is added to the lattice.

### Gravity / Resolution Gradient

Cell-size equilibration pressure. Regions with small $L$ (high $\lambda$, near mass) have
a $u$-field gradient that draws adjacent cells toward smaller $L$. Objects are carried
toward higher-$\lambda$ (smaller-cell) regions. No separate force law is needed.

Equilibrium $u$-field outside a spherical mass: $u(r) = GM/(rc^2)$, giving
$$L(r) = L_\text{vacuum} \times \left(1 - \frac{GM}{rc^2}\right)$$

This is the first-order Schwarzschild result. GPS corrections, Pound–Rebka, and
Hafele–Keating all follow directly.

### Time dilation

Small cells ($\sigma = L/(1+\lambda)$ small) update slower — local time runs slow.
- Earth surface vs GPS orbit: $\Delta L / L \approx 5.3 \times 10^{-10}$ (= 45.9 μs/day)
- With $L_\text{vacuum} \approx 10^{30}\,\ell_P \approx 16\,\mu\text{m}$, this corresponds
  to a cell-size difference of $\approx 8.6$ fm between sea level and GPS orbit.

### Vacuum hierarchy

The "vacuum" is not uniform. The substrate is compressed by every gravitational source:

| Step | $\Delta L$ (for $L_\text{vacuum}=10^{30}\,\ell_P$) | Scale |
|---|---|---|
| Extragalactic → intragalactic (galaxy) | 8.7 pm | sub-atomic |
| Intragalactic → solar system (Sun at 1 AU) | 159.5 fm | ~190 proton radii |
| Solar system → Earth surface | 11.25 fm | ~13 proton radii |
| Sea level → GPS orbit | 8.55 fm | ~10 proton radii |

All Earth-based experiments measure relative to the intragalactic+solar background.

### Per-body throughput rates

Absolute $u = GM/(c^2 R)$ and throughput suppression $\Delta L = \ell_P \cdot u$ for solar-system
bodies plus the S2/Sgr A* anchor (same $N_\text{SAGA} \approx 2880$ calibration as above):

| Body | $u = GM/c^2R$ | Clock deficit (s/s) | $\Delta L$ (m) |
|---|---|---|---|
| Moon | $3.139\times10^{-11}$ | $3.631\times10^{-16}$ | $5.073\times10^{-46}$ |
| Mercury | $1.000\times10^{-10}$ | $1.157\times10^{-15}$ | $1.616\times10^{-45}$ |
| Mars | $1.400\times10^{-10}$ | $1.620\times10^{-15}$ | $2.263\times10^{-45}$ |
| Venus | $5.973\times10^{-10}$ | $6.912\times10^{-15}$ | $9.653\times10^{-45}$ |
| Earth | $6.960\times10^{-10}$ | $6.960\times10^{-10}$ | $1.125\times10^{-44}$ |
| Earth core | $1.043\times10^{-9}$ | $1.043\times10^{-9}$ | $1.686\times10^{-44}$ |
| Uranus | $2.542\times10^{-9}$ | $2.542\times10^{-9}$ | $4.108\times10^{-44}$ |
| Neptune | $3.088\times10^{-9}$ | $3.088\times10^{-9}$ | $4.992\times10^{-44}$ |
| Saturn | $7.247\times10^{-9}$ | $7.247\times10^{-9}$ | $1.171\times10^{-43}$ |
| Jupiter | $2.016\times10^{-8}$ | $2.016\times10^{-8}$ | $3.258\times10^{-43}$ |
| Sun | $2.123\times10^{-6}$ | $2.123\times10^{-6}$ | $3.431\times10^{-41}$ |
| S2 @ Sgr A* peri | $3.414\times10^{-4}$ | $3.414\times10^{-4}$ | $5.518\times10^{-39}$ |

For every solar-system body $L_\text{local} \approx \ell_P$ to any practical precision — $\Delta L$
is what distinguishes them. GPS anchor: the 45 μs/day orbit-surface clock gap gives
$\Delta L = 8.42\times10^{-45}$ m independently, confirming the formula without relying on the S2
calibration.

### Casimir effect — vacuum pressure from lattice dispersion

The exact Kelvin lattice dispersion relation,
$$\omega^2(K) = 8 W_\text{HEX}^2\bigl(1-\cos(hK_x)\cos(hK_y)\cos(hK_z)\bigr) + W_\text{SQ}^2\bigl(6-2(\cos(aK_x)+\cos(aK_y)+\cos(aK_z))\bigr),$$
with $h = a/2$, $W_\text{HEX} = \sqrt3$, $W_\text{SQ} = 1$, reproduces the Casimir effect between two
parallel plates with no imposed vacuum-energy postulate. Stacking the lattice along $z$ (layer spacing
$h$), fixing $u = 0$ on two layers at each plate (the square faces reach two layers) and summing
$\tfrac12\omega$ over the exact modes between them, the separation-dependent part of the energy per
area is $-\pi^2 c/(1440\,d^3)$ -- the scalar Dirichlet continuum value with $c = \sqrt{W_\text{HEX}^2 +
W_\text{SQ}^2}\,a$ the lattice wave speed -- to $2\times10^{-4}$, for any face weights. The $1/d^3$ term is set
by the long-wavelength end alone, so it carries no lattice-scale information: the lattice appears only
in the $d \sim a$ corrections ($1/d^4$ effective plate position, $1/d^5$), the surface term, and the
bulk zero-point energy density. (`equilibrium/substrate/casimir.py`, `tst/substrate/test_casimir.py`,
`exp/2026/09/29/casimir/status.md`; an earlier Euler--Maclaurin version took derivatives at the zone
edge and reported a spurious $0.72\times$.)

The same dispersion relation gives the point-scatterer (Casimir–Polder) analogue
(`casimir_polder.py`): two knots feel a $\propto 1/r^3$ attractive potential from the identical
vacuum mode structure — no separate force law, no separate vacuum-energy calculation.
**Unverified:** `casimir_polder.py` integrates only half the BCC zone, and the real
Casimir–Polder law is $1/r^6$–$1/r^7$; this paragraph stands until redone (see `dst_caveats.md`).

### Magnetism — $\varphi$ ring-mode cycling, not bulk rotation

A magnetic field is not sourced by a static high-$u$ shell, nor by bulk rotation of a knot's $u$
distribution — both give zero net chirality (opposite winding signs produce identical
deflections, confirmed by direct test). It is sourced by the knot's internal $\varphi$ ring modes
**cycling** within the static tension landscape: that cycling is a local $u$-translation, and the
outward-projected cycling wake is the magnetic dipole field.

Confirmed by direct simulation: seeding $\varphi$ as a vector-potential-like face pattern
($\varphi[i,f] \propto B\cdot(-y_\text{mid}\,dx + x_\text{mid}\,dy)$) and measuring drift under a
background field gives mirror-image trajectories for the two winding chiralities — a genuine
Lorentz signature, not a lattice artefact. The radial profile of the cycling ring's own $\varphi$
field scales as $\varphi_\text{rms} \propto r^{-2.14}$ (target $r^{-2}$ for a vector potential;
the $B$-field itself, $\propto \text{curl}\,\varphi$, would instead fall as $r^{-3}$) — the ring's
own $\varphi$ field is the vector potential, not $B$ itself.

**Venus/Earth anchor**: a body whose bulk rotation is too slow to keep its internal ring modes
coherently phase-aligned (Venus) radiates no net dipole wake; sustained bulk rotation (Earth)
keeps the modes aligned and sustains the dipole. No dynamo mechanism is imposed — the field is
emergent from ring-mode coherence alone.

### Earth's magnetic dipole from circulating dipole-bearing matter, not spin-alignment

This is **not** a claim that Earth is itself one giant toroidal+poloidal winding, structurally
identical to a proton at planetary $L$ — that would be a single-object scale-invariance claim,
not what is proposed here. It is also **not** a bar-magnet-style claim ("many small proton-scale
dipoles held in phase, summed like aligned atomic moments") — that analogy is ruled out by a
simple empirical fact: molten/liquid metal has essentially no net ferromagnetic ordering. Bulk
alignment of that kind needs a rigid lattice with exchange coupling between fixed neighbours;
a fluid has neither. Since Earth's field-generating region is a convecting, molten outer core,
"many little dipoles phase-locked in place" cannot be the mechanism regardless of rotation.

**The actual proposed mechanism has two independent rotational degrees of freedom, not one:**

1. **Primary/intrinsic** — each dipole-bearing knot's own ring-mode cycling (the "Magnetism"
   section above): spin-like, present whether or not the knot goes anywhere.
2. **Secondary/translational** — the bulk *translation* of dipole-bearing matter itself, organized
   into a large-scale circulating loop (convective circulation in a molten, rotating body). This
   is structurally a current loop: a loop of *translating* dipole-bearing matter, not a lattice of
   *aligned, stationary* dipoles. Crucially, a current loop does not require its charge/knot
   carriers to be internally ordered at all — disordered, molten material circulating in an
   organized loop still sources a dipole field. This is what makes the mechanism compatible with
   molten, non-ferromagnetic source material, unlike the ruled-out bar-magnet reading.

**Venus/Earth anchor, restated correctly**: the relevant rotation is not what phase-locks
individual dipoles — it is what organizes bulk convective motion into a coherent large-scale loop
(the DST analogue of Coriolis-organized helical convection in classical dynamo theory). Venus
rotates too slowly to organize its convection this way; Earth's rotation does. This ends up
**structurally equivalent to the classical convective (MHD) dynamo mechanism**, expressed in
DST's own primitives (knots, ring-mode cycling, $\varphi$) rather than Maxwell's equations and
Ohm's law — not a distinct alternative to it. That convergence is expected, not a weakness: DST
should reproduce known mechanisms in its own vocabulary before claiming to go beyond them.

**Open, not resolved**: Mercury's weak-but-real field despite slow rotation, and Mars's absent
present-day field, both plausibly reduce to "is there an active, organized circulating loop in a
still-molten region" — the same open question real planetary dynamo theory has for both bodies.
DST does not yet offer a criterion that resolves either case more sharply than the classical
account does; it has not been shown to diverge from it anywhere yet.

### Nuclear structure / Island of stability

Nuclear magic numbers (2, 8, 20, 28, 50, 82, 126) are stable resonant modes of the
discrete tension field at $\lambda=1$. Two **independent** populations fill these modes:

- **Protons** (open knots): fill spherical shell modes, feel long-range $u$-field
  repulsion from other protons.
- **Neutrons** (toroidal knots, spheromaks): fill toroidal shell modes, charge-neutral,
  no mutual $u$-repulsion, but still couple to the nuclear substrate.

Because the two topologies are different, their mode structures in the nuclear potential
are not identical — this is the source of the deviation from the simple harmonic oscillator
beyond magic number 20.

**Scalar harmonic oscillator** on the Kelvin lattice (proton-only, one sector):
- Closures at 1, 4, 10, 20 (scalar) → ×2 chirality: **2, 8, 20** ← first three nuclear magic numbers ✓

**Where the deviation begins** (magic 28 vs harmonic oscillator 40):
The conventional shell model adds a spin-orbit term $\ell \cdot s$ by hand. DST does not.
Instead, the splitting arises from the **topological interaction** between the toroidal
neutron field and the open-knot nuclear potential — the spheromak's axis couples to the
radial gradient, breaking the degeneracy of the $N=3$ harmonic shell. This is the DST
candidate for the levels that produce 28, 50, 82, 126.

This is a **falsifiable prediction**: compute the energy levels of a toroidal flux loop
(spheromak mode) in a spherical harmonic potential on the Kelvin lattice; if the
anisotropy of the 2-hex+1-sq topology correctly splits the $N=3$ shell to give 8 states
below the rest, magic number 28 emerges without any imposed coupling term.

**Island of stability** (Z≈114, N≈184): the simultaneous shell closure of both the
open (proton) and toroidal (neutron) populations. The specific closure numbers are
determined entirely by the Kelvin cell topology.

### Fission, fusion, and the Coulomb barrier

A nucleus is a multi-winding pattern in a stable omnibalanced state — the net phi
circulation across all 14 faces of every constituent cell sums to zero.  No translation,
no spontaneous reorganisation.  **Overcoming that omnibalance is what gives rise to
fission and fusion.**

**Fission**: a sufficiently large winding complex (heavy nucleus) can be driven by an
external phi imbalance (neutron impact, photon absorption) into a state where no single
omnibalanced topology exists for the combined pattern.  The winding splits into two
daughter patterns, each of which individually finds omnibalance.  The $u$ that was
maintaining the overlap between the two partial windings propagates away as photons and
as translational phi tilt of the daughters (kinetic drift).

**Fusion**: two proton windings approach carrying translational phi tilt (kinetic energy
in classical terms).  Their overlapping $u$ fields resist further compression —
same-sign windings push back.  This is the Coulomb barrier in DST: not an imposed force
law but the omnibalancing restoring force of each winding's own $u$ field acting on the
other's phi circulation.  If the incoming tilt is large enough to compress the shared
region past the point where the phi loops can topologically interlock, the combined
winding finds a new omnibalanced state at lower total $u$ than the two separate states.
The $u$ difference radiates.

**Binding energy** is therefore the $u$ difference between the combined omnibalanced
winding and the sum of its parts.  The alpha particle ($^4$He) is the first geometry
where 2 open-knot + 2 toroidal windings can close into an unusually tight omnibalance —
the DST explanation for its anomalously large binding energy per nucleon.

**The threshold is topological**: below it the winding absorbs the perturbation
elastically and returns to its prior omnibalanced state (scattering).  At threshold, a
new topological closure becomes accessible.  There is no need to import a separate
strong-force law.

### Black hole / galactic centre

The galactic centre is $\lambda_\text{max}$ — cells so compressed and so tightly coupled
that from outside, the update rate $\sigma(\lambda_\text{max})$ is negligibly slow. This
is the "sintered macro-cell" state: the cells have effectively merged into a single unit
whose internal dynamics are invisible to external observers on human timescales. The
substrate is still ticking; only the relative rate ratio makes it appear frozen. There is no
true singularity: the discrete floor imposes a minimum cell volume and $\lambda_\text{max}$ is
finite, so density stays finite. What GR reads as a curvature divergence is a saturated
substrate knot at the substrate's own hard limit, not an infinity.

### Chirality and galactic polarisation

The BCC Kelvin lattice has no intrinsic chirality preference — the topology is
$\mathbb{Z}_2$ symmetric.  A clockwise toroidal winding and a counterclockwise one are
equally valid solutions to the update rule.  The observed chirality of matter is not
written into the substrate; it is **imposed by the galactic background**.

The galactic rotation (Sgr A* spin, large-scale substrate flow) breaks the $\mathbb{Z}_2$
symmetry globally, polarising all stable winding closures in one direction across the
observable patch.  Our universe is the patch where the galactic substrate selected
one chirality for proton-topology windings and the opposite for electron-topology windings.
An antimatter-dominated region would be the substrate patch with the opposite polarisation.

**Consequences:**

- **Matter/antimatter asymmetry** is not a property of the particle interactions but an
  inherited background: the galactic substrate broke the symmetry during large-scale
  structure formation.  No separate baryogenesis mechanism is required.

- **CP violation** is a leakage of the galactic polarisation background into nuclear-scale
  winding interactions — not a fundamental asymmetry of the update rule itself.

- **Neutrino handedness** (left-handed only in the observed universe) follows the same
  logic: the galactic chirality selects which helicity of $\phi$-wave radiation couples
  coherently to winding transitions.

- **Lepton generations** (electron, muon, tau) are the same toroidal topology at
  increasing $\lambda$ excitation levels — same chirality, different winding density.
  The muon ($\lambda \approx 207\,\lambda_e$) and tau ($\lambda \approx 3477\,\lambda_e$)
  are unstable because their elevated $\lambda$ is not the topological ground state; they
  shed $\lambda$ as $\phi$ radiation (neutrinos) until reaching the stable electron level.
  The ratios 207 and 3477 are not expected to be clean integers — they are resonant
  $\lambda$ levels of the toroidal winding in a galactically polarised substrate, and
  galaxy motion breaks the symmetry that would otherwise produce clean ratios.

- **Decay rate = distance from nearest resonance.**  A winding whose $\lambda$ sits
  exactly at a BCC topological eigenstate achieves perfect omnibalance — no residual
  $\phi$ leakage, indefinite lifetime.  A winding at an off-resonance $\lambda$ cannot
  close perfectly; a small persistent $\phi$ imbalance bleeds out as radiation
  (neutrinos) tick by tick until the winding sheds enough $\lambda$ to reach the next
  lower resonance.  The further from resonance, the faster the decay.  This gives a
  unified stability table:

  | Particle | $\lambda$ level | Status | Decays to |
  |---|---|---|---|
  | Electron | clean resonance | stable | — |
  | Muon | $207\,\lambda_e$ (off-resonance) | unstable, ~2.2 μs | electron + neutrinos |
  | Tau | $3477\,\lambda_e$ (further off-resonance) | unstable, ~0.29 ps | muon/electron + neutrinos |
  | Proton | clean resonance (open-knot topology) | stable | — |
  | Neutron | slightly above proton $\lambda$ | unstable, ~10 min | proton + electron + neutrino |

  Tau decays ${\sim}7600\times$ faster than the muon because it lies further from any
  clean BCC closure.  The lifetime ratio is a falsifiable prediction: compute the
  resonant $\lambda$ spectrum of the toroidal winding on the BCC lattice and verify
  that 207 and 3477 fall in the gaps, with lifetimes scaling as their distance from
  the nearest eigenstate.

- **Simulation convention**: the choice of which amplitude sign in
  `init_toroidal_winding` is "proton" vs "electron" is not free — it reflects the
  galactic polarisation of the substrate patch being simulated.

---

## Multi-resolution simulation

The same update rule applies at all scales. A simulation cell does not have to be
Planck-scale:

- **Cell size $L$** (in Planck lengths) sets the local accounting: $d_{ij} \to L$,
  $A_{ij} \to L^2 A_0$, throughput $\sigma \to L/(1+\lambda)$.
- A cell in deep space may have $L \sim 10^{30}\,\ell_P \approx 16\,\mu\text{m}$.
- A proton cell has $L \sim 5 \times 10^{19}\,\ell_P \approx 1\,\text{fm}$.
- The **Kelvin topology (14 neighbours) never changes** — only the accounting per cell.
- **Consistency check**: resonant mode radii in cell-units must be invariant under
  changes in $L$.

---

## Physical constants in Planck units

| Constant | Value | Role |
|---|---|---|
| $m_e / m_P$ | $4.185 \times 10^{-23}$ | Electron mass |
| $\alpha$ | $7.297 \times 10^{-3}$ | Fine structure constant |
| $a_0 = 1/(m_e \alpha)$ | $\approx 3.28 \times 10^{24}\,\ell_P$ | Bohr radius |
| $K = \ell_P / (t_P Z_0)$ | 1 (normalised) | Substrate stiffness |
| $Z_0 \approx 377\,\Omega$ | — | Substrate impedance of free space |

### Parameters must come from relationships, not borrowed constants

DST substrate parameters ($\alpha$, $\kappa$, $\sigma$, …) must be derivable from
lattice geometry and internal relationships alone.  If a formula requires $G$, $c$, or
$\hbar$ as *inputs*, it is an approximation or a consistency check — not a
first-principles derivation.  $G$, $c$, and $\hbar$ are *outputs* of the substrate: they
emerge from the Kelvin geometry and the update rule coefficients, and can be verified
against measurement once the lattice is fully specified.  Smuggling them back in as inputs
creates circular dependencies that obscure what the theory actually predicts.

**Example — the chi feedback coupling $\alpha$**: the formula
$\alpha = (G/c^2) \cdot m_p / (\ell_P \cdot W_\text{HEX})$ used in early simulations
is circular (G and c are both DST outputs).  The correct route is the relationship
$$\alpha \approx \frac{\chi_\text{eq}}{0.30 \times u_\text{local}}$$
where $\chi_\text{eq}$ is the equilibrium chi at the ring tiles (set by the SR kinematic
bound $\chi \lesssim 10^{-8}$) and $u_\text{local}$ is the local Coulomb field amplitude
— both measurable within the simulation.  This gives $\alpha \approx 2.6 \times 10^{-7}$
with no reference to $G$ or $c$.  The factor 0.30 is a lattice geometry constant
derivable in principle from $W_\text{HEX}$, $W_\text{SQ}$, and the face structure.

---

## Quantum correlations — entanglement, Bell, decoherence

### The canal

Two open knots (protons, or any two entangled particles) radiate $u$ outward through
all 14 faces.  Where their fields overlap, the intermediate substrate cells carry
elevated $u$ from *both* sources simultaneously.  This elevated corridor is the
**canal** — it is not a third entity but the direct product of the two overlapping
cell-tension fields in the substrate.

The canal is maintained as long as both knots exist and their fields overlap.  Its
depth decays with separation as the overlap diminishes.  It is non-factorisable: the
joint $u$-field cannot be written as $u_A(\mathbf{r}) + u_B(\mathbf{r})$ — it carries
correlations that belong to neither knot alone.

### Entanglement

Two particles are *entangled* when they share a canal: a non-factorisable joint
$u$-field state in the substrate connecting them.  The canal is established during
close interaction (when their fields deeply overlap) and persists after separation.
No additional mechanism is required.

### Decoherence — regression to the mean

The DST update rule is fundamentally regression to the (weighted) mean of 14
neighbours.  Every tick drives each cell toward its local average.

- **Coherence** — a structured deviation from the mean maintained in the substrate
  (the canal).
- **Decoherence** — the update rule winning: the canal diffuses back to ambient.
  Each surrounding substrate cell that has no memory of the original interaction pulls
  the canal cells toward the environmental mean.
- **Decoherence rate** — set by $\sigma(\lambda)$ and the density of unstructured
  neighbouring cells (temperature ≈ amplitude of ambient $u$-fluctuations).

Macroscopic superpositions decohere instantaneously because they would require
maintaining a structured deviation across $\sim 10^{58}$ Planck cells simultaneously
against continuous mean-regression.  A single canal between two protons has a narrow
cross-section; it survives.

### Polarisation — a polariser is a selective re-emitter

A polariser is a substrate structure that regresses one transverse $u$-component to
mean (absorbed) and re-emits the other as a new $u$-wave at the polariser's axis.
It does not merely filter — it **reconstructs** the canal in its own orientation.

This resolves the three-polariser paradox (0°→45°→90° transmits; 0°→90° does not):
the 45° polariser creates an intermediate canal bridging two otherwise-orthogonal
directions.  There is no paradox — only sequential canal reconstruction.

### DST is outside Bell's theorem — not a new way to violate it

Bell's derivation requires this factorisation:

$$P(A, B \mid \theta_A, \theta_B, \lambda) = P(A \mid \theta_A, \lambda)\; P(B \mid \theta_B, \lambda)$$

This factorisation is only possible if $\lambda$ can be assigned *separately* to each
particle — i.e. the two systems are independent subsystems that carry their own hidden
variables.  Bell never states this as an explicit assumption because in every theory
he was considering, particles are independent entities that separate at the source.

**In DST there are no particles.**

Both "particles" are patterns in the same $u$-field — the same substrate tension, the
same connected Kelvin lattice.  The hidden variable is not $\lambda_A$ for particle A
and $\lambda_B$ for particle B.  It is $u(\mathbf{r})$ everywhere — a single global
field that both measurement events interact with locally.

The factorisation fails not because of faster-than-light signalling and not because
of superdeterminism (the measurement choices are still free).  It fails because the
premise is false: there is no $\lambda$ that can be partitioned into one piece
belonging to A and another belonging to B.  The canal is not carried by either
pattern — it is the substrate itself.

Bell's inequality is a theorem about theories of *independent particles with separable
hidden variables*.  DST is not such a theory.  The theorem simply does not apply.

| Model | $C(\Delta\theta)$ | CHSH $S$ | Why |
|-------|-------------------|----------|-----|
| Classical LHV | $-\tfrac{1}{2}\cos(2\Delta\theta)$ | $\sqrt{2}$ | independent particles, separable λ |
| DST joint canal | $-\cos(2\Delta\theta)$ | $2\sqrt{2}$ | one substrate, non-factorisable field |

The $S$ values follow directly from the standard CHSH angles $a=0$, $a'=\pi/4$,
$b=\pi/8$, $b'=3\pi/8$: every setting pair has $|\Delta\theta| \in \{\pi/8,\, 3\pi/8\}$,
so $\cos(2\Delta\theta) = \pm\tfrac{\sqrt{2}}{2}$, with the $(a,b')$ term
entering the sum sign-flipped.  All four terms then add coherently:
$S = |C(a,b) - C(a,b') + C(a',b) + C(a',b')|
   = 4 \cdot \tfrac{\sqrt{2}}{2} \cdot A = 2\sqrt{2}\,A$
where $A$ is the correlation amplitude ($\tfrac{1}{2}$ or $1$ above).
Note $\sqrt{2}$ is the value of *this specific* classical model (stochastic
Malus-law response, which wastes correlation), not the LHV *bound*:
deterministic instruction-set models can reach the Bell bound $S = 2$, and no
separable-λ model can exceed it.  $2\sqrt{2}$ is the Tsirelson bound — the
joint-canal model saturates the quantum maximum.  Both values are verified
numerically in `bell.py` (`chsh()`, 200k samples).

**Bell violation = canal intact.  Inequality satisfied = canal decohered.**

The degree of violation measures how far the joint $u$-field deviates from a
factorisable (fully decohered) state.

### Bohmian analogy — structural, not ontological

The DST update rule is the discrete Laplacian of $u$:

$$\frac{\Delta u_i}{\Delta t} = K\sigma \sum_j \frac{A_{ij}}{d_{ij}}(u_j - u_i)$$

This is the same equation as the Bohmian quantum potential
$Q = -(\hbar^2/2m)\,\nabla^2 R / R$.
The mathematical structure is identical.

**But DST is not Bohmian mechanics.**  Bohm retains a dualism: real particles guided
by a real separate wave.  In DST there are no particles and no pilot wave.  Knots are
patterns in $u$; the canal is the non-factorisable joint $u$-field between two knots.
The Bohmian analogy is structural (same Laplacian dynamics) not ontological.
Do not reify the pilot wave — it is not a DST primitive.

The non-locality is in the *topology* of the pre-existing canal, not in any
measurement-time signal.  It was established causally, propagated at $c$, and is
simply being read at measurement time.

---

## Theoretical status — classical, quantum, or new?

**At the substrate level DST is deterministic and real-valued.**  The update rule
is a fixed scalar equation over a fixed topology.  There is no probability, no
complex amplitude, no wavefunction, no observer.  In that narrow sense it
resembles a classical field theory.

**Quantum phenomena are emergent, not axiomatic.**

| QM feature | DST origin |
|---|---|
| Discrete energy levels | Boundary conditions on resonant $u$-modes; no $\hbar$ needed |
| Quantised charge | Integer winding number — topology, not axiom |
| Wave-particle duality | $u$-field propagates as a wave; knots are localised and topologically stable |
| Uncertainty | Planck-scale cell size is the irreducible position–gradient resolution limit |
| Spin | Lattice chirality — two winding handednesses |
| Entanglement | Non-factorisable joint $u$-field (canal) between two knots |

**The quantum/classical boundary is emergent.**  At macroscopic scales the canal
decoheres (mean-regression wins) and classical behaviour appears.  At atomic
scales the canal survives and quantum behaviour appears.  Both regimes are
consequences of the same update rule at different scales.

**DST is not Bohmian mechanics.**  Bohmian theory has a dualism: real particles
guided by a real pilot wave.  DST has no particles and no separate guiding field.
Knots are patterns in $u$; the pilot-wave analogy is structural (both use a
Laplacian equation) not ontological.

**DST is not a quantisation of a classical theory.**  It does not start from
Lagrangians, path integrals, or canonical commutation relations.  The Schrödinger
equation is a continuum low-energy limit of the update rule, not its foundation.

### Modelling principles (working model, 2026-10)

- No time, only rate of change, set by local cell size. No $\pi$ and no periodic systems:
  $2\pi$ is pressure equalisation, repetition is a pattern of context.
- No vectors, spinors or dimensions as physical objects — only cell/face quantities and their
  relations. No complex numbers in the rule (fine in notation).
- No E/B/dipole fields: only substrate disturbances; deviation from the mean is the
  observation. Look for the face/loop/hop relation whose effects E and B describe.
- Mesh spacing outranks structure: a cell-set rate is physics, and its dependence on
  $a/\ell$ is a prediction. The true lattice is undefined (perhaps not Kelvin), so mechanisms
  must be lattice-agnostic (loops, adjacency, content, hand-off).
- Nature is not sharp: no infinities, no kinks, no noise (only context). A result that does
  not converge under refinement argues against the structure physically.
- One model, one rule ($E = mc^2$; no separate matter sector). Linearity is the
  about-the-average limit. Reversibility is not an axiom; locality means bounded speed.
- Mass, momentum, friction and loss must emerge from the rule, never be added.
- The scale gap is bridged by axiom, not by verification: from the floor cell (~1e-44 m) to any physical
  object is ~30 decades, so nothing physical (not even a photon) fits on a simulable discrete lattice. That
  structure-scale physics depends on the floor only through coarse coefficients is assumed. It can be made
  plausible (coarse rules that converge as L grows across the reachable ladder, analytic discrete results
  carried up as power laws) and tested only through its consequences at observable scale.

---

## Continuum limits — Schrödinger and Feynman as derived results

DST is the substrate.  Schrödinger and the Feynman path integral are not inputs —
they should emerge as limits.  Demonstrating this explicitly is a key validation.

### Schrödinger limit

The DST update rule in the continuum limit ($L \to 0$, $\sigma \to$ const) becomes:

$$\frac{\partial u}{\partial t} = K\sigma\,\nabla^2 u$$

This is the real diffusion equation.  Under Wick rotation ($t \to -i\tau$) it becomes:

$$-i\frac{\partial u}{\partial \tau} = K\sigma\,\nabla^2 u$$

which is the free Schrödinger equation with $\hbar/m \leftrightarrow 2K\sigma L^2$.
Adding a fixed-source knot introduces the Coulomb potential as the steady-state
$u$-profile around the knot — the full hydrogen Schrödinger equation emerges with
no free parameters beyond the identification of $K\sigma$ with $\hbar/2m$.

The Wick rotation is not a postulate — it is the statement that DST time is real
and discrete, and the imaginary-time (statistical mechanics) path integral is its
natural continuum image.

### Feynman path integral limit

The Feynman path integral sums $e^{iS/\hbar}$ over all paths.  The heat kernel
(continuum limit of DST) sums $e^{-S_E}$ over Brownian paths (Wiener measure).
Wick rotation connects the two: DST $\xrightarrow{L\to 0}$ heat kernel
$\xrightarrow{t\to -i\tau}$ Feynman propagator.

More concretely: a discrete random walk on the Kelvin BCC lattice, weighted by
face areas $A_{ij}$ and distances $d_{ij}$, defines a specific lattice
regularisation of the path integral.  In the limit $L \to 0$ the walk recovers
the Wiener measure; Wick-rotated it recovers the Feynman measure.  The 14-neighbour
Kelvin topology sets the lattice regularisation — it is not arbitrary, it is the
geometry dictated by the substrate.

The Feynman phase $e^{iS/\hbar}$ accumulated along a path is, in DST terms, the
cumulative tension gradient $\sum_{\text{path}} (u_j - u_i)$ along that path —
a direct count of the substrate stress traversed.  The interference of paths is
not a postulate; it is the substrate's response to overlapping $u$-gradients from
multiple propagation routes.  DST is therefore closer in spirit to the path
integral than to the wavefunction: both describe *all routes simultaneously*, and
the observable is their aggregate tension balance.

### Status of Schrödinger/Hamiltonian in the codebase

The Hamiltonian eigensolver (`bohr_kelvin.py`, `nuclear_kelvin.py`) is a
**verification tool**, not a DST primitive.  It confirms that the Kelvin lattice
geometry recovers the correct energy levels as $\delta \to 0$.  The correct
long-term DST test is to run the update rule with a proton knot, perturb the
$u$-field, and measure the resonant frequencies of the moat directly — no
Schrödinger equation imported.  The Hamiltonian approach is valid scaffolding
during development; it is not the final formalism.

**The honest category**: a deterministic emergent substrate theory.  Classical in
its mechanics (real scalar, fixed rule, no probability).  Quantum in its
observables (because knots, moats, and canals produce quantised, wave-like,
entangled behaviour at the appropriate scale).  The distinction between classical
and quantum is not fundamental in DST — it is a scale-dependent phenomenon.

---

## Problems and paradoxes addressed by DST

Ported and rewritten from the retired `doc/eq/dst_into.md` (2026-08-13) — its Bell's-theorem
and black-hole/void-limit paragraphs are dropped here as strictly superseded by this document's
own, more developed treatments (`Quantum correlations` above; `Black hole/galactic centre`
above). The items below have no counterpart elsewhere in this document.

### The N-body problem

Classical gravity has no closed-form solution for three or more bodies, usually treated as a
computational difficulty. The deeper issue is conceptual: instantaneous action at a distance
couples every body to every other simultaneously. DST has no action at a distance — each cell
responds only to its 14 face-neighbours, and the resolution gradient propagates causally at the
local throughput rate. The N-body system is a well-posed local evolution problem on the lattice:
computationally expensive, not fundamentally ill-defined.

### Galactic rotation curves

Observed rotation curves are flat at large radii; Newtonian gravity from visible mass predicts a
Keplerian decline, motivating dark matter halos. In DST the resolution-gradient field of a galaxy
is not the sum of point-knot $1/r$ contributions — flux stored in extended winding configurations
creates long-range gradients beyond point-mass accounting, so the effective field at large radius
draws on the galaxy's whole substrate depletion pattern. **Not yet quantitatively derived from the
lattice** — a falsifiable target (compute the winding-sourced far-field gradient for a realistic
galactic mass distribution and compare to observed flat curves), not a settled result.

### Olbers' paradox

In an infinite, eternal, static universe uniformly filled with stars, every line of sight should
terminate on a stellar surface and the night sky should be uniformly bright. It is not. The
substrate has finite total capacity — the global mean $u$ is bounded. A universe filled with
tension knots progressively depletes the background substrate; sufficiently depleted regions have
reduced local throughput, which suppresses effective emission rate directly, not merely dilutes
flux by distance. The sky is dark because distant substrate is depleted, not because the universe
is necessarily young or spatially finite.

### Bell's spaceship paradox

Two ships accelerate identically in special relativity, yet the string connecting them breaks
because their frames lose simultaneity. In DST the string is a chain of $\varphi$-linked cells;
acceleration is a resolution gradient imposed on those cells. The front and rear genuinely
experience different local $u$ environments during acceleration and evolve at different rates —
the string does not stay rigid. The break is a direct consequence of differential substrate
throughput along the object's length, not an artefact of how simultaneity is defined.

### The measurement problem

Quantum mechanics has no dynamical account of how superpositions become definite outcomes. DST
dissolves the problem rather than solving it: the lattice always has a definite, fully determined
microstate — there is no wavefunction and no superposition in the substrate itself. What the
wavefunction encodes is the observer's partial knowledge of substrate state, coarse-grained below
the cell scale. "Measurement" is an interaction that couples the apparatus to the substrate; the
outcome was already fixed by the microstate, and the interaction merely reveals it. Born-rule
probabilities emerge from the distribution of substrate microstates consistent with the observer's
coarse-grained context — the same move statistical mechanics makes deriving thermodynamic
probabilities from molecular ignorance, not from any fundamental randomness in the update rule.

### The ultraviolet catastrophe and renormalisation

Continuum QFT requires renormalisation to remove divergences from arbitrarily short-wavelength
modes. DST has a hard ultraviolet cutoff at the discrete floor — there is no sub-floor resolution,
so the spectrum of substrate excitations is finite by construction. Renormalisation-group flow in
the continuum limit corresponds to coarse-graining the discrete lattice, not to removing
unphysical infinities.

### The cosmological constant and void expansion

The vacuum energy density predicted by QFT is roughly 120 orders of magnitude above the observed
cosmological constant. DST has no vacuum energy in the QFT sense: the ground state is a uniform
$u$ field with no excitations, zero energy density by construction. The observed accelerating
expansion corresponds to the secular growth of void regions — the void limit of the update rule,
where throughput is unconstrained and undepleted substrate corresponds to an effectively
inflating spatial scale — as tension is progressively locked into stable knots elsewhere. The
"constant" is not constant; it tracks the evolving global substrate state.

### The Standard Model as a catalogue of substrate events

The Standard Model lists particles and couplings without explaining why they exist or why the
coupling constants take the values they do. In DST it is an effective macroscopic description of
substrate tension and diffusion events, not a fundamental ontology:

- **The Higgs mechanism** — the SM gives particles mass through coupling to the Higgs field. In
  DST, mass is substrate tension: the amplitude of a winding knot in $u$. No separate scalar field
  is required; a Higgs-like excitation, if it exists, would be a substrate breathing mode of the
  knot, not a mass-generation mechanism.
- **The vacuum as a quantum field** — the SM assigns nonzero vacuum energy through zero-point
  fluctuations of each field. DST's ground state is a uniform $u$ distribution with zero energy
  density — the same root fix as the cosmological constant problem above.
- **Fundamental point particles** — the SM treats electrons, quarks, and neutrinos as dimensionless
  points. In DST every stable entity has spatial extent set by its winding topology; point-particle
  self-energy divergences are artefacts of applying the continuum approximation beyond its domain
  of validity.

---

## Open questions

1. **What fixes $L_\text{vacuum}$?** The extragalactic cell size ($\sim 10^{30}\,\ell_P$)
   is not yet derived from first principles. It may emerge from the cosmological expansion
   history.

2. **Does the 8+6 anisotropy produce nuclear magic numbers 28, 50, 82, 126?**
   The scalar lattice gives 2, 8, 20. The higher magic numbers require the lattice
   chirality to act as spin-orbit coupling — testable by extending the tension-mode
   solver to use anisotropic face weights.

3. **What is $\lambda_\text{max}$?** Is it set by the Schwarzschild condition, by a
   discrete saturation of the face-area accounting, or by something else?

4. **What is the mechanism for cell subdivision?** When a cell promotes to higher $\lambda$,
   it borrows from surrounding $u$-gradients rather than from an absolute baseline — but
   the exact accounting for subdivision energy is not yet specified.
