# 2026-10-16 nonrad -- imposed neutral rotating cores at c (Philip: core = rotating + translating charge structure,
# wake = its field). Constraint: no tired light -> core must not radiate.

nr.py: hop / triangle DEC (vacuum), P 140, dt 0.25, T 40, gaussian sigma 3, rotation omega 0.5 in xy, v 1.
Core = electric dipole P (J = dP/dt, bound charge only) + alpha x toroidal dipole (M = alpha sig^2 curl(T g)), T
leads by psi. Strength ramped from 0 over t 5 (without the ramp the IC leaves a ghost compensating charge behind).
Metric: E_slab = energy within |z - z_s| < 6 (co-moving: at c the bound field is a thin pancake that keeps forming
sideways, so a sphere metric is wrong); radiation = growth of energy behind the slab per unit distance (t >= 20).

| core | dE_behind / distance | E_slab | ratio |
| non-rotating P (control) | 5.97 | 752 (still forming) | 0.8 % |
| rotating P | 12.5 | 136 | 9.2 % |
| rotating T only (alpha 1) | 274 | 1984 | 13.8 % |
| P + T scan: R = 12.5 + 8.6 x + 112 y + 274 (x^2 + y^2), x,y = alpha (cos, sin) psi; T coefficient matches T-only |
| P + T at predicted minimum alpha 0.205 psi 266 | 1.02 (pred 1.05) | 56 | 1.8 % (still falling: 1.4 -> 0.65 / unit late) |
Rotating P + T (toroidal a quarter turn behind, anapole-like) cuts the radiation of the rotating dipole by 92 %.
Not zero: one mixing number cannot cancel every on-shell harmonic component of a gaussian shape; a shaped core might.
Open: residual vs shape / sigma / omega; wake pitch + handedness of the bound field; then what holds such a core.

## Item 1: can the residual reach zero? -- yes: exact anapole P = dT/dt (--anapole)
On the light cone the radiating (transverse) current of T is -d2T/dt2, of P is dP/dt: P = dT/dt cancels it for ANY
shape (continuum exact). The scan's optimum (alpha 0.205, quarter turn) was the rotation half of dT/dt (predicted
1/(sig^2 omega) = 0.222); the missing half is the translation term -v dT/dz.
| core (T 30) | dE_behind / distance | E_slab | ratio | rho_rms (slab) |
| rotating P (dipole) | 14.8 | 138 | 10.7 % | 5.28 |
| anapole sig 3, omega 0.5 | 0.059 | 95.6 (steady) | 0.06 % | 3.22 |
| anapole sig 4.5, omega 0.333 | < 0 (behind energy falling: only the ramp transient leaving) | 285 (steady) | none detectable | 4.80 |
Residual at sig 3 is lattice light-cone error / transient; gone at sig 4.5. A rotating, translating neutral core at c
can be exactly nonradiating: no tired light.
BUT: the exact anapole's field is confined to the core (A = T inside, B = curl T): rho_rms ~1.07 sig, no field beyond
the core -> no extended wake. Extended bound field (no radiation) appears only from non-rotating structure (control
pancake, 0.8 % leak, keeps forming sideways at c). For Philip: wake extent vs nonradiation is a real tension.
