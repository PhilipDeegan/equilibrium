# Working principles

The stance this project works under when building and judging models. It is a working stance,
revised as the model changes; [dst.md](dst.md) holds the theory itself. Code and workflow
conventions are in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Goal: mechanism, not description

The aim is a plausible causal mechanism, not a model that reproduces known numbers.

- "We found a parameter that reproduces X" is descriptive. "X follows from the substrate's own
  rules with no free choice" is mechanistic. Say which one a result is.
- Derive parameters from the lattice's structure (the way isotropy forces the 14-face weights);
  don't pick them to fit.
- Method: guess a rule, compute its consequences, compare with observation. Approximate,
  calibrate, repeat.

## Model facts are not physical facts

- A result that is stable or exact inside the discrete model (a surviving fixed point, an
  identity checked to float64) is a fact about the model. It is not evidence that the
  mechanism is real. Keep the two registers separate when reporting.
- A field held fixed by construction (pinned uniform, externally imposed) to make something
  stable signals an isolated condition, not the real object. Holding parts constant "like a
  partial derivative" is a legitimate method; label what is held, and treat it as a control,
  not a result.
- Tests come in two kinds: "survives these conditions" (a stepping stone) and "survives real
  expectations" (the goal). Real expectations aren't known yet, so every result so far is
  conditional. State the conditions.
- A mechanism added by hand only reports its own behaviour. Scope it as a one-off check; don't
  turn it into an object of study or call it a body.
- Requirements on the rule (holds under context, deforms under pressure, translates) are
  properties of the whole model. Don't use them as tests to select between structures until the
  rule meets them.
- The model's own choices (`u` as a scalar, `phi` on 14 faces, the Kelvin-cell lattice) are
  hypotheses too. Scope conclusions to "within this construction"; a result that depends on the
  lattice orientation is a property of the approximation.

## Judge against phenomena

- Match against measured outcomes (photovoltaic conversion, Bell correlations, radiation
  pressure, spectra), not against a textbook narrative of how they happen. A mechanism can
  reproduce the phenomenon without reproducing the story.
- Observation is interpreted; it doesn't imply a cause. State the raw outcome, then the readings
  it allows.
- Textbook idealisations (black body, point dipole, smooth falloff) are signposts, not targets.
  A non-smooth result isn't wrong just because the idealised formula is smooth.
- Make no claims beyond what has been observed: a proton or electron is a "preferred state,
  observed persistent within human timescales", not something with a quoted lifetime.

## Discrete all the way down

- There is no continuum underneath the update rule. Don't diagnose a kernel by shrinking dt
  toward zero; the limit is the discrete tick.
- Prefer exact algebraic identities at the actual tick over limits, derivatives, or trig and
  exponential maps that assume continuous periodicity (e.g. a Cayley transform, not Rodrigues).
- Reproduce classical physics (mass, momentum, rotation, vectors, pi, derivatives) as a
  coarse-grained limit. Don't treat any of it as separately real at the substrate level. Noise,
  chaos, turbulence and eddy terms are names for what isn't resolved; don't add them as
  mechanisms.

## Resolution and references

- Structures at a few cells across measure lattice dispersion, not the rule. State the cells
  across a structure and the planned convergence check before running it. A spheromak needed
  ~220 cells across; an electron candidate below radius 20 cells is invalid, and R20 is only a
  screening scale.
- A too-small domain can invalidate a result. Go larger when in doubt, and state the box/body
  ratio.
- Compare against the continuum answer for the source actually seeded (a gaussian tube, not a
  filament formula) and require convergence toward it before calling a gap a model deficit.

## Vocabulary

Words carry ontology. Use these:

| avoid | use |
|---|---|
| proton, neutron (as primitives) | open knot, closed knot; knot/moat pair |
| mass | translation cost, omnibalance |
| field, "its own field" | the substrate; a disturbance and its effect radius |
| time (for step counts) | rate of change, ticks |
| information | state |
| fluid (for the substrate) | substrate; MHD is a fluid description on top of it |
| wave, reflection (as objects) | aggregate flux pattern; face-flux mismatch |
| 2D system | planar (confined in 3D) |
| matter vs EM sector | the one rule at core amplitudes vs its linear limit |
| dimension (as physical) | state counts and relations |
| E, B, Poynting (as mechanism) | the face, loop and hop relations they describe |
| lam as a substance | depletion; flux unavailable for processing |
