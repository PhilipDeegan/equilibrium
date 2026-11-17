# Contributing

How code, experiments and results are organised in this repo. The research stance behind the
work (what counts as evidence, which words to avoid) is in [doc/principles.md](doc/principles.md).

## Layout

| path | what |
|---|---|
| `equilibrium/` | the Python package; `eq` is a symlink to it |
| `inc/`, `src/` | C++ (lattice, substrate step); built into `equilibrium/*.so` by mkn |
| `tst/` | tests, run by phlop |
| `run/` | reusable run scripts |
| `exp/YYYY/MM/DD/` | dated experiment work: scripts, logs, plots, `status.md` |
| `vis/` | visualisation scripts and their notebooks |
| `doc/` | theory and status notes; `doc/dst/failures/`, `doc/dst/resolution_requirements.txt` |

## Build

[mkn](https://github.com/mkn/mkn) builds the C++ extensions. CI uses:

```sh
mkn clean build -dtKOqp mod,mod_lattice test -Wx res/mkn/mpi
```

Locally, a release build of both extensions with warnings as errors:

```sh
KLOG=1 mkn build -Oqp mod,mod_lattice -x mpi -g 0 -W 9
```

`-g 0` also defines `NDEBUG`, which disables `EQUILIBRIUM_DEBUG_DO` assertions. To keep
assertions with optimisation, drop `-g 0` and keep `-O`: `mkn build -KOqp mod,mod_lattice,test_lattice -x mpi -gW 9`.
Rebuild after every C++ change; a running Python process keeps the old `.so`.

## Tests

```sh
./sh/test.sh
```

- CI runners have 2 cores (maybe 1 logical) and run with `-c 2` and `EQ_TEST_N_THREADS=1`.
  Any thread or core count committed to the repo must be safe at 1 thread, or be opt-in.
- A test is documentation of a result. When an exploration in `exp/` confirms something durable
  (an identity, an exact zero, a bug in a diagnostic), move the reusable logic into
  `equilibrium/` and pin the result with a `tst/` test that asserts, not prints.
- When a test fails, first ask whether it should fail. If a simpler sibling case meets the
  strict bar, the bar is right: leave the test red. Don't mark it `expectedFailure`/skip or
  loosen its tolerance to whatever passes. Delete a test only if its premise is wrong.
- Don't bound a ratio whose denominator can legitimately reach zero; check the numerator and
  denominator (or their sources) separately.

## Code

- Comments state present-tense invariants and landmines ("must stay float64: X relies on exact
  cancellation"). Never narrate an edit or its history; that belongs in the commit or PR.
  Default to no comment.
- Add a third-party dependency only for something genuinely hard to own (a Krylov eigensolver,
  HDF5). mkn and mkn.avx are in-house and always fine.
- Python: create directories with `Path(p).mkdir(parents=True, exist_ok=True)`; launch scripts
  with `python3 -u` (`-Ou` with optimisation) so output streams.
- Lint: `./sh/lint.sh` (black, ruff, autoflake).

## Visualisation (`vis/`)

Logic lives in an importable module with plain functions (`run`, `figure`, `animation`) and a
`main(argv=None)` under `if __name__ == "__main__"`. A notebook next to it only imports and
calls those functions. Animations return `eq.plot.Animation`, whose `.show()` gives HTML in
Jupyter and `plt.show()` from a script. Sort scripts into topic directories, not catch-alls.

## Experiments and runs

- New work gets a dated directory, `exp/YYYY/MM/DD/HH_MM_<name>/`. Start a fresh one for a new
  day rather than growing an old one. Notes for a day go in a single `status.md`.
- Plots made on a cadence go in one directory per run, the run name split on `_` into nested
  directories (`.../N4/S1/r40/t01000.png`), with that run's logs and `rec.npz` alongside.
- Every plot or figure worth looking at goes in `exp/`, never a temp directory.
- Large checkpoints (HDF5 states, GB-scale) go outside the repo, e.g. `/bk/dst/<project>/`.
  Logs, `.npy` traces and small PNGs stay in `exp/`.
- When saving data for later plots, save the full state (`phi`, `u`, `lam`, `pos`), not only
  derived scalars.
- Name domains in cells, R/h, not physical R: R20 at h=0.125 is "R160". Keep h alongside.
- Long runs print early and often: a t=0 estimate of the measured quantity, then the running
  value against an `--expect` prediction, invariant drift, and wall time with ETA. State the
  expected value and the failure signature before launching.

## Recording results

- `doc/dst/failures/`: wrong hypotheses, category errors, mismatched mechanisms, bugs.
- `doc/dst/resolution_requirements.txt`: "this number isn't trustworthy below resolution X".
  If running bigger, finer or longer with nothing else changed would move the number, it is a
  resolution requirement, not a failure.
- Tag survival results as conditional (state what was held) or real; see
  [doc/principles.md](doc/principles.md).
- `doc/dst.md` and the other theory notes are preliminary and change often.
