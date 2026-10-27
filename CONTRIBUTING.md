# Contributing

## Running the tests

The suite runs on two dependency stacks and CI checks both
(`.github/workflows/CI.yml`): SciMLBase 3 + OrdinaryDiffEq 7 + ModelingToolkit >= 11.27
(what a plain registry resolve gives today) and SciMLBase 2 + OrdinaryDiffEq 6 +
ModelingToolkit <= 11.26.

```julia
# plain stack: whatever the registry resolves under [compat]
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

```julia
# SciMLBase 2 stack: narrow the compat bound in a THROWAWAY copy of the checkout
# (Pkg.compat rewrites Project.toml; the bound is honoured by every resolve Pkg.test does)
julia --project=. -e 'using Pkg; Pkg.compat("SciMLBase", "2"); Pkg.resolve(); Pkg.test()'
```

Test files under `test/` are included from `test/runtests.jl` and share its fixtures
(`tank_pm`, `tank_mdp`, the `*_PROB` problems); a new file should state at the top what
it pins and why. Performance assertions stay out of `Pkg.test` (CI timing is unreliable);
put benchmarks next to the examples instead.

## Conventions

- `src/` and `ext/` are kept comment-free. Design rationale lives in
  docstrings, in the test files (each one names the behaviour it protects), and in the
  documentation (`docs/src/`). A pull request that adds comments to `src/` will be asked
  to move the text to one of those places.
- The ten invariants the package maintains (round-trip law, parameter isolation,
  `u_modified!` after every action, ...) are listed in the Guide. A change that touches
  one of them needs a test that would fail without the change.

## Issues and pull requests

- Bug reports: a minimal `ODEProblem` (or `System`) plus the wrapper call that misbehaves,
  the Julia version, and `Pkg.status()` for SciMLBase, OrdinaryDiffEq and ModelingToolkit.
  Those three versions decide which stack you are on.
- Pull requests: one topic per PR, tests included, suite green on both stacks in CI, and
  docs updated when the public surface changes. Please keep `Manifest.toml` out of the
  diff unless the change is about dependency versions.
- Keep the scope: the package is a thin bridge between SciML problems and POMDPs.jl, not
  a reinforcement-learning framework. Features that belong to a solver, a belief updater
  or a modelling language are better proposed there.
