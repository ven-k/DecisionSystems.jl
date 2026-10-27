# Examples

Three consumers of the package, smallest first. Each file's header comment gives its
own run command; all are run from the repo root.

- [`tank.jl`](tank.jl) -- the README toy: a one-state tank as a raw `ODEProblem`,
  wrapped as a `DecisionMDP` and solved with MCTS-DPW (needs `MCTS` in the env).
- [`hidden_unknown_tank.jl`](hidden_unknown_tank.jl) -- the hidden-unknown modelling
  idiom on an MTK-compiled tank: the outflow coefficient is a per-episode hidden
  constant carried as a der = 0 unknown (zero-rate-parameter form), the prototype is
  built with `build_initializeprob = false`, and a bootstrap particle filter plus a
  one-block belief lookahead run through the same compiled model. Prints the
  k-posterior contracting; optional CSV argument.
- [`crack_inspection/`](crack_inspection/) -- the flagship hand-rolled-physics POMDP:
  Paris-law fatigue-crack growth with a within-block mission load profile and
  net-section feedback (an `ODEProblem` integrated by the real solver every duty
  block), unknown per-component material rate, `:run`/`:inspect`/`:repair` actions,
  rupture caught mid-block by a `ContinuousCallback`, and a belief-space planner
  benchmarked against fixed inspection schedules. See its README for the run
  commands and results.

The Dyad/ModelingToolkit flagship (warehouse robots at an occluded crossing) lives
outside `examples/` in `WareHouse/warehouse_robots/`.
