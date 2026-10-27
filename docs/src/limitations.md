# [When not to use this](@id limitations)

This package has a specific niche and these are outside it:

- **Fast control loops.** If decisions happen at kHz (robot locomotion, power
  electronics), you want a controller, not a planner; per-decision tree search cannot
  keep up.
- **Problems a well-tuned MPC already owns.** Smooth dynamics + quadratic cost + full
  observability = MPC's home turf. This package earns its keep when there are
  discrete/hybrid events, non-smooth objectives (windows, thresholds), genuine partial
  observability, or costly information.
- **Variable decision intervals.** `dt_decision` is constant. State-dependent epochs
  make the problem a semi-Markov decision process; POMDPs.jl solvers bake in a constant
  per-step discount and would silently mis-discount.
- **Reward that lives between epochs.** `reward_fn(s, a, sp)` sees endpoints. Integral
  (running) costs: add a quadrature state to the physics (`du[i] = L(u, p, t)`, reset in
  `apply_action!`) and read it in `reward_fn`; the ODE solver then computes your reward
  integral at solver accuracy. First-class support is planned; the pattern works today.
- **Offline solvers wanting explicit models.** SARSOP and value iteration need
  enumerable state spaces; this bridge is sampling-only (`ImplicitDistribution`, no
  `pdf` on transitions). Online sampling solvers (the MCTS family, POMCPOW) and RL are
  the target; discretization layers are future work.

## Known gaps

Features that are not in version 0.1.0, listed so nobody has to discover them by
experiment:

- **Transition noise.** A transition is a deterministic ODE solve; the only randomness
  is in `initialstate`, in `observe_fn`, and in the policy. Process noise today has to
  be carried in the state (for example a disturbance component drawn per episode);
  SDE support, with seeds threaded through the pool and the parallel helpers, is future
  work.
- **Hidden unknowns in raw `ODEProblem`s.** The ModelingToolkit constructors take a
  `hidden =` keyword; for a
  raw `ODEProblem` you write the `apply_state!` hook or the zero-rate component yourself
  (see [Hidden unknowns](@ref hidden-unknowns)).
- **Event-triggered decisions.** Decisions happen on a fixed grid. Variable epochs need
  semi-Markov discounting, which the POMDPs.jl solver contract does not provide.
- **Running reward.** The quadrature-state pattern above is documented, not an API.
- **Paired evaluation.** Separating the environment's and the policy's random streams
  for paired policy comparisons is left to the user; a harness is planned.
- **Belief tooling.** No particle-filter helpers ship with the package; the stock
  `ParticleFilters.BootstrapFilter` runs on a `DecisionPOMDP` unmodified.
- **Offline-solver path.** No discretization layer for SARSOP or value iteration.
