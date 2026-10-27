# DecisionSystems.jl

[![CI](https://github.com/ven-k/DecisionSystems.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ven-k/DecisionSystems.jl/actions/workflows/CI.yml)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ven-k.github.io/DecisionSystems.jl/stable/)
[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ven-k.github.io/DecisionSystems.jl/dev/)

Turn a SciML simulator into a decision problem. `DecisionSystems.jl` wraps an
`ODEProblem` as a POMDPs.jl `MDP` or `POMDP`, so the POMDPs.jl solver
ecosystem — MCTS, MCTS-DPW, POMCPOW, particle filters, RL bridges — plans
directly against your physics, with the ODE solver (adaptive stepping,
events/callbacks) doing the work inside every decision interval.

## Installation

Julia 1.10 or newer. Until the package is registered in the General registry:

```julia
pkg> add https://github.com/ven-k/DecisionSystems.jl
```

After registration: `pkg> add DecisionSystems`. Documentation:
[stable](https://ven-k.github.io/DecisionSystems.jl/stable/) ·
[dev](https://ven-k.github.io/DecisionSystems.jl/dev/). Worked examples live in
[`examples/`](examples/).

```julia
using DecisionSystems, OrdinaryDiffEq, POMDPs, POMDPTools

tank!(du, u, p, t) = (du[1] = p[1] - p[2] * sqrt(max(u[1], 0.0)))

mdp = DecisionMDP(
    prob          = ODEProblem(tank!, [5.0], (0.0, 1.0), [1.0, 0.5]),
    alg           = Tsit5(),
    dt_decision   = 1.0,                                  # decide once per second
    apply_action! = (integ, a) -> set_param!(integ, 1, a),
    reward_fn     = (s, a, sp) -> -abs(sp[1] - 10.0),
    actions       = [0.0, 0.5, 1.0, 1.5, 2.0],
    discount      = 0.95,
    initialstate  = Deterministic([5.0]),
)

using MCTS
policy = POMDPs.solve(DPWSolver(n_iterations=500, depth=20), mdp)   # qualified: OrdinaryDiffEq exports a `solve` too
```

Add a sensor model and it is a POMDP:

```julia
pomdp = DecisionPOMDP(; ...,                       # same physics keywords
    observe_fn = sp -> Normal(sp[1], 0.5))         # noisy level sensor
# or action-dependent sensing:
#   observe_fn = (a, sp) -> a == :inspect ? Normal(sp[1], σ(sp)) : Deterministic(-1.0)
```

## What you get

- **Validated construction.** One probe step runs your model at construction:
  argument checks, the `state_to_u0`/`state_from_integrator` round-trip law,
  state/observation type inference, and the observation-distribution contract
  all fail loudly here — not 40 frames inside a solver. (`validate=false` to
  skip.)
- **Tree-search-safe transitions.** Each `transition` is stateless: planners
  can branch counterfactuals freely; parameters never alias between branches.
- **Action vocabulary.** `set_param!`, `impulse!`, `set_state!` compose inside
  `apply_action!`; raw integrator access stays available.
- **Continuous action spaces.** `actions` is any object supporting
  `rand(rng, actions)` — pre-grid nothing; MCTS-DPW's progressive widening
  works natively. Collections remain the common case.
- **Hidden states and events.** Carry unknown physical parameters in the
  state (`du = 0` components) for belief-space planning, or, with
  ModelingToolkit, `hidden = [sys.k]` to carry a parameter per particle
  through the `apply_state!` hook with no `der = 0` equation; put safety
  limits in a `ContinuousCallback` with `terminate!` so violations inside a
  decision interval end the step at the crossing.
- **RL bridge.** POMDPTools' `MDPCommonRLEnv`/`POMDPCommonRLEnv` work
  unmodified (`initialobs` included).
- **`IntegratorPool`.** `pool=n` reuses integrators via `reinit!` — measured
  2.9× per transition on init-dominated models and 2.6× on the MTK-compiled
  warehouse block (28 → 11 µs, 14 kB → 1 kB per transition; the
  `MTKParameters` object is restored from the prototype on every checkout),
  bitwise-identical trajectories, safe under threaded search. (When the
  block's integration itself dominates, the pool does not help — measure
  yours.)

A complete worked example — hidden fatigue-crack growth, costly inspection,
belief-space planning beating tuned inspection schedules — lives in
[`examples/crack_inspection/`](examples/crack_inspection/).

## Discounting

Think in a continuous rate ρ per unit of physical time and set
`discount = exp(-ρ * dt_decision)`. This keeps reward scales comparable if
you later change `dt_decision`, and makes explicit that POMDPs.jl solvers
assume one constant per-step γ (which is why `dt_decision` is constant — see
below).

## When NOT to use this

Honesty section — this package has a specific niche and these are outside it:

- **Fast control loops.** If decisions happen at kHz (robot locomotion,
  power electronics), you want a controller, not a planner; per-decision tree
  search cannot keep up.
- **Problems a well-tuned MPC already owns.** Smooth dynamics + quadratic
  cost + full observability = MPC's home turf. This package earns its keep
  when there are discrete/hybrid events, non-smooth objectives (windows,
  thresholds), genuine partial observability, or costly information.
- **Variable decision intervals.** `dt_decision` is constant. State-dependent
  epochs make the problem a semi-Markov decision process; POMDPs.jl solvers
  bake in a constant per-step γ and would silently mis-discount.
- **Reward that lives between epochs.** `reward_fn(s, a, sp)` sees endpoints.
  Integral (running) costs: add a quadrature state to the physics
  (`du[i] = L(u, p, t)`, reset in `apply_action!`) and read it in
  `reward_fn` — the ODE solver then computes your reward integral at solver
  accuracy. First-class support is planned; the pattern works today.
- **Offline solvers wanting explicit models.** SARSOP/value iteration need
  enumerable state spaces; this bridge is sampling-only (`ImplicitDistribution`,
  no `pdf` on transitions). Online sampling solvers (MCTS family, POMCPOW) and
  RL are the target; discretization layers are future work.

## Observation contract (POMDPs)

`observe_fn` must return a *distribution* (something with `rand`, ideally
`pdf`), not a sampled value. Sampled-value observations have no honest
likelihood, which silently breaks POMCPOW, DESPOT-family solvers, and
particle-filter belief updates. Noiseless sensors are fine:
`sp -> Deterministic(g(sp))`. You will get a construction-time warning if
your observation distribution has no `pdf`.

## Status

Pre-registration. Core bridge, validated constructors, action helpers,
continuous action spaces, RL env compatibility, the integrator pool,
thread-parallel evaluation (`propagate`, `parallel_rollouts`), and the
ModelingToolkit extension — symbolic state/parameter access compiled at
construction, observed-variable sensors, events firing inside decision
intervals — are implemented and tested (250 tests, passing on both the
SciMLBase 2 and SciMLBase 3 stacks with MTK 11). The integrator pool
supports MTK models: the `MTKParameters` object is restored from the
prototype before every checkout (2.6× per transition on the warehouse
physics).
