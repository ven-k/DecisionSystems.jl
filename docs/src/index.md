# DecisionSystems.jl

Turn a SciML simulator into a decision problem. DecisionSystems.jl wraps an
`ODEProblem` — or a ModelingToolkit `System` — as a POMDPs.jl `MDP` or `POMDP`, so the
POMDPs.jl solver ecosystem (MCTS, MCTS-DPW, POMCPOW, particle filters, RL bridges) plans
directly against your physics. The ODE solver, with its adaptive stepping, events and
callbacks, does the work inside every decision interval.

## Statement of need

Sampling-based planners need a generative transition model: given a state and an
action, produce the next state. For physical systems that model usually exists already
as a validated simulation model, and the common practice is to hand-rewrite it into a
planner-shaped `step` function. The copy drifts from the original, cannot be set to an
arbitrary hypothesis state, and steps over events that happen between samples.

DecisionSystems.jl removes the rewrite. One constructor turns the simulation problem into
the transition kernel; every transition is stateless, so tree-search branches never
share integrator state or alias parameters; and a probe transition at construction
checks what otherwise fails silently inside a solver: the Markov round-trip law between
your state mapping and the integrator, the observation-distribution contract, and the
state and observation types.

The niche is deliberate: hidden physical unknowns, sparse decision epochs (seconds, not
milliseconds), discrete or hybrid actions, and events inside the interval. See
[When not to use this](@ref limitations) for what lies outside it.

## Installation

Julia 1.10 or newer. Until the package is registered in the General registry:

```julia
pkg> add https://github.com/ven-k/DecisionSystems.jl
```

After registration: `pkg> add DecisionSystems`. The test suite runs on both current
dependency stacks (SciMLBase 2 + OrdinaryDiffEq 6, and SciMLBase 3 + OrdinaryDiffEq 7).
ModelingToolkit 11 is optional; loading it activates the
[ModelingToolkit extension](@ref mtk).

## A tank, decided once per second

An `ODEProblem`, a solver, a decision interval, a way to apply an action, and a reward
are all a `DecisionMDP` needs:

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
policy = POMDPs.solve(DPWSolver(n_iterations=500, depth=20), mdp)
```

`solve` is exported by both POMDPs and OrdinaryDiffEq (via SciMLBase), so it has to be
qualified once both are loaded. Add a sensor model and it is a POMDP:

```julia
pomdp = DecisionPOMDP(; ...,                       # same physics keywords
    observe_fn = sp -> Normal(sp[1], 0.5))         # noisy level sensor (Distributions.jl)
# or action-dependent sensing:
#   observe_fn = (a, sp) -> a == :inspect ? Normal(sp[1], σ(sp)) : Deterministic(-1.0)
```

`observe_fn` returns a *distribution*, never a sample; see the
[observation contract](@ref observation-contract).

## Where next

- [Guide](@ref guide): the constructor, what a transition does, action helpers,
  continuous action spaces, the observation contract, discounting, the integrator pool,
  and the threaded `propagate` / `parallel_rollouts` helpers.
- [ModelingToolkit models](@ref mtk): wrapping a compiled `System`, the initialization
  trap, and the hidden-unknowns idiom.
- [When not to use this](@ref limitations): the boundary of the niche and the known gaps.
- [API](@ref api): the exported surface.
- Worked examples in the repository:
  [`examples/`](https://github.com/ven-k/DecisionSystems.jl/tree/main/examples)
  (the tank above, and a fatigue-crack inspection POMDP with belief-space planning).

## Status

Pre-registration, version 0.1.0. The source files carry no comments by design: the
reasoning lives in docstrings, in the tests (each file names the behaviour it protects),
and in these pages.
