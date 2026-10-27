# [API](@id api)

Everything below is exported. The `sys::System` methods added by the ModelingToolkit
extension are described in [ModelingToolkit models](@ref mtk); their docstrings live in
the extension module and are available in the REPL once ModelingToolkit is loaded.

## Wrappers

```@docs
DecisionMDP
DecisionPOMDP
```

### [`DecisionSystem`](@id decisionsystem)

```julia
const DecisionSystem = Union{DecisionMDP, DecisionPOMDP}
```

The union both wrappers belong to; `propagate` and `parallel_rollouts` accept it. Both
structs hold the same fields: `model::PhysicsModel`, `actions`, `discount_factor`,
`initial_state_dist`, and `pool` (`nothing` or an [`IntegratorPool`](@ref)).

### [`PhysicsModel`](@id physicsmodel)

```julia
PhysicsModel(; prob, alg, dt_decision, apply_action!, reward_fn,
               apply_state! = (integ, s) -> nothing,
               observe_fn = identity, terminal_fn = s -> false,
               state_from_integrator = integ -> copy(integ.u),
               state_to_u0 = <vector: s; NamedTuple: s.u>,
               integrator_kwargs = (;))
```

The physics half of a decision problem: everything a transition needs, nothing about
action spaces, discounting or initial states (those belong to the wrapper). Keyword
constructor; the fields are

| field | required | meaning |
|---|---|---|
| `prob` | yes | the `ODEProblem` prototype; `u0` and `tspan` are replaced per transition, `p` is copied per transition |
| `alg` | yes | the OrdinaryDiffEq algorithm |
| `dt_decision` | yes | model time between decisions; finite and strictly positive |
| `apply_action!` | yes | `(integ, a) -> nothing`; writes the action into the integrator before the interval is integrated |
| `apply_state!` | no | `(integ, s) -> nothing`; writes the parts of the state that do not live in `integ.u` (hidden constants carried as parameters) into the integrator, after `init`/`reinit!` and before `apply_action!`; default no-op. The ModelingToolkit `hidden =` keyword builds one for you |
| `reward_fn` | yes | `(s, a, sp) -> Real`; sees the endpoints of the interval |
| `observe_fn` | POMDP only | `sp -> distribution` or `(a, sp) -> distribution`; the default `identity` is rejected by `DecisionPOMDP` |
| `terminal_fn` | no | `s -> Bool`; default never terminal |
| `state_from_integrator` | no | `integ -> s`; default copies `integ.u` |
| `state_to_u0` | no | `s -> u0`; default passes a vector through, or reads field `u` of a NamedTuple |
| `integrator_kwargs` | no | NamedTuple forwarded to `init` (tolerances, `callback`, a pinned `dt`, ...) |

`state_to_u0` and `state_from_integrator` must be mutual inverses; the wrapper's probe
transition asserts `state_to_u0(state_from_integrator(integ)) ≈ integ.u` at
construction (see [What a transition does](@ref transition)).

## Action helpers

Each helper flags the integrator state as modified so the solver's first-stage
derivative cache is refreshed; a raw write such as `integ.p[1] = a` is also allowed
because the transition path flags `u_modified!` itself after `apply_action!` returns.

```@docs
set_param!
impulse!
set_state!
```

## Integrator pool and threaded evaluation

```@docs
IntegratorPool
propagate
parallel_rollouts
```

## POMDPs.jl interface

Methods implemented for every `DecisionSystem` (`m`), with the field or function each
one delegates to:

| method | behaviour |
|---|---|
| `transition(m, s, a)` | `ImplicitDistribution`; `rand` runs one decision interval (pooled when `m.pool !== nothing`). No `pdf`. |
| `actions(m)` | `m.actions`, state-independent |
| `reward(m, s, a, sp)` | `model.reward_fn(s, a, sp)` |
| `initialstate(m)` | the `initialstate` object passed at construction |
| `discount(m)` | the `discount` passed at construction |
| `isterminal(m, s)` | `model.terminal_fn(s)` |
| `observation(m, s, a, sp)` | `DecisionPOMDP` only: `observe_fn(a, sp)` if that method exists, else `observe_fn(sp)` |
| `initialobs(m, s)` | `DecisionPOMDP` only: `observe_fn(s)`; errors for an action-dependent sensor without a 1-argument method (define `POMDPs.initialobs` yourself) |
| `convert_s`, `convert_o` | for `NamedTuple` states and observations: flatten to a vector of numbers (nested tuples in order) and rebuild from one, so array-typed CommonRL envs work; a field whose length the type does not fix (an array) converts forward only, and the inverse raises an `ArgumentError` saying so |

`statetype`, `actiontype` and `obstype` are inferred from the probe transition at
construction, so solvers see concrete types (`Vector{Float64}`, a `NamedTuple` type, a
`Symbol` action type, ...).
