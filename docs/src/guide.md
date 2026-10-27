# [Guide](@id guide)

## The constructor, two ways

A decision problem is a physics half and a decision half. The physics half is a
[`PhysicsModel`](@ref physicsmodel): the `ODEProblem` prototype, the solver, the decision
interval, how an action is written into the integrator, and the reward. The decision half
is the wrapper: the action set, the discount, the initial-state distribution, and
optionally a pool. Both forms below build the same object:

```julia
pm = PhysicsModel(
    prob          = ODEProblem(tank!, [5.0], (0.0, 1.0), [1.0, 0.5]),
    alg           = Tsit5(),
    dt_decision   = 1.0,
    apply_action! = (integ, a) -> set_param!(integ, 1, a),
    reward_fn     = (s, a, sp) -> -abs(sp[1] - 10.0))

mdp = DecisionMDP(model = pm, actions = [0.0, 1.0, 2.0], discount = 0.95,
                  initialstate = Deterministic([5.0]))

# flat form: PhysicsModel keywords and wrapper keywords in one call
mdp = DecisionMDP(prob = ..., alg = Tsit5(), dt_decision = 1.0, apply_action! = ...,
                  reward_fn = ..., actions = [0.0, 1.0, 2.0], discount = 0.95,
                  initialstate = Deterministic([5.0]))
```

Passing both `model` and physics keywords is an error. [`DecisionPOMDP`](@ref) takes
the same keywords plus `observe_fn`.

### What construction checks

With `validate = true` (the default) the wrapper runs **one real transition** from
`rand(initialstate)` with the first action (or a random one for continuous spaces) and
fails there, with an `ArgumentError`, instead of forty frames inside a solver:

- `dt_decision` is finite and strictly positive; `actions` is non-empty (or supports
  `rand`); `discount` is in `(0, 1]`; `initialstate` supports `rand(rng, d)`.
- The **round-trip law** `state_to_u0(state_from_integrator(integ)) ≈ integ.u`. This is
  a Markov check: if your state mapping drops a component the integrator carries (a
  controller's integral term is the classic miss), the planner solves a different problem
  than the one that runs.
- The state type returned by the transition equals the type `initialstate` samples;
  otherwise a warning, because solvers key on `statetype`.
- For a POMDP: `observe_fn` returns something `rand` can sample (error otherwise), and
  something with a `pdf` method (a warning otherwise, see the
  [observation contract](@ref observation-contract)).

`validate = false` skips the probe; the type parameters are then taken from
`rand(initialstate)` alone.

## [What a transition does](@id transition)

`rand(transition(m, s, a))` performs, in this order:

1. `remake(prob; u0 = state_to_u0(s), tspan = (t0, t0 + dt_decision), p = copy(prob.p))`
   where `t0` is the state's `t` field if it has one, else `0`. The prototype's
   parameters are **copied every time**, so an action that writes `integ.p` can never
   leak into another branch of a search tree or into the prototype.
2. `init(prob, alg; save_everystep = false, save_start = false, dense = false,
   integrator_kwargs...)`.
3. `apply_state!(integ, s)` (default no-op): writes state fields that are not part of
   `integ.u`, such as hidden constants carried as parameters, into the fresh integrator.
4. `apply_action!(integ, a)`, then `u_modified!(integ, true)` unconditionally, so the
   solver's cached first-stage derivative is recomputed even if your action wrote to
   `integ.u` or `integ.p` directly.
5. `auto_dt_reset!(integ)` unless `integrator_kwargs` pins `dt`, so the first step is
   sized from the post-action state (an impulse from `u = 1` to `u = 3` then lands within
   `1e-9` of a transition that starts at `3`, which the tests assert).
6. `step!(integ, dt_decision, true)`: the whole interval, with the solver's adaptive
   steps, events and callbacks.
7. `state_from_integrator(integ)` is the next state.

The transition is stateless and deterministic given `(s, a)`; the `rng` the
`ImplicitDistribution` receives is not consumed. Chained transitions reproduce a
monolithic solve over the same span to solver tolerance.

### States

The default state is `copy(integ.u)`, a `Vector`. A `NamedTuple` with a field `u` is
also understood by the default `state_to_u0`; add a `t` field to carry absolute time,
which non-autonomous models need so that the next interval resumes at the right time:

```julia
pm = PhysicsModel(prob = SEASONAL_PROB, alg = Tsit5(), dt_decision = 1.0,
    state_from_integrator = integ -> (u = copy(integ.u), t = integ.t),
    apply_action! = (integ, a) -> set_param!(integ, 1, a),
    reward_fn = (s, a, sp) -> -sp.u[1]^2)
mdp = DecisionMDP(model = pm, actions = [1.0], discount = 0.9,
                  initialstate = Deterministic((u = [1.0], t = 0.0)))
```

Any other layout works as long as `state_to_u0` and `state_from_integrator` are mutual
inverses; the probe enforces it. The [ModelingToolkit extension](@ref mtk) uses
NamedTuples keyed by unknown name.

## Actions

`apply_action!(integ, a)` receives the live integrator. Three helpers cover the common
cases and each flags `u_modified!` itself:

| helper | effect | typical use |
|---|---|---|
| [`set_param!`](@ref)`(integ, i, v)` | `integ.p[i] = v` | piecewise-constant control: thrust level, inflow rate, quota |
| [`impulse!`](@ref)`(integ, i, dv)` | `integ.u[i] += dv` | impulsive control: dose, injection, deposit |
| [`set_state!`](@ref)`(integ, i, v)` | `integ.u[i] = v` | resets and replacements: component swap, tank drain |

Raw writes (`integ.p[1] = a`) and any other integrator surgery remain available. Actions
need not be numbers: with `actions = [:low, :high]` the mapping lives in `apply_action!`
and `actiontype` becomes `Symbol`.

### Continuous action spaces

`actions` can be any object supporting `rand(rng, actions)`; nothing is pre-gridded,
and progressive-widening solvers (MCTS-DPW, POMCPOW) sample from it natively:

```julia
using Random

struct IntervalSpace
    lo::Float64
    hi::Float64
end
Base.rand(rng::Random.AbstractRNG, s::IntervalSpace) = s.lo + (s.hi - s.lo) * rand(rng)

mdp = DecisionMDP(model = pm, actions = IntervalSpace(0.0, 2.0), discount = 0.95,
                  initialstate = Deterministic([5.0]))
```

Enumerating solvers need a collection (`Vector` or `Tuple`), which stays the common
case. An object with no `rand` method is rejected at construction.

## [Observations](@id observation-contract)

`observe_fn` must return a *distribution* (something with `rand`, ideally `pdf`), not a
sampled value. Sampled-value observations have no honest likelihood, which silently
breaks POMCPOW, DESPOT-family solvers, and particle-filter belief updates. Noiseless
sensors are fine: `sp -> Deterministic(g(sp))`. You get a construction-time warning if
your observation distribution has no `pdf`.

Two call forms are supported and chosen by `hasmethod`:

- `observe_fn(sp)`: the sensor depends on the new state only.
- `observe_fn(a, sp)`: the sensor depends on the action taken (measure only when the
  action inspects). `initialobs` then has no 1-argument method to call; define
  `POMDPs.initialobs(::DecisionPOMDP, s)` for your model, or add a 1-argument method
  describing the pre-action observation.

## Rewards, terminal states, discounting

`reward_fn(s, a, sp)` sees the endpoints of the interval; `terminal_fn(s)` makes a state
absorbing (default: never). Put safety limits in a `ContinuousCallback` with `terminate!`
(via `integrator_kwargs = (callback = cb,)`) so a violation inside the interval ends the
step at the crossing, and let `terminal_fn` recognise the resulting state.

Think of the discount as a continuous rate ρ per unit of physical time and set
`discount = exp(-ρ * dt_decision)`. This keeps reward scales comparable if you later
change `dt_decision`, and makes explicit that POMDPs.jl solvers assume one constant
per-step discount, which is why `dt_decision` is constant (see
[When not to use this](@ref limitations)). Rewards that accrue *between* epochs are
integrated by the solver itself if you add a quadrature state (`du[i] = L(u, p, t)`,
reset in `apply_action!`) and read it in `reward_fn`.

## Hidden states and events

Carry unknown physical quantities in the state rather than in `p`: parameters are
frozen per transition, state components are what belief particles vary. Two ways: a
zero-derivative component (`du[i] = 0`), or a state field that `apply_state!` writes into
the fresh integrator's parameter copy before each interval, which keeps the ODE as
written. Belief-space planners and particle filters then propagate the hidden quantity
for free through the same transition. The ModelingToolkit page describes
[both forms for compiled models](@ref hidden-unknowns), where the second is the
`hidden =` keyword.

## Integrator pool

`init` dominates a short interval on a small model. `pool = n` (or `pool =
IntegratorPool(pm; size = n)`, which two wrappers may share) pre-builds `n` integrators
and routes every transition through `reinit!` on a checked-out one:

```julia
mdp = DecisionMDP(model = pm, actions = [0.0, 1.0, 2.0], discount = 0.95,
                  initialstate = Deterministic([5.0]), pool = Threads.nthreads())
```

Before each use the integrator's parameters are restored in full from the prototype
(plain vectors, and ModelingToolkit's `MTKParameters` via the extension), then
`reinit!(integ, u0; t0, tf, erase_sol = true, reset_dt = true)`, then the same
action/`u_modified!`/`dt`-reset/`step!` sequence as the unpooled path. The tests
assert that pooled and unpooled transitions agree **bitwise**, that concurrent
checkouts do not corrupt each other, and that parameter writes never leak between
checkouts.

Measured: 2.9× per transition on an init-dominated toy model, and 2.6× on the
MTK-compiled warehouse block (28 → 11 µs, 14 kB → 1 kB per transition). When the
interval's integration itself dominates, the pool does not help; measure yours. Restrictions: callbacks must be stateless (the integrator, and with it any
callback-held state, is reused); the pool does not support SDE noise or changes of state
dimension between transitions.

## Threaded evaluation

[`propagate`](@ref)`(m, states, a; rng)` moves a whole particle set one interval forward
under `a`, chunked over `Threads.nthreads()` tasks; with a pool each task holds one
integrator for its entire chunk. [`parallel_rollouts`](@ref)`(m, policy, s; n, depth,
rng)` runs `n` independent depth-`depth` rollouts from `s` and returns the `n` discounted
returns (average them for a leaf estimate); `policy` is `s -> a`, `(rng, s) -> a`, or a
`POMDPs.Policy`, and rollouts stop at terminal states. Both derive one child RNG per
particle or rollout from `rng`, so results are identical for any thread count. Start
Julia with `-t auto` (or `-t N`) to use them. Measured: 256 rollouts × depth 20 went
from 7.7 ms to 1.6 ms at 16 threads; on ~1 µs intervals the spawn overhead dominates.

## RL bridge

POMDPTools' `MDPCommonRLEnv(mdp)` and `POMDPCommonRLEnv(pomdp)` work unmodified
(`initialobs` is implemented), so CommonRL-based agents can train against the physics.
`NamedTuple` states and observations are flattened to vectors for array-typed envs and
rebuilt from them (`convert_s`, `convert_o`), nested tuples included.

## What the package keeps true

The invariants the implementation maintains, each one a trap that otherwise produces
a silently wrong answer:

1. **Round-trip law**, asserted at construction (the Markov check).
2. **Parameter isolation**: every transition gets a copy of `prob.p`; tree branches
   never alias.
3. **Hidden per-episode quantities travel in the state, not in `p`**, because `p` is
   frozen per transition (as zero-rate unknowns, or written back by `apply_state!`).
4. **`u_modified!` after every action**, so the solver's first-stage cache is refreshed.
5. **`auto_dt_reset!` after the action** unless you pinned `dt`.
6. **`observe_fn` returns a distribution**; construction warns when it has no `pdf`.
7. **`dt_decision` is constant**: POMDPs.jl solvers assume one discount per step.
8. **ModelingToolkit initialization is disabled on the prototype** where it would
   rederive `u0` (see [the initialization trap](@ref init-trap)).
9. **`POMDPs.solve` and `SciMLBase.solve` are both in scope**: qualify.
10. **Pooled integrators are reinitialised and their parameters restored** before every
    use; only stateless callbacks are safe in a pool.
