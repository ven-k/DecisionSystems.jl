# [ModelingToolkit models](@id mtk)

Loading ModelingToolkit activates a package extension that adds two methods:

```julia
DecisionMDP(sys::System; inputs, dt_decision, alg, reward_fn,
            actions, discount, initialstate,
            observed = [], hidden = [], op = [], terminal_fn = s -> false,
            integrator_kwargs = (;), validate = true, pool = nothing,
            prob_kwargs...)

DecisionPOMDP(sys::System; observe_fn, <the same keywords>)
```

`sys` is a compiled `System` (`mtkcompile`), which is also what a Dyad-compiled component
is. The extension builds the prototype with
`ODEProblem(sys, op, (0.0, dt_decision); prob_kwargs...)` and compiles symbolic access
once, at construction, so transitions do not pay for symbolic lookups.

| keyword | meaning |
|---|---|
| `inputs` | the parameter(s) the action writes, through a compiled `setp`. One parameter: the action is a scalar. Several: the action is a vector in the same order. |
| `observed` | observed variables (eliminated by `mtkcompile`, still computable) appended to the state as read-only sensor values, through a compiled `getu`. |
| `hidden` | *parameters* of `sys` that are unknown per episode but constant within it; each becomes a state field written into the per-transition parameter copy before every interval (see [Hidden unknowns](@ref hidden-unknowns)). |
| `op` | the operating point for the prototype: parameter values and initial conditions, as `[sys.h => 5.0, sys.k => 0.5]`. |
| `prob_kwargs...` | forwarded to `ODEProblem`; this is where `build_initializeprob = false` goes (see below). |

The state is a `NamedTuple` laid out as `(unknowns…, hidden…, observed…)`: the names
of `unknowns(sys)` in compiled order, then the `hidden` names, then the `observed`
names; `initialstate` must sample that layout, e.g. `Deterministic((h = 5.0, sensor =
5.1))`. `reward_fn`, `observe_fn` and `terminal_fn` see it by name (`sp.h`). The
round-trip law uses the unknown fields only; hidden fields are written in and read back,
observed fields are recomputed.

## Loading both packages

- `solve` is exported by POMDPs and by ModelingToolkit (and OrdinaryDiffEq, via
  SciMLBase). Qualify: `POMDPs.solve(solver, mdp)`, `SciMLBase.solve(prob, alg)`.
- ModelingToolkit 10 exported `transition` (its state-machine API), which collided with
  `POMDPs.transition`. ModelingToolkit 11 does not, and this package requires 11; the
  qualified import below is still the safest form and is what the package's own tests
  use.
- `@mtkmodel` is deprecated in ModelingToolkit 11 (moved to SciCompDSL.jl, no longer
  maintained by the core developers) and is not defined in recent 11.x releases; the
  examples here use the explicit `@variables` / `@parameters` / `System` / `mtkcompile`
  API, which exists on every 11.x.

## Example

The tank of the front page as a compiled system, with an observed sensor variable:

```julia
using DecisionSystems, OrdinaryDiffEq, POMDPs, POMDPTools
import ModelingToolkit as MTK
using ModelingToolkit: t_nounits as t, D_nounits as D

MTK.@variables h(t)=5.0 sensor(t)
MTK.@parameters inflow=1.0 k=0.5
sys = MTK.mtkcompile(MTK.System(
    [D(h) ~ inflow - k * sqrt(max(h, 0)), sensor ~ h + 0.1], t; name = :tank))

mdp = DecisionMDP(sys;
    inputs       = [sys.inflow],                  # the action sets the inflow
    dt_decision  = 1.0, alg = Tsit5(),
    reward_fn    = (s, a, sp) -> -abs(sp.h - 10.0),
    observed     = [sys.sensor],
    actions      = [0.0, 1.0, 2.0], discount = 0.95,
    initialstate = Deterministic((h = 5.0, sensor = 5.1)))

statetype(mdp)                                    # @NamedTuple{h::Float64, sensor::Float64}
sp = rand(transition(mdp, (h = 5.0, sensor = 5.1), 2.0))
sp.sensor ≈ sp.h + 0.1                            # true

pomdp = DecisionPOMDP(sys;
    inputs = [sys.inflow], dt_decision = 1.0, alg = Tsit5(),
    reward_fn    = (s, a, sp) -> -abs(sp.h - 10.0),
    observe_fn   = sp -> Deterministic(sp.sensor), # or any Distributions.jl distribution
    observed     = [sys.sensor],
    actions      = [0.0, 1.0], discount = 0.95,
    initialstate = Deterministic((h = 5.0, sensor = 5.1)))
```

Symbolic continuous events fire inside the interval like any other callback:
`System(eqs, t; continuous_events = [[h ~ 8.0] => [h ~ 4.0]], name = :drain)` empties
the tank to 4 whenever the level crosses 8, and a transition over a long interval shows
it.

## [The initialization trap](@id init-trap)

`ODEProblem(sys, op, tspan)` attaches an *initialization problem* to the prototype
whenever the system has defaults or initialization equations. That problem rederives
`u0` from `Initial(x)` parameters stored in `p`. The bridge remakes the prototype every
transition with the state as `u0` and a copy of the prototype's `p` — and the
initialization then overwrites that `u0` with the prototype's initial values. Measured
on ModelingToolkit 11.38.2 with the tank above (default `h = 5`): after
`remake(prob; u0 = [7.0], p = copy(prob.p))` and `init`, the integrator starts at
`h = 5.0`. Nothing warns, and the construction probe cannot catch it (the round-trip law
holds for whatever the integrator ended up with).

What the extension does: its transition path checks `has_initializeprob(prob.f)` and,
when the prototype carries one, remakes with `u0` only so ModelingToolkit rebuilds the
`Initial` parameters from the state you passed; the parameter object is still fresh per
transition. You do not have to do anything for the default prototype to be correct.

What you can do better: build the prototype with an explicit operating point and no
initialization problem,

```julia
mdp = DecisionMDP(sys; op = [sys.h => 5.0], build_initializeprob = false, ...)
```

which skips the initialization solve on every transition. The state carries every
unknown, so `u0` is always fully determined; the only requirement is that the states you
sample in `initialstate` are consistent for systems with algebraic constraints. The
warehouse model in this repository (`WareHouse/warehouse_robots/model.jl`) builds its
prototype this way. Whichever form you use, the one-line check is a transition from a
state that differs from the prototype's initial values:
`rand(transition(mdp, (h = 7.0, sensor = 7.1), a)).h` must depend on the `7.0`.

## [Hidden unknowns](@id hidden-unknowns)

A quantity that is unknown per episode but constant within it — the payload mass of a
robot, the outflow coefficient of a tank — is what a belief is *about*. It cannot simply
stay in `p`: parameters are frozen per transition and restored from the prototype, so
every belief particle would carry the same value. Two forms carry it in the state.

### `hidden =`: the parameter stays a parameter

```julia
MTK.@variables h(t)=5.0
MTK.@parameters inflow=1.0 k=0.5
sys = MTK.mtkcompile(MTK.System([D(h) ~ inflow - k * sqrt(max(h, 0))], t; name = :tank))

pomdp = DecisionPOMDP(sys;
    inputs = [sys.inflow], hidden = [sys.k],
    dt_decision = 1.0, alg = Tsit5(),
    reward_fn = (s, a, sp) -> -abs(sp.h - 4.0),
    observe_fn = sp -> Deterministic(sp.h),
    actions = [0.0, 1.0, 2.0], discount = 0.95,
    initialstate = Deterministic((h = 5.0, k = 0.5)))   # layout (unknowns…, hidden…)
```

The state is `(h, k)`. Before every interval the extension writes the particle's `k`
into the fresh parameter copy through a compiled `setp` (the `apply_state!` hook of the
[`PhysicsModel`](@ref physicsmodel)) and reads it back unchanged, so each particle and
each tree branch carries its own value while the solver integrates the ODE exactly as
written. Construction checks that every `hidden` symbol is a parameter (not an
unknown), is not also an input, and comes back unchanged from the probe transition
(an event that writes it is an error). Hidden values are written after initialization,
so a DAE whose consistent initialization depends on a hidden parameter needs
`build_initializeprob = false` or the second form.

### `der = 0`: the parameter becomes an unknown

The form that needs nothing from the bridge, and the one Dyad/Modelica users can apply
without it: an unknown with a zero rate.

```julia
MTK.@variables h(t)=5.0 k(t)=0.5          # k: hidden outflow coefficient, now a state
MTK.@parameters inflow=1.0 kdot=0.0       # kdot: zero-valued rate parameter, see below
sys = MTK.mtkcompile(MTK.System(
    [D(h) ~ inflow - k * sqrt(max(h, 0)), D(k) ~ kdot], t; name = :tank))
MTK.unknowns(sys)                          # [k(t), h(t)]: k is part of the state
# a particle filter varies k across particles and the same compiled model propagates
# state and belief — the plant doubles as the likelihood model
```

The rate parameter is not decoration. Recent structural simplification treats a literal
zero derivative as a constant and removes the variable from the state: with `D(k) ~ 0`
instead of `D(k) ~ kdot`, `unknowns(sys)` is `[h(t)]` on ModelingToolkit 11.38.2 (`k`
survives only as an observed variable), while 11.26.8 still keeps `k` as an unknown. The
rate-parameter form keeps `k` in the state on every 11.x. The Dyad source of the
warehouse robot (`WareHouse/dyad/WarehouseRobot.dyad`, component `Robot`) uses the same
form for the hidden payload mass: `der(m_payload) = mdot_payload` with
`mdot_payload = 0.0`.

A filtered-and-planned instance of the `der = 0` form is
[`examples/hidden_unknown_tank.jl`](https://github.com/ven-k/DecisionSystems.jl/blob/main/examples/hidden_unknown_tank.jl)
(a bootstrap particle filter and a one-block belief lookahead, both through the same
compiled model; the printed table shows the posterior of `k` contracting). Which form to
use: `hidden =` keeps the compiled system and its initialization untouched and costs one
`setp` call per transition; `der = 0` adds one unknown per hidden quantity and changes
the compiled system, but is portable to any tool that consumes the model.

## Pools with compiled models

`pool = n` works with `System`-backed problems: before every checkout the pool restores
the integrator's `MTKParameters` in full from the prototype, so the compiled `setp` write
of the action, or a parameter change made by an event, never leaks into the next
transition. The tests assert pooled and unpooled transitions agree bitwise, with events
included. Measured on the warehouse block: 2.6× per transition (28 → 11 µs). If a
parameter portion has a type the restore does not know, the first pooled transition
fails with a message naming it; build the model without `pool` then.

Performance, measured on the eight-unknown warehouse model (Tsit5, reltol 1e-6): one
half-second block costs 29.7 µs through the compiled accessors the extension installs,
against 346 µs for the naive `remake(p = [sym => v])` path (12×).

## [Hidden unknowns: `hidden=` vs `der = 0`](@id hidden-api)

The two forms of the [Hidden unknowns](@ref hidden-unknowns) section, side by side. The
same tank, carried both ways, gives the same `h` after a block (`rtol = 1e-8` in
`test/test_hidden_api.jl`), pooled or not, under `propagate` or `transition`.

| | `hidden = [sys.k]` | `der = 0` (`D(k) ~ kdot`, `kdot = 0`) |
|---|---|---|
| where `k` lives | a parameter of `sys`; a state field written into the transition's own parameter copy by `apply_state!`, after `init`/`reinit!` and before `apply_action!` | an unknown; part of `u0` like every other state |
| what the solver sees | the physical ODE as written: one unknown fewer per hidden quantity | one extra unknown and a zero-rate parameter per hidden quantity |
| model source | unchanged | one equation and one parameter per quantity |
| works without the bridge | no (the hook is the bridge's) | yes: pure Dyad/Modelica, any consumer of the model |
| initialization | written *after* initialization: a DAE whose consistent initialization depends on `k` needs `build_initializeprob = false` (and consistent sampled states), or the other form; for an ODE both prototypes agree, initializeprob attached or not (tested) | part of `u0`, so initialization sees it; the prototype still needs `build_initializeprob = false` to honour `u0` at all (see [the initialization trap](@ref init-trap)) |
| cost per transition | one compiled `setp` write (tens of nanoseconds, no allocation); on the tank example the two forms are within noise of each other (≈1.2 µs per pooled transition) | integrating the extra unknown |

Use `hidden=` when you build with the bridge; use `der = 0` when the quantity must be in
the state vector for other consumers of the model, or when a DAE's consistent
initialization depends on it. Two runnable twins differ only in their model block and
print the same posterior contraction of `k` (not bitwise: different compiled systems):
[`examples/hidden_unknown_tank_api.jl`](https://github.com/ven-k/DecisionSystems.jl/blob/main/examples/hidden_unknown_tank_api.jl)
(`hidden = [sys.k]`) and
[`examples/hidden_unknown_tank.jl`](https://github.com/ven-k/DecisionSystems.jl/blob/main/examples/hidden_unknown_tank.jl)
(`der = 0`). Outside ModelingToolkit the same hook is a `PhysicsModel` field:
`apply_state! = (integ, s) -> set_param!(integ, 2, s.k)` with a state `(u = …, k = …)` read
back by `state_from_integrator`; `test/test_apply_state.jl` is that raw-`ODEProblem`
version.
