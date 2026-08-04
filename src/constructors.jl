function _validate_physics(model::PhysicsModel)
  (; dt_decision) = model
  (isfinite(dt_decision) && dt_decision > 0) ||
    throw(ArgumentError("dt_decision = $dt_decision. Pass a finite, strictly positive time between decision points."))
  return nothing
end

function _validate_decision(actions, discount)
  (actions isa Union{AbstractArray,Tuple}) && isempty(actions) &&
    throw(ArgumentError("`actions` is empty. Pass at least one action."))
  (0 < discount <= 1) ||
    throw(ArgumentError("discount = $discount. Pass a value in (0, 1]."))
  return nothing
end

_probe_action(actions::Union{AbstractArray,Tuple}) = first(actions)
function _probe_action(actions)
  try
    return rand(_probe_rng(), actions)
  catch
    throw(ArgumentError(
      "`actions` must be a non-empty collection or an action space supporting " *
      "rand(rng, actions); got $(typeof(actions)). Continuous spaces work with " *
      "progressive-widening solvers (MCTS-DPW); enumerating solvers need a collection."))
  end
end

_action_eltype(actions::Union{AbstractArray,Tuple}) = eltype(actions)
_action_eltype(actions) = typeof(_probe_action(actions))

_probe_rng() = Xoshiro(0x0DEC15105)

function _probe_initialstate(initialstate)
  try
    return rand(_probe_rng(), initialstate)
  catch
    throw(ArgumentError("`initialstate` must support rand(rng, d) (e.g. Deterministic([5.0])); got $(typeof(initialstate))."))
  end
end

function _probe_step(model::PhysicsModel, s0, a0)
  sp, integ = _simulate_step_core(model, s0, a0, nothing)
  u_rt = model.state_to_u0(sp)
  isapprox(u_rt, integ.u) || throw(ArgumentError(
    "state_to_u0(state_from_integrator(integ)) must reproduce integ.u, " *
    "got $u_rt vs $(integ.u). Check that the two mappings are mutual inverses."))
  typeof(sp) == typeof(s0) ||
    @warn "transition returns states of type $(typeof(sp)) but initialstate samples $(typeof(s0)); solvers may fail on the mismatch."
  return sp
end

_has_pdf(d, o) = try; pdf(d, o); true; catch; false; end

function _resolve_model(model, physics)
  model === nothing && return PhysicsModel(; physics...)
  isempty(physics) && return model
  throw(ArgumentError("pass either `model` or the PhysicsModel keywords ($(join(keys(physics), ", "))), not both."))
end

_resolve_pool(pool::Nothing, pm) = nothing
_resolve_pool(pool::Integer, pm) = IntegratorPool(pm; size=pool)
_resolve_pool(pool::IntegratorPool, pm) = pool

"""
    DecisionMDP(; model, actions, discount, initialstate, validate=true)
    DecisionMDP(; prob, alg, dt_decision, apply_action!, reward_fn, ...,
                  actions, discount, initialstate, validate=true)

Wrap a `PhysicsModel` as a `POMDPs.MDP` — either an existing `model`, or the
flat form where the `PhysicsModel` keywords are given directly. With
`validate=true` (default), runs one probe step at construction: argument
checks, the state round-trip law, and state-type inference all happen here
instead of inside a solver.

`actions` is a collection (`Vector`, `Tuple`) for discrete action sets, or any
object supporting `rand(rng, actions)` for continuous spaces (usable with
progressive-widening solvers such as MCTS-DPW).
"""
function DecisionMDP(; model=nothing, actions, discount, initialstate, validate=true,
                       pool=nothing, physics...)
  pm = _resolve_model(model, physics)
  s0 = _probe_initialstate(initialstate)
  sp = s0
  if validate
    _validate_physics(pm)
    _validate_decision(actions, discount)
    sp = _probe_step(pm, s0, _probe_action(actions))
  end
  pl = _resolve_pool(pool, pm)
  DecisionMDP{typeof(sp),typeof(pm),_action_eltype(actions),typeof(actions),
              typeof(discount),typeof(initialstate),typeof(pl)}(pm, actions, discount, initialstate, pl)
end

"""
    DecisionPOMDP(; model, actions, discount, initialstate, validate=true)
    DecisionPOMDP(; prob, alg, dt_decision, apply_action!, reward_fn, observe_fn, ...,
                    actions, discount, initialstate, validate=true)

Wrap a `PhysicsModel` as a `POMDPs.POMDP` — either an existing `model`, or the
flat form where the `PhysicsModel` keywords are given directly. Requires
`observe_fn` to return an observation *distribution* — `sp -> dist`, or the
2-arg form `(a, sp) -> dist` for sensors that depend on the action taken.
"""
function DecisionPOMDP(; model=nothing, actions, discount, initialstate, validate=true,
                         pool=nothing, physics...)
  pm = _resolve_model(model, physics)
  pm.observe_fn === identity && throw(ArgumentError(
    "DecisionPOMDP requires an observe_fn returning an observation distribution " *
    "(e.g. sp -> Normal(sp[1], 0.5)); the default `identity` is only valid for DecisionMDP."))
  s0 = _probe_initialstate(initialstate)
  sp = s0
  if validate
    _validate_physics(pm)
    _validate_decision(actions, discount)
    sp = _probe_step(pm, s0, _probe_action(actions))
  end
  A = _action_eltype(actions)
  obs_dist = hasmethod(pm.observe_fn, Tuple{A,typeof(sp)}) ?
             pm.observe_fn(_probe_action(actions), sp) : pm.observe_fn(sp)
  o = try
    rand(_probe_rng(), obs_dist)
  catch
    throw(ArgumentError("observe_fn must return a sampleable distribution; rand(rng, observe_fn(sp)) failed for $(typeof(obs_dist))."))
  end
  validate && !_has_pdf(obs_dist, o) &&
    @warn "observe_fn returns $(typeof(obs_dist)), which has no pdf method; POMCPOW-, DESPOT-family solvers and particle-filter belief updaters will not work with it." maxlog = 1
  pl = _resolve_pool(pool, pm)
  DecisionPOMDP{typeof(sp),typeof(o),typeof(pm),_action_eltype(actions),typeof(actions),
                typeof(discount),typeof(initialstate),typeof(pl)}(pm, actions, discount, initialstate, pl)
end
