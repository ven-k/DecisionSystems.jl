module DecisionSystemsModelingToolkitExt

using DecisionSystems
using DecisionSystems: _copy_params, _remake_step, _restore_params!, _default_apply_state,
                       _simulate_step_core, _probe_initialstate, _probe_action
using ModelingToolkit
using ModelingToolkit: System, MTKParameters, getu, setp, unknowns, parameters, getname,
                       is_parameter
using SciMLBase: ODEProblem, remake, u_modified!, has_initializeprob
import SciMLBase

DecisionSystems._copy_params(p::MTKParameters) = copy(p)

_unwrap_p(p) = p
if isdefined(SciMLBase, :DespecializedParameters)
  _unwrap_p(p::SciMLBase.DespecializedParameters) = getfield(p, :params)
end

function DecisionSystems._remake_step(prob, u0, tspan, p::MTKParameters)
  has_initializeprob(prob.f) || return remake(prob; u0=u0, tspan=tspan, p=copy(p))
  new = remake(prob; u0=u0, tspan=tspan)
  new.p === p ? remake(prob; u0=u0, tspan=tspan, p=copy(p)) : new
end

function _restore_buffer!(dst::AbstractArray, src::AbstractArray)
  dst === src && return nothing
  if isbitstype(eltype(src))
    isempty(src) || copyto!(dst, src)
  else
    for i in eachindex(dst, src)
      x = src[i]
      dst[i] = x isa AbstractArray ? copy(x) : x
    end
  end
  nothing
end
_restore_buffer!(dst::Union{Tuple,NamedTuple}, src::Union{Tuple,NamedTuple}) =
  (map(_restore_buffer!, dst, src); nothing)
_restore_buffer!(dst, src) = dst === src ? nothing :
  error("IntegratorPool cannot restore an MTKParameters portion of type $(typeof(dst)) " *
        "(ModelingToolkit $(pkgversion(ModelingToolkit))); construct the model without `pool`.")

@generated function _restore_fields!(p, p0)
  fieldcount(p) == fieldcount(p0) || return :(error(
    "IntegratorPool: integrator parameters $(typeof(p)) and template $(typeof(p0)) differ in layout."))
  Expr(:block, (:(_restore_buffer!(getfield(p, $i), getfield(p0, $i))) for i in 1:fieldcount(p))...,
       :(return nothing))
end

function DecisionSystems._restore_params!(integ, p0::MTKParameters)
  p = _unwrap_p(integ.p)
  p isa typeof(p0) && return _restore_fields!(p, p0)
  _restore_fields!(p, p0)
end

struct _StateReader{names,G}
  get_state::G
end
_StateReader(names::Tuple, get_state) = _StateReader{names,typeof(get_state)}(get_state)
(r::_StateReader{names})(integ) where {names} = NamedTuple{names}(Tuple(r.get_state(integ)))

struct _StateToU0{names} end
(::_StateToU0{names})(s::NamedTuple) where {names} = [values(NamedTuple{names}(s))...]
(::_StateToU0{names})(s) where {names} = throw(ArgumentError(
  "System-backed models take NamedTuple states keyed by unknown name $(names) " *
  "(then hidden and observed fields); got $(typeof(s))."))

struct _HiddenWriter{names,S}
  set_hidden::S
end
_HiddenWriter(names::Tuple, set_hidden) = _HiddenWriter{names,typeof(set_hidden)}(set_hidden)
(w::_HiddenWriter{names})(integ, s) where {names} =
  (w.set_hidden(integ, values(NamedTuple{names}(s))); nothing)

_symnames(xs) = Tuple(Symbol.(getname.(xs)))

_is_unknown(sys, x) = any(u -> isequal(u, x), unknowns(sys))

function _validate_hidden(sys, hidden, inputs, observed)
  for h in hidden
    is_parameter(sys, h) && continue
    _is_unknown(sys, h) && throw(ArgumentError(
      "hidden: `$h` is an unknown of `$(nameof(sys))`, so it is already part of the state; " *
      "`hidden` lists parameters of the system that vary per episode."))
    throw(ArgumentError(
      "hidden: `$h` is not a parameter of `$(nameof(sys))` (its parameters are " *
      "$(join(string.(parameters(sys)), ", ")))."))
  end
  for h in hidden, i in inputs
    isequal(h, i) && throw(ArgumentError(
      "`$h` is listed in both `hidden` and `inputs`; a hidden quantity is a per-episode " *
      "constant written from the state, an input is written by the action. Pick one."))
  end
  names = [_symnames(unknowns(sys))..., _symnames(hidden)..., _symnames(observed)...]
  dups = unique(filter(n -> count(==(n), names) > 1, names))
  isempty(dups) || throw(ArgumentError(
    "state field names must be unique across unknowns, `hidden` and `observed`; " *
    "$(join(dups, ", ")) repeat$(length(dups) == 1 ? "s" : "")."))
  return nothing
end

function _mtk_physics(sys::System; inputs, dt_decision, alg, reward_fn,
                      observed=[], hidden=[], op=[], terminal_fn=DecisionSystems._default_terminal,
                      observe_fn=identity, integrator_kwargs=(;), prob_kwargs...)
  hs = collect(hidden)
  os = collect(observed)
  _validate_hidden(sys, hs, inputs, os)
  prob = ODEProblem(sys, op, (0.0, dt_decision); prob_kwargs...)

  us = unknowns(sys)
  state_syms = (_symnames(us)..., _symnames(hs)..., _symnames(os)...)
  set_inputs = setp(prob, length(inputs) == 1 ? only(inputs) : collect(inputs))

  state_from_integrator = _StateReader(state_syms, getu(prob, (us..., hs..., os...)))
  state_to_u0 = _StateToU0{_symnames(us)}()
  apply_action! = (integ, a) -> (set_inputs(integ, a); u_modified!(integ, true); nothing)
  apply_state! = isempty(hs) ? _default_apply_state : _HiddenWriter(_symnames(hs), setp(prob, hs))

  PhysicsModel(; prob, alg, dt_decision, state_from_integrator, state_to_u0,
                 apply_action!, apply_state!, reward_fn, observe_fn, terminal_fn, integrator_kwargs)
end

function _check_hidden_fields(hidden, initialstate)
  s0 = _probe_initialstate(initialstate)
  for n in _symnames(hidden)
    hasproperty(s0, n) || throw(ArgumentError(
      "initialstate samples $(typeof(s0)), which has no field `$n`; with `hidden` the state " *
      "layout is (unknowns…, hidden…, observed…) and every sample must carry each hidden field."))
  end
  return nothing
end

function _probe_hidden(pm::PhysicsModel, hidden, initialstate, actions)
  s0 = _probe_initialstate(initialstate)
  sp, _ = _simulate_step_core(pm, s0, _probe_action(actions), nothing)
  for n in _symnames(hidden)
    v0, v1 = getproperty(s0, n), getproperty(sp, n)
    v1 == v0 || throw(ArgumentError(
      "hidden parameter `$n` was $v0 before the probe transition and read back as $v1 after " *
      "it. A hidden quantity must be a constant the model never writes: check events or " *
      "discrete updates that assign `$n`, or drop it from `hidden`."))
  end
  return nothing
end

"""
    DecisionMDP(sys::System; inputs, dt_decision, alg, reward_fn,
                actions, discount, initialstate, observed=[], hidden=[], op=[], ...)

Wrap an `@mtkcompile`-ed `System` as an MDP. `inputs` lists the parameter(s)
actions drive (single parameter: action is a scalar; several: a vector).
States are NamedTuples keyed by unknown name, in `unknowns(sys)` order, then
the `hidden` names, then the `observed` names: `(unknowns…, hidden…, observed…)`.

`hidden` lists *parameters* of `sys` that are unknown per episode but constant
within it — a payload mass, an outflow coefficient. Each one becomes a state
field: it is written into the integrator's (per-transition) parameter object
before every block through a compiled `setp`, and read back unchanged, so
belief particles and tree branches each carry their own value while the solver
integrates the physical ODE as written, with no `D(k) ~ 0` equation. A hidden
symbol must be a parameter (not an unknown), must not be an input, and must
come back unchanged from the construction probe; each violation is an
`ArgumentError`. Hidden values are written after initialization, so a DAE
whose consistent initialization depends on a hidden parameter must be built
with `build_initializeprob = false` (or keep the `der = 0` form).

`observed` variables are appended to the state as compiled read-only sensor
values. `op` is the operating point for the prototype; remaining keywords are
forwarded to `ODEProblem` (this is where `build_initializeprob = false` goes).
"""
function DecisionSystems.DecisionMDP(sys::System;
    inputs, dt_decision, alg, reward_fn, actions, discount, initialstate,
    validate=true, pool=nothing, hidden=[], kwargs...)
  pm = _mtk_physics(sys; inputs, dt_decision, alg, reward_fn, hidden, kwargs...)
  validate && !isempty(hidden) && _check_hidden_fields(collect(hidden), initialstate)
  m = DecisionMDP(; model=pm, actions, discount, initialstate, validate, pool)
  validate && !isempty(hidden) && _probe_hidden(pm, collect(hidden), initialstate, actions)
  m
end

"""
    DecisionPOMDP(sys::System; observe_fn, hidden=[], ...)

POMDP variant: `observe_fn(sp)` (or `(a, sp)`) sees the NamedTuple state —
`(unknowns…, hidden…, observed…)`, including any `hidden` parameters and
`observed` sensor variables — and returns a distribution. `hidden` carries
per-episode constant parameters as state fields (see [`DecisionMDP`](@ref));
a belief over them is then a belief over state, and the compiled model is the
likelihood model for free. Hidden values are written after initialization: a
DAE whose consistent initialization depends on a hidden parameter needs
`build_initializeprob = false` or the `der = 0` form.
"""
function DecisionSystems.DecisionPOMDP(sys::System;
    inputs, dt_decision, alg, reward_fn, observe_fn, actions, discount, initialstate,
    validate=true, pool=nothing, hidden=[], kwargs...)
  pm = _mtk_physics(sys; inputs, dt_decision, alg, reward_fn, observe_fn, hidden, kwargs...)
  validate && !isempty(hidden) && _check_hidden_fields(collect(hidden), initialstate)
  m = DecisionPOMDP(; model=pm, actions, discount, initialstate, validate, pool)
  validate && !isempty(hidden) && _probe_hidden(pm, collect(hidden), initialstate, actions)
  m
end

end
