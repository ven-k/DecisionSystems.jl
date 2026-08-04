module DecisionSystemsModelingToolkitExt

using DecisionSystems
using DecisionSystems: _copy_params
using ModelingToolkit
using ModelingToolkit: System, getu, setp, unknowns, getname
using SciMLBase: ODEProblem, remake, u_modified!

DecisionSystems._copy_params(p::ModelingToolkit.MTKParameters) = copy(p)

DecisionSystems._restore_params!(integ, p0::ModelingToolkit.MTKParameters) =
  error("IntegratorPool does not support MTK models yet (MTKParameters restore). " *
        "Construct the model without `pool` for now.")

function _mtk_physics(sys::System; inputs, dt_decision, alg, reward_fn,
                      observed=[], op=[], terminal_fn=DecisionSystems._default_terminal,
                      observe_fn=identity, integrator_kwargs=(;), prob_kwargs...)
  prob = ODEProblem(sys, op, (0.0, dt_decision); prob_kwargs...)

  us = unknowns(sys)
  nu = length(us)
  state_syms = (Symbol.(getname.(us))..., Symbol.(getname.(observed))...)
  get_state = getu(prob, vcat(us, collect(observed)))
  set_inputs = setp(prob, length(inputs) == 1 ? only(inputs) : collect(inputs))

  state_from_integrator = integ -> NamedTuple{state_syms}(Tuple(get_state(integ)))
  state_to_u0 = s -> [Tuple(s)[1:nu]...]
  apply_action! = (integ, a) -> (set_inputs(integ, a); u_modified!(integ, true); nothing)

  PhysicsModel(; prob, alg, dt_decision, state_from_integrator, state_to_u0,
                 apply_action!, reward_fn, observe_fn, terminal_fn, integrator_kwargs)
end

"""
    DecisionMDP(sys::System; inputs, dt_decision, alg, reward_fn,
                actions, discount, initialstate, observed=[], op=[], ...)

Wrap an `@mtkcompile`-ed `System` as an MDP. `inputs` lists the parameter(s)
actions drive (single parameter: action is a scalar; several: a vector).
States are NamedTuples keyed by unknown name; `observed` variables are
appended to the state as compiled read-only sensor values.
"""
function DecisionSystems.DecisionMDP(sys::System;
    inputs, dt_decision, alg, reward_fn, actions, discount, initialstate,
    validate=true, pool=nothing, kwargs...)
  pm = _mtk_physics(sys; inputs, dt_decision, alg, reward_fn, kwargs...)
  DecisionMDP(; model=pm, actions, discount, initialstate, validate, pool)
end

"""
    DecisionPOMDP(sys::System; observe_fn, ...)

POMDP variant: `observe_fn(sp)` (or `(a, sp)`) sees the NamedTuple state —
including any `observed` sensor variables — and returns a distribution.
"""
function DecisionSystems.DecisionPOMDP(sys::System;
    inputs, dt_decision, alg, reward_fn, observe_fn, actions, discount, initialstate,
    validate=true, pool=nothing, kwargs...)
  pm = _mtk_physics(sys; inputs, dt_decision, alg, reward_fn, observe_fn, kwargs...)
  DecisionPOMDP(; model=pm, actions, discount, initialstate, validate, pool)
end

end
