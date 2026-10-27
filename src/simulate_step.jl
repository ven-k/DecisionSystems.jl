_copy_params(p::AbstractVector) = copy(p)
_copy_params(p::Union{Nothing,SciMLBase.NullParameters}) = p
_copy_params(p) = copy(p)

_remake_step(prob, u0, tspan, p) = remake(prob; u0=u0, tspan=tspan, p=_copy_params(p))

function _simulate_step_core(model::PhysicsModel, s, a, rng)
  (; state_from_integrator, state_to_u0, prob, alg, apply_state!, apply_action!,
     integrator_kwargs, dt_decision) = model
  u0 = state_to_u0(s)
  t0 = _get_t0(s)

  prob = _remake_step(prob, u0, (t0, t0 + dt_decision), prob.p)
  integ = init(prob, alg; save_everystep=false, save_start=false, dense=false,
               integrator_kwargs...)
  apply_state!(integ, s)
  apply_action!(integ, a)
  u_modified!(integ, true)
  haskey(integrator_kwargs, :dt) || auto_dt_reset!(integ)
  step!(integ, dt_decision, true)
  sp = state_from_integrator(integ)
  return sp, integ
end

_simulate_step(model::PhysicsModel, s, a, rng) = first(_simulate_step_core(model, s, a, rng))
