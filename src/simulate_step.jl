_copy_params(p::AbstractVector) = copy(p)
_copy_params(p::Union{Nothing,SciMLBase.NullParameters}) = p
_copy_params(p) = copy(p)

function _simulate_step_core(model::PhysicsModel, s, a, rng)
  (; state_from_integrator, state_to_u0, prob, alg, apply_action!, integrator_kwargs, dt_decision) = model
  u0 = state_to_u0(s)
  t0 = _get_t0(s)

  prob = remake(prob; u0=u0, tspan=(t0, t0 + dt_decision), p=_copy_params(prob.p))
  integ = init(prob, alg; save_everystep=false, save_start=false, dense=false,
               integrator_kwargs...)
  apply_action!(integ, a)
  u_modified!(integ, true)
  haskey(integrator_kwargs, :dt) || auto_dt_reset!(integ)
  step!(integ, dt_decision, true)
  sp = state_from_integrator(integ)
  return sp, integ
end

_simulate_step(model::PhysicsModel, s, a, rng) = first(_simulate_step_core(model, s, a, rng))
