struct IntegratorPool{I}
  channel::Channel{I}
  size::Int
end

_restore_params!(integ, p0::AbstractVector) = copyto!(integ.p, p0)
_restore_params!(integ, p0) =
  error("IntegratorPool does not know how to restore parameters of type $(typeof(p0)); " *
        "define DecisionSystems._restore_params!(integ, p0) for it.")

"""
    IntegratorPool(model::PhysicsModel; size=Threads.nthreads())

Pre-build `size` integrators for `model` so transitions reuse them via
`reinit!` instead of paying `init` each call. Pass the pool to `DecisionMDP`/
`DecisionPOMDP` via the `pool` keyword. Safe under parallel tree search: each
transition holds a checked-out integrator exclusively.
"""
function IntegratorPool(model::PhysicsModel; size::Int=Threads.nthreads())
  size >= 1 || throw(ArgumentError("pool size must be >= 1, got $size"))
  mk() = init(remake(model.prob; p=_copy_params(model.prob.p)), model.alg;
              save_everystep=false, save_start=false, dense=false,
              model.integrator_kwargs...)
  first_integ = mk()
  ch = Channel{typeof(first_integ)}(size)
  put!(ch, first_integ)
  for _ in 2:size
    put!(ch, mk())
  end
  IntegratorPool{typeof(first_integ)}(ch, size)
end

function _transition_on!(integ, model::PhysicsModel, s, a)
  (; state_from_integrator, state_to_u0, prob, apply_action!, integrator_kwargs, dt_decision) = model
  u0 = state_to_u0(s)
  t0 = _get_t0(s)
  _restore_params!(integ, prob.p)
  reinit!(integ, u0; t0=t0, tf=t0 + dt_decision, erase_sol=true, reset_dt=true)
  apply_action!(integ, a)
  u_modified!(integ, true)
  haskey(integrator_kwargs, :dt) || auto_dt_reset!(integ)
  step!(integ, dt_decision, true)
  state_from_integrator(integ)
end

function _simulate_step_pooled(pool::IntegratorPool, model::PhysicsModel, s, a, rng)
  integ = take!(pool.channel)
  try
    return _transition_on!(integ, model, s, a)
  finally
    put!(pool.channel, integ)
  end
end
