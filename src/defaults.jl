_state_to_u0(s::AbstractVector) = s
function _state_to_u0(s::NamedTuple)
  hasproperty(s, :u) ||
    throw(ArgumentError("NamedTuple states must carry the integrator state in field `u`; got fields $(keys(s))."))
  s.u
end

_default_state_from_integrator(integ) = copy(integ.u)

_timed_state_from_integrator(integ) = (u = copy(integ.u), t = integ.t)

_get_t0(s::NamedTuple) = get(s, :t, 0.0)
_get_t0(s) = 0.0

_default_terminal(s) = false
