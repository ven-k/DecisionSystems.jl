@kwdef struct PhysicsModel{P,A,T,C,SI,R,O,TR,U,K<:NamedTuple}
  prob::P
  alg::A
  dt_decision::T
  state_from_integrator::SI = _default_state_from_integrator
  apply_action!::C
  reward_fn::R
  observe_fn::O = identity
  terminal_fn::TR = _default_terminal
  state_to_u0::U = _state_to_u0
  integrator_kwargs::K = (;)
end

struct DecisionMDP{S,PM<:PhysicsModel,A,AS,G,I,PL} <: MDP{S,A}
  model::PM
  actions::AS
  discount_factor::G
  initial_state_dist::I
  pool::PL
end

struct DecisionPOMDP{S,O,PM<:PhysicsModel,A,AS,G,I,PL} <: POMDP{S,A,O}
  model::PM
  actions::AS
  discount_factor::G
  initial_state_dist::I
  pool::PL
end

const DecisionSystem = Union{DecisionMDP,DecisionPOMDP}
