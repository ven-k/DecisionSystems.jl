function POMDPs.transition(m::DecisionSystem, s, a)
  ImplicitDistribution(m, s, a) do m, s, a, rng
    m.pool === nothing ? _simulate_step(m.model, s, a, rng) :
                         _simulate_step_pooled(m.pool, m.model, s, a, rng)
  end
end

POMDPs.actions(m::DecisionSystem) = m.actions

POMDPs.reward(m::DecisionSystem, s, a, sp) = _reward(m.model, s, a, sp)

POMDPs.initialstate(m::DecisionSystem) = m.initial_state_dist

POMDPs.discount(m::DecisionSystem) = m.discount_factor

POMDPs.isterminal(m::DecisionSystem, s) = m.model.terminal_fn(s)

function POMDPs.observation(m::DecisionPOMDP, s, a, sp)
  f = m.model.observe_fn
  hasmethod(f, Tuple{typeof(a),typeof(sp)}) ? f(a, sp) : f(sp)
end

function POMDPs.initialobs(m::DecisionPOMDP, s)
  f = m.model.observe_fn
  hasmethod(f, Tuple{typeof(s)}) && return f(s)
  error("initialobs is undefined for an action-dependent (2-arg) observe_fn: " *
        "define POMDPs.initialobs(::$(typeof(m).name.name), s) for your model, or add a " *
        "1-arg observe_fn(sp) method describing the pre-action observation.")
end
