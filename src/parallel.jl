_child_rngs(rng, n) = [Xoshiro(rand(rng, UInt64)) for _ in 1:n]

_chunks(idxs) = Iterators.partition(idxs, cld(length(idxs), Threads.nthreads()))

"""
    propagate(m::DecisionSystem, states, a; rng=Random.default_rng())

Propagate every state in `states` one decision block forward under action `a`,
in parallel across `Threads.nthreads()` tasks. This is the particle-filter /
belief-planner inner loop. Build the wrapper with `pool = Threads.nthreads()`
(or more) so each task reuses one pooled integrator for its whole chunk;
without a pool the call is still correct, each task just inits per transition.
Deterministic for a given `rng` regardless of thread count.

Threads pay only when the block itself carries real integration work (tens of
µs); on ~1 µs blocks the spawn overhead dominates — measure before adopting.
"""
function propagate(m::DecisionSystem, states::AbstractVector, a;
                   rng=default_rng())
  out = Vector{POMDPs.statetype(typeof(m))}(undef, length(states))
  isempty(states) && return out
  rngs = _child_rngs(rng, length(states))
  @sync for chunk in _chunks(eachindex(states))
    Threads.@spawn _propagate_chunk!(out, m, states, a, rngs, chunk)
  end
  out
end

function _propagate_chunk!(out, m::DecisionSystem, states, a, rngs, chunk)
  if m.pool === nothing
    for i in chunk
      out[i] = _simulate_step(m.model, states[i], a, rngs[i])
    end
  else
    integ = take!(m.pool.channel)
    try
      for i in chunk
        out[i] = _transition_on!(integ, m.model, states[i], a)
      end
    finally
      put!(m.pool.channel, integ)
    end
  end
  nothing
end

_act(p::POMDPs.Policy, rng, s) = POMDPs.action(p, s)
_act(f, rng, s) = hasmethod(f, Tuple{typeof(rng),typeof(s)}) ? f(rng, s) : f(s)

function _rollout_with(m::DecisionSystem, policy, s, depth, rng, integ)
  γ = m.discount_factor
  total = 0.0
  disc = 1.0
  for _ in 1:depth
    m.model.terminal_fn(s) && break
    a = _act(policy, rng, s)
    sp = integ === nothing ? _simulate_step(m.model, s, a, rng) :
                             _transition_on!(integ, m.model, s, a)
    total += disc * _reward(m.model, s, a, sp)
    disc *= γ
    s = sp
  end
  total
end

function _rollouts_chunk!(returns, m::DecisionSystem, policy, s0, depth, rngs, chunk)
  if m.pool === nothing
    for i in chunk
      returns[i] = _rollout_with(m, policy, s0, depth, rngs[i], nothing)
    end
  else
    integ = take!(m.pool.channel)
    try
      for i in chunk
        returns[i] = _rollout_with(m, policy, s0, depth, rngs[i], integ)
      end
    finally
      put!(m.pool.channel, integ)
    end
  end
  nothing
end

"""
    parallel_rollouts(m::DecisionSystem, policy, s; n, depth,
                      rng=Random.default_rng())

Run `n` independent depth-`depth` rollouts of `policy` from state `s` on
threads and return the `n` discounted returns (average them for a leaf value
estimate). `policy` is `s -> a`, `(rng, s) -> a`, or a `POMDPs.Policy`.
Serial leaf rollouts are the dominant planner cost (~70% measured on the crack
demo); this is the lever the pool's thread safety buys. Deterministic for a
given `rng` regardless of thread count.
"""
function parallel_rollouts(m::DecisionSystem, policy, s; n::Int, depth::Int,
                           rng=default_rng())
  rngs = _child_rngs(rng, n)
  returns = Vector{Float64}(undef, n)
  @sync for chunk in _chunks(1:n)
    Threads.@spawn _rollouts_chunk!(returns, m, policy, s, depth, rngs, chunk)
  end
  returns
end
