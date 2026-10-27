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

_flatten!(v, x::Number) = (push!(v, x); v)
_flatten!(v, x::Union{Tuple,NamedTuple,AbstractArray}) = (foreach(y -> _flatten!(v, y), x); v)
_flatten!(v, x) = throw(ArgumentError(
  "cannot flatten a field of type $(typeof(x)) into a vector; define POMDPs.convert_s / " *
  "POMDPs.convert_o for this state or observation type to use array-typed CommonRL envs."))
_flatten(x) = _flatten!(Float64[], x)

_unflatten(::Type{T}, v, i) where {T<:Number} = (convert(T, v[i]), i + 1)
function _unflatten(::Type{T}, v, i) where {T<:Union{Tuple,NamedTuple}}
  vals = ()
  for F in fieldtypes(T)
    x, i = _unflatten(F, v, i)
    vals = (vals..., x)
  end
  T(vals), i
end
_unflatten(::Type{T}, v, i) where {T} = throw(ArgumentError(
  "cannot rebuild a field of type $T from a vector (its length is not fixed by the type); " *
  "define POMDPs.convert_s / POMDPs.convert_o for this state or observation type."))

function _from_vector(::Type{T}, v::AbstractArray) where {T}
  x, i = _unflatten(T, v, firstindex(v))
  i == lastindex(v) + 1 || throw(ArgumentError(
    "vector of length $(length(v)) does not match the layout of $T ($(i - firstindex(v)) numbers)."))
  x
end

POMDPs.convert_s(::Type{A}, s::S, ::DecisionMDP{S}) where {A<:AbstractArray,S<:NamedTuple} =
  convert(A, _flatten(s))
POMDPs.convert_s(::Type{A}, s::S, ::DecisionPOMDP{S}) where {A<:AbstractArray,S<:NamedTuple} =
  convert(A, _flatten(s))
POMDPs.convert_s(::Type{S}, v::AbstractArray, ::DecisionMDP{S}) where {S<:NamedTuple} =
  _from_vector(S, v)
POMDPs.convert_s(::Type{S}, v::AbstractArray, ::DecisionPOMDP{S}) where {S<:NamedTuple} =
  _from_vector(S, v)
POMDPs.convert_o(::Type{A}, o::O, ::DecisionPOMDP{S,O}) where {A<:AbstractArray,S,O<:NamedTuple} =
  convert(A, _flatten(o))
POMDPs.convert_o(::Type{O}, v::AbstractArray, ::DecisionPOMDP{S,O}) where {S,O<:NamedTuple} =
  _from_vector(O, v)

POMDPs.convert_s(::Type{A}, s::S, m::UnderlyingMDP{<:DecisionPOMDP{S},S}) where {A<:AbstractArray,S<:NamedTuple} =
  POMDPs.convert_s(A, s, m.pomdp)
POMDPs.convert_s(::Type{S}, v::AbstractArray, m::UnderlyingMDP{<:DecisionPOMDP{S},S}) where {S<:NamedTuple} =
  POMDPs.convert_s(S, v, m.pomdp)
