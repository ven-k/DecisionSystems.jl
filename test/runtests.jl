using DecisionSystems
using Test
using OrdinaryDiffEq, POMDPs, POMDPTools
using Random

# --- Shared fixtures ---------------------------------------------------------

# Exponential decay: closed form u(t) = u0 * exp(-k t). k = p[1].
decay!(du, u, p, t) = (du[1] = -p[1] * u[1])
const DECAY_PROB = ODEProblem(decay!, [1.0], (0.0, 1.0), [0.5])

# Tank: dh/dt = inflow - k_out * sqrt(h). Action sets inflow = p[1].
tank!(du, u, p, t) = (du[1] = p[1] - p[2] * sqrt(max(u[1], 0.0)))
const TANK_PROB = ODEProblem(tank!, [5.0], (0.0, 1.0), [1.0, 0.5])

# Seasonal (non-autonomous): forcing depends on absolute time.
seasonal!(du, u, p, t) = (du[1] = p[1] * sin(2pi * t / 10) - p[2] * u[1])
const SEASONAL_PROB = ODEProblem(seasonal!, [1.0], (0.0, 10.0), [1.0, 0.1])

# Fixture builders. Splatted kwargs come last, so callers can override any field.
tank_pm(; kwargs...) = PhysicsModel(;
  prob=TANK_PROB, alg=Tsit5(), dt_decision=1.0,
  (apply_action!)=(integ, a) -> (integ.p[1] = a),
  reward_fn=(s, a, sp) -> -abs(sp[1] - 10.0),
  kwargs...)

tank_mdp(; kwargs...) = DecisionMDP(;
  model=tank_pm(), actions=[0.0, 1.0, 2.0], discount=0.95,
  initialstate=Deterministic([5.0]), kwargs...)

@testset "DecisionSystems" begin
  include("test_construction.jl")
  include("test_bridge.jl")
  include("test_interfaces.jl")
  include("test_action_spaces.jl")
  include("test_actions.jl")
  include("test_flat_constructor.jl")
  include("test_rl_env.jl")
  include("test_pool.jl")
  include("test_parallel.jl")
  include("test_mtk_ext.jl")
  isfile(joinpath(@__DIR__, "test_hidden_unknowns.jl")) && include("test_hidden_unknowns.jl")
  include("test_apply_state.jl")
  include("test_hidden_api.jl")
end
