# Phase 1b (rnd/14 F5, Q7): actions beyond Vector — continuous spaces work,
# collections stay the common case, and failures are loud at construction.

# Minimal continuous action space: supports rand only — the exact contract
# MCTS-DPW's progressive widening and RandomPolicy require.
struct IntervalSpace
  lo::Float64
  hi::Float64
end
Base.rand(rng::Random.AbstractRNG, s::IntervalSpace) = s.lo + (s.hi - s.lo) * rand(rng)
Base.rand(s::IntervalSpace) = rand(Random.default_rng(), s)

struct NotASpace end  # no rand method (struct defs must sit outside @testset)

@testset "Continuous action space (rand-only contract)" begin
  mdp = tank_mdp(actions=IntervalSpace(0.0, 2.0))
  @test actiontype(mdp) == Float64
  @test actions(mdp) isa IntervalSpace
  a = rand(actions(mdp))
  @test 0.0 <= a <= 2.0

  # transition accepts a sampled continuous action end-to-end
  sp = rand(transition(mdp, [5.0], rand(actions(mdp))))
  @test sp isa Vector{Float64}

  # RandomPolicy exercises rand(rng, actions(m, s)) — the DPW contract
  steps = collect(stepthrough(mdp, RandomPolicy(mdp), "s,a,r"; max_steps=3))
  @test length(steps) == 3
  @test all(0.0 <= s.a <= 2.0 for s in steps)
end

@testset "Collections still the common case" begin
  @test actiontype(tank_mdp()) == Float64                       # Vector
  mdp_t = tank_mdp(actions=(0.0, 1.0, 2.0))                     # Tuple
  @test actiontype(mdp_t) == Float64
  # Symbol actions need an action mapping that resolves them to a rate —
  # the tank fixture's default writes the action into a Float64 parameter.
  pm_sym = tank_pm((apply_action!)=(integ, a) -> (integ.p[1] = a === :low ? 0.0 : 2.0))
  mdp_sym = tank_mdp(model=pm_sym, actions=[:low, :high])
  @test mdp_sym isa DecisionMDP
  @test actiontype(mdp_sym) == Symbol
end

@testset "Validation across space kinds" begin
  @test_throws ArgumentError tank_mdp(actions=Float64[])        # empty collection
  @test_throws ArgumentError tank_mdp(actions=())               # empty tuple
  @test_throws ArgumentError tank_mdp(actions=NotASpace())      # no rand method
end

@testset "Continuous space on the POMDP path" begin
  pmo = tank_pm(observe_fn=sp -> Deterministic(sp[1]))
  pomdp = DecisionPOMDP(model=pmo, actions=IntervalSpace(0.0, 2.0),
                        discount=0.95, initialstate=Deterministic([5.0]))
  @test actiontype(pomdp) == Float64
  psteps = collect(stepthrough(pomdp, RandomPolicy(pomdp), "s,a,o"; max_steps=3))
  @test length(psteps) == 3
end
