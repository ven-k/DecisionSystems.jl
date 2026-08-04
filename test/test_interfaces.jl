# POMDPs.jl interface conformance and end-to-end simulation.

@testset "Trivial delegations" begin
  mdp = tank_mdp()
  @test actions(mdp) == [0.0, 1.0, 2.0]
  @test discount(mdp) == 0.95
  @test initialstate(mdp) isa Deterministic
  @test rand(initialstate(mdp)) == [5.0]

  # reward delegates to reward_fn
  pm = tank_pm(reward_fn=(s, a, sp) -> s[1] + a + sp[1])
  mdp2 = DecisionMDP(model=pm, actions=[1.0], discount=0.9, initialstate=Deterministic([5.0]))
  @test reward(mdp2, [1.0], 2.0, [3.0]) == 6.0
end

@testset "isterminal: default and custom" begin
  @test !isterminal(tank_mdp(), [1.0e9])   # default terminal_fn is always false
  mdp = tank_mdp(model=tank_pm(terminal_fn=s -> s[1] > 50.0))
  @test isterminal(mdp, [51.0])
  @test !isterminal(mdp, [5.0])
end

@testset "transition returns a sampleable ImplicitDistribution" begin
  mdp = tank_mdp()
  d = transition(mdp, [5.0], 1.0)
  @test d isa ImplicitDistribution
  @test rand(d) isa Vector{Float64}
end

@testset "observation returns the distribution unwrapped (Q1)" begin
  pmo = tank_pm(observe_fn=sp -> Deterministic(sp[1]))
  pomdp = DecisionPOMDP(model=pmo, actions=[0.0, 1.0], discount=0.95,
                        initialstate=Deterministic([5.0]))
  d = observation(pomdp, [5.0], 1.0, [5.5])
  @test d isa Deterministic
  @test rand(d) == 5.5
  @test pdf(d, 5.5) == 1.0
end

# Action-dependent sensors: 2-arg observe_fn(a, sp) is dispatched to when the
# method exists (crack-demo finding 1 — "measure only if the action inspects").
obs_2arg(a, sp) = Deterministic(a >= 1.0 ? sp[1] : -1.0)

@testset "observation with 2-arg observe_fn (action-dependent sensor)" begin
  pmo = tank_pm(observe_fn=obs_2arg)
  pomdp = DecisionPOMDP(model=pmo, actions=[0.0, 1.0], discount=0.95,
                        initialstate=Deterministic([5.0]))
  @test obstype(pomdp) == Float64
  @test rand(observation(pomdp, [5.0], 1.0, [5.5])) == 5.5    # measuring action
  @test rand(observation(pomdp, [5.0], 0.0, [5.5])) == -1.0   # non-measuring action
end

@testset "End-to-end simulation" begin
  mdp = tank_mdp()
  steps = collect(stepthrough(mdp, RandomPolicy(mdp), "s,a,r"; max_steps=5))
  @test length(steps) == 5

  pmo = tank_pm(observe_fn=sp -> Deterministic(sp[1]))
  pomdp = DecisionPOMDP(model=pmo, actions=[0.0, 1.0], discount=0.95,
                        initialstate=Deterministic([5.0]))
  psteps = collect(stepthrough(pomdp, RandomPolicy(pomdp), "s,a,o,r"; max_steps=5))
  @test length(psteps) == 5

  total = simulate(RolloutSimulator(max_steps=10), mdp, RandomPolicy(mdp))
  @test total isa Float64
end
