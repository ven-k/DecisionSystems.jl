# IntegratorPool gates.
#   Gate 1: pooled path produces IDENTICAL trajectories to the init path.
#   Gate 2: parallel tree-search-style access does not corrupt state.
#   Gate 3: parameter mutations by actions do not leak between checkouts.
# (Perf assertions live outside Pkg.test — CI timing is too flaky; the
#  speedup benchmark lives with the examples.)

@testset "Gate 1: pooled == unpooled trajectories" begin
  plain = tank_mdp()
  pooled = tank_mdp(pool=2)
  s = [5.0]
  for a in [0.0, 2.0, 1.0, 0.0, 2.0]
    sp_plain = rand(transition(plain, s, a))
    sp_pool  = rand(transition(pooled, s, a))
    @test sp_pool == sp_plain          # bitwise, not ≈: same code path contract
    s = sp_plain
  end

  # non-autonomous resume (G2) through the pool
  pm = PhysicsModel(prob=SEASONAL_PROB, alg=Tsit5(), dt_decision=1.0,
    state_from_integrator=DecisionSystems._timed_state_from_integrator,
    (apply_action!)=(integ, a) -> set_param!(integ, 1, a),
    reward_fn=(s, a, sp) -> 0.0)
  mdp_np = DecisionMDP(model=pm, actions=[1.0], discount=0.9,
                       initialstate=Deterministic((u=[1.0], t=0.0)))
  mdp_p = DecisionMDP(model=pm, actions=[1.0], discount=0.9,
                      initialstate=Deterministic((u=[1.0], t=0.0)), pool=2)
  s1 = rand(transition(mdp_np, (u=[1.0], t=0.0), 1.0))
  s1p = rand(transition(mdp_p, (u=[1.0], t=0.0), 1.0))
  @test s1p.t == s1.t == 1.0
  @test s1p.u == s1.u
end

@testset "Gate 2: parallel access does not corrupt" begin
  pooled = tank_mdp(pool=4)
  ref = rand(transition(pooled, [5.0], 1.0))
  results = Vector{Vector{Float64}}(undef, 64)
  Threads.@threads for i in 1:64
    results[i] = rand(transition(pooled, [5.0], 1.0))
  end
  @test all(r == ref for r in results)
end

@testset "Gate 3: no parameter leakage between checkouts" begin
  # size-1 pool forces every transition through the SAME integrator; a leaked
  # p[1] from the a=2.0 call would corrupt the following a=0.0 call.
  pooled = tank_mdp(pool=1)
  plain = tank_mdp()
  sp_ref = rand(transition(plain, [5.0], 0.0))
  rand(transition(pooled, [5.0], 2.0))       # mutates integ.p[1] = 2.0
  @test rand(transition(pooled, [5.0], 0.0)) == sp_ref
end

@testset "Pool construction and API" begin
  @test_throws ArgumentError IntegratorPool(tank_pm(); size=0)
  shared = IntegratorPool(tank_pm(); size=2)
  m1 = tank_mdp(pool=shared)
  m2 = tank_mdp(pool=shared)                 # two wrappers, one pool
  @test rand(transition(m1, [5.0], 1.0)) == rand(transition(m2, [5.0], 1.0))
end