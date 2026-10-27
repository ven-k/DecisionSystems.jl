# _simulate_step correctness: golden trajectories, tree-search safety, time handling.

@testset "Golden trajectory: closed-form decay" begin
  pm = PhysicsModel(prob=DECAY_PROB, alg=Tsit5(), dt_decision=1.0,
    (apply_action!)=(integ, a) -> (integ.p[1] = a),
    reward_fn=(s, a, sp) -> 0.0)
  mdp = DecisionMDP(model=pm, actions=[0.5], discount=0.9, initialstate=Deterministic([1.0]))
  sp = rand(transition(mdp, [1.0], 0.5))
  @test sp[1] ≈ exp(-0.5) rtol = 5e-3
end

@testset "Impulse action modifies state before the step" begin
  pm = PhysicsModel(prob=DECAY_PROB, alg=Tsit5(), dt_decision=1.0,
    (apply_action!)=(integ, dose) -> (integ.u[1] += dose),
    reward_fn=(s, a, sp) -> 0.0)
  mdp = DecisionMDP(model=pm, actions=[2.0], discount=0.9, initialstate=Deterministic([1.0]))
  sp = rand(transition(mdp, [1.0], 2.0))
  @test sp[1] ≈ 3.0 * exp(-0.5) rtol = 1e-2   # (1 + dose) decays for dt at k = 0.5
end

@testset "Chained transitions match a monolithic solve" begin
  mdp = tank_mdp()
  s1 = rand(transition(mdp, [5.0], 1.0))
  s2 = rand(transition(mdp, s1, 1.0))
  mono = OrdinaryDiffEq.solve(remake(TANK_PROB; u0=[5.0], tspan=(0.0, 2.0), p=[1.0, 0.5]), Tsit5())
  @test s2[1] ≈ mono.u[end][1] rtol = 1e-2
end

@testset "Tree-search safety: statelessness and no parameter aliasing" begin
  mdp = tank_mdp()
  p0 = copy(mdp.model.prob.p)

  # same (s, a) -> same sp, regardless of interleaved counterfactuals
  sp_first = rand(transition(mdp, [5.0], 0.0))
  for a in [2.0, 1.0, 2.0, 0.0, 1.0]
    rand(transition(mdp, [5.0], a))
    rand(transition(mdp, [3.0], a))
  end
  sp_again = rand(transition(mdp, [5.0], 0.0))
  @test sp_first == sp_again

  # the prototype problem's parameters were never mutated
  @test mdp.model.prob.p == p0
end

@testset "Non-autonomous time handling (G2)" begin
  pm = PhysicsModel(prob=SEASONAL_PROB, alg=Tsit5(), dt_decision=1.0,
    state_from_integrator=DecisionSystems._timed_state_from_integrator,
    (apply_action!)=(integ, a) -> (integ.p[1] = a),
    reward_fn=(s, a, sp) -> -sp.u[1]^2)
  mdp = DecisionMDP(model=pm, actions=[1.0], discount=0.9,
                    initialstate=Deterministic((u=[1.0], t=0.0)))

  s0 = rand(initialstate(mdp))
  s1 = rand(transition(mdp, s0, 1.0))
  s2 = rand(transition(mdp, s1, 1.0))
  @test s1.t == 1.0
  @test s2.t == 2.0

  # Regression: if transition restarted at t = 0, the time-dependent forcing
  # would integrate sin over [0,1] twice and diverge from the monolithic solve.
  mono = OrdinaryDiffEq.solve(remake(SEASONAL_PROB; u0=[1.0], tspan=(0.0, 2.0), p=[1.0, 0.1]), Tsit5())
  @test s2.u[1] ≈ mono.u[end][1] rtol = 1e-2
end
