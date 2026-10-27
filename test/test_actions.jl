# Action helpers and the post-action dt reset.

@testset "Action helpers reproduce raw-closure behavior" begin
  # set_param!: golden decay — helper writes the rate like the raw closure did
  pm = PhysicsModel(prob=DECAY_PROB, alg=Tsit5(), dt_decision=1.0,
    (apply_action!)=(integ, a) -> set_param!(integ, 1, a),
    reward_fn=(s, a, sp) -> 0.0)
  mdp = DecisionMDP(model=pm, actions=[0.5], discount=0.9, initialstate=Deterministic([1.0]))
  sp = rand(transition(mdp, [1.0], 0.5))
  @test sp[1] ≈ exp(-0.5) rtol = 5e-3

  # impulse!: dose then decay
  pm2 = PhysicsModel(prob=DECAY_PROB, alg=Tsit5(), dt_decision=1.0,
    (apply_action!)=(integ, dose) -> impulse!(integ, 1, dose),
    reward_fn=(s, a, sp) -> 0.0)
  mdp2 = DecisionMDP(model=pm2, actions=[2.0], discount=0.9, initialstate=Deterministic([1.0]))
  sp2 = rand(transition(mdp2, [1.0], 2.0))
  @test sp2[1] ≈ 3.0 * exp(-0.5) rtol = 1e-2

  # set_state!: hard reset before the step
  pm3 = PhysicsModel(prob=DECAY_PROB, alg=Tsit5(), dt_decision=1.0,
    (apply_action!)=(integ, v) -> set_state!(integ, 1, v),
    reward_fn=(s, a, sp) -> 0.0)
  mdp3 = DecisionMDP(model=pm3, actions=[4.0], discount=0.9, initialstate=Deterministic([1.0]))
  sp3 = rand(transition(mdp3, [9.9], 4.0))       # pre-action state must not matter
  @test sp3[1] ≈ 4.0 * exp(-0.5) rtol = 5e-3
end

@testset "Post-action auto_dt_reset: impulse == equivalent initial state" begin
  # An impulse from u0=1 to u0=3 must integrate the same trajectory as
  # starting at 3 with a no-op action. With the dt reset, both first steps are
  # sized from the same (post-action) state, so the endpoints agree tightly —
  # without it they differ by tolerance-level drift (crack demo finding 6).
  pm_imp = PhysicsModel(prob=DECAY_PROB, alg=Tsit5(), dt_decision=1.0,
    (apply_action!)=(integ, dose) -> impulse!(integ, 1, dose),
    reward_fn=(s, a, sp) -> 0.0)
  pm_noop = PhysicsModel(prob=DECAY_PROB, alg=Tsit5(), dt_decision=1.0,
    (apply_action!)=(integ, a) -> nothing,
    reward_fn=(s, a, sp) -> 0.0)
  mdp_imp = DecisionMDP(model=pm_imp, actions=[2.0], discount=0.9, initialstate=Deterministic([1.0]))
  mdp_noop = DecisionMDP(model=pm_noop, actions=[0.0], discount=0.9, initialstate=Deterministic([3.0]))
  sp_imp = rand(transition(mdp_imp, [1.0], 2.0))
  sp_noop = rand(transition(mdp_noop, [3.0], 0.0))
  @test sp_imp[1] ≈ sp_noop[1] rtol = 1e-9
end

@testset "User-pinned dt is respected (no auto reset)" begin
  pm = tank_pm(integrator_kwargs=(dt=0.01, adaptive=false))
  mdp = tank_mdp(model=pm)
  sp = rand(transition(mdp, [5.0], 1.0))
  @test sp isa Vector{Float64}
end
