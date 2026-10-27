# Hidden per-episode constants carried as PARAMETERS written per transition
# through `apply_state!` (runs after init / reinit!, before apply_action!). Raw
# ODEProblem, no ModelingToolkit: the tank's outflow coefficient k = p[2] becomes
# a state field. Fixtures `TANK_PROB`, `tank_pm`, `tank_mdp` come from runtests.jl.

hidden_pm(; kwargs...) = PhysicsModel(;
  prob=TANK_PROB, alg=Tsit5(), dt_decision=1.0,
  state_from_integrator=integ -> (u=copy(integ.u), k=integ.p[2]),
  state_to_u0=s -> s.u,
  (apply_state!)=(integ, s) -> set_param!(integ, 2, s.k),
  (apply_action!)=(integ, a) -> set_param!(integ, 1, a),
  reward_fn=(s, a, sp) -> -abs(sp.u[1] - 10.0),
  kwargs...)

hidden_mdp(; kwargs...) = DecisionMDP(;
  model=hidden_pm(), actions=[0.0, 1.0, 2.0], discount=0.95,
  initialstate=Deterministic((u=[5.0], k=0.5)), kwargs...)

@testset "apply_state!: default is a no-op and the field is optional" begin
  @test tank_pm().apply_state! === DecisionSystems._default_apply_state
  @test DecisionSystems._default_apply_state(nothing, nothing) === nothing
  @test rand(transition(tank_mdp(), [5.0], 1.0)) == rand(transition(tank_mdp(pool=1), [5.0], 1.0))
end

@testset "apply_state!: the hidden k is honoured and comes back unchanged" begin
  mdp = hidden_mdp()
  @test statetype(mdp) == @NamedTuple{u::Vector{Float64}, k::Float64}
  s3 = rand(transition(mdp, (u=[5.0], k=0.3), 1.0))
  s8 = rand(transition(mdp, (u=[5.0], k=0.8), 1.0))
  @test s3.k == 0.3 && s8.k == 0.8
  @test s3.u[1] > s8.u[1]                  # larger k drains faster
  @test TANK_PROB.p == [1.0, 0.5]          # the prototype's parameters are never written

  # golden: k = 0.5 is the prototype's value, so the write is a no-op and the
  # trajectory is bitwise the plain fixture's
  @test rand(transition(mdp, (u=[5.0], k=0.5), 1.0)).u == rand(transition(tank_mdp(), [5.0], 1.0))
  # and k = 0.3 is bitwise a problem whose p[2] is 0.3 from the start
  pm3 = tank_pm(prob=remake(TANK_PROB; p=[1.0, 0.3]), (apply_action!)=(integ, a) -> set_param!(integ, 1, a))
  m3 = DecisionMDP(model=pm3, actions=[1.0], discount=0.95, initialstate=Deterministic([5.0]))
  @test s3.u == rand(transition(m3, [5.0], 1.0))
end

@testset "apply_state!: round-trip law and probe order (state write before action write)" begin
  pm = hidden_pm()
  sp, integ = DecisionSystems._simulate_step_core(pm, (u=[5.0], k=0.7), 2.0, nothing)
  @test pm.state_to_u0(sp) == integ.u
  @test integ.p == [2.0, 0.7]              # action overwrote p[1], the state write p[2]
  # an apply_action! that reads what apply_state! wrote sees the new value
  pm2 = hidden_pm((apply_action!)=(integ, a) -> set_param!(integ, 1, a * integ.p[2]))
  sp2, integ2 = DecisionSystems._simulate_step_core(pm2, (u=[5.0], k=0.7), 2.0, nothing)
  @test integ2.p[1] == 1.4
end

@testset "apply_state!: pooled == unpooled bitwise, alternating k through one integrator" begin
  plain = hidden_mdp()
  pooled = hidden_mdp(pool=1)
  s = (u=[5.0], k=0.3)
  for (k, a) in zip([0.3, 0.8, 0.3, 0.5, 0.8, 0.2], [1.0, 0.0, 2.0, 1.0, 0.0, 2.0])
    s = (u=s.u, k=k)
    sp_plain = rand(transition(plain, s, a))
    sp_pool = rand(transition(pooled, s, a))
    @test sp_pool == sp_plain
    s = sp_plain
  end
  @test TANK_PROB.p == [1.0, 0.5]
  # the k written for one checkout never leaks into the next (size-1 pool)
  sp_ref = rand(transition(plain, (u=[5.0], k=0.5), 0.0))
  rand(transition(pooled, (u=[5.0], k=0.9), 2.0))
  @test rand(transition(pooled, (u=[5.0], k=0.5), 0.0)) == sp_ref
end

@testset "apply_state!: propagate over particles with varying k" begin
  plain = hidden_mdp()
  pooled = hidden_mdp(pool=2)
  states = [(u=[5.0], k=0.2 + 0.8 * rand(Xoshiro(i))) for i in 1:60]
  ref = [rand(transition(plain, s, 1.0)) for s in states]
  @test propagate(plain, states, 1.0) == ref
  @test propagate(pooled, states, 1.0) == ref
  @test all(o.k == s.k for (o, s) in zip(ref, states))
  @test length(unique(o.u[1] for o in ref)) == 60     # every particle integrated its own k
  @test parallel_rollouts(pooled, s -> 1.0, (u=[5.0], k=0.3); n=4, depth=3, rng=Xoshiro(1)) ==
        parallel_rollouts(plain, s -> 1.0, (u=[5.0], k=0.3); n=4, depth=3, rng=Xoshiro(1))
end

@testset "apply_state!: flat constructor accepts it" begin
  flat = DecisionMDP(
    prob=TANK_PROB, alg=Tsit5(), dt_decision=1.0,
    state_from_integrator=integ -> (u=copy(integ.u), k=integ.p[2]),
    (apply_state!)=(integ, s) -> set_param!(integ, 2, s.k),
    (apply_action!)=(integ, a) -> set_param!(integ, 1, a),
    reward_fn=(s, a, sp) -> -abs(sp.u[1] - 10.0),
    actions=[0.0, 1.0, 2.0], discount=0.95, initialstate=Deterministic((u=[5.0], k=0.5)))
  @test statetype(flat) == @NamedTuple{u::Vector{Float64}, k::Float64}
  @test rand(transition(flat, (u=[5.0], k=0.3), 1.0)) == rand(transition(hidden_mdp(), (u=[5.0], k=0.3), 1.0))
  pomdp = DecisionPOMDP(
    prob=TANK_PROB, alg=Tsit5(), dt_decision=1.0,
    state_from_integrator=integ -> (u=copy(integ.u), k=integ.p[2]),
    (apply_state!)=(integ, s) -> set_param!(integ, 2, s.k),
    (apply_action!)=(integ, a) -> set_param!(integ, 1, a),
    reward_fn=(s, a, sp) -> 0.0, observe_fn=sp -> Deterministic(sp.u[1]),
    actions=[0.0, 1.0], discount=0.95, initialstate=Deterministic((u=[5.0], k=0.5)))
  @test pomdp isa DecisionPOMDP
  @test rand(transition(pomdp, (u=[5.0], k=0.3), 1.0)).k == 0.3
end
