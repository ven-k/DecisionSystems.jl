# 1d (rnd/14 F6c): flat constructor — PhysicsModel keywords passed directly
# to the wrapper. Same vocabulary as the two-tier form.

@testset "Flat DecisionMDP equals two-tier" begin
  flat = DecisionMDP(
    prob=TANK_PROB, alg=Tsit5(), dt_decision=1.0,
    (apply_action!)=(integ, a) -> set_param!(integ, 1, a),
    reward_fn=(s, a, sp) -> -abs(sp[1] - 10.0),
    actions=[0.0, 1.0, 2.0], discount=0.95, initialstate=Deterministic([5.0]))
  @test flat isa DecisionMDP
  @test statetype(flat) == Vector{Float64}
  # identical dynamics to the two-tier fixture
  @test rand(transition(flat, [5.0], 1.0)) == rand(transition(tank_mdp(), [5.0], 1.0))
end

@testset "Flat DecisionPOMDP" begin
  flat = DecisionPOMDP(
    prob=TANK_PROB, alg=Tsit5(), dt_decision=1.0,
    (apply_action!)=(integ, a) -> set_param!(integ, 1, a),
    reward_fn=(s, a, sp) -> -abs(sp[1] - 10.0),
    observe_fn=sp -> Deterministic(sp[1]),
    actions=[0.0, 1.0], discount=0.95, initialstate=Deterministic([5.0]))
  @test flat isa DecisionPOMDP
  @test obstype(flat) == Float64
end

@testset "Flat form rejects mixing with model" begin
  @test_throws ArgumentError DecisionMDP(
    model=tank_pm(), prob=TANK_PROB,
    actions=[1.0], discount=0.9, initialstate=Deterministic([5.0]))
end
