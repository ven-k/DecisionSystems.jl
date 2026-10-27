# POMDPTools' generic CommonRL converter works on our
# models — no native env needed for the interface; a native env would only
# add performance (stateful integrator), which is what the integrator pool addresses.

const RL = POMDPTools.CommonRLIntegration.RL

@testset "MDP as CommonRL env" begin
  env = MDPCommonRLEnv(tank_mdp())
  RL.reset!(env)
  @test RL.observe(env) isa AbstractArray
  r = RL.act!(env, 1.0)
  @test r isa Real
  @test RL.terminated(env) == false
end

@testset "POMDP as CommonRL env (needs initialobs)" begin
  pmo = tank_pm(observe_fn=sp -> Deterministic(sp[1]))
  pomdp = DecisionPOMDP(model=pmo, actions=[0.0, 1.0], discount=0.95,
                        initialstate=Deterministic([5.0]))
  penv = POMDPCommonRLEnv(pomdp)
  RL.reset!(penv)
  @test RL.observe(penv) isa AbstractArray
  @test RL.act!(penv, 1.0) isa Real

  # 1-arg sensors give initialobs for free; action-dependent sensors must
  # define it themselves and get told so.
  @test initialobs(pomdp, [5.0]) isa Deterministic
  pomdp2 = DecisionPOMDP(model=tank_pm(observe_fn=obs_2arg), actions=[0.0, 1.0],
                         discount=0.95, initialstate=Deterministic([5.0]))
  @test_throws ErrorException initialobs(pomdp2, [5.0])
end

@testset "NamedTuple states/observations convert to vectors for array-typed envs" begin
  nt_pm(; kwargs...) = tank_pm(state_from_integrator=integ -> (h=integ.u[1],),
                               state_to_u0=s -> [s.h]; kwargs...)
  mdp = DecisionMDP(model=nt_pm(), actions=[0.0, 1.0], discount=0.95,
                    initialstate=Deterministic((h=5.0,)))
  env = MDPCommonRLEnv(mdp)                       # default RLO = AbstractArray
  RL.reset!(env)
  @test RL.observe(env) == [5.0]
  RL.setstate!(env, [7.0])
  @test RL.state(env) == [7.0]
  @test convert_s(statetype(mdp), [7.0], mdp) === (h=7.0,)
  @test_throws ArgumentError convert_s(statetype(mdp), [7.0, 1.0], mdp)

  # nested tuples (the warehouse's NTuple mover fields) flatten and rebuild in order
  pomdp = DecisionPOMDP(model=nt_pm(observe_fn=sp -> Deterministic((level=sp.h, flags=(1.0, 2.0)))),
                        actions=[0.0, 1.0], discount=0.95, initialstate=Deterministic((h=5.0,)))
  penv = POMDPCommonRLEnv(pomdp)
  RL.reset!(penv)
  @test RL.observe(penv) == [5.0, 1.0, 2.0]
  @test RL.state(penv) == [5.0]
  @test convert_o(obstype(pomdp), [4.0, 1.0, 2.0], pomdp) === (level=4.0, flags=(1.0, 2.0))
  @test RL.act!(penv, 1.0) isa Real

  # the POMDPTools full-state wrapper reaches the same methods
  uenv = MDPCommonRLEnv(UnderlyingMDP(pomdp))
  RL.reset!(uenv)
  @test RL.observe(uenv) == [5.0]
  RL.setstate!(uenv, [6.0])
  @test RL.state(uenv) == [6.0]

  # array-valued fields have no fixed length: the forward direction works, the inverse says why not
  tpm = PhysicsModel(prob=SEASONAL_PROB, alg=Tsit5(), dt_decision=1.0,
    state_from_integrator=DecisionSystems._timed_state_from_integrator,
    (apply_action!)=(integ, a) -> set_param!(integ, 1, a), reward_fn=(s, a, sp) -> 0.0)
  tmdp = DecisionMDP(model=tpm, actions=[1.0], discount=0.9, initialstate=Deterministic((u=[1.0], t=0.0)))
  @test convert_s(AbstractArray, (u=[1.0], t=0.0), tmdp) == [1.0, 0.0]
  @test_throws ArgumentError convert_s(statetype(tmdp), [1.0, 0.0], tmdp)
end
