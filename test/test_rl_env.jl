# Phase 2 (rnd/16): POMDPTools' generic CommonRL converter works on our
# models — no native env needed for the interface; a native env would only
# add performance (stateful integrator), which is the Phase 4 pool question.

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
