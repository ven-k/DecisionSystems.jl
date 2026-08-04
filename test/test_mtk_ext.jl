# Phase 3 (rnd/09): ModelingToolkit extension.
#
# Import style matters: `using ModelingToolkit` would clash with POMDPs on
# `transition` (MTK exports one for its state machines) and break every other
# test file, so only qualified/selective imports appear here. Document this
# clash for users who load both.

import ModelingToolkit as MTK
using ModelingToolkit: t_nounits as t, D_nounits as D

MTK.@mtkmodel MTKTank begin
  @parameters begin
    inflow = 1.0
    k = 0.5
  end
  @variables begin
    h(t) = 5.0
    sensor(t)
  end
  @equations begin
    D(h) ~ inflow - k * sqrt(max(h, 0))
    sensor ~ h + 0.1              # eliminated by mtkcompile, still observable
  end
end

MTK.@mtkmodel MTKDrainTank begin
  @parameters begin
    inflow = 2.0
    k = 0.5
  end
  @variables begin
    h(t) = 5.0
  end
  @equations begin
    D(h) ~ inflow - k * sqrt(max(h, 0))
  end
  @continuous_events begin
    [h ~ 8.0] => [h ~ 4.0]        # emergency drain at the safety line
  end
end

MTK.@mtkcompile mtk_sys = MTKTank()
MTK.@mtkcompile mtk_dsys = MTKDrainTank()

@testset "MTK: goal API, S inference, observed sensor in state" begin
  mdp = DecisionMDP(mtk_sys;
    inputs=[mtk_sys.inflow], dt_decision=1.0, alg=Tsit5(),
    reward_fn=(s, a, sp) -> -abs(sp.h - 10.0),
    observed=[mtk_sys.sensor],
    actions=[0.0, 1.0, 2.0], discount=0.95,
    initialstate=Deterministic((h=5.0, sensor=5.1)))
  @test statetype(mdp) == @NamedTuple{h::Float64, sensor::Float64}
  sp = rand(transition(mdp, (h=5.0, sensor=5.1), 2.0))
  @test sp.sensor ≈ sp.h + 0.1

  # golden: identical physics through the raw-ODEProblem path
  sp_raw = rand(transition(tank_mdp(), [5.0], 2.0))
  @test sp.h ≈ sp_raw[1] rtol = 1e-6
end

@testset "MTK: no MTKParameters aliasing across branches (B1)" begin
  mdp = DecisionMDP(mtk_sys;
    inputs=[mtk_sys.inflow], dt_decision=1.0, alg=Tsit5(),
    reward_fn=(s, a, sp) -> 0.0, observed=[mtk_sys.sensor],
    actions=[0.0, 2.0], discount=0.95,
    initialstate=Deterministic((h=5.0, sensor=5.1)))
  s0 = (h=5.0, sensor=5.1)
  sp_a = rand(transition(mdp, s0, 0.0))
  rand(transition(mdp, s0, 2.0))
  @test rand(transition(mdp, s0, 0.0)) == sp_a
end

@testset "MTK: symbolic events fire inside the decision interval" begin
  dmdp = DecisionMDP(mtk_dsys;
    inputs=[mtk_dsys.inflow], dt_decision=10.0, alg=Tsit5(),
    reward_fn=(s, a, sp) -> 0.0,
    actions=[2.0], discount=0.95, initialstate=Deterministic((h=5.0,)))
  sd = rand(transition(dmdp, (h=5.0,), 2.0))
  @test sd.h < 8.5      # no-event equilibrium is h = 16; the drain must fire
end

@testset "MTK: POMDP with observed-variable sensor" begin
  pomdp = DecisionPOMDP(mtk_sys;
    inputs=[mtk_sys.inflow], dt_decision=1.0, alg=Tsit5(),
    reward_fn=(s, a, sp) -> -abs(sp.h - 10.0),
    observe_fn=sp -> Deterministic(sp.sensor),
    observed=[mtk_sys.sensor],
    actions=[0.0, 1.0], discount=0.95,
    initialstate=Deterministic((h=5.0, sensor=5.1)))
  steps = collect(stepthrough(pomdp, RandomPolicy(pomdp), "s,a,o"; max_steps=3))
  @test length(steps) == 3
  @test steps[1].o ≈ steps[1].s.h + 0.1
end

@testset "MTK: pool refuses with instructive error" begin
  pooled = DecisionMDP(mtk_sys;
    inputs=[mtk_sys.inflow], dt_decision=1.0, alg=Tsit5(),
    reward_fn=(s, a, sp) -> 0.0, observed=[mtk_sys.sensor],
    actions=[1.0], discount=0.95,
    initialstate=Deterministic((h=5.0, sensor=5.1)), pool=1)
  err = try
    rand(transition(pooled, (h=5.0, sensor=5.1), 1.0)); nothing
  catch e
    sprint(showerror, e)
  end
  @test err !== nothing && occursin("MTKParameters restore", err)
end
