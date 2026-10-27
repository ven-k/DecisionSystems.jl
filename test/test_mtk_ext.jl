# ModelingToolkit extension.
#
# Import style matters: `using ModelingToolkit` would clash with POMDPs on
# `transition` (MTK exports one for its state machines) and break every other
# test file, so only qualified/selective imports appear here. Document this
# clash for users who load both.
#
# Fixtures use the explicit `@variables`/`@parameters`/`System`/`mtkcompile`
# API, which exists on every MTK 11: `@mtkmodel` moved to SciCompDSL.jl in
# MTK 11.3x and is not available here. Other test files reuse `MTK`, `t`, `D`,
# `mtk_sys` and `mtk_dsys` — keep those names.

import ModelingToolkit as MTK
using ModelingToolkit: t_nounits as t, D_nounits as D

MTK.@variables h(t)=5.0 sensor(t)
MTK.@parameters inflow=1.0 k=0.5

# Tank with a sensor that mtkcompile eliminates but keeps observable.
mtk_sys = MTK.mtkcompile(MTK.System(
  [D(h) ~ inflow - k * sqrt(max(h, 0)), sensor ~ h + 0.1], t; name=:mtk_tank))

# Tank with an emergency drain at the safety line (symbolic continuous event).
mtk_dsys = MTK.mtkcompile(MTK.System(
  [D(h) ~ inflow - k * sqrt(max(h, 0))], t; name=:mtk_drain_tank,
  continuous_events=[[h ~ 8.0] => [h ~ 4.0]]))

# Fixture builders mirroring runtests.jl's tank_mdp(); kwargs override.
mtk_tank_mdp(; kwargs...) = DecisionMDP(mtk_sys;
  inputs=[mtk_sys.inflow], dt_decision=1.0, alg=Tsit5(),
  reward_fn=(s, a, sp) -> -abs(sp.h - 10.0), observed=[mtk_sys.sensor],
  actions=[0.0, 1.0, 2.0], discount=0.95,
  initialstate=Deterministic((h=5.0, sensor=5.1)), kwargs...)

mtk_drain_mdp(; kwargs...) = DecisionMDP(mtk_dsys;
  inputs=[mtk_dsys.inflow], dt_decision=10.0, alg=Tsit5(),
  reward_fn=(s, a, sp) -> 0.0, actions=[2.0], discount=0.95,
  initialstate=Deterministic((h=5.0,)), kwargs...)

@testset "MTK: goal API, S inference, observed sensor in state" begin
  mdp = mtk_tank_mdp()
  @test statetype(mdp) == @NamedTuple{h::Float64, sensor::Float64}
  sp = rand(transition(mdp, (h=5.0, sensor=5.1), 2.0))
  @test sp.sensor ≈ sp.h + 0.1

  # golden: identical physics through the raw-ODEProblem path
  sp_raw = rand(transition(tank_mdp(), [5.0], 2.0))
  @test sp.h ≈ sp_raw[1] rtol = 1e-6
end

@testset "MTK: no MTKParameters aliasing across branches (B1)" begin
  mdp = mtk_tank_mdp(reward_fn=(s, a, sp) -> 0.0, actions=[0.0, 2.0])
  s0 = (h=5.0, sensor=5.1)
  sp_a = rand(transition(mdp, s0, 0.0))
  rand(transition(mdp, s0, 2.0))
  @test rand(transition(mdp, s0, 0.0)) == sp_a
  @test MTK.getp(mdp.model.prob, mtk_sys.inflow)(mdp.model.prob) == 1.0   # template untouched
end

@testset "MTK: initial state honored with an initializeprob attached" begin
  # ODEProblem(sys, op, tspan) with defaults attaches an initializeprob that
  # rederives u0 from the Initial(x) parameters in `p`. Passing `p` explicitly to `remake` suppresses the Initial(x) sync, so
  # every transition would restart from the default h = 5.0; the bridge lets
  # `remake` copy `p` itself in that case so the planner's state is used.
  mdp = mtk_tank_mdp()
  @test DecisionSystems.SciMLBase.has_initializeprob(mdp.model.prob.f)
  sp7 = rand(transition(mdp, (h=7.0, sensor=7.1), 0.0))
  @test sp7.h ≈ rand(transition(tank_mdp(), [7.0], 0.0))[1] rtol = 1e-6
  @test sp7 != rand(transition(mdp, (h=5.0, sensor=5.1), 0.0))
  @test MTK.getp(mdp.model.prob, mtk_sys.inflow)(mdp.model.prob) == 1.0   # template untouched
  @test MTK.getp(mdp.model.prob, MTK.Initial(h))(mdp.model.prob) == 5.0
end

@testset "MTK: symbolic events fire inside the decision interval" begin
  sd = rand(transition(mtk_drain_mdp(), (h=5.0,), 2.0))
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
  steps = collect(stepthrough(pomdp, RandomPolicy(pomdp), "s,a,sp,o"; max_steps=3))
  @test length(steps) == 3
  @test steps[1].o ≈ steps[1].sp.h + 0.1     # o is generated from sp, the post-step state
  @test steps[2].s == steps[1].sp
end

# --- IntegratorPool × MTK: the test_pool.jl gates on MTKParameters -----------
# The pooled path restores the WHOLE parameter object from the prototype before
# every checkout, so anything that wrote to `integ.p` — the compiled `setp` in
# apply_action!, or an event affect — is undone, and a pooled transition runs
# the exact code path of an unpooled one from `copy(prob.p)`.

@testset "MTK pool Gate 1: pooled == unpooled trajectories" begin
  plain = mtk_tank_mdp()
  pooled = mtk_tank_mdp(pool=2)
  s = (h=7.0, sensor=7.1)        # non-default h: a stale Initial(h) would show here
  for a in [0.0, 2.0, 1.0, 0.0, 2.0]
    sp_plain = rand(transition(plain, s, a))
    sp_pool = rand(transition(pooled, s, a))
    @test sp_pool == sp_plain    # bitwise, including the observed sensor field
    s = sp_plain
  end
end

@testset "MTK pool Gate 2: parallel access does not corrupt" begin
  pooled = mtk_tank_mdp(pool=4)
  s0 = (h=5.0, sensor=5.1)
  ref = rand(transition(pooled, s0, 1.0))
  results = Vector{typeof(ref)}(undef, 64)
  Threads.@threads for i in 1:64
    results[i] = rand(transition(pooled, s0, 1.0))
  end
  @test all(r == ref for r in results)
end

@testset "MTK pool Gate 3: no parameter leakage between checkouts" begin
  # size-1 pool forces every transition through the SAME integrator; the
  # inflow = 2.0 left in integ.p by setp would corrupt the following a=0.0 call.
  pooled = mtk_tank_mdp(pool=1)
  plain = mtk_tank_mdp()
  s0 = (h=5.0, sensor=5.1)
  sp_ref = rand(transition(plain, s0, 0.0))
  rand(transition(pooled, s0, 2.0))
  @test rand(transition(pooled, s0, 0.0)) == sp_ref
  @test MTK.getp(pooled.model.prob, mtk_sys.inflow)(pooled.model.prob) == 1.0  # template untouched
end

@testset "MTK pool: restore brings the whole parameter object back" begin
  pm = mtk_tank_mdp().model
  pool = IntegratorPool(pm; size=1)
  integ = take!(pool.channel)
  try
    MTK.setp(pm.prob, mtk_sys.inflow)(integ, 2.0)
    MTK.setp(pm.prob, mtk_sys.k)(integ, 0.9)    # not a declared input: an affect could do this
    # SciMLBase >= 3.5x wraps integ.p in DespecializedParameters; compare the wrapped object
    unwrap = Base.get_extension(DecisionSystems, :DecisionSystemsModelingToolkitExt)._unwrap_p
    @test unwrap(integ.p) != pm.prob.p
    DecisionSystems._restore_params!(integ, pm.prob.p)
    @test unwrap(integ.p) == pm.prob.p
    @test unwrap(integ.p) !== pm.prob.p
    @test @allocated(DecisionSystems._restore_params!(integ, pm.prob.p)) == 0
  finally
    put!(pool.channel, integ)
  end
end

@testset "MTK pool: symbolic event fires through the pooled path" begin
  sd = rand(transition(mtk_drain_mdp(), (h=5.0,), 2.0))
  dpool = mtk_drain_mdp(pool=1)
  @test rand(transition(dpool, (h=5.0,), 2.0)) == sd
  @test rand(transition(dpool, (h=5.0,), 2.0)) == sd   # same integrator: the event re-arms
end

@testset "MTK pool: IntegratorPool from the ext-built PhysicsModel, shared" begin
  pm = mtk_tank_mdp().model
  shared = IntegratorPool(pm; size=2)
  s0 = (h=5.0, sensor=5.1)
  m1 = mtk_tank_mdp(pool=shared)
  m2 = DecisionMDP(model=pm, actions=[0.0, 1.0, 2.0], discount=0.95,
                   initialstate=Deterministic(s0), pool=shared)   # two wrappers, one pool
  @test m1.pool === shared && m2.pool === shared
  @test rand(transition(m1, s0, 1.0)) == rand(transition(m2, s0, 1.0))
  @test rand(transition(m1, s0, 1.0)) == rand(transition(mtk_tank_mdp(), s0, 1.0))
end
