# The `hidden = [sys.k]` keyword of the ModelingToolkit extension. k is a
# PARAMETER of the system with no D(k) equation; the ext writes each state's k into
# the per-transition parameter object (apply_state!) and reads it back. Fixtures
# `MTK`, `t`, `D`, `mtk_sys` (test_mtk_ext.jl; mtk_sys has parameters inflow, k and
# the observed sensor) and `tank_mdp` (runtests.jl) are in scope. The der = 0
# counterpart for the golden comparison is built locally.

_api_op = [mtk_sys.h => 5.0, mtk_sys.k => 0.5]

api_mdp(; kwargs...) = DecisionMDP(mtk_sys;
  inputs=[mtk_sys.inflow], hidden=[mtk_sys.k], observed=[mtk_sys.sensor],
  dt_decision=1.0, alg=Tsit5(), reward_fn=(s, a, sp) -> -abs(sp.h - 10.0),
  actions=[0.0, 1.0, 2.0], discount=0.95,
  initialstate=Deterministic((h=5.0, k=0.5, sensor=5.1)),
  op=_api_op, build_initializeprob=false, kwargs...)

function _api_zerorate_tank()               # the der = 0 form of the same tank
  MTK.@variables h(t)=5.0 k(t)=0.5
  MTK.@parameters inflow=1.0 kdot=0.0
  MTK.mtkcompile(MTK.System([D(h) ~ inflow - k * sqrt(max(h, 0)), D(k) ~ kdot], t;
                            name=:api_zr))
end

@testset "hidden=: state layout (unknowns…, hidden…, observed…) and S inference" begin
  mdp = api_mdp()
  @test statetype(mdp) == @NamedTuple{h::Float64, k::Float64, sensor::Float64}
  @test mdp.model.apply_state! !== DecisionSystems._default_apply_state
  plain = api_mdp(hidden=[], initialstate=Deterministic((h=5.0, sensor=5.1)))
  @test plain.model.apply_state! === DecisionSystems._default_apply_state
  @test statetype(plain) == @NamedTuple{h::Float64, sensor::Float64}
end

@testset "hidden=: the value is honoured and comes back unchanged" begin
  mdp = api_mdp()
  s3 = rand(transition(mdp, (h=5.0, k=0.3, sensor=5.1), 1.0))
  s8 = rand(transition(mdp, (h=5.0, k=0.8, sensor=5.1), 1.0))
  @test s3.k == 0.3 && s8.k == 0.8
  @test s3.h > s8.h                        # larger k drains faster
  @test s3.sensor ≈ s3.h + 0.1             # observed field computed from the new h
  @test MTK.getp(mdp.model.prob, mtk_sys.k)(mdp.model.prob) == 0.5   # template untouched

  # golden: k = 0.5 is the raw-ODEProblem fixture's k
  s5 = rand(transition(mdp, (h=5.0, k=0.5, sensor=5.1), 1.0))
  @test s5.h ≈ rand(transition(tank_mdp(), [5.0], 1.0))[1] rtol = 1e-6

  # fields are read by name: a differently ordered state gives the same transition
  @test rand(transition(mdp, (k=0.3, sensor=5.1, h=5.0), 1.0)) == s3
end

@testset "hidden=: POMDP with a hidden parameter and an observed sensor" begin
  pomdp = DecisionPOMDP(mtk_sys;
    inputs=[mtk_sys.inflow], hidden=[mtk_sys.k], observed=[mtk_sys.sensor],
    dt_decision=1.0, alg=Tsit5(), reward_fn=(s, a, sp) -> -abs(sp.h - 10.0),
    observe_fn=sp -> Deterministic(sp.sensor),
    actions=[0.0, 1.0], discount=0.95,
    initialstate=Deterministic((h=5.0, k=0.5, sensor=5.1)),
    op=_api_op, build_initializeprob=false)
  @test statetype(pomdp) == @NamedTuple{h::Float64, k::Float64, sensor::Float64}
  steps = collect(stepthrough(pomdp, RandomPolicy(pomdp), "s,a,sp,o"; max_steps=3))
  @test length(steps) == 3
  @test steps[1].o ≈ steps[1].sp.h + 0.1
  @test all(st.sp.k == 0.5 for st in steps)
  @test steps[2].s == steps[1].sp
  # the sensor does not see k directly: two particles with different k but the same
  # h observe the same thing, yet transition differently (what a filter resolves)
  @test observation(pomdp, (h=5.0, k=0.3, sensor=5.1), 1.0, (h=5.0, k=0.3, sensor=5.1)) ==
        observation(pomdp, (h=5.0, k=0.8, sensor=5.1), 1.0, (h=5.0, k=0.8, sensor=5.1))
end

@testset "hidden=: construction-time validation" begin
  @test_throws ArgumentError api_mdp(hidden=[mtk_sys.inflow])            # hidden ∩ inputs
  @test_throws ArgumentError api_mdp(hidden=[mtk_sys.h])                 # an unknown
  @test_throws ArgumentError api_mdp(hidden=[mtk_sys.k, mtk_sys.k])      # duplicate
  @test_throws ArgumentError api_mdp(hidden=[mtk_sys.k], observed=[mtk_sys.k])   # name clash
  MTK.@parameters not_in_sys
  @test_throws ArgumentError api_mdp(hidden=[not_in_sys])                # not a parameter
  @test_throws ArgumentError api_mdp(initialstate=Deterministic((h=5.0, sensor=5.1)))  # no k field
  err = try api_mdp(hidden=[mtk_sys.h]); nothing catch e; e end
  @test occursin("unknown", sprint(showerror, err))
  err = try api_mdp(hidden=[not_in_sys]); nothing catch e; e end
  @test occursin("not a parameter", sprint(showerror, err))
end

@testset "hidden=: pooled == unpooled bitwise, alternating k through one integrator" begin
  plain = api_mdp()
  pooled = api_mdp(pool=1)
  s = (h=5.0, k=0.3, sensor=5.1)
  for (k, a) in zip([0.3, 0.8, 0.3, 0.5, 0.8, 0.2], [1.0, 0.0, 2.0, 1.0, 0.0, 2.0])
    s = merge(s, (k=k,))
    sp_plain = rand(transition(plain, s, a))
    sp_pool = rand(transition(pooled, s, a))
    @test sp_pool == sp_plain
    s = sp_plain
  end
  @test MTK.getp(pooled.model.prob, mtk_sys.k)(pooled.model.prob) == 0.5   # template untouched
  sp_ref = rand(transition(plain, (h=5.0, k=0.5, sensor=5.1), 0.0))
  rand(transition(pooled, (h=5.0, k=0.9, sensor=5.1), 2.0))
  @test rand(transition(pooled, (h=5.0, k=0.5, sensor=5.1), 0.0)) == sp_ref
end

@testset "hidden=: propagate over 60 particles with varying k" begin
  plain = api_mdp()
  pooled = api_mdp(pool=2)
  states = [(h=5.0, k=0.2 + 0.8 * rand(Xoshiro(i)), sensor=5.1) for i in 1:60]
  ref = [rand(transition(plain, s, 1.0)) for s in states]
  @test propagate(plain, states, 1.0) == ref
  @test propagate(pooled, states, 1.0) == ref
  @test all(o.k == s.k for (o, s) in zip(ref, states))
  @test length(unique(o.h for o in ref)) == 60
end

@testset "hidden=: an initializeprob attached to the prototype still honours h and k" begin
  attached = api_mdp(build_initializeprob=true)
  @test DecisionSystems.SciMLBase.has_initializeprob(attached.model.prob.f)
  flag = api_mdp()
  for (k, a) in ((0.3, 1.0), (0.8, 0.0))
    s = (h=7.0, k=k, sensor=7.1)           # non-default h: a stale Initial(h) would show here
    sp_att = rand(transition(attached, s, a))
    sp_flag = rand(transition(flag, s, a))
    @test sp_att.k == k
    @test sp_att.h ≈ sp_flag.h rtol = 1e-10
    @test rand(transition(api_mdp(build_initializeprob=true, pool=1), s, a)) == sp_att
  end
  @test MTK.getp(attached.model.prob, MTK.Initial(mtk_sys.h))(attached.model.prob) == 5.0
end

@testset "hidden=: golden against the der = 0 form of the same tank" begin
  zs = _api_zerorate_tank()
  zkeys = Tuple(MTK.getname.(MTK.unknowns(zs)))
  zstate(; h, k) = NamedTuple{zkeys}((h=h, k=k))
  @test sort(collect(zkeys)) == [:h, :k]   # one more unknown than the hidden= form
  zm = DecisionMDP(zs; inputs=[zs.inflow], dt_decision=1.0, alg=Tsit5(),
    reward_fn=(s, a, sp) -> 0.0, actions=[0.0, 1.0, 2.0], discount=0.95,
    initialstate=Deterministic(zstate(h=5.0, k=0.5)),
    op=[zs.h => 5.0, zs.k => 0.5], build_initializeprob=false)
  hm = api_mdp()
  @test length(MTK.unknowns(mtk_sys)) == 1
  for (k, a) in ((0.3, 1.0), (0.8, 2.0), (0.5, 0.0), (0.95, 1.0))
    z = rand(transition(zm, zstate(h=5.0, k=k), a))
    p = rand(transition(hm, (h=5.0, k=k, sensor=5.1), a))
    @test p.h ≈ z.h rtol = 1e-8
    @test p.k == z.k == k
  end
end
