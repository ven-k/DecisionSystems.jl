# Hidden per-episode quantities as der = 0 unknowns, asserted as they behave on the
# installed ModelingToolkit. The worked example: examples/hidden_unknown_tank.jl.
#
# Fixtures `MTK`, `t`, `D` (test_mtk_ext.jl) and `Tsit5`, `Deterministic`, `tank_mdp`
# (runtests.jl) are in scope. Systems are built inside functions so the symbolic
# variables stay local and the globals test_mtk_ext.jl defines are untouched.

function _literal_tank()                    # D(k) ~ 0 written literally
  MTK.@variables h(t)=5.0 k(t)=0.5
  MTK.@parameters inflow=1.0
  MTK.mtkcompile(MTK.System([D(h) ~ inflow - k * sqrt(max(h, 0)), D(k) ~ 0], t;
                            name=:hidden_lit))
end

function _zerorate_tank()                   # D(k) ~ kdot with a parameter kdot = 0
  MTK.@variables h(t)=5.0 k(t)=0.5
  MTK.@parameters inflow=1.0 kdot=0.0
  MTK.mtkcompile(MTK.System([D(h) ~ inflow - k * sqrt(max(h, 0)), D(k) ~ kdot], t;
                            name=:hidden_zr))
end

_names(xs) = MTK.getname.(xs)
_index(sys, name) = findfirst(==(name), _names(MTK.unknowns(sys)))
_state(sys; h, k) = NamedTuple{Tuple(_names(MTK.unknowns(sys)))}((h=h, k=k))

_hidden_pomdp(sys; kwargs...) = DecisionPOMDP(sys;
  inputs=[sys.inflow], dt_decision=1.0, alg=Tsit5(),
  reward_fn=(s, a, sp) -> 0.0, observe_fn=sp -> Deterministic(sp.h),
  actions=[1.0], discount=0.95, initialstate=Deterministic(_state(sys; h=5.0, k=0.5)),
  op=[sys.h => 5.0, sys.k => 0.5], kwargs...)

# Version-dependent: ModelingToolkit 11.38 (ModelingToolkitBase >= 1.54, field
# `System.analytically_integrated`) integrates `D(k) ~ 0` analytically -- k leaves
# `u`, an observed alias of a generated constant parameter remains; MTK 11.26
# keeps k as an unknown. Either way the consequences must be consistent; the
# outcome is recorded in the testset name.
_lit_sys = _literal_tank()
_lit_eliminated = :k ∉ _names(MTK.unknowns(_lit_sys))

@testset "hidden unknowns (i): literal D(k) ~ 0 is $(_lit_eliminated ? "eliminated" : "kept") by mtkcompile (MTK $(pkgversion(MTK)))" begin
  sys = _lit_sys
  if _lit_eliminated
    @test _names(MTK.unknowns(sys)) == [:h]                # k is gone from u
    @test length(MTK.parameters(sys)) == 2                 # inflow + generated constant
    @test any(eq -> MTK.getname(eq.lhs) == :k, MTK.observed(sys))   # k survives as observed
  else
    @test sort(_names(MTK.unknowns(sys))) == [:h, :k]      # literal form still carries k
    @test _names(MTK.parameters(sys)) == [:inflow]
    @test isempty(MTK.observed(sys))
  end
end

@testset "hidden unknowns (i): zero-rate parameter keeps k in u" begin
  sys = _zerorate_tank()
  @test sort(_names(MTK.unknowns(sys))) == [:h, :k]
  @test :kdot in _names(MTK.parameters(sys))
  @test length(MTK.equations(sys)) == 2
end

@testset "hidden unknowns (ii): initializeprob trap and its one-line defense" begin
  sys = _zerorate_tank()
  op = [sys.h => 5.0, sys.k => 0.5]
  pdef = ODEProblem(sys, op, (0.0, 1.0))
  pnip = ODEProblem(sys, op, (0.0, 1.0); build_initializeprob=false)
  @test pdef.f.initialization_data !== nothing
  @test pnip.f.initialization_data === nothing

  u0n = copy(pdef.u0); u0n[_index(sys, :h)] = 2.0; u0n[_index(sys, :k)] = 0.9
  bridge(prob) = init(remake(prob; u0=u0n, tspan=(0.0, 1.0), p=DecisionSystems._copy_params(prob.p)),
                      Tsit5(); save_everystep=false, save_start=false, dense=false)
  @test bridge(pdef).u == pdef.u0          # the trap: remade u0 replaced by the prototype's
  @test bridge(pnip).u == u0n              # the defense: u0 flows through verbatim

  integ = bridge(pnip); step!(integ, 1.0, true)
  ref = OrdinaryDiffEq.solve(ODEProblem(sys, [sys.h => 2.0, sys.k => 0.9], (0.0, 1.0)), Tsit5();
                             save_everystep=false).u[end]
  @test integ.u ≈ ref rtol = 1e-6         # the honoured u0 integrates like a fresh problem
end

@testset "hidden unknowns: a der = 0 state passed through transition is honoured by the ext" begin
  sys = _zerorate_tank()
  pom = _hidden_pomdp(sys; build_initializeprob=false)
  s3 = rand(transition(pom, _state(sys; h=5.0, k=0.3), 1.0))
  s8 = rand(transition(pom, _state(sys; h=5.0, k=0.8), 1.0))
  @test s3.k == 0.3 && s8.k == 0.8        # the hidden constant comes back untouched
  @test s3.h > s8.h                        # larger k drains faster: different h after one block

  # golden: k = 0.5 matches the raw-ODEProblem fixture, whose k is an ordinary parameter
  s5 = rand(transition(pom, _state(sys; h=5.0, k=0.5), 1.0))
  @test s5.h ≈ rand(transition(tank_mdp(), [5.0], 1.0))[1] rtol = 1e-6

  # without the flag the prototype carries an initializeprob; the ext then lets
  # `remake` copy `p` itself so the Initial(x) parameters follow the particle's u0
  # (ext _remake_step) -- same trajectory, no silent reset to the operating point
  attached = _hidden_pomdp(sys)
  @test DecisionSystems.SciMLBase.has_initializeprob(attached.model.prob.f)
  sp = rand(transition(attached, _state(sys; h=5.0, k=0.3), 1.0))
  @test sp.k == 0.3
  @test sp.h ≈ s3.h rtol = 1e-10
end
