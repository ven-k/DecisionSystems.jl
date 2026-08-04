# Construction-time behavior: kwdef defaults, validation, probe, type inference.
# Pins the July 2026 review findings (rnd/00_status.md): B5, N1-N5, G3.

struct NotADist end  # has no rand method

@testset "PhysicsModel: kwdef defaults and keyword coverage" begin
  pm = tank_pm()
  @test pm.observe_fn === identity
  @test pm.terminal_fn === DecisionSystems._default_terminal
  @test pm.state_to_u0 === DecisionSystems._state_to_u0
  @test pm.state_from_integrator === DecisionSystems._default_state_from_integrator
  @test pm.integrator_kwargs === (;)

  # Regression (N3): every mapping field is settable by keyword.
  myobs = sp -> Deterministic(sp[1])
  mystate = integ -> copy(integ.u)
  myterm = s -> s[1] > 50.0
  pm2 = tank_pm(observe_fn=myobs, state_from_integrator=mystate, terminal_fn=myterm)
  @test pm2.observe_fn === myobs
  @test pm2.state_from_integrator === mystate
  @test pm2.terminal_fn === myterm
end

@testset "DecisionMDP: validation fires with ArgumentError (N1, G3)" begin
  @test tank_mdp() isa DecisionMDP{Vector{Float64}}

  @test_throws ArgumentError tank_mdp(actions=Float64[])
  @test_throws ArgumentError tank_mdp(discount=0.0)
  @test_throws ArgumentError tank_mdp(discount=-0.5)
  @test_throws ArgumentError tank_mdp(discount=1.5)
  @test tank_mdp(discount=1.0) isa DecisionMDP        # contract is (0, 1]

  @test_throws ArgumentError tank_mdp(model=tank_pm(dt_decision=-1.0))
  @test_throws ArgumentError tank_mdp(model=tank_pm(dt_decision=0.0))
  @test_throws ArgumentError tank_mdp(model=tank_pm(dt_decision=Inf))

  @test_throws ArgumentError tank_mdp(initialstate=NotADist())
end

@testset "Probe: round-trip law and type inference (14 F2)" begin
  # state_from_integrator not inverted by state_to_u0 -> caught at construction
  pm_rt = tank_pm(state_from_integrator=integ -> 2 .* integ.u)
  @test_throws ArgumentError tank_mdp(model=pm_rt)
  # escape hatch skips the probe
  @test tank_mdp(model=pm_rt, validate=false) isa DecisionMDP

  # S inferred from the probe step, not from rand(initialstate) alone
  @test statetype(tank_mdp()) == Vector{Float64}
  @test actiontype(tank_mdp()) == Float64

  # state type produced by transition differs from initialstate's sample -> warn
  pm_nt = tank_pm(state_from_integrator=DecisionSystems._timed_state_from_integrator)
  @test_logs (:warn, r"solvers may fail") match_mode = :any tank_mdp(model=pm_nt)
end

@testset "DecisionPOMDP: constructible, O inferred, observe contract (B5, Q1)" begin
  pmo = tank_pm(observe_fn=sp -> Deterministic(sp[1]))
  pomdp = DecisionPOMDP(model=pmo, actions=[0.0, 1.0], discount=0.95,
                        initialstate=Deterministic([5.0]))
  @test pomdp isa DecisionPOMDP
  @test statetype(pomdp) == Vector{Float64}
  @test obstype(pomdp) == Float64

  # default identity observe_fn is only valid for an MDP
  @test_throws ArgumentError DecisionPOMDP(model=tank_pm(), actions=[1.0],
                                           discount=0.9, initialstate=Deterministic([5.0]))

  # observe_fn returning something non-sampleable
  pm_bad = tank_pm(observe_fn=sp -> NotADist())
  @test_throws ArgumentError DecisionPOMDP(model=pm_bad, actions=[1.0],
                                           discount=0.9, initialstate=Deterministic([5.0]))

  # sampleable but pdf-less observation distribution -> warn, not error
  pm_nopdf = tank_pm(observe_fn=sp -> [sp[1], sp[1] + 1.0])
  @test_logs (:warn, r"no pdf method") match_mode = :any DecisionPOMDP(
    model=pm_nopdf, actions=[1.0], discount=0.9, initialstate=Deterministic([5.0]))
end
