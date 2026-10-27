# Threaded batch APIs on top of the pool.
#   Gate 1: propagate == per-state transition, pool and init paths.
#   Gate 2: parallel_rollouts is deterministic in the caller rng and matches
#           a serial recomputation with the same child streams.
#   Gate 3: the three policy conventions (s->a, (rng,s)->a, POMDPs.Policy)
#           produce identical returns.
# (Speedup assertions live outside Pkg.test, same policy as test_pool.jl.)

@testset "Gate 1: propagate == per-state transition" begin
  states = [[h] for h in 2.0:1.0:9.0]
  for m in (tank_mdp(), tank_mdp(pool=2))
    sps = propagate(m, states, 1.0; rng=Xoshiro(3))
    @test sps == [rand(transition(m, s, 1.0)) for s in states]
  end
  @test propagate(tank_mdp(), Vector{Float64}[], 1.0) == Vector{Float64}[]
end

@testset "Gate 2: rollouts deterministic and serially reproducible" begin
  m = tank_mdp(pool=2)
  policy = s -> s[1] < 10.0 ? 2.0 : 0.0
  r1 = parallel_rollouts(m, policy, [5.0]; n=16, depth=8, rng=Xoshiro(11))
  r2 = parallel_rollouts(m, policy, [5.0]; n=16, depth=8, rng=Xoshiro(11))
  @test r1 == r2                       # same caller rng => same returns

  # serial reference with the same child-stream derivation
  seeds = [rand(Xoshiro(11), UInt64) for _ in 1:16]
  ref = map(seeds) do seed
    rng = Xoshiro(seed)
    s = [5.0]; total = 0.0; disc = 1.0
    for _ in 1:8
      isterminal(m, s) && break
      a = policy(s)
      sp = rand(rng, transition(m, s, a))
      total += disc * reward(m, s, a, sp)
      disc *= discount(m)
      s = sp
    end
    total
  end
  @test r1 == ref
end

@testset "Gate 3: policy conventions agree; terminal is absorbing" begin
  m = tank_mdp()
  f1 = s -> 1.0
  f2 = (rng, s) -> 1.0
  p = FunctionPolicy(s -> 1.0)
  r1 = parallel_rollouts(m, f1, [5.0]; n=4, depth=5, rng=Xoshiro(1))
  r2 = parallel_rollouts(m, f2, [5.0]; n=4, depth=5, rng=Xoshiro(1))
  r3 = parallel_rollouts(m, p,  [5.0]; n=4, depth=5, rng=Xoshiro(1))
  @test r1 == r2 == r3

  mterm = tank_mdp(model=tank_pm(terminal_fn=s -> true))
  @test parallel_rollouts(mterm, f1, [5.0]; n=4, depth=5) == zeros(4)
end
