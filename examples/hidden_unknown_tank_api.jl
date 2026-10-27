# examples/hidden_unknown_tank_api.jl -- hidden_unknown_tank.jl with `hidden = [sys.k]` instead of
# the der = 0 idiom: k stays a PARAMETER (no `D(k) ~ kdot`, no `kdot`), the solver sees a one-state
# tank, and the ext writes each particle's k into its per-transition parameter copy before each block.
#
# Run, from the repo root (same deps as hidden_unknown_tank.jl; ~1-2 min, mostly compilation):
#   julia --project=<env> examples/hidden_unknown_tank_api.jl [posterior.csv]
# Everything below the model block is hidden_unknown_tank.jl verbatim: the state is touched only
# by field name (s.h, s.k) and `merge`, so the filter, policy and table do not know which form
# carries k. The two runs print the same table here (the transitions agree to ~1e-13); in
# general they need not be bitwise identical: different compiled systems, different steps.

using DecisionSystems, OrdinaryDiffEq, POMDPs, POMDPTools, Random, Printf
import ModelingToolkit as MTK                          # never `using ModelingToolkit`: it
using ModelingToolkit: t_nounits as t, D_nounits as D  # exports `transition` too (POMDPs clash)

# ── Model block: the only part that differs from hidden_unknown_tank.jl ───────────────────
const DT, H_TARGET, SIG_H = 1.0, 4.0, 0.05           # block length [s], level setpoint, sensor σ
const INFLOWS = [0.0, 0.5, 1.0, 1.5, 2.0]            # discrete pump settings

struct LevelObs; mu::Float64; end                    # Gaussian level sensor: rand + pdf
Base.rand(rng::AbstractRNG, d::LevelObs) = d.mu + SIG_H * randn(rng)
POMDPs.pdf(d::LevelObs, o) = exp(-0.5 * ((o - d.mu) / SIG_H)^2) / (SIG_H * sqrt(2pi))

MTK.@variables h(t) = 5.0                            # the one physical unknown
MTK.@parameters inflow = 1.0 k = 0.5                 # k: hidden per-episode constant, a parameter
sys = MTK.mtkcompile(MTK.System([D(h) ~ inflow - k * sqrt(max(h, 0))], t; name = :tank))
mkstate(; h, k) = (h = h, k = k)                     # state layout: (unknowns…, hidden…)
prior = ImplicitDistribution(rng -> mkstate(h = 5.0, k = 0.2 + 0.8 * rand(rng)))  # k ~ U(0.2, 1)

pomdp = DecisionPOMDP(sys; inputs = [sys.inflow], hidden = [sys.k], dt_decision = DT, alg = Tsit5(),
  reward_fn = (s, a, sp) -> -abs(sp.h - H_TARGET), observe_fn = sp -> LevelObs(sp.h),
  actions = INFLOWS, discount = 0.95, initialstate = prior,
  op = [sys.h => 5.0], build_initializeprob = false)  # hidden values are written after init
# ──────────────────────────────────────────────────────────────────────────────────────────

# Estimator: bootstrap particle filter; each particle's own k rides through `transition`.
function pf_update(pomdp, b, a, o, rng; hidden = :k, rough = 0.01)
  n = length(b); bp = similar(b); w = zeros(n)
  for i in 1:n
    sp = rand(rng, transition(pomdp, b[i], a))
    jit = getproperty(sp, hidden) * exp(rough * randn(rng))   # roughening: no collapse
    bp[i] = merge(sp, NamedTuple{(hidden,)}((jit,)))
    w[i] = pdf(observation(pomdp, b[i], a, bp[i]), o)
  end
  sum(w) > 0 ? (w ./= sum(w)) : fill!(w, 1 / n)
  c = cumsum(w); c[end] = 1.0; u = (rand(rng) .+ (0:n-1)) ./ n   # systematic resampling
  out = similar(b); j = 1
  for i in 1:n
    while u[i] > c[j]; j += 1; end
    out[i] = bp[j]
  end
  return out
end

belief_stats(b, f) = (m = sum(f, b) / length(b); (m, sqrt(sum(x -> (f(x) - m)^2, b) / length(b))))

# Policy: one-block lookahead on a belief subsample, expected squared level error.
function policy(pomdp, b, rng; nsub = 20)
  sub = b[rand(rng, eachindex(b), nsub)]
  cost(a) = sum(s -> (rand(rng, transition(pomdp, s, a)).h - H_TARGET)^2, sub) / nsub
  return argmin(cost, actions(pomdp))
end

# Episode: true tank, filter, policy, per-block table (and optional CSV of the k posterior).
function run_episode(pomdp, rng; steps = 12, n = 200, hidden = :k, csv = nothing)
  s = rand(rng, initialstate(pomdp)); k_true = getproperty(s, hidden)
  b = [rand(rng, initialstate(pomdp)) for _ in 1:n]
  kof(x) = getproperty(x, hidden)
  m, sd = belief_stats(b, kof); rows = [(0.0, m, sd)]
  @printf("%5s %7s %8s %8s %8s %8s\n", "t", "inflow", "h_true", "k_mean", "k_std", "k_true")
  @printf("%5.1f %7s %8.3f %8.3f %8.4f %8.3f\n", 0.0, "-", s.h, m, sd, k_true)
  for i in 1:steps
    a = policy(pomdp, b, rng)
    sp = rand(rng, transition(pomdp, s, a))
    o = rand(rng, observation(pomdp, s, a, sp))
    b = pf_update(pomdp, b, a, o, rng; hidden)
    s = sp
    m, sd = belief_stats(b, kof); push!(rows, (i * DT, m, sd))
    @printf("%5.1f %7.2f %8.3f %8.3f %8.4f %8.3f\n", i * DT, a, s.h, m, sd, k_true)
  end
  @printf("k posterior after %d blocks: %.3f ± %.4f (true %.3f, prior std %.4f)\n",
          steps, rows[end][2], rows[end][3], k_true, rows[1][3])
  csv === nothing || open(csv, "w") do io
    println(io, "t,k_mean,k_std"); foreach(r -> println(io, join(r, ",")), rows)
  end
  return rows
end

run_episode(pomdp, Xoshiro(2026); csv = isempty(ARGS) ? nothing : ARGS[1])
