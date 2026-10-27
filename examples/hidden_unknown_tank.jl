# examples/hidden_unknown_tank.jl -- the hidden-unknown idiom on a two-state tank.
#
# What it shows. A per-episode hidden physical constant -- the outflow coefficient k --
# carried as a der = 0 UNKNOWN, not a parameter. The bridge copies and freezes `p` on
# every transition (src/simulate_step.jl), so anything that must differ between belief
# particles has to ride in `u`; then the one compiled model propagates the true state,
# every particle of the filter and every lookahead sample (model as estimator). The
# printed table shows the posterior std of k contracting towards the true k.
#
# Two ModelingToolkit facts the model block encodes (measured on MTK 11.38.2):
#   * a literal `D(k) ~ 0` is eliminated by `mtkcompile` (k becomes an observed alias of
#     a generated parameter), so the equation is written `D(k) ~ kdot` with a parameter
#     kdot = 0 that is never written -- the zero-rate-parameter trick;
#   * the prototype must be built with an explicit operating point AND
#     `build_initializeprob = false`; otherwise MTK's initialization rederives u0 from the
#     `Initial(x)` parameters inside the copied `p`, and every u0 a particle passes in is
#     silently replaced by the prototype's.
# The model uses the plain `System` API because `@mtkmodel` is deprecated in MTK 11
# (moved to SciCompDSL.jl).
#
# Below the model block the state is touched only by field name (s.h, s.k) and `merge`,
# so the sibling built with the `hidden = [sys.k]` keyword (hidden_unknown_tank_api.jl)
# reuses the filter, policy and table unchanged.
#
# Run, from the repo root, in any env with ModelingToolkit, OrdinaryDiffEq, POMDPs,
# POMDPTools and DecisionSystems:
#   julia --project=<env> examples/hidden_unknown_tank.jl [posterior.csv]
# Runtime 1-2 min (72 s measured), nearly all package load and compilation; the 12-block
# episode itself is < 1 s.

using DecisionSystems, OrdinaryDiffEq, POMDPs, POMDPTools, Random, Printf
import ModelingToolkit as MTK                          # never `using ModelingToolkit`: it
using ModelingToolkit: t_nounits as t, D_nounits as D  # exports `transition` too (POMDPs clash)

# ── Model block: the idiom lives here; a `hidden = [sys.k]` sibling swaps only this ────────
const DT, H_TARGET, SIG_H = 1.0, 4.0, 0.05           # block length [s], level setpoint, sensor σ
const INFLOWS = [0.0, 0.5, 1.0, 1.5, 2.0]            # discrete pump settings

struct LevelObs; mu::Float64; end                    # Gaussian level sensor: rand + pdf
Base.rand(rng::AbstractRNG, d::LevelObs) = d.mu + SIG_H * randn(rng)
POMDPs.pdf(d::LevelObs, o) = exp(-0.5 * ((o - d.mu) / SIG_H)^2) / (SIG_H * sqrt(2pi))

MTK.@variables h(t) = 5.0 k(t) = 0.5                 # k: hidden per-episode constant, in u
MTK.@parameters inflow = 1.0 kdot = 0.0              # kdot: zero-rate parameter, never written
sys = MTK.mtkcompile(MTK.System([D(h) ~ inflow - k * sqrt(max(h, 0)),
                                 D(k) ~ kdot], t; name = :tank))
const KEYS = Tuple(MTK.getname.(MTK.unknowns(sys)))  # mtkcompile chooses the state order
mkstate(; h, k) = NamedTuple{KEYS}((h = h, k = k))
prior = ImplicitDistribution(rng -> mkstate(h = 5.0, k = 0.2 + 0.8 * rand(rng)))  # k ~ U(0.2, 1)

pomdp = DecisionPOMDP(sys; inputs = [sys.inflow], dt_decision = DT, alg = Tsit5(),
  reward_fn = (s, a, sp) -> -abs(sp.h - H_TARGET), observe_fn = sp -> LevelObs(sp.h),
  actions = INFLOWS, discount = 0.95, initialstate = prior,
  op = [sys.h => 5.0, sys.k => 0.5], build_initializeprob = false)  # the one-line defense
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
