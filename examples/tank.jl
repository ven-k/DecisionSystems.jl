# @mtkmodel Tank begin
#   @variables begin
#     h(t)
#     inflow(t)
#   end
#   @parameters begin
#     k
#   end
#   @equations begin
#     D(h) = inflow - k * sqrt(h)
#   end
# end

# @mtkmodel FullSystem begin
#   @components begin
#     tank = Tank()
#   end
#   @equations begin
#     tank.inflow ~ 2
#   end
# end

using DecisionSystems, OrdinaryDiffEq, POMDPs, POMDPTools, MCTS

tank!(du, u, p, t) = (du[1] = p[1] - p[2] * sqrt(max(u[1], 0)))

u0 = [1.0]
p = [1.0, 1.0]
tspan = (0, 1)

pm = PhysicsModel(
  prob=ODEProblem(tank!, u0, tspan, p),
  alg=Rodas5P(),
  dt_decision=0.1,
  (apply_action!)=(integ, a) -> (integ.p[1] = a),
  reward_fn=(s, a, sp) -> -abs(sp[1][1] - 10.0),
  terminal_fn=(s) -> s[1] < 0.0 || s[1] > 50
)

mdp = DecisionMDP(
  model=pm,
  actions=[1.0, 2.0, 3.0, 4.0],
  discount=0.95,
  initialstate=Deterministic(u0)
);

solver = DPWSolver(
  n_iterations=1000,
  depth=10,
  exploration_constant=5.0
);

policy = POMDPs.solve(solver, mdp);

for (s, a, r) in stepthrough(mdp, policy, "s,a,r"; max_steps=50)
  @show s, a, r
end