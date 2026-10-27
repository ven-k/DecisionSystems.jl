using Documenter
using DecisionSystems

makedocs(;
  modules=[DecisionSystems],
  sitename="DecisionSystems.jl",
  authors="Venkateshprasad",
  format=Documenter.HTML(;
    canonical="https://ven-k.github.io/DecisionSystems.jl",
    edit_link="main",
    prettyurls=get(ENV, "CI", nothing) == "true"),
  pages=[
    "Home" => "index.md",
    "Guide" => "guide.md",
    "ModelingToolkit models" => "mtk.md",
    "When not to use this" => "limitations.md",
    "API" => "api.md",
  ],
  # Every exported name with a docstring must appear in an @docs block; internal
  # docstrings may stay out of the manual. All other checks are strict (warnonly=false).
  checkdocs=:exports,
)

deploydocs(;
  repo="github.com/ven-k/DecisionSystems.jl",
  devbranch="main",
)
