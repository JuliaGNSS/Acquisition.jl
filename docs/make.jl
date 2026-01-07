using Documenter
using Acquisition

makedocs(
    sitename = "Acquisition.jl",
    modules = [Acquisition],
    authors = "Soeren Schoenbrod",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://JuliaGNSS.github.io/Acquisition.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Usage Guide" => "guide.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/JuliaGNSS/Acquisition.jl.git",
    devbranch = "master",
)
