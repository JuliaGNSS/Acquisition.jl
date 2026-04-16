using Documenter
using Acquisition

makedocs(
    sitename = "Acquisition.jl",
    modules = [Acquisition],
    warnonly = [:missing_docs],
    authors = "Soeren Schoenbrod",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://JuliaGNSS.github.io/Acquisition.jl",
        example_size_threshold = nothing,
        size_threshold = 50 * 1024 * 1024,
    ),
    pages = [
        "Home" => "index.md",
        "Usage Guide" => "guide.md",
        "API Reference" => "api.md",
    ],
)

# Post-process HTML files to fix PlotlyJS + Documenter require.js conflict.
# Problem: Documenter loads require.js (AMD), so when the Plotly CDN script loads,
# it detects `define.amd` and registers as an AMD module instead of setting `window.Plotly`.
# Then the embedded plot functions fail with "Plotly is not defined".
# Fix: wrap Plotly CDN loads to temporarily hide `define` so Plotly registers as a global,
# and replace the require.js loader trigger with a direct function call.
for (root, _, files) in walkdir(joinpath(@__DIR__, "build"))
    for file in filter(endswith(".html"), files)
        path = joinpath(root, file)
        content = read(path, String)
        # Wrap Plotly CDN script to hide AMD define so Plotly sets window.Plotly
        content = replace(content,
            "<script src=\"https://cdn.plot.ly/" =>
            "<script>window.__define_tmp=window.define;window.define=undefined;</script><script src=\"https://cdn.plot.ly/")
        # After each Plotly CDN load, restore define
        content = replace(content,
            r"(plotly-[\d.]+\.min\.js\"></script>)" =>
            s"\1<script>window.define=window.__define_tmp;delete window.__define_tmp;</script>")
        # Replace the require.js loader pattern with a direct call
        content = replace(content,
            r"let plotlyloader = window\.document\.createElement\(\"script\"\)\s*let src=\"https://requirejs\.org/docs/release/[\d.]+/minified/require\.js\"\s*plotlyloader\.addEventListener\(\"load\", (plots_jl_plotly_\w+)\);\s*plotlyloader\.src = src\s*document\.querySelector\(\"[^\"]+\"\)\.appendChild\(plotlyloader\)"
            => s"\1()")
        write(path, content)
    end
end

deploydocs(
    repo = "github.com/JuliaGNSS/Acquisition.jl.git",
    devbranch = "master",
)
