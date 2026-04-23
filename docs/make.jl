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
#
# The three patterns below target Documenter's Plotly-embedding template. We count
# hits per pattern; on CI, if any stops matching (template changed upstream) we
# error out so the silent regression surfaces before broken docs ship.
cdn_wrap_pat  = "<script src=\"https://cdn.plot.ly/"
cdn_after_pat = r"(plotly-[\d.]+\.min\.js\"></script>)"
requirejs_pat = r"let plotlyloader = window\.document\.createElement\(\"script\"\)\s*let src=\"https://requirejs\.org/docs/release/[\d.]+/minified/require\.js\"\s*plotlyloader\.addEventListener\(\"load\", (plots_jl_plotly_\w+)\);\s*plotlyloader\.src = src\s*document\.querySelector\(\"[^\"]+\"\)\.appendChild\(plotlyloader\)"
cdn_wrap_hits = cdn_after_hits = requirejs_hits = 0
for (root, _, files) in walkdir(joinpath(@__DIR__, "build"))
    for file in filter(endswith(".html"), files)
        path = joinpath(root, file)
        content = read(path, String)
        cdn_wrap_hits  += count(cdn_wrap_pat,  content)
        cdn_after_hits += count(cdn_after_pat, content)
        requirejs_hits += count(requirejs_pat, content)
        content = replace(content, cdn_wrap_pat =>
            "<script>window.__define_tmp=window.define;window.define=undefined;</script><script src=\"https://cdn.plot.ly/")
        content = replace(content, cdn_after_pat =>
            s"\1<script>window.define=window.__define_tmp;delete window.__define_tmp;</script>")
        content = replace(content, requirejs_pat => s"\1()")
        write(path, content)
    end
end
# CI guard: if any pattern matched nothing, the Documenter/Plots template likely
# changed — fail loudly. Skipped locally (local builds may not include Plotly).
if get(ENV, "CI", "false") == "true"
    for (name, hits) in (("cdn_wrap", cdn_wrap_hits),
                         ("cdn_after", cdn_after_hits),
                         ("requirejs_loader", requirejs_hits))
        hits == 0 && error(
            "docs/make.jl: Plotly post-process pattern `$name` matched 0 times. " *
            "Documenter or Plots template likely changed — update the pattern.")
    end
end

deploydocs(
    repo = "github.com/JuliaGNSS/Acquisition.jl.git",
    devbranch = "master",
)
