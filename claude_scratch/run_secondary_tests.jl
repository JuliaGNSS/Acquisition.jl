# Run just the secondary_code_search testset.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Test, FFTW, Acquisition, GNSSSignals, Random, Unitful, RecipesBase
import Unitful: Hz

include(joinpath(@__DIR__, "..", "test", "secondary_code_search.jl"))
