# Run the package test suite excluding Aqua (which has a dep missing in the env).
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Test
const RUN_AQUA = get(ENV, "RUN_AQUA", "0") == "1"
# We mimic test/runtests.jl but skip Aqua. The remaining 454+ tests must pass.
using FFTW, Acquisition, GNSSSignals, Random, Unitful, RecipesBase
import Unitful: Hz

@testset "All (Aqua skipped)" begin
    @testset "est_signal_noise_power" begin
        include(joinpath(@__DIR__, "..", "test", "est_signal_noise_power.jl"))
    end
    @testset "cfar" begin
        include(joinpath(@__DIR__, "..", "test", "cfar.jl"))
    end
    @testset "sign_patterns" begin
        include(joinpath(@__DIR__, "..", "test", "sign_patterns.jl"))
    end
    @testset "secondary_code_search" begin
        include(joinpath(@__DIR__, "..", "test", "secondary_code_search.jl"))
    end
    @testset "FM-DBZP" begin
        include(joinpath(@__DIR__, "..", "test", "plan.jl"))
        include(joinpath(@__DIR__, "..", "test", "coherent_integration.jl"))
        include(joinpath(@__DIR__, "..", "test", "noncoherent_integration.jl"))
        include(joinpath(@__DIR__, "..", "test", "acquire.jl"))
        include(joinpath(@__DIR__, "..", "test", "plot.jl"))
    end
    @testset "recommend_sampling_freq" begin
        include(joinpath(@__DIR__, "..", "test", "recommend_sampling_freq.jl"))
    end
end
