using Test, FFTL, Acquisition, GNSSSignals, Random, Aqua, Unitful, RecipesBase
import Unitful: Hz

# Each new plan shape would otherwise compile its plan-dependent kernels in
# `plan_acquire` (see `_precompile_search`); the suite builds dozens of shapes.
# test/plan.jl re-enables it for the test that covers it.
Acquisition._PRECOMPILE_SEARCH[] = false

@testset "Aqua" begin
    Aqua.test_all(Acquisition)
end

include("est_signal_noise_power.jl")
include("cfar.jl")
include("sign_patterns.jl")
include("secondary_code_search.jl")

@testset "FM-DBZP" begin
    include("plan.jl")
    include("coherent_integration.jl")
    include("noncoherent_integration.jl")
    include("acquire.jl")
    include("doppler_passes.jl")
    include("code_doppler.jl")
    include("tune_layout.jl")
    include("plot.jl")
end

include("recommend_sampling_freq.jl")
