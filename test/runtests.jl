using Test, FFTW, Acquisition, GNSSSignals, Random, Aqua, Unitful, RecipesBase
import Unitful: Hz

@testset "Aqua" begin
    Aqua.test_all(Acquisition)
end

include("est_signal_noise_power.jl")
include("cfar.jl")

@testset "FM-DBZP" begin
    include("plan.jl")
    include("coherent_integration.jl")
    include("noncoherent_integration.jl")
    include("acquire.jl")
    include("plot.jl")
end
