using Test, FFTW, Acquisition, GNSSSignals, Random, Aqua
import Unitful: Hz

@testset "Aqua" begin
    Aqua.test_all(Acquisition; ambiguities=false)
end

include("downconvert.jl")
include("est_signal_noise_power.jl")
include("calc_powers.jl")
include("acquire.jl")
include("ka_acquire.jl")