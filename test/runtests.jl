using Test, FFTW, Acquisition, GNSSSignals, Random
import Unitful: Hz

include("downconvert.jl")
include("est_signal_noise_power.jl")
include("calc_powers.jl")
include("acquire.jl")