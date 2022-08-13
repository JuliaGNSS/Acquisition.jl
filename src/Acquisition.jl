module Acquisition

    using DocStringExtensions, GNSSSignals, RecipesBase, FFTW, Statistics, LinearAlgebra, LoopVectorization, Unitful
    import Unitful: s, Hz

    export acquire, plot_acquisition_results, coarse_fine_acquire

    struct AcquisitionResults{S<:AbstractGNSS, T}
        system::S
        prn::Int
        sampling_frequency::typeof(1.0Hz)
        carrier_doppler::typeof(1.0Hz)
        code_phase::Float64
        CN0::Float64
        power_bins::Array{T, 2}
        dopplers::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}
    end

    include("downconvert.jl")
    include("plot.jl")
    include("calc_powers.jl")
    include("est_signal_noise_power.jl")
    include("acquire.jl")
end
