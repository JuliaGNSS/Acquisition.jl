module Acquisition

using DocStringExtensions,
    GNSSSignals, RecipesBase, FFTW, Statistics, LinearAlgebra, LoopVectorization, Unitful

import Unitful: s, Hz

using PrecompileTools

export acquire,
    plot_acquisition_results,
    coarse_fine_acquire,
    coarse_fine_acquire!,
    acquire!,
    AcquisitionPlan,
    CoarseFineAcquisitionPlan

struct AcquisitionResults{S<:AbstractGNSS,T}
    system::S
    prn::Int
    sampling_frequency::typeof(1.0Hz)
    carrier_doppler::typeof(1.0Hz)
    code_phase::Float64
    CN0::Float64
    noise_power::T
    power_bins::Matrix{T}
    dopplers::StepRangeLen{
        Float64,
        Base.TwicePrecision{Float64},
        Base.TwicePrecision{Float64},
    }
end

include("plan_acquire.jl")
include("downconvert.jl")
include("plot.jl")
include("calc_powers.jl")
include("est_signal_noise_power.jl")
include("acquire.jl")


@setup_workload begin
    # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
    # precompile file and potentially make loading faster.
    rate = 2.048e6 

    #common real data
    data_i8 = rand(Int8,Int(round(rate*0.01)))
    data_i16 = rand(Int16,Int(round(rate*0.01)))
    data_f32 = rand(Float32,Int(round(rate*0.01)))
    data_f64 = rand(Float64,Int(round(rate*0.01)))


    #common complex data
    data_ci8 = rand(Complex{Int8},Int(round(rate*0.01)))
    data_ci16 = rand(Complex{Int16},Int(round(rate*0.01)))
    data_cf32 = rand(Complex{Float32},Int(round(rate*0.01)))
    data_cf64 = rand(Complex{Float64},Int(round(rate*0.01)))


    @compile_workload begin
        # all calls in this block will be precompiled, regardless of whether
        # they belong to your package or not (on Julia 1.8 and higher)
        #TODO add support for other GNSS constellations
        samplerate = rate*Hz
        acquire(GPSL1(),data_ci8[1:2048], samplerate, 1:32; interm_freq=0*Hz, max_doppler=10e3*Hz);
        acquire(GPSL1(),data_ci16[1:2048], samplerate, 1:32; interm_freq=0*Hz, max_doppler=10e3*Hz);
        acquire(GPSL1(),data_cf32[1:2048], samplerate, 1:32; interm_freq=0*Hz, max_doppler=10e3*Hz);
        acquire(GPSL1(),data_cf64[1:2048], samplerate, 1:32; interm_freq=0*Hz, max_doppler=10e3*Hz);
        acquire(GPSL1(),data_i8[1:2048], samplerate, 1:32; interm_freq=10*Hz, max_doppler=10e3*Hz);
        acquire(GPSL1(),data_i16[1:2048], samplerate, 1:32; interm_freq=10*Hz, max_doppler=10e3*Hz);
        acquire(GPSL1(),data_f32[1:2048], samplerate, 1:32; interm_freq=10*Hz, max_doppler=10e3*Hz);
        acquire(GPSL1(),data_f64[1:2048], samplerate, 1:32; interm_freq=10*Hz, max_doppler=10e3*Hz);
    end
end


 
end
