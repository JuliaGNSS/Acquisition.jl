module Acquisition

using DocStringExtensions,
    GNSSSignals, RecipesBase, FFTW, Statistics, LinearAlgebra, LoopVectorization, Unitful

import Unitful: s, Hz

using PrecompileTools

using ThreadsX

using PrettyTables

export acquire,
    plot_acquisition_results,
    coarse_fine_acquire,
    coarse_fine_acquire!,
    acquire!,
    AcquisitionPlan,
    CoarseFineAcquisitionPlan,
    noncoherent_integrate,
    noncoherent_integrate_manual_timeshift,
    noncoherent_integrate_manual_timeshift_dopplers

struct AcquisitionResults{S<:AbstractGNSS,T}
    system::S
    prn::Int
    sampling_frequency::typeof(1.0Hz)
    carrier_doppler::typeof(1.0Hz)
    code_phase::Float64
    CN0::Float64
    noise_power::T
    power_bins::Matrix{T}
    complex_signal::Matrix{Complex{T}}
    dopplers::StepRangeLen{
        Float64,
        Base.TwicePrecision{Float64},
        Base.TwicePrecision{Float64},
    }
end

function Base.show(io::IO, ::MIME"text/plain", acq_channels::Vector{Acquisition.AcquisitionResults{T1,T2}}) where {T1,T2}
    header = ["PRN"; "CN0"; "Carrier doppler (Hz)"; "Code phase (samples)"]
    data = Matrix{Any}(undef, length(acq_channels),length(header))

    for (idx,acq) in enumerate(acq_channels)
        data[idx,1] = acq.prn
        data[idx,2] = acq.CN0
        data[idx,3] = acq.carrier_doppler
        data[idx,4] = acq.code_phase
    end
    hl_good = Highlighter((data,i,j)->(j==2) &&(data[i,j] > 42),crayon"green")
    hl_bad = Highlighter((data,i,j)->(j==2) &&(data[i,j] < 42),crayon"red")
    
    pretty_table(io,data,header=header,highlighters=(hl_good,hl_bad))
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

    data_ci8 = rand(Complex{Int8},Int(round(rate*0.01)))
    data_ci16 = rand(Complex{Int16},Int(round(rate*0.01)))
    data_cf32 = rand(Complex{Float32},Int(round(rate*0.01)))
    data_cf64 = rand(Complex{Float64},Int(round(rate*0.01)))

    @compile_workload begin
        # all calls in this block will be precompiled, regardless of whether
        # they belong to your package or not (on Julia 1.8 and higher)
        samplerate = rate*Hz
        coarse_fine_acquire(GPSL1(),data_ci8[1:2048], samplerate, 1:32; interm_freq=0*Hz, max_doppler=40e3*Hz);
        coarse_fine_acquire(GPSL1(),data_ci16[1:2048], samplerate, 1:32; interm_freq=0*Hz, max_doppler=40e3*Hz);
        coarse_fine_acquire(GPSL1(),data_cf32[1:2048], samplerate, 1:32; interm_freq=0*Hz, max_doppler=40e3*Hz);
        coarse_fine_acquire(GPSL1(),data_cf64[1:2048], samplerate, 1:32; interm_freq=0*Hz, max_doppler=40e3*Hz);

    end
end


 
end
