module Acquisition

using DocStringExtensions,
    GNSSSignals, RecipesBase, FFTW, Statistics, LinearAlgebra, LoopVectorization, Unitful

import Unitful: s, Hz
using PrettyTables

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
end
