module Acquisition

using DocStringExtensions,
    GNSSSignals, RecipesBase, FFTW, Statistics, LinearAlgebra, LoopVectorization, Unitful

import Unitful: s, Hz
using PrettyTables: pretty_table, TextHighlighter, @crayon_str

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
    column_labels = ["PRN", "CN0 (dBHz)", "Carrier Doppler (Hz)", "Code phase (chips)"]
    data = reduce(vcat, map(acq -> [acq.prn, acq.CN0, acq.carrier_doppler, acq.code_phase]', acq_channels))
    hl_good = TextHighlighter((data,i,j)->(j==2) && (data[i,j] > 42), crayon"green")
    hl_bad = TextHighlighter((data,i,j)->(j==2) && (data[i,j] < 42), crayon"red")

    pretty_table(io, data, column_labels=column_labels, highlighters=[hl_good, hl_bad])
end




include("plan_acquire.jl")
include("downconvert.jl")
include("plot.jl")
include("calc_powers.jl")
include("est_signal_noise_power.jl")
include("acquire.jl")
end
