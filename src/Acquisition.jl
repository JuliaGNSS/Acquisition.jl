module Acquisition

using DocStringExtensions,
    GNSSSignals, RecipesBase, FFTW, Statistics, LinearAlgebra, LoopVectorization, Unitful

import Unitful: s, Hz
using Unitful: ustrip
using AbstractFFTs: fft
using Scratch: @get_scratch!
using PrettyTables: pretty_table, AnsiTextCell

export acquire,
    coarse_fine_acquire,
    coarse_fine_acquire!,
    acquire!,
    AcquisitionPlan,
    CoarseFineAcquisitionPlan,
    AcquisitionResults,
    KAAcquisitionPlan

"""
    AcquisitionResults{S,T}

Results from GNSS signal acquisition for a single PRN.

# Fields
- `system::S`: GNSS system used for acquisition
- `prn::Int`: PRN number of the satellite
- `sampling_frequency`: Sampling frequency of the signal
- `carrier_doppler`: Estimated carrier Doppler frequency
- `code_phase::Float64`: Estimated code phase in chips
- `CN0::Float64`: Carrier-to-noise density ratio in dB-Hz
- `noise_power::T`: Estimated noise power
- `power_bins::Matrix{T}`: Correlation power over code phase and Doppler (for plotting)
- `dopplers`: Doppler frequencies searched

# Plotting
`AcquisitionResults` can be plotted directly using Plots.jl:
```julia
using Plots
plot(result)  # 3D surface plot of correlation power
plot(result, true)  # Use log scale (dB)
```

# See also
[`acquire`](@ref), [`coarse_fine_acquire`](@ref)
"""
struct AcquisitionResults{S<:AbstractGNSS,T,D<:AbstractRange}
    system::S
    prn::Int
    sampling_frequency::typeof(1.0Hz)
    carrier_doppler::typeof(1.0Hz)
    code_phase::Float64
    CN0::Float64
    noise_power::T
    power_bins::Matrix{T}
    dopplers::D
end

function Base.show(io::IO, ::MIME"text/plain", acq::AcquisitionResults)
    print(io, "AcquisitionResults: PRN $(acq.prn), ")
    print(io, "CN0 = $(round(acq.CN0, digits=2)) dB-Hz, ")
    print(io, "Doppler = $(acq.carrier_doppler), ")
    print(io, "Code phase = $(round(acq.code_phase, digits=3)) chips")
end

function _format_cn0(cn0, use_color::Bool)
    # TODO: Remove trailing space workaround once PrettyTables.jl PR #289 is merged
    # (https://github.com/ronisbr/PrettyTables.jl/pull/289)
    # The trailing space prevents PrettyTables from stripping the reset code due to
    # a precompilation bug with Crayons.jl.
    if !use_color
        return cn0
    elseif cn0 > 42
        return AnsiTextCell("\e[32m$(cn0)\e[0m ")
    elseif cn0 < 42
        return AnsiTextCell("\e[31m$(cn0)\e[0m ")
    else
        return cn0
    end
end

function Base.show(io::IO, ::MIME"text/plain", acq_channels::Vector{<:Acquisition.AcquisitionResults})
    column_labels = ["PRN", "CN0 (dBHz)", "Carrier Doppler (Hz)", "Code phase (chips)"]
    use_color = get(io, :color, false)
    data = reduce(vcat, map(acq -> permutedims([acq.prn, _format_cn0(acq.CN0, use_color), acq.carrier_doppler, acq.code_phase]), acq_channels))

    pretty_table(io, data; column_labels=column_labels)
end




include("plan_acquire.jl")
include("downconvert.jl")
include("plot.jl")
include("calc_powers.jl")
include("est_signal_noise_power.jl")
include("acquire.jl")
include("ka_acquire.jl")
end
