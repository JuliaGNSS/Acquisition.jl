module Acquisition

using DocStringExtensions,
    GNSSSignals, RecipesBase, FFTW, LinearAlgebra, Unitful,
    SpecialFunctions, Random, Polyester

import Unitful: Hz
using Unitful: ustrip
using PrettyTables: pretty_table, TextHighlighter
using CPUSummary: num_cores

export acquire,
    acquire!,
    plan_acquire,
    AcquisitionPlan,
    AcquisitionResults,
    AbstractNoiseEstimator,
    OppositeRowNoiseEstimator,
    GlobalMeanNoiseEstimator,
    cfar_threshold,
    get_num_cells,
    is_detected,
    generate_test_signal,
    recommend_sampling_freqs,
    SamplingFreqRecommendation,
    AbstractSDRClockPlan,
    AD9361ClockPlan,
    is_valid_sample_rate,
    sample_rate_range

"""
    AcquisitionResults{S,T}

Results from GNSS signal acquisition for a single PRN.

# Fields

  - `system::S`: GNSS system used for acquisition
  - `prn::Int`: PRN number of the satellite
  - `sampling_frequency`: Sampling frequency of the signal
  - `carrier_doppler`: Estimated carrier Doppler frequency
  - `code_phase::Float64`: Estimated code phase in chips
  - `secondary_code_phase::Union{Int,Nothing}`: Secondary-code phase index — the
    secondary-code chip the start of the coherent window aligned to, in
    `0:get_secondary_code_length(system)-1`. Estimated exactly by despreading the
    per-period prompt correlations at the recovered `(carrier_doppler, code_phase)`.
    `nothing` when no rotation search ran (no secondary code, `use_secondary_code = false`,
    or `num_coherently_integrated_code_periods == 1`) or when the peak does not clear the
    CFAR detection threshold (a non-detected peak has no meaningful secondary phase).
    Useful for seeding a tracking loop's secondary-code alignment.
  - `CN0::Float64`: Carrier-to-noise density ratio in dB-Hz
  - `noise_power::T`: Estimated noise power
  - `peak_to_noise_ratio::T`: Ratio of peak correlation power to estimated noise power
    (`peak_power / noise_power`). Compare against [`cfar_threshold`](@ref) to decide
    if a satellite is present.
  - `num_noncoherent_integrations::Int`: Number of non-coherent integrations performed
  - `power_bins::Matrix{T}`: Correlation power over Doppler × code phase (for plotting)
  - `dopplers`: Doppler frequencies searched
  - `num_blocks::Int`: FM-DBZP number of blocks per code period
  - `block_size::Int`: FM-DBZP samples per block

# Plotting

`AcquisitionResults` can be plotted directly using Plots.jl:

```julia
using Plots
plot(result)  # heatmap of correlation power (chip-axis sorted)
plot(result, true)  # Use log scale (dB)
```

# See also

[`acquire`](@ref), [`plan_acquire`](@ref)
"""
struct AcquisitionResults{S<:AbstractGNSSSignal,T,D<:AbstractRange}
    system::S
    prn::Int
    sampling_frequency::typeof(1.0Hz)
    carrier_doppler::typeof(1.0Hz)
    code_phase::Float64
    # Secondary-code phase index (which secondary-code chip the start of the
    # coherent integration window aligned to) recovered by the secondary-code
    # rotation search. `nothing` when no rotation search ran — i.e. the signal
    # has no secondary code, `use_secondary_code = false`, or
    # `num_coherently_integrated_code_periods == 1`. When populated it is an
    # integer in `0:get_secondary_code_length(system)-1`, suitable for seeding a
    # tracking loop's secondary-code alignment.
    secondary_code_phase::Union{Int, Nothing}
    CN0::Float64
    noise_power::T
    peak_to_noise_ratio::T
    num_noncoherent_integrations::Int
    power_bins::Union{Matrix{T}, Nothing}
    dopplers::D
    num_blocks::Int
    block_size::Int
end

"""
    get_num_cells(result::AcquisitionResults) -> Int

Return the number of search cells (Doppler bins × code phases) in the acquisition
result. This is the `num_cells` argument expected by [`cfar_threshold`](@ref).
"""
get_num_cells(result::AcquisitionResults) = length(result.dopplers) * result.num_blocks * result.block_size

"""
    is_detected(result::AcquisitionResults; pfa=0.01) -> Bool

Return `true` if the satellite signal is detected at the given probability of false alarm.
Compares `result.peak_to_noise_ratio` against [`cfar_threshold`](@ref), using the
number of search cells and non-coherent integrations stored in the result.
"""
function is_detected(result::AcquisitionResults; pfa = 0.01)
    threshold = cfar_threshold(
        pfa,
        get_num_cells(result);
        num_noncoherent_integrations = result.num_noncoherent_integrations,
    )
    result.peak_to_noise_ratio > threshold
end

function Base.show(io::IO, ::MIME"text/plain", acq::AcquisitionResults)
    print(io, "AcquisitionResults: PRN $(acq.prn), ")
    print(io, "CN0 = $(round(acq.CN0, digits=2)) dB-Hz, ")
    print(io, "Doppler = $(acq.carrier_doppler), ")
    print(io, "Code phase = $(round(acq.code_phase, digits=3)) chips")
end

function Base.show(
    io::IO,
    ::MIME"text/plain",
    acq_channels::Vector{<:Acquisition.AcquisitionResults},
)
    column_labels = ["PRN", "CN0 (dBHz)", "Carrier Doppler (Hz)", "Code phase (chips)"]
    detected = map(is_detected, acq_channels)
    data = reduce(
        vcat,
        map(
            acq -> permutedims([
                acq.prn,
                acq.CN0,
                acq.carrier_doppler,
                acq.code_phase,
            ]),
            acq_channels,
        ),
    )

    use_color = get(io, :color, false)
    highlighters = use_color ? TextHighlighter[
        TextHighlighter((_, i, j) -> j == 2 && detected[i], foreground = :green),
        TextHighlighter((_, i, j) -> j == 2 && !detected[i], foreground = :red),
    ] : TextHighlighter[]

    pretty_table(io, data; column_labels = column_labels, highlighters = highlighters)
end

include("est_signal_noise_power.jl")
include("cfar.jl")
include("plan.jl")
include("coherent_integration.jl")
include("sign_patterns.jl")
include("noncoherent_integration.jl")
include("acquire.jl")
include("plot.jl")
include("generate_test_signal.jl")
include("sdr_clock_plans.jl")
include("recommend_sampling_freq.jl")
end
