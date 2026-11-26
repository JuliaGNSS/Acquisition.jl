module Acquisition

using GNSSSignals, RecipesBase, FFTL, LinearAlgebra, Unitful, SpecialFunctions, Random

import Unitful: Hz
using Unitful: ustrip
using PrettyTables: pretty_table, TextHighlighter
using CPUSummary: num_cores

include("sign_search_layout.jl")
include("sign_search_null.jl")

export acquire,
    acquire!,
    acquire_stream,
    acquire_stream!,
    plan_acquire,
    AcquisitionPlan,
    AcquisitionResults,
    OppositeRowNoiseEstimator,
    GlobalMeanNoiseEstimator,
    cfar_threshold,
    get_num_cells,
    is_detected

"""
    AcquisitionResults{S,T}

Results from GNSS signal acquisition for a single PRN.

# Fields

  - `signal::S`: GNSS signal acquired
  - `prn::Int`: PRN number of the satellite
  - `sampling_frequency`: Sampling frequency of the signal
  - `carrier_doppler`: Estimated carrier Doppler frequency
  - `code_phase::Float64`: Estimated code phase in chips
  - `secondary_code_phase::Union{Int,Nothing}`: Secondary-code phase index — the
    secondary-code chip the start of the coherent window aligned to, in
    `0:get_secondary_code_length(signal, prn)-1`. Estimated exactly by despreading the
    per-period prompt correlations at the recovered `(carrier_doppler, code_phase)`.
    `nothing` when no rotation search ran (no secondary code for the PRN, as for a BeiDou
    GEO satellite; `use_secondary_code = false`; or
    `num_coherently_integrated_code_periods == 1`) or when the peak does not clear the
    CFAR detection threshold (a non-detected peak has no meaningful secondary phase).
    Useful for seeding a tracking loop's secondary-code alignment.
  - `cn0::Float64`: Carrier-to-noise density ratio in dB-Hz
  - `noise_power::T`: Estimated noise power
  - `peak_to_noise_ratio::T`: Ratio of peak correlation power to estimated noise power
    (`peak_power / noise_power`). Compare against [`cfar_threshold`](@ref) to decide
    if a satellite is present.
  - `num_noncoherent_integrations::Int`: Number of non-coherent integrations performed
  - `power_bins::Union{Matrix{T}, Nothing}`: Correlation power over Doppler × code phase
    (for plotting). Only populated when `acquire!` was called with
    `store_power_bins = true`; `nothing` otherwise — by default the pipeline
    streams the reduction and never materialises this surface.
  - `doppler_bins`: Doppler bins searched (their centre frequencies)
  - `num_blocks::Int`: FM-DBZP number of blocks per code period
  - `block_size::Int`: FM-DBZP samples per block
  - `num_secondary_rotations::Int`: Secondary-code rotation hypotheses per cell (`1` off
    the rotation-search path). Part of the CFAR cell count — see [`get_num_cells`](@ref).
  - `sign_search_null::Union{SignSearchNull,Nothing}`: Null distribution of
    `peak_to_noise_ratio` when each cell is the maximum over several sign hypotheses
    (bit polarities, bit-edge alignments); `nothing` when each cell is a single
    hypothesis. Used by [`cfar_threshold`](@ref)`(result)` and [`is_detected`](@ref).

# Construction

`acquire` and `acquire!` return these. To build one yourself — a test fake, a
detection replayed from a log — use the keyword constructor, which keeps working when
fields are added:

```julia
AcquisitionResults(; signal, prn, sampling_frequency, carrier_doppler, code_phase,
    cn0, noise_power, peak_to_noise_ratio, doppler_bins)
```

The other fields are optional and default to a single-hypothesis search over one code
period: `secondary_code_phase = nothing`, `num_noncoherent_integrations = 1`,
`power_bins = nothing`, `num_blocks = 1`, `block_size` = samples per code period,
`num_secondary_rotations = 1` and `sign_search_null = nothing`.

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
    signal::S
    prn::Int
    sampling_frequency::typeof(1.0Hz)
    carrier_doppler::typeof(1.0Hz)
    code_phase::Float64
    # Secondary-code phase index (which secondary-code chip the start of the
    # coherent integration window aligned to) recovered by the secondary-code
    # rotation search. `nothing` when no rotation search ran — i.e. the PRN
    # has no secondary code, `use_secondary_code = false`, or
    # `num_coherently_integrated_code_periods == 1`. When populated it is an
    # integer in `0:get_secondary_code_length(signal, prn)-1`, suitable for seeding a
    # tracking loop's secondary-code alignment.
    secondary_code_phase::Union{Int, Nothing}
    cn0::Float64
    noise_power::T
    peak_to_noise_ratio::T
    num_noncoherent_integrations::Int
    power_bins::Union{Matrix{T}, Nothing}
    doppler_bins::D
    num_blocks::Int
    block_size::Int
    # Number of secondary-code rotation hypotheses searched per (Doppler, code
    # phase) cell. `1` on every non-rotation path; `get_secondary_code_length(signal, prn)`
    # when the secondary-code rotation search is active. The peak is taken over the
    # expanded `num_doppler_bins × (samples_per_code × num_secondary_rotations)`
    # surface, so this factor is part of the CFAR cell count (see `get_num_cells`).
    num_secondary_rotations::Int
    # Null distribution of `peak_to_noise_ratio` for a cell that is the max over
    # several sign hypotheses (see `SignSearchNull`); `nothing` for one hypothesis.
    sign_search_null::Union{SignSearchNull,Nothing}
end

# Keyword constructor for results not produced by `acquire!` (test fakes,
# replayed detections). The fields that describe the search default to a
# single-hypothesis search over one code period, so a caller names only what
# it cares about and keeps compiling when a field is added.
function AcquisitionResults(;
    signal::AbstractGNSSSignal,
    prn::Integer,
    sampling_frequency,
    carrier_doppler,
    code_phase,
    cn0,
    noise_power,
    peak_to_noise_ratio,
    doppler_bins::AbstractRange,
    secondary_code_phase = nothing,
    num_noncoherent_integrations = 1,
    power_bins = nothing,
    num_blocks = 1,
    block_size = ceil(Int,
        get_code_length(signal) / get_chip_rate(signal) * sampling_frequency) ÷ num_blocks,
    num_secondary_rotations = 1,
    sign_search_null = nothing,
)
    T = promote_type(typeof(noise_power), typeof(peak_to_noise_ratio))
    return AcquisitionResults{typeof(signal),T,typeof(doppler_bins)}(
        signal, prn, sampling_frequency, carrier_doppler, code_phase,
        secondary_code_phase, cn0, noise_power, peak_to_noise_ratio,
        num_noncoherent_integrations, power_bins, doppler_bins, num_blocks, block_size,
        num_secondary_rotations, sign_search_null)
end

"""
    get_num_cells(result::AcquisitionResults) -> Int

Return the number of search cells in the acquisition result. This is the `num_cells`
argument expected by [`cfar_threshold`](@ref).

The base grid is Doppler bins × code phases (`num_blocks * block_size == samples_per_code`).
On the secondary-code rotation-search path the peak is taken over an expanded surface that
carries `num_secondary_rotations` independent code-phase hypotheses per cell, so the count
is multiplied by `num_secondary_rotations` (`1` on every non-rotation path). Omitting this
factor understates the cell count `num_secondary_rotations`-fold, which sets the CFAR
threshold too low and inflates the realised false-alarm rate (issue #70).
"""
get_num_cells(result::AcquisitionResults) =
    length(result.doppler_bins) * result.num_blocks * result.block_size *
    result.num_secondary_rotations

"""
    is_detected(result::AcquisitionResults; pfa=0.01) -> Bool

Return `true` if the satellite signal is detected at the given probability of false alarm.
Compares `result.peak_to_noise_ratio` against [`cfar_threshold`](@ref)`(result; pfa)`,
which uses the number of search cells, the non-coherent integrations and, after a
sign search, the null distribution stored in the result.
"""
is_detected(result::AcquisitionResults; pfa = 0.01) =
    result.peak_to_noise_ratio > cfar_threshold(result; pfa)

function Base.show(io::IO, ::MIME"text/plain", acq::AcquisitionResults)
    print(io, "AcquisitionResults: PRN $(acq.prn), ")
    print(io, "CN0 = $(round(acq.cn0, digits=2)) dB-Hz, ")
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
                acq.cn0,
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

    # Show every column and every channel rather than letting pretty_table crop
    # to the display and emit a "N columns/rows omitted" summary.
    pretty_table(
        io,
        data;
        column_labels = column_labels,
        highlighters = highlighters,
        fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically = false,
    )
end

include("est_signal_noise_power.jl")
include("cfar.jl")
include("plan.jl")
include("doppler_passes.jl")
include("coherent_integration.jl")
include("sign_patterns.jl")
include("noncoherent_integration.jl")
include("acquire.jl")
include("tune_layout.jl")
include("plot.jl")
include("generate_test_signal.jl")
include("precompile.jl")
end
