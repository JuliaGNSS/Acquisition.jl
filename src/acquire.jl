"""
    acquire(system, signal, sampling_freq, prns; kwargs...) -> Vector{AcquisitionResults}

Perform parallel code phase search acquisition for multiple satellites.

Searches for GNSS signals by correlating the input signal with locally generated
replica codes across a grid of Doppler frequencies and code phases.

# Arguments
- `system`: GNSS system (e.g., `GPSL1()` from GNSSSignals.jl)
- `signal`: Complex baseband signal samples
- `sampling_freq`: Sampling frequency of the signal
- `prns`: PRN numbers to search (e.g., `1:32`)

# Keyword Arguments
- `interm_freq`: Intermediate frequency (default: `0.0Hz`)
- `max_doppler`: Maximum Doppler frequency (default: `7000Hz`)
- `min_doppler`: Minimum Doppler frequency (default: `-max_doppler`)
- `dopplers`: Custom Doppler search range (default: `min_doppler:250Hz:max_doppler`)

# Returns
Vector of [`AcquisitionResults`](@ref), one per PRN, containing:
- `carrier_doppler`: Estimated Doppler frequency
- `code_phase`: Estimated code phase in chips
- `CN0`: Carrier-to-noise density ratio in dB-Hz
- `power_bins`: Correlation power matrix for plotting

# Example
```julia
using Acquisition, GNSSSignals
results = acquire(GPSL1(), signal, 5e6Hz, 1:32)
```

# See also
[`acquire!`](@ref), [`coarse_fine_acquire`](@ref), [`AcquisitionPlan`](@ref)
"""
function acquire(
    system::AbstractGNSS,
    signal,
    sampling_freq,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    min_doppler = -max_doppler,
    dopplers = min_doppler:250Hz:max_doppler,
)
    acq_plan = AcquisitionPlan(
        system,
        length(signal),
        sampling_freq;
        dopplers,
        prns,
        fft_flag = FFTW.ESTIMATE,
    )
    acquire!(acq_plan, signal, prns; interm_freq)
end

"""
    acquire!(acq_plan, signal, prns; kwargs...) -> Vector{AcquisitionResults}

Perform acquisition using a pre-computed [`AcquisitionPlan`](@ref).

Using a pre-computed plan avoids repeated memory allocation and FFT planning,
which significantly improves performance when acquiring multiple signals.

# Arguments
- `acq_plan`: Pre-computed [`AcquisitionPlan`](@ref)
- `signal`: Complex baseband signal samples
- `prns`: PRN numbers to search (must be subset of `acq_plan.avail_prn_channels`)

# Keyword Arguments
- `interm_freq`: Intermediate frequency (default: `0.0Hz`)
- `doppler_offset`: Offset added to Doppler search range (default: `0.0Hz`)
- `noise_power`: Pre-computed noise power, or `nothing` to estimate (default: `nothing`)

# Returns
Vector of [`AcquisitionResults`](@ref), one per PRN.

# Example
```julia
using Acquisition, GNSSSignals
plan = AcquisitionPlan(GPSL1(), 10000, 5e6Hz; prns=1:32)
results = acquire!(plan, signal, 1:32)
```

# See also
[`acquire`](@ref), [`AcquisitionPlan`](@ref)
"""
function acquire!(
    acq_plan::AcquisitionPlan,
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
    noise_power = nothing,
)
    all(map(prn -> prn in acq_plan.avail_prn_channels, prns)) ||
        throw(ArgumentError("You'll need to plan every PRN"))
    code_period = get_code_length(acq_plan.system) / get_code_frequency(acq_plan.system)
    powers_per_sats =
        power_over_doppler_and_codes!(acq_plan, signal, prns, interm_freq, doppler_offset)
    map(powers_per_sats, prns) do powers, prn
        signal_power, noise_power, code_index, doppler_index = est_signal_noise_power(
            powers,
            acq_plan.sampling_freq,
            get_code_frequency(acq_plan.system),
            noise_power,
        )
        CN0 = 10 * log10(signal_power / noise_power / code_period / 1.0Hz)
        doppler =
            (doppler_index - 1) * step(acq_plan.dopplers) +
            first(acq_plan.dopplers) +
            doppler_offset
        code_phase =
            (code_index - 1) /
            (acq_plan.sampling_freq / get_code_frequency(acq_plan.system))
        AcquisitionResults(
            acq_plan.system,
            prn,
            acq_plan.sampling_freq,
            doppler,
            code_phase,
            CN0,
            noise_power,
            powers,
            (acq_plan.dopplers .+ doppler_offset) / 1.0Hz,
        )
    end
end

function acquire!(
    acq_plan::CoarseFineAcquisitionPlan,
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
)
    acq_res = acquire!(acq_plan.coarse_plan, signal, prns; interm_freq)
    map(acq_res, prns) do res, prn
        acquire!(
            acq_plan.fine_plan,
            signal,
            prn;
            interm_freq,
            doppler_offset = res.carrier_doppler,
            noise_power = res.noise_power,
        )
    end
end

"""
    acquire(system, signal, sampling_freq, prn::Integer; kwargs...) -> AcquisitionResults

Perform acquisition for a single satellite PRN.

Convenience method that calls the multi-PRN version and returns a single result.

# Arguments
- `system`: GNSS system (e.g., `GPSL1()`)
- `signal`: Complex baseband signal samples
- `sampling_freq`: Sampling frequency of the signal
- `prn`: Single PRN number to search

# Keyword Arguments
- `interm_freq`: Intermediate frequency (default: `0.0Hz`)
- `max_doppler`: Maximum Doppler frequency (default: `7000Hz`)
- `min_doppler`: Minimum Doppler frequency (default: `-max_doppler`)
- `dopplers`: Custom Doppler search range (default: `min_doppler:250Hz:max_doppler`)

# Returns
Single [`AcquisitionResults`](@ref) for the requested PRN.

# Example
```julia
using Acquisition, GNSSSignals
result = acquire(GPSL1(), signal, 5e6Hz, 1)
```

# See also
[`acquire`](@ref), [`coarse_fine_acquire`](@ref)
"""
function acquire(
    system::AbstractGNSS,
    signal,
    sampling_freq,
    prn::Integer;
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    min_doppler = -max_doppler,
    dopplers = min_doppler:250Hz:max_doppler,
)
    only(acquire(system, signal, sampling_freq, [prn]; interm_freq, dopplers))
end

function acquire!(
    acq_plan::AcquisitionPlan,
    signal,
    prn::Integer;
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
    noise_power = nothing,
)
    only(acquire!(acq_plan, signal, [prn]; interm_freq, doppler_offset, noise_power))
end

function acquire!(
    acq_plan::CoarseFineAcquisitionPlan,
    signal,
    prn::Integer;
    interm_freq = 0.0Hz,
)
    only(acquire!(acq_plan, signal, [prn]; interm_freq))
end

"""
    coarse_fine_acquire(system, signal, sampling_freq, prns; kwargs...) -> Vector{AcquisitionResults}

Perform two-stage coarse-fine acquisition for multiple satellites.

First performs a coarse search with large Doppler steps, then refines the estimate
with a fine search around the detected Doppler. This approach provides high Doppler
resolution while reducing computational cost compared to a single high-resolution search.

# Arguments
- `system`: GNSS system (e.g., `GPSL1()`)
- `signal`: Complex baseband signal samples
- `sampling_freq`: Sampling frequency of the signal
- `prns`: PRN numbers to search (e.g., `1:32`)

# Keyword Arguments
- `interm_freq`: Intermediate frequency (default: `0.0Hz`)
- `max_doppler`: Maximum Doppler frequency (default: `7000Hz`)
- `min_doppler`: Minimum Doppler frequency (default: `-max_doppler`)
- `coarse_step`: Doppler step for coarse search (default: `250Hz`)
- `fine_step`: Doppler step for fine search (default: `25Hz`)

# Returns
Vector of [`AcquisitionResults`](@ref), one per PRN, with refined Doppler estimates.

# Example
```julia
using Acquisition, GNSSSignals
results = coarse_fine_acquire(GPSL1(), signal, 5e6Hz, 1:32)
```

# See also
[`acquire`](@ref), [`coarse_fine_acquire!`](@ref), [`CoarseFineAcquisitionPlan`](@ref)
"""
function coarse_fine_acquire(
    system::AbstractGNSS,
    signal,
    sampling_freq,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    min_doppler = -max_doppler,
    coarse_step = 250Hz,
    fine_step = 25Hz,
)
    acq_plan = CoarseFineAcquisitionPlan(
        system,
        length(signal),
        sampling_freq;
        max_doppler,
        min_doppler,
        coarse_step,
        fine_step,
        prns,
        fft_flag = FFTW.ESTIMATE,
    )
    acquire!(acq_plan, signal, prns; interm_freq)
end

"""
    coarse_fine_acquire(system, signal, sampling_freq, prn::Integer; kwargs...) -> AcquisitionResults

Perform two-stage coarse-fine acquisition for a single satellite PRN.

Convenience method that calls the multi-PRN version and returns a single result.

# Arguments
- `system`: GNSS system (e.g., `GPSL1()`)
- `signal`: Complex baseband signal samples
- `sampling_freq`: Sampling frequency of the signal
- `prn`: Single PRN number to search

# Keyword Arguments
- `interm_freq`: Intermediate frequency (default: `0.0Hz`)
- `max_doppler`: Maximum Doppler frequency (default: `7000Hz`)
- `min_doppler`: Minimum Doppler frequency (default: `-max_doppler`)
- `coarse_step`: Doppler step for coarse search (default: `250Hz`)
- `fine_step`: Doppler step for fine search (default: `25Hz`)

# Returns
Single [`AcquisitionResults`](@ref) for the requested PRN.

# Example
```julia
using Acquisition, GNSSSignals
result = coarse_fine_acquire(GPSL1(), signal, 5e6Hz, 1)
```

# See also
[`coarse_fine_acquire`](@ref), [`acquire`](@ref)
"""
function coarse_fine_acquire(
    system::AbstractGNSS,
    signal,
    sampling_freq,
    prn::Integer;
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    min_doppler = -max_doppler,
    coarse_step = 250Hz,
    fine_step = 25Hz,
)
    only(
        coarse_fine_acquire(
            system,
            signal,
            sampling_freq,
            [prn];
            interm_freq,
            max_doppler,
            min_doppler,
            coarse_step,
            fine_step,
        ),
    )
end

"""
    coarse_fine_acquire!(acq_plan::CoarseFineAcquisitionPlan, signal, prns; kwargs...)

Perform two-stage coarse-fine acquisition using a pre-computed plan.

Alias for `acquire!(acq_plan::CoarseFineAcquisitionPlan, ...)`.

# Arguments
- `acq_plan`: Pre-computed [`CoarseFineAcquisitionPlan`](@ref)
- `signal`: Complex baseband signal samples
- `prns`: PRN numbers to search

# Keyword Arguments
- `interm_freq`: Intermediate frequency (default: `0.0Hz`)

# Returns
Vector of [`AcquisitionResults`](@ref), one per PRN.

# Example
```julia
using Acquisition, GNSSSignals
plan = CoarseFineAcquisitionPlan(GPSL1(), 10000, 5e6Hz; prns=1:32)
results = coarse_fine_acquire!(plan, signal, 1:32)
```

# See also
[`coarse_fine_acquire`](@ref), [`CoarseFineAcquisitionPlan`](@ref)
"""
const coarse_fine_acquire! = acquire!