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
  - `samples_to_integrate_coherently`: Number of samples per coherent integration chunk
    (default: `ceil(Int, sampling_freq / get_data_frequency(system))`, i.e. one bit period)
  - `doppler_step_factor`: Factor for computing Doppler step from integration time (default: `1//3`).
    The step is `doppler_step_factor / T` where `T = samples_to_integrate_coherently / sampling_freq`.
  - `dopplers`: Custom Doppler search range (default: computed from `doppler_step_factor`)

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
    samples_to_integrate_coherently = ceil(Int, sampling_freq / get_data_frequency(system)),
    doppler_step_factor = 1//3,
    dopplers = min_doppler:(doppler_step_factor * sampling_freq / samples_to_integrate_coherently):max_doppler,
)
    acq_plan = AcquisitionPlan(
        system,
        min(length(signal), samples_to_integrate_coherently),
        sampling_freq;
        dopplers,
        prns,
        fft_flag = FFTW.MEASURE,
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
plan = AcquisitionPlan(GPSL1(), 10000, 5e6Hz; prns = 1:32)
results = acquire!(plan, signal, 1:32)
```

# See also

[`acquire`](@ref), [`AcquisitionPlan`](@ref)
"""
function acquire!(
    acq_plan::AcquisitionPlan{T},
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
    noise_power = nothing,
) where {T}
    all(prn -> prn in acq_plan.avail_prn_channels, prns) ||
        throw(ArgumentError("You'll need to plan every PRN"))
    S = typeof(acq_plan.system)
    DS = typeof(acq_plan.dopplers)
    isempty(prns) && return AcquisitionResults{S,Float32,DS}[]
    code_period = get_code_length(acq_plan.system) / get_code_frequency(acq_plan.system)

    chunk_samples = acq_plan.num_samples_to_integrate_coherently
    num_signal_samples = length(signal)

    # Process signal in chunks, accumulating powers non-coherently
    # First chunk overwrites (accumulate=false), subsequent chunks add (accumulate=true)
    num_chunks = cld(num_signal_samples, chunk_samples)
    for chunk_idx = 1:num_chunks
        start_idx = (chunk_idx - 1) * chunk_samples + 1
        end_idx = min(chunk_idx * chunk_samples, num_signal_samples)
        signal_chunk = view(signal, start_idx:end_idx)

        power_over_doppler_and_codes!(
            acq_plan,
            signal_chunk,
            prns,
            interm_freq,
            doppler_offset;
            accumulate = chunk_idx > 1,
        )
    end
    powers_per_sats = view(acq_plan.signal_powers, acq_plan.prn_indices)

    # resize! does not allocate when shrinking or staying within original capacity
    resize!(acq_plan.output_results, length(prns))
    for (i, (powers, prn, prn_idx)) in
        enumerate(zip(powers_per_sats, prns, acq_plan.prn_indices))
        signal_power, noise_power_est, code_index, doppler_index = est_signal_noise_power(
            powers,
            acq_plan.sampling_freq,
            get_code_frequency(acq_plan.system),
            noise_power,
        )
        # CN0 includes coherent integration gain (normalized by code_period).
        # Non-coherent integration improves detection probability but doesn't increase
        # the measured SNR ratio, so no explicit gain is added here.
        CN0 = 10 * log10(signal_power / noise_power_est / code_period / 1.0Hz)
        doppler =
            (doppler_index - 1) * step(acq_plan.dopplers) +
            first(acq_plan.dopplers) +
            doppler_offset
        code_phase =
            (code_index - 1) /
            (acq_plan.sampling_freq / get_code_frequency(acq_plan.system))
        result = AcquisitionResults(
            acq_plan.system,
            prn,
            acq_plan.sampling_freq,
            doppler,
            code_phase,
            CN0,
            noise_power_est,
            powers,
            acq_plan.dopplers,
        )
        acq_plan.results[prn_idx] = result
        acq_plan.output_results[i] = result
    end
    return acq_plan.output_results
end

function acquire!(
    acq_plan::CoarseFineAcquisitionPlan,
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
)
    coarse_results = acquire!(acq_plan.coarse_plan, signal, prns; interm_freq)
    fine_plan = acq_plan.fine_plan

    # Build prn_indices for fine plan
    # resize! does not allocate when shrinking or staying within original capacity
    resize!(fine_plan.prn_indices, length(prns))
    for (i, prn) in enumerate(prns)
        fine_plan.prn_indices[i] = findfirst(==(prn), fine_plan.avail_prn_channels)
    end

    code_period = get_code_length(fine_plan.system) / get_code_frequency(fine_plan.system)

    chunk_samples = fine_plan.num_samples_to_integrate_coherently
    num_signal_samples = length(signal)
    num_chunks = cld(num_signal_samples, chunk_samples)

    # resize! does not allocate when shrinking or staying within original capacity
    resize!(fine_plan.output_results, length(prns))
    for (i, (res, prn)) in enumerate(zip(coarse_results, prns))
        prn_idx = fine_plan.prn_indices[i]
        doppler_offset = res.carrier_doppler
        noise_power = res.noise_power

        # Use views with explicit single-element range
        signal_powers_view = view(fine_plan.signal_powers, prn_idx:prn_idx)
        codes_freq_domain_view = view(fine_plan.codes_freq_domain, prn_idx:prn_idx)

        @inbounds for (doppler_idx, doppler) in enumerate(fine_plan.dopplers)
            # Process signal in chunks, accumulating powers non-coherently
            for chunk_idx = 1:num_chunks
                start_idx = (chunk_idx - 1) * chunk_samples + 1
                end_idx = min(chunk_idx * chunk_samples, num_signal_samples)
                signal_chunk = view(signal, start_idx:end_idx)

                power_over_code!(
                    signal_powers_view,
                    doppler_idx,
                    fine_plan.signal_baseband,
                    fine_plan.signal_baseband_freq_domain,
                    fine_plan.code_freq_baseband_freq_domain,
                    fine_plan.code_baseband,
                    signal_chunk,
                    fine_plan.fft_plan,
                    fine_plan.ifft_plan,
                    codes_freq_domain_view,
                    doppler + doppler_offset,
                    fine_plan.sampling_freq,
                    interm_freq;
                    accumulate = chunk_idx > 1,
                )
            end
        end

        # Compute result
        powers = fine_plan.signal_powers[prn_idx]
        signal_power, noise_power_est, code_index, doppler_index = est_signal_noise_power(
            powers,
            fine_plan.sampling_freq,
            get_code_frequency(fine_plan.system),
            noise_power,
        )
        # CN0 includes coherent integration gain (normalized by code_period).
        # Non-coherent integration improves detection probability but doesn't increase
        # the measured SNR ratio, so no explicit gain is added here.
        CN0 = 10 * log10(signal_power / noise_power_est / code_period / 1.0Hz)
        doppler =
            (doppler_index - 1) * step(fine_plan.dopplers) +
            first(fine_plan.dopplers) +
            doppler_offset
        code_phase =
            (code_index - 1) /
            (fine_plan.sampling_freq / get_code_frequency(fine_plan.system))
        result = AcquisitionResults(
            fine_plan.system,
            prn,
            fine_plan.sampling_freq,
            doppler,
            code_phase,
            CN0,
            noise_power_est,
            powers,
            fine_plan.dopplers,
        )
        fine_plan.results[prn_idx] = result
        fine_plan.output_results[i] = result
    end

    return fine_plan.output_results
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
  - `samples_to_integrate_coherently`: Number of samples per coherent integration chunk
    (default: `ceil(Int, sampling_freq / get_data_frequency(system))`, i.e. one bit period)
  - `doppler_step_factor`: Factor for computing Doppler step from integration time (default: `1//3`)
  - `dopplers`: Custom Doppler search range (default: computed from `doppler_step_factor`)

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
    samples_to_integrate_coherently = ceil(Int, sampling_freq / get_data_frequency(system)),
    doppler_step_factor = 1//3,
    dopplers = min_doppler:(doppler_step_factor * sampling_freq / samples_to_integrate_coherently):max_doppler,
)
    only(acquire(system, signal, sampling_freq, [prn]; interm_freq, dopplers, samples_to_integrate_coherently))
end

function acquire!(
    acq_plan::AcquisitionPlan{T,S,DS,CS,P,IP,PS},
    signal,
    prn::Integer;
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
    noise_power = nothing,
)::AcquisitionResults{S,Float32,DS} where {T,S,DS,CS,P,IP,PS}
    only(acquire!(acq_plan, signal, [prn]; interm_freq, doppler_offset, noise_power))
end

function acquire!(
    acq_plan::CoarseFineAcquisitionPlan{C,AcquisitionPlan{T,S,DS,CS,P,IP,PS}},
    signal,
    prn::Integer;
    interm_freq = 0.0Hz,
)::AcquisitionResults{S,Float32,DS} where {C,T,S,DS,CS,P,IP,PS}
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
  - `samples_to_integrate_coherently`: Number of samples per coherent integration chunk
    (default: `ceil(Int, sampling_freq / get_data_frequency(system))`, i.e. one bit period)
  - `doppler_step_factor`: Factor for computing coarse step from integration time (default: `1//3`).
    The coarse step is `doppler_step_factor / T` where `T = samples_to_integrate_coherently / sampling_freq`.
  - `coarse_step`: Doppler step for coarse search (default: computed from `doppler_step_factor`)
  - `fine_step`: Doppler step for fine search (default: `coarse_step / 10`)

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
    samples_to_integrate_coherently = ceil(Int, sampling_freq / get_data_frequency(system)),
    doppler_step_factor = 1//3,
    coarse_step = doppler_step_factor * sampling_freq / samples_to_integrate_coherently,
    fine_step = coarse_step / 10,
)
    acq_plan = CoarseFineAcquisitionPlan(
        system,
        min(length(signal), samples_to_integrate_coherently),
        sampling_freq;
        max_doppler,
        min_doppler,
        coarse_step,
        fine_step,
        prns,
        fft_flag = FFTW.MEASURE,
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
  - `samples_to_integrate_coherently`: Number of samples per coherent integration chunk
    (default: `ceil(Int, sampling_freq / get_data_frequency(system))`, i.e. one bit period)
  - `doppler_step_factor`: Factor for computing coarse step from integration time (default: `1//3`)
  - `coarse_step`: Doppler step for coarse search (default: computed from `doppler_step_factor`)
  - `fine_step`: Doppler step for fine search (default: `coarse_step / 10`)

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
    samples_to_integrate_coherently = ceil(Int, sampling_freq / get_data_frequency(system)),
    doppler_step_factor = 1//3,
    coarse_step = doppler_step_factor * sampling_freq / samples_to_integrate_coherently,
    fine_step = coarse_step / 10,
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
            samples_to_integrate_coherently,
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
plan = CoarseFineAcquisitionPlan(GPSL1(), 10000, 5e6Hz; prns = 1:32)
results = coarse_fine_acquire!(plan, signal, 1:32)
```

# See also

[`coarse_fine_acquire`](@ref), [`CoarseFineAcquisitionPlan`](@ref)
"""
const coarse_fine_acquire! = acquire!