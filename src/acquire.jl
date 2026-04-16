# src/acquire.jl — public API layer for FM-DBZP acquisition

"""
    _parabolic_interp(left, peak, right) -> Float64

Parabolic interpolation of a discrete peak.  Returns the fractional
offset in bins relative to the peak bin that maximises the quadratic fit.
"""
function _parabolic_interp(left::Real, peak::Real, right::Real)
    denom = 2 * (2 * peak - left - right)
    iszero(denom) && return 0.0
    (right - left) / denom
end

function _acquire_prn!(plan::AcquisitionPlan, scratch, prn::Int, accumulation_step_index::Int)
    prn_idx = findfirst(==(prn), plan.avail_prns)
    prn_fft_matrix = plan.prn_conj_ffts[prn]

    _build_coherent_integration_matrix!(
        scratch.coherent_integration_matrix,
        plan.sig_buf,
        prn_fft_matrix,
        plan.samples_per_code,
        plan.num_blocks,
        plan.block_size,
        plan.num_coherently_integrated_code_periods,
        scratch.double_block_buf,
        scratch.corr_buf,
        plan.double_block_fft_plan,
        plan.double_block_bfft_plan,
    )

    _accumulate_noncoherent_integration_step!(
        plan.noncoherent_integration_matrices[prn_idx],
        scratch.coherent_integration_matrix,
        plan,
        scratch,
        accumulation_step_index,
    )
end

function _acquire_step_threaded!(plan::AcquisitionPlan, prns, accumulation_step_index::Int)
    Threads.@threads for i in eachindex(prns)
        prn = @inbounds prns[i]
        scratch = plan.thread_scratch[Threads.threadid()]
        _acquire_prn!(plan, scratch, prn, accumulation_step_index)
    end
end

function _acquire_step!(plan::AcquisitionPlan, prns, accumulation_step_index::Int)
    if length(prns) > 1 && Threads.nthreads() > 1
        _acquire_step_threaded!(plan, prns, accumulation_step_index)
    else
        scratch = plan.thread_scratch[Threads.threadid()]
        for prn in prns
            _acquire_prn!(plan, scratch, prn, accumulation_step_index)
        end
    end
end

"""
    acquire!(plan::AcquisitionPlan, signal, prns; interm_freq=0.0Hz, subsample_interpolation=false) -> Vector{AcquisitionResults}

Perform FM-DBZP acquisition using a pre-computed [`AcquisitionPlan`](@ref).

Reuses all pre-allocated buffers in `plan`.  Multiple PRNs are processed in a single
pass.

# Arguments

  - `plan`: Pre-computed [`AcquisitionPlan`](@ref) (from [`plan_acquire`](@ref))
  - `signal`: Complex baseband signal samples
  - `prns`: PRN numbers to search (must be a subset of `plan.avail_prns`)

# Keyword Arguments

  - `interm_freq`: Intermediate frequency (default: `0.0Hz`)
  - `subsample_interpolation`: When `true`, apply parabolic interpolation to refine
    the code phase and Doppler estimates below the grid resolution (default: `false`)

# Returns

`Vector{AcquisitionResults}`, one entry per PRN in `prns`.

# See also

[`acquire`](@ref), [`plan_acquire`](@ref)
"""
function acquire!(
    plan::AcquisitionPlan,
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    subsample_interpolation::Bool = false,
    store_power_bins::Bool = false,
)
    all(prn -> prn in plan.avail_prns, prns) ||
        throw(ArgumentError("All requested PRNs must be in plan.avail_prns. Got: $prns, available: $(plan.avail_prns)"))

    segment_length = plan.num_coherently_integrated_code_periods * plan.samples_per_code
    num_segments = length(signal) ÷ segment_length

    num_segments >= plan.num_noncoherent_accumulations ||
        throw(ArgumentError(
            "Signal has $(length(signal)) samples → $num_segments full segments of $segment_length, " *
            "but plan.num_noncoherent_accumulations=$(plan.num_noncoherent_accumulations). " *
            "Provide a longer signal."))

    interm_freq_hz = ustrip(Hz, interm_freq)
    sampling_freq_hz = ustrip(Hz, plan.sampling_freq)

    # Reset per-PRN noncoherent accumulators for the requested PRNs
    for prn in prns
        prn_idx = findfirst(==(prn), plan.avail_prns)
        fill!(plan.noncoherent_integration_matrices[prn_idx], 0f0)
    end

    for step_idx in 1:plan.num_noncoherent_accumulations
        seg_start = (step_idx - 1) * segment_length + 1

        # Downconvert: apply intermediate frequency rotation into sig_buf
        if iszero(interm_freq_hz)
            plan.sig_buf .= ComplexF32.(view(signal, seg_start:seg_start + segment_length - 1))
        else
            phase_step = Float32(-2π * interm_freq_hz / sampling_freq_hz)
            phase_offset = Float32((seg_start - 1) * phase_step)
            @inbounds for sample_idx in 1:segment_length
                phase = phase_offset + (sample_idx - 1) * phase_step
                s, c = sincos(phase)
                plan.sig_buf[sample_idx] = ComplexF32(signal[seg_start + sample_idx - 1]) * Complex(c, s)
            end
        end

        _acquire_step!(plan, prns, step_idx - 1)
    end

    # Build results into pre-allocated buffer (concrete-typed to avoid boxing).
    # resize! to the requested PRN count is non-allocating when shrinking.
    results = resize!(plan.acq_results_buf, length(prns))
    for (result_idx, prn) in enumerate(prns)
        prn_idx = findfirst(==(prn), plan.avail_prns)
        power_bins = plan.noncoherent_integration_matrices[prn_idx]

        signal_power, noise_power, code_bin_idx, doppler_bin_idx = est_signal_noise_power(
            power_bins,
            sampling_freq_hz,
            ustrip(Hz, get_code_frequency(plan.system)),
            nothing,
            plan.col_sums_buf,
        )

        peak_to_noise = (signal_power + noise_power) / noise_power
        code_period = get_code_length(plan.system) / get_code_frequency(plan.system)
        CN0 = 10 * log10(signal_power / noise_power / code_period / 1.0Hz)

        # Decode code phase from column index (0-indexed column → delay in samples)
        scrambled_col = code_bin_idx - 1  # convert to 0-indexed
        delay_samples = _fmdbzp_column_to_tau(scrambled_col, plan.num_blocks, plan.block_size)
        code_freq_hz = ustrip(Hz, get_code_frequency(plan.system))
        code_length = get_code_length(plan.system)
        code_phase = mod(-delay_samples * code_freq_hz / sampling_freq_hz, code_length)

        if subsample_interpolation
            num_code_bins = plan.samples_per_code
            col_left  = mod(scrambled_col - 1, num_code_bins)
            col_right = mod(scrambled_col + 1, num_code_bins)
            power_left  = power_bins[doppler_bin_idx, col_left + 1]
            power_peak  = power_bins[doppler_bin_idx, scrambled_col + 1]
            power_right = power_bins[doppler_bin_idx, col_right + 1]
            if max(power_left, power_right) > sqrt(noise_power)
                fractional_col_offset = _parabolic_interp(power_left, power_peak, power_right)
                delay_samples_interp = delay_samples + fractional_col_offset
                code_phase = mod(-delay_samples_interp * code_freq_hz / sampling_freq_hz, code_length)
            end
        end

        doppler = plan.doppler_freqs[doppler_bin_idx]
        if subsample_interpolation
            num_doppler_bins = length(plan.doppler_freqs)
            dop_left  = power_bins[doppler_bin_idx == 1 ? num_doppler_bins : doppler_bin_idx - 1, code_bin_idx]
            dop_peak  = power_bins[doppler_bin_idx, code_bin_idx]
            dop_right = power_bins[doppler_bin_idx == num_doppler_bins ? 1 : doppler_bin_idx + 1, code_bin_idx]
            if max(dop_left, dop_right) > sqrt(noise_power)
                fractional_doppler_offset = _parabolic_interp(dop_left, dop_peak, dop_right)
                doppler = doppler + fractional_doppler_offset * step(plan.doppler_freqs)
            end
        end

        result_buf = if store_power_bins
            copyto!(plan.result_buffers[prn_idx], power_bins)
        else
            nothing
        end
        results[result_idx] = AcquisitionResults(
            plan.system,
            prn,
            plan.sampling_freq,
            doppler,
            code_phase,
            CN0,
            Float32(noise_power),
            Float32(peak_to_noise),
            plan.num_noncoherent_accumulations,
            result_buf,
            plan.doppler_freqs,
            plan.num_blocks,
            plan.block_size,
        )
    end
    return results
end

"""
    acquire!(plan::AcquisitionPlan, signal, prn::Integer; kwargs...) -> AcquisitionResults

Single-PRN convenience method. Calls the multi-PRN `acquire!` and returns the single result.
"""
function acquire!(
    plan::AcquisitionPlan,
    signal,
    prn::Integer;
    interm_freq = 0.0Hz,
    subsample_interpolation::Bool = false,
    store_power_bins::Bool = false,
)
    only(acquire!(plan, signal, [prn]; interm_freq, subsample_interpolation, store_power_bins))
end

"""
    acquire(system, signal, sampling_freq, prns; kwargs...) -> Vector{AcquisitionResults}

Convenience wrapper: calls [`plan_acquire`](@ref) then [`acquire!`](@ref).

# Arguments

  - `system`: GNSS system (e.g. `GPSL1()`)
  - `signal`: Complex baseband signal samples
  - `sampling_freq`: Sampling frequency
  - `prns`: PRN numbers to search

# Keyword Arguments forwarded to `plan_acquire`:

  - `min_doppler_coverage`: Minimum one-sided Doppler coverage (default: `7000Hz`)
  - `num_coherently_integrated_code_periods`: Code periods per coherent block (default: `1`)
  - `bit_edge_search_steps`: Bit edge search positions (default: `1`)
  - `num_noncoherent_accumulations`: Non-coherent integration steps (default: `1`)
  - `fft_flag`: FFTW planning flag (default: `FFTW.MEASURE`)

# Keyword Arguments forwarded to `acquire!`:

  - `interm_freq`: Intermediate frequency (default: `0.0Hz`)
  - `subsample_interpolation`: Enable parabolic interpolation (default: `false`)

# Returns

`Vector{AcquisitionResults}`, one per PRN.

# See also

[`plan_acquire`](@ref), [`acquire!`](@ref)
"""
function acquire(
    system::AbstractGNSS,
    signal,
    sampling_freq,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    min_doppler_coverage = 7000Hz,
    num_coherently_integrated_code_periods::Int = 1,
    bit_edge_search_steps::Int = 1,
    num_noncoherent_accumulations::Int = 1,
    fft_flag = FFTW.MEASURE,
    subsample_interpolation::Bool = false,
    store_power_bins::Bool = false,
)
    plan = plan_acquire(
        system,
        sampling_freq,
        collect(Int, prns);
        min_doppler_coverage,
        num_coherently_integrated_code_periods,
        bit_edge_search_steps,
        num_noncoherent_accumulations,
        fft_flag,
    )
    acquire!(plan, signal, collect(Int, prns); interm_freq, subsample_interpolation, store_power_bins)
end

"""
    acquire(system, signal, sampling_freq, prn::Integer; kwargs...) -> AcquisitionResults

Single-PRN convenience method. Returns a single [`AcquisitionResults`](@ref).
"""
function acquire(
    system::AbstractGNSS,
    signal,
    sampling_freq,
    prn::Integer;
    kwargs...,
)
    only(acquire(system, signal, sampling_freq, [prn]; kwargs...))
end
