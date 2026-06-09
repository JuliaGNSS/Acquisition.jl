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

    # plan.signal_block_ffts is filled by _precompute_signal_block_ffts! in acquire!
    # before this loop fires; we read from it here.
    _build_coherent_integration_matrix!(
        scratch.coherent_integration_matrix,
        plan.signal_block_ffts,
        prn_fft_matrix,
        plan.samples_per_code,
        plan.num_blocks,
        plan.block_size,
        plan.num_coherently_integrated_code_periods,
        scratch.corr_buf,
        plan.double_block_bfft_plan,
    )

    _accumulate_noncoherent_integration_step!(
        plan.noncoherent_integration_matrices[prn_idx],
        scratch.coherent_integration_matrix,
        plan,
        scratch,
        prn,
        accumulation_step_index,
    )
end

function _acquire_step_threaded!(plan::AcquisitionPlan, prns, accumulation_step_index::Int)
    @batch per=core for i in eachindex(prns)
        prn = @inbounds prns[i]
        scratch, slot = _claim_scratch!(plan)
        try
            _acquire_prn!(plan, scratch, prn, accumulation_step_index)
        finally
            _release_scratch!(plan, slot)
        end
    end
end

function _acquire_step!(plan::AcquisitionPlan, prns, accumulation_step_index::Int)
    if length(prns) > 1 && Threads.nthreads() > 1
        _acquire_step_threaded!(plan, prns, accumulation_step_index)
    else
        # Single-threaded: one scratch serves the whole PRN loop.
        scratch, slot = _claim_scratch!(plan)
        try
            for prn in prns
                _acquire_prn!(plan, scratch, prn, accumulation_step_index)
            end
        finally
            _release_scratch!(plan, slot)
        end
    end
end

"""
    acquire!(plan::AcquisitionPlan, signal, prns; interm_freq=0.0Hz, subsample_interpolation=false, store_power_bins=false) -> Vector{AcquisitionResults}

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
  - `store_power_bins`: When `true`, copy the full Doppler × code-phase correlation
    power matrix into each result's `power_bins` field (required for plotting).
    When `false`, `power_bins` is `nothing` and no extra copy is made (default: `false`)

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

    # The N_nc==1 case fuses build → accumulate → extract per PRN against a
    # single per-thread accumulator, dropping the 32-PRN-wide noncoherent matrix
    # vector. The multistep path (N_nc>1) still needs that vector because the
    # accumulator carries state across signal segments shared between PRNs.
    if plan.num_noncoherent_accumulations == 1
        _acquire_sequential!(plan, signal, prns, segment_length, interm_freq_hz,
            sampling_freq_hz, subsample_interpolation, store_power_bins)
    else
        _acquire_multistep!(plan, signal, prns, segment_length, interm_freq_hz,
            sampling_freq_hz, subsample_interpolation, store_power_bins)
    end
end

# Fill `sig_buf` with one downconverted code segment starting at `seg_start`.
function _downconvert!(sig_buf, signal, seg_start, segment_length, interm_freq_hz, sampling_freq_hz)
    if iszero(interm_freq_hz)
        sig_buf .= ComplexF32.(view(signal, seg_start:seg_start + segment_length - 1))
    else
        phase_step = Float32(-2π * interm_freq_hz / sampling_freq_hz)
        phase_offset = Float32((seg_start - 1) * phase_step)
        @inbounds for sample_idx in 1:segment_length
            phase = phase_offset + (sample_idx - 1) * phase_step
            s, c = sincos(phase)
            sig_buf[sample_idx] = ComplexF32(signal[seg_start + sample_idx - 1]) * Complex(c, s)
        end
    end
end

# Multistep path (current pre-slice-4 behaviour, factored out unchanged).
function _acquire_multistep!(plan, signal, prns, segment_length, interm_freq_hz,
                              sampling_freq_hz, subsample_interpolation, store_power_bins)
    # Reset per-PRN noncoherent accumulators for the requested PRNs.
    for prn in prns
        prn_idx = findfirst(==(prn), plan.avail_prns)
        fill!(plan.noncoherent_integration_matrices[prn_idx], 0f0)
    end

    # The dwell-level scratch (downconverted segment + per-FFT temp) lives in
    # thread 1's slot — `_default_scratch(plan)` names that convention.
    main_scratch = _default_scratch(plan)
    sig_buf = main_scratch.sig_buf

    for step_idx in 1:plan.num_noncoherent_accumulations
        seg_start = (step_idx - 1) * segment_length + 1
        _downconvert!(sig_buf, signal, seg_start, segment_length, interm_freq_hz, sampling_freq_hz)
        # Precompute signal-block FFTs once per dwell. They do not depend on
        # PRN, so the per-PRN inner loop reads from `plan.signal_block_ffts`
        # instead of redoing this O(num_coh*num_blocks) FFT batch per PRN.
        _precompute_signal_block_ffts!(
            plan.signal_block_ffts,
            sig_buf,
            plan.samples_per_code,
            plan.num_blocks,
            plan.block_size,
            plan.num_coherently_integrated_code_periods,
            main_scratch.double_block_buf,
            plan.double_block_fft_plan,
        )
        _acquire_step!(plan, prns, step_idx - 1)
    end

    results = resize!(plan.acq_results_buf, length(prns))
    code_freq_hz = ustrip(Hz, get_code_frequency(plan.system))
    code_length = get_code_length(plan.system)
    code_period = code_length / get_code_frequency(plan.system)
    num_doppler_bins = length(plan.doppler_freqs)
    doppler_step = step(plan.doppler_freqs)
    @batch per=core for result_idx in eachindex(prns)
        prn = @inbounds prns[result_idx]
        prn_idx = findfirst(==(prn), plan.avail_prns)
        power_bins = plan.noncoherent_integration_matrices[prn_idx]
        scratch, slot = _claim_scratch!(plan)
        try
            results[result_idx] = _extract_result!(plan, scratch, prn, prn_idx, power_bins,
                signal, interm_freq_hz,
                sampling_freq_hz, code_freq_hz, code_length, code_period,
                num_doppler_bins, doppler_step, subsample_interpolation, store_power_bins)
        finally
            _release_scratch!(plan, slot)
        end
    end
    return results
end

# Sequential path used when num_noncoherent_accumulations == 1: one segment,
# per-PRN fused build → accumulate → extract against a per-thread accumulator.
function _acquire_sequential!(plan, signal, prns, segment_length, interm_freq_hz,
                               sampling_freq_hz, subsample_interpolation, store_power_bins)
    main_scratch = _default_scratch(plan)
    sig_buf = main_scratch.sig_buf

    _downconvert!(sig_buf, signal, 1, segment_length, interm_freq_hz, sampling_freq_hz)
    _precompute_signal_block_ffts!(
        plan.signal_block_ffts,
        sig_buf,
        plan.samples_per_code,
        plan.num_blocks,
        plan.block_size,
        plan.num_coherently_integrated_code_periods,
        main_scratch.double_block_buf,
        plan.double_block_fft_plan,
    )

    results = resize!(plan.acq_results_buf, length(prns))
    code_freq_hz = ustrip(Hz, get_code_frequency(plan.system))
    code_length = get_code_length(plan.system)
    code_period = code_length / get_code_frequency(plan.system)
    num_doppler_bins = length(plan.doppler_freqs)
    doppler_step = step(plan.doppler_freqs)

    # When the simple/pilot routing applies, fuse the column FFT + |x|² + fftshift
    # row permutation into one pass directly into the accumulator — see
    # `_accumulate_fftshifted_power_pilot!`. Otherwise (bit edge / data bit search
    # or secondary-code rotation search) fall back to the unfused
    # `_accumulate_noncoherent_integration_step!`.
    rotation_search_active = plan.num_secondary_rotations > 1 &&
                             plan.num_coherently_integrated_code_periods > 1
    simple_path = plan.num_data_bits == 1 && plan.bit_edge_search_steps == 1 &&
                  !rotation_search_active
    @batch per=core for result_idx in eachindex(prns)
        prn = @inbounds prns[result_idx]
        prn_idx = findfirst(==(prn), plan.avail_prns)
        scratch, slot = _claim_scratch!(plan)
        try
            accumulator = scratch.noncoherent_integration_accumulator
            fill!(accumulator, 0f0)

            _build_coherent_integration_matrix!(
                scratch.coherent_integration_matrix,
                plan.signal_block_ffts,
                plan.prn_conj_ffts[prn],
                plan.samples_per_code,
                plan.num_blocks,
                plan.block_size,
                plan.num_coherently_integrated_code_periods,
                scratch.corr_buf,
                plan.double_block_bfft_plan,
            )
            if simple_path
                if num_doppler_bins <= BATCH_FFT_THRESHOLD
                    _accumulate_fftshifted_power_pilot_batched!(
                        accumulator, scratch.coherent_integration_matrix,
                        plan.col_batch_fft_plan, plan.samples_per_code, num_doppler_bins)
                else
                    _accumulate_fftshifted_power_pilot!(
                        accumulator, scratch.coherent_integration_matrix,
                        scratch.col_buf, plan.col_fft_plan,
                        plan.samples_per_code, num_doppler_bins)
                end
            else
                _accumulate_noncoherent_integration_step!(accumulator, scratch.coherent_integration_matrix,
                    plan, scratch, prn, 0)
            end

            results[result_idx] = _extract_result!(plan, scratch, prn, prn_idx, accumulator,
                signal, interm_freq_hz,
                sampling_freq_hz, code_freq_hz, code_length, code_period,
                num_doppler_bins, doppler_step, subsample_interpolation, store_power_bins)
        finally
            _release_scratch!(plan, slot)
        end
    end
    return results
end

# Exact secondary-code phase estimate. The FM-DBZP rotation index (`rotation_block`)
# is ±1 at worst-case code phases — the chip and the sub-chip code phase entangle in
# the factored search. So instead, with the peak's `(doppler, code_phase)` in hand,
# we despread the first `L` per-coherent-period prompt correlations against each of the
# `L` secondary-code rotations and take the strongest. A full secondary-code period of
# coherent despread is the unambiguous case (bounded periodic autocorrelation), so this
# is exact at every code phase. `L = num_secondary_rotations` and only the first `L`
# periods are used, so an unknown data-bit flip at a symbol boundary (N > L) never
# enters — the NH-phase is constant across symbols anyway (N is a multiple of L).
#
# Carrier (interm + doppler) wipe-off uses an incremental complex phasor reset each
# period (one `sincos` per period, not per sample) to stay cheap and bound Float32
# drift to within a primary-code period. The primary-code reference is generated once
# per call; for signals whose `gen_code` bakes the secondary code (e.g. GPS L5I) it
# carries a single constant chip, a global ±1 that cancels in the magnitude `argmax`.
function _estimate_secondary_code_phase(plan, prn, signal, interm_freq_hz,
                                        sampling_freq_hz, code_phase, doppler_hz)
    system = plan.system
    L = plan.num_secondary_rotations
    spc = plan.samples_per_code
    sec = get_secondary_code(system)
    code = gen_code(spc, system, prn, plan.sampling_freq, get_code_frequency(system), code_phase)
    phase_step = -2.0 * π * (interm_freq_hz + doppler_hz) / sampling_freq_hz
    dphi = ComplexF32(cos(phase_step), sin(phase_step))   # per-sample carrier rotation
    zs = Vector{ComplexF32}(undef, L)
    @inbounds for k in 0:L-1
        base = k * spc
        s0, c0 = sincos(phase_step * base)                # reset phasor each period
        carr = ComplexF32(c0, s0)
        acc = zero(ComplexF32)
        for n in 0:spc-1
            acc += ComplexF32(signal[base + n + 1]) * carr * ComplexF32(code[n + 1])
            carr *= dphi
        end
        zs[k + 1] = acc
    end
    best_r = 0
    best_mag = -1.0f0
    @inbounds for r in 0:L-1
        s = zero(ComplexF32)
        for k in 0:L-1
            s += Float32(GNSSSignals.secondary_value(sec, prn, mod(k + r, L))) * zs[k + 1]
        end
        mag = abs2(s)
        if mag > best_mag
            best_mag = mag
            best_r = r
        end
    end
    return best_r
end

# Read the peak out of `power_bins` and assemble an AcquisitionResults. Used by
# both the sequential and multistep paths; in the sequential path `power_bins`
# is the per-thread accumulator (and is reused by the next PRN), so the
# store_power_bins copy must happen here before this function returns.
function _extract_result!(plan, scratch, prn, prn_idx, power_bins, signal, interm_freq_hz,
                          sampling_freq_hz, code_freq_hz, code_length, code_period,
                          num_doppler_bins, doppler_step, subsample_interpolation, store_power_bins)
    col_sums_buf = scratch.col_sums_buf

    signal_power, noise_power, code_bin_idx, doppler_bin_idx = est_signal_noise_power(
        power_bins,
        sampling_freq_hz,
        code_freq_hz,
        col_sums_buf,
        plan.noise_estimator,
    )

    peak_to_noise = (signal_power + noise_power) / noise_power
    CN0 = 10 * log10(signal_power / noise_power / code_period / 1.0Hz)

    # On the rotation path the cp axis is expanded to
    # `samples_per_code * num_secondary_rotations` and the peak col encodes
    # `(cp_within, rotation_idx)` together. We decode `cp_within` for the public
    # `code_phase`; the secondary-code phase is recovered exactly further below
    # via `_estimate_secondary_code_phase` (the raw `rotation_idx` is ±1 at
    # worst-case code phases, so it is not used for the public field).
    samples_per_code = plan.samples_per_code
    rotation_block = (code_bin_idx - 1) ÷ samples_per_code
    scrambled_col = (code_bin_idx - 1) % samples_per_code
    delay_samples = _fmdbzp_column_to_tau(scrambled_col, plan.num_blocks, plan.block_size)
    code_phase = mod(-delay_samples * code_freq_hz / sampling_freq_hz, code_length)

    if subsample_interpolation
        # Neighbouring columns for parabolic interpolation must stay within the
        # SAME rotation block (different blocks correspond to different NH10
        # hypotheses and aren't physically adjacent in cp).
        num_code_bins = samples_per_code
        col_left  = mod(scrambled_col - 1, num_code_bins) + rotation_block * samples_per_code
        col_right = mod(scrambled_col + 1, num_code_bins) + rotation_block * samples_per_code
        power_left  = power_bins[doppler_bin_idx, col_left + 1]
        power_peak  = power_bins[doppler_bin_idx, code_bin_idx]
        power_right = power_bins[doppler_bin_idx, col_right + 1]
        if max(power_left, power_right) > sqrt(noise_power)
            fractional_col_offset = _parabolic_interp(power_left, power_peak, power_right)
            delay_samples_interp = delay_samples + fractional_col_offset
            code_phase = mod(-delay_samples_interp * code_freq_hz / sampling_freq_hz, code_length)
        end
    end

    doppler = plan.doppler_freqs[doppler_bin_idx]
    if subsample_interpolation
        dop_left  = power_bins[doppler_bin_idx == 1 ? num_doppler_bins : doppler_bin_idx - 1, code_bin_idx]
        dop_peak  = power_bins[doppler_bin_idx, code_bin_idx]
        dop_right = power_bins[doppler_bin_idx == num_doppler_bins ? 1 : doppler_bin_idx + 1, code_bin_idx]
        if max(dop_left, dop_right) > sqrt(noise_power)
            fractional_doppler_offset = _parabolic_interp(dop_left, dop_peak, dop_right)
            doppler = doppler + fractional_doppler_offset * doppler_step
        end
    end

    # Exact secondary-code phase — only on the rotation path AND only when the peak
    # clears the CFAR detection threshold. A secondary phase for a non-detected PRN is
    # meaningless (it would despread noise), so it stays `nothing`; this also means
    # absent PRNs in a wide search pay nothing for the estimator (it's the few detected
    # PRNs that run it). The gate uses the same default pfa as [`is_detected`](@ref).
    secondary_code_phase = if plan.num_secondary_rotations > 1
        # True searched-cell count INCLUDING the rotation expansion: the rotation
        # path's power_bins is `num_doppler_bins × (samples_per_code ×
        # num_secondary_rotations)`. Omitting the `× num_secondary_rotations`
        # understates the cell count L-fold, dropping the CFAR threshold enough
        # that pure-noise peaks clear it and the estimator fires on absent PRNs —
        # that was the AcquireSignals/L5I regression. (`num_blocks × block_size ==
        # samples_per_code`.) This matches `get_num_cells` of the result built below
        # — the public `is_detected` undercounted the same way until issue #70.
        num_cells = num_doppler_bins * plan.samples_per_code * plan.num_secondary_rotations
        threshold = cfar_threshold(0.01, num_cells;
            num_noncoherent_integrations = plan.num_noncoherent_accumulations)
        if peak_to_noise > threshold
            _estimate_secondary_code_phase(plan, prn, signal, interm_freq_hz,
                sampling_freq_hz, code_phase, ustrip(Hz, doppler))
        else
            nothing
        end
    else
        nothing
    end

    result_buf = if store_power_bins
        cached = plan.result_buffers[prn_idx]
        buf = cached === nothing ? similar(power_bins) : cached
        plan.result_buffers[prn_idx] = buf
        copyto!(buf, power_bins)
    else
        nothing
    end

    return AcquisitionResults(
        plan.system,
        prn,
        plan.sampling_freq,
        doppler,
        code_phase,
        secondary_code_phase,
        CN0,
        Float32(noise_power),
        Float32(peak_to_noise),
        plan.num_noncoherent_accumulations,
        result_buf,
        plan.doppler_freqs,
        plan.num_blocks,
        plan.block_size,
        plan.num_secondary_rotations,
    )
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

  - `system`: GNSS system (e.g. `GPSL1CA()`)
  - `signal`: Complex baseband signal samples
  - `sampling_freq`: Sampling frequency
  - `prns`: PRN numbers to search

# Keyword Arguments forwarded to `plan_acquire`:

  - `min_doppler_coverage`: Minimum one-sided Doppler coverage (default: `7000Hz`)
  - `num_coherently_integrated_code_periods`: Code periods per coherent block (default: `1`)
  - `bit_edge_search_steps`: Bit edge search positions (default: `1`)
  - `num_noncoherent_accumulations`: Non-coherent integration steps (default: `1`)
  - `use_secondary_code`: enable the secondary-code rotation search (default: `true`).
    Requires `num_coherently_integrated_code_periods` to be a whole multiple of the
    secondary-code length `L`; a partial period is rejected to avoid the ±Doppler sign
    ambiguity (issue #68). See [`plan_acquire`](@ref).
  - `fft_flag`: FFTW planning flag (default: `FFTW.MEASURE`)

# Keyword Arguments forwarded to `acquire!`:

  - `interm_freq`: Intermediate frequency (default: `0.0Hz`)
  - `subsample_interpolation`: Enable parabolic interpolation (default: `false`)
  - `store_power_bins`: Retain the full correlation power surface in each result
    for plotting (default: `false`)

# Returns

`Vector{AcquisitionResults}`, one per PRN.

# See also

[`plan_acquire`](@ref), [`acquire!`](@ref)
"""
function acquire(
    system::AbstractGNSSSignal,
    signal,
    sampling_freq,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    min_doppler_coverage = 7000Hz,
    num_coherently_integrated_code_periods::Int = 1,
    bit_edge_search_steps::Int = 1,
    num_noncoherent_accumulations::Int = 1,
    use_secondary_code::Bool = true,
    max_secondary_code_rotations::Int = 32,
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
        use_secondary_code,
        max_secondary_code_rotations,
        fft_flag,
    )
    acquire!(plan, signal, collect(Int, prns); interm_freq, subsample_interpolation, store_power_bins)
end

"""
    acquire(system, signal, sampling_freq, prn::Integer; kwargs...) -> AcquisitionResults

Single-PRN convenience method. Returns a single [`AcquisitionResults`](@ref).
"""
function acquire(
    system::AbstractGNSSSignal,
    signal,
    sampling_freq,
    prn::Integer;
    kwargs...,
)
    only(acquire(system, signal, sampling_freq, [prn]; kwargs...))
end
