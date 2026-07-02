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

# The (double_block_size, num_coh*num_blocks) slice of the plan's all-segment
# signal-block FFT cache belonging to 1-based accumulation step `step_idx`.
@inline function _signal_block_ffts_for_step(plan::AcquisitionPlan, step_idx::Int)
    cols_per_segment = plan.num_coherently_integrated_code_periods * plan.num_blocks
    view(plan.signal_block_ffts, :,
        (step_idx - 1) * cols_per_segment + 1 : step_idx * cols_per_segment)
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

    # The N_nc==1 case streams build → reduce → extract per PRN: result
    # statistics are reduced tile by tile and no power surface is materialised
    # (unless the caller opts in via store_power_bins). The multistep path
    # (N_nc>1) keeps per-PRN accumulation matrices because the surface carries
    # state across signal segments.
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

# Multistep path (N_nc > 1), PRN-outer: the per-segment signal-block FFTs are
# precomputed once for ALL segments (same total FFT work as the former
# per-segment cache — no recompute anywhere), then each parallel chunk carries
# one PRN through every accumulation step against its claimed scratch slot's
# accumulator and extracts the result before moving to the next PRN. This
# bounds the number of live power surfaces by `min(nthreads, num_cores,
# num_prns)` instead of one per PRN, and removes the per-step thread barriers
# the former segment-outer loop needed.
function _acquire_multistep!(plan, signal, prns, segment_length, interm_freq_hz,
                              sampling_freq_hz, subsample_interpolation, store_power_bins)
    # Fill the all-segment signal-block FFT cache. The downconverted segment
    # lives in the plan's single `sig_buf`; the per-FFT temp comes from thread
    # 1's scratch slot — `_default_scratch(plan)` names that convention.
    main_scratch = _default_scratch(plan)
    sig_buf = plan.sig_buf
    for step_idx in 1:plan.num_noncoherent_accumulations
        seg_start = (step_idx - 1) * segment_length + 1
        _downconvert!(sig_buf, signal, seg_start, segment_length, interm_freq_hz, sampling_freq_hz)
        _precompute_signal_block_ffts!(
            _signal_block_ffts_for_step(plan, step_idx),
            sig_buf,
            plan.samples_per_code,
            plan.num_blocks,
            plan.block_size,
            plan.num_coherently_integrated_code_periods,
            main_scratch.double_block_buf,
            plan.double_block_fft_plan,
        )
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
        scratch, slot = _claim_scratch!(plan)
        try
            accumulator = scratch.noncoherent_accumulator
            fill!(accumulator, 0f0)
            for step_idx in 1:plan.num_noncoherent_accumulations
                _accumulate_prn_step_tiled!(accumulator,
                    _signal_block_ffts_for_step(plan, step_idx),
                    plan, scratch, prn, step_idx - 1)
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

# Sequential path used when num_noncoherent_accumulations == 1: per PRN, the
# tiled build+reduce streams the power cells straight into the result
# statistics (global peak, per-row sums for the noise estimate) — the
# (num_doppler_bins × samples_per_code_eff) power surface is only materialised
# into the per-PRN result buffer when the caller requests it via
# `store_power_bins = true`.
function _acquire_sequential!(plan, signal, prns, segment_length, interm_freq_hz,
                               sampling_freq_hz, subsample_interpolation, store_power_bins)
    main_scratch = _default_scratch(plan)
    sig_buf = plan.sig_buf

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

    @batch per=core for result_idx in eachindex(prns)
        prn = @inbounds prns[result_idx]
        prn_idx = findfirst(==(prn), plan.avail_prns)
        scratch, slot = _claim_scratch!(plan)
        try
            store_buf = store_power_bins ? _get_result_buffer!(plan, prn_idx) : nothing

            peak_power, peak_doppler_bin, peak_col =
                _acquire_prn_streamed!(plan, scratch, prn, store_buf)

            results[result_idx] = _extract_result_streamed!(plan, scratch, prn,
                peak_power, peak_doppler_bin, peak_col, store_buf,
                signal, interm_freq_hz,
                sampling_freq_hz, code_freq_hz, code_length, code_period,
                num_doppler_bins, doppler_step, subsample_interpolation)
        finally
            _release_scratch!(plan, slot)
        end
    end
    return results
end

# Fetch (allocating on first use) the cached per-PRN power-surface buffer used
# when the caller opts in via `store_power_bins = true`. This is the ONLY place
# the full (num_doppler_bins × samples_per_code_eff) surface is materialised on
# the sequential path.
function _get_result_buffer!(plan::AcquisitionPlan, prn_idx::Int)
    cached = plan.result_buffers[prn_idx]
    buf = cached === nothing ?
        Matrix{Float32}(undef, length(plan.doppler_freqs), plan.samples_per_code_eff) :
        cached
    plan.result_buffers[prn_idx] = buf
    return buf
end

# Streamed per-PRN pipeline for the sequential (N_nc == 1) path: build each
# coherent tile, transform its columns to the Doppler domain, and reduce every
# power cell on the fly — global peak (value, row, column), per-row power sums
# (feeding the OppositeRow noise estimate) and, when the plan's estimator needs
# them, per-column sums (GlobalMean). Cell values are written out only when
# `store_buf` is provided. Row/column iteration order matches the sorted-order
# scan of `_findmax_and_colsums!` exactly, so peak tie-breaking and Float32
# accumulation order are bit-identical to reducing a materialised surface.
#
# Returns `(peak_power, peak_doppler_bin, peak_col)` with `peak_doppler_bin` in
# sorted-Doppler row order and `peak_col` indexing the effective cp axis
# (including the rotation-slice offset).
function _acquire_prn_streamed!(
    plan::AcquisitionPlan,
    scratch,
    prn::Int,
    store_buf::Union{Nothing,Matrix{Float32}},
)
    num_doppler_bins = length(plan.doppler_freqs)
    block_size = plan.block_size
    rotation_active = plan.num_secondary_rotations > 1 &&
                      plan.num_coherently_integrated_code_periods > 1
    simple_path = plan.num_data_bits == 1 && plan.bit_edge_search_steps == 1 &&
                  !rotation_active
    row_sums = scratch.row_sums_buf
    fill!(row_sums, 0f0)
    col_sums = scratch.col_sums_buf
    collect_col_sums = !isempty(col_sums)
    prn_conj_fft = plan.prn_conj_ffts[prn]
    tile = scratch.coherent_tile

    peak_power = -Inf32
    peak_doppler_bin = 1
    peak_col = 1

    if simple_path
        batched = num_doppler_bins <= BATCH_FFT_THRESHOLD
        for col_block_idx in 0:plan.num_blocks-1
            _build_coherent_tile!(tile, plan.signal_block_ffts, prn_conj_fft,
                col_block_idx, plan.num_blocks, block_size,
                plan.num_coherently_integrated_code_periods,
                scratch.corr_buf, plan.double_block_bfft_plan)
            dest_col_offset = col_block_idx * block_size
            if batched
                mul!(tile, plan.col_batch_fft_plan, tile)
                for local_col in 1:block_size
                    peak_power, peak_doppler_bin, peak_col = _stream_power_column!(
                        row_sums, col_sums, collect_col_sums, store_buf,
                        view(tile, :, local_col), dest_col_offset + local_col,
                        num_doppler_bins, peak_power, peak_doppler_bin, peak_col)
                end
            else
                col_buf = scratch.col_buf
                for local_col in 1:block_size
                    copyto!(col_buf, 1, tile, (local_col - 1) * num_doppler_bins + 1, num_doppler_bins)
                    mul!(col_buf, plan.col_fft_plan, col_buf)
                    peak_power, peak_doppler_bin, peak_col = _stream_power_column!(
                        row_sums, col_sums, collect_col_sums, store_buf,
                        col_buf, dest_col_offset + local_col,
                        num_doppler_bins, peak_power, peak_doppler_bin, peak_col)
                end
            end
        end
    else
        stage = scratch.stage_buf
        num_slices = size(stage, 2)
        samples_per_code = plan.samples_per_code
        num_data_combos = 1 << (plan.num_data_bits - 1)
        bit_period_codes = plan.num_coherently_integrated_code_periods ÷ plan.num_data_bits
        edge_step = bit_period_codes * plan.num_blocks ÷ plan.bit_edge_search_steps
        use_max = num_data_combos > 1 || plan.bit_edge_search_steps > 1
        patterns = plan.sign_patterns_by_prn[prn]
        tiled_re = rotation_active ? plan.tiled_phase_patterns_re_by_prn[prn] :
            _EMPTY_TILED_PATTERNS
        tiled_im = rotation_active ? plan.tiled_phase_patterns_im_by_prn[prn] :
            _EMPTY_TILED_PATTERNS
        for col_block_idx in 0:plan.num_blocks-1
            _build_coherent_tile!(tile, plan.signal_block_ffts, prn_conj_fft,
                col_block_idx, plan.num_blocks, block_size,
                plan.num_coherently_integrated_code_periods,
                scratch.corr_buf, plan.double_block_bfft_plan)
            for local_col in 1:block_size
                cp_col = col_block_idx * block_size + local_col
                fill!(stage, 0f0)
                _fill_sign_stage_column!(stage, tile, local_col, cp_col, plan, scratch,
                    patterns, tiled_re, tiled_im, rotation_active, num_data_combos,
                    edge_step, use_max)
                for s in 0:num_slices-1
                    peak_power, peak_doppler_bin, peak_col = _stream_stage_column!(
                        row_sums, col_sums, collect_col_sums, store_buf,
                        view(stage, :, s + 1), cp_col + s * samples_per_code,
                        num_doppler_bins, peak_power, peak_doppler_bin, peak_col)
                end
            end
        end
    end
    return peak_power, peak_doppler_bin, peak_col
end

# Reduce one Doppler-domain column (raw FFT row order) into the streaming
# statistics, folding the fftshift row permutation inline. Iterates destination
# rows in ascending sorted order — first half [1, shift] from raw bins
# [front+1, N], second half from [1, front] — so peak tie-breaking and the
# Float32 order of `col_sum`/`row_sums` additions are identical to scanning a
# materialised, fftshifted surface row by row.
@inline function _stream_power_column!(
    row_sums::Vector{Float32},
    col_sums::Vector{Float32},
    collect_col_sums::Bool,
    store_buf::Union{Nothing,Matrix{Float32}},
    x::AbstractVector{ComplexF32},
    dest_col::Int,
    num_doppler_bins::Int,
    peak_power::Float32,
    peak_doppler_bin::Int,
    peak_col::Int,
)
    shift = num_doppler_bins ÷ 2
    front = num_doppler_bins - shift
    col_sum = 0f0
    col_max = -Inf32
    col_max_row = 1
    @inbounds for r in front+1:num_doppler_bins
        v = abs2(x[r])
        dst = r - front
        row_sums[dst] += v
        col_sum += v
        if v > col_max
            col_max = v
            col_max_row = dst
        end
    end
    @inbounds for r in 1:front
        v = abs2(x[r])
        dst = r + shift
        row_sums[dst] += v
        col_sum += v
        if v > col_max
            col_max = v
            col_max_row = dst
        end
    end
    collect_col_sums && (col_sums[dest_col] = col_sum)
    if store_buf !== nothing
        @inbounds for r in front+1:num_doppler_bins
            store_buf[r - front, dest_col] = abs2(x[r])
        end
        @inbounds for r in 1:front
            store_buf[r + shift, dest_col] = abs2(x[r])
        end
    end
    if col_max > peak_power
        return col_max, col_max_row, dest_col
    end
    return peak_power, peak_doppler_bin, peak_col
end

# Streaming consumer for one sign-search stage column (already |x|² in sorted
# row order): same statistics update as `_stream_power_column!` without the
# fftshift fold.
@inline function _stream_stage_column!(
    row_sums::Vector{Float32},
    col_sums::Vector{Float32},
    collect_col_sums::Bool,
    store_buf::Union{Nothing,Matrix{Float32}},
    stage_col::AbstractVector{Float32},
    dest_col::Int,
    num_doppler_bins::Int,
    peak_power::Float32,
    peak_doppler_bin::Int,
    peak_col::Int,
)
    col_sum = 0f0
    col_max = -Inf32
    col_max_row = 1
    @inbounds for r in 1:num_doppler_bins
        v = stage_col[r]
        row_sums[r] += v
        col_sum += v
        if v > col_max
            col_max = v
            col_max_row = r
        end
    end
    collect_col_sums && (col_sums[dest_col] = col_sum)
    if store_buf !== nothing
        @inbounds for r in 1:num_doppler_bins
            store_buf[r, dest_col] = stage_col[r]
        end
    end
    if col_max > peak_power
        return col_max, col_max_row, dest_col
    end
    return peak_power, peak_doppler_bin, peak_col
end

# Recompute the Doppler-power column `global_col` (index into the effective cp
# axis, i.e. including the rotation-slice offset) into `dest`, in sorted-Doppler
# row order. Rebuilds the one tile the column lives in — num_doppler_bins
# double-block IFFTs plus one column reduction, ~1/block_size of a full PRN
# pass — so subsample interpolation can read peak-neighbour cells without the
# power surface ever having been materialised. Used only when store_power_bins
# is off (otherwise the neighbours are read back from the stored buffer).
function _recompute_column_power!(
    dest::Vector{Float32},
    plan::AcquisitionPlan,
    scratch,
    prn::Int,
    global_col::Int,
)
    num_doppler_bins = length(plan.doppler_freqs)
    samples_per_code = plan.samples_per_code
    block_size = plan.block_size
    cp_col0 = (global_col - 1) % samples_per_code       # 0-based cp column
    rotation_idx = (global_col - 1) ÷ samples_per_code
    col_block_idx = cp_col0 ÷ block_size
    local_col = cp_col0 % block_size + 1
    tile = scratch.coherent_tile
    _build_coherent_tile!(tile, plan.signal_block_ffts, plan.prn_conj_ffts[prn],
        col_block_idx, plan.num_blocks, block_size,
        plan.num_coherently_integrated_code_periods,
        scratch.corr_buf, plan.double_block_bfft_plan)
    rotation_active = plan.num_secondary_rotations > 1 &&
                      plan.num_coherently_integrated_code_periods > 1
    simple_path = plan.num_data_bits == 1 && plan.bit_edge_search_steps == 1 &&
                  !rotation_active
    if simple_path
        col_buf = scratch.col_buf
        copyto!(col_buf, 1, tile, (local_col - 1) * num_doppler_bins + 1, num_doppler_bins)
        mul!(col_buf, plan.col_fft_plan, col_buf)
        shift = num_doppler_bins ÷ 2
        front = num_doppler_bins - shift
        @inbounds for r in front+1:num_doppler_bins
            dest[r - front] = abs2(col_buf[r])
        end
        @inbounds for r in 1:front
            dest[r + shift] = abs2(col_buf[r])
        end
    else
        stage = scratch.stage_buf
        fill!(stage, 0f0)
        num_data_combos = 1 << (plan.num_data_bits - 1)
        bit_period_codes = plan.num_coherently_integrated_code_periods ÷ plan.num_data_bits
        edge_step = bit_period_codes * plan.num_blocks ÷ plan.bit_edge_search_steps
        use_max = num_data_combos > 1 || plan.bit_edge_search_steps > 1
        patterns = plan.sign_patterns_by_prn[prn]
        tiled_re = rotation_active ? plan.tiled_phase_patterns_re_by_prn[prn] :
            _EMPTY_TILED_PATTERNS
        tiled_im = rotation_active ? plan.tiled_phase_patterns_im_by_prn[prn] :
            _EMPTY_TILED_PATTERNS
        _fill_sign_stage_column!(stage, tile, local_col, cp_col0 + 1, plan, scratch,
            patterns, tiled_re, tiled_im, rotation_active, num_data_combos,
            edge_step, use_max)
        dest .= view(stage, :, rotation_idx + 1)
    end
    return dest
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
        copyto!(_get_result_buffer!(plan, prn_idx), power_bins)
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

# Streamed-path counterpart of `_extract_result!`: assembles an
# AcquisitionResults from the statistics reduced on the fly by
# `_acquire_prn_streamed!` — no power surface is read. The noise power comes
# from the streamed per-row sums (OppositeRow: mean of the row half the grid
# away from the peak) or per-column sums (GlobalMean); peak-neighbour cells for
# subsample interpolation are read back from the stored buffer when available,
# else recomputed on demand for the up-to-three columns involved
# (`_recompute_column_power!`).
function _extract_result_streamed!(plan, scratch, prn,
                                   peak_power, peak_doppler_bin, peak_col,
                                   store_buf, signal, interm_freq_hz,
                                   sampling_freq_hz, code_freq_hz, code_length, code_period,
                                   num_doppler_bins, doppler_step, subsample_interpolation)
    num_cp_cols = plan.samples_per_code_eff
    noise_power = if plan.noise_estimator isa GlobalMeanNoiseEstimator
        samples_per_chip = floor(Int, sampling_freq_hz / code_freq_hz)
        _global_mean_from_colsums(scratch.col_sums_buf, num_doppler_bins,
            peak_col, samples_per_chip)
    else
        # OppositeRow: mean of the Doppler row exactly half the search grid away
        # from the peak — noise-only by construction (see est_signal_noise_power).
        opp_row = mod(peak_doppler_bin - 1 + num_doppler_bins ÷ 2, num_doppler_bins) + 1
        scratch.row_sums_buf[opp_row] / Float32(num_cp_cols)
    end
    signal_power = peak_power - noise_power

    peak_to_noise = (signal_power + noise_power) / noise_power
    CN0 = 10 * log10(signal_power / noise_power / code_period / 1.0Hz)

    # Decode the effective-cp-axis peak column into (cp_within, rotation) — see
    # `_extract_result!` for the layout.
    samples_per_code = plan.samples_per_code
    rotation_block = (peak_col - 1) ÷ samples_per_code
    scrambled_col = (peak_col - 1) % samples_per_code
    delay_samples = _fmdbzp_column_to_tau(scrambled_col, plan.num_blocks, plan.block_size)
    code_phase = mod(-delay_samples * code_freq_hz / sampling_freq_hz, code_length)

    doppler = plan.doppler_freqs[peak_doppler_bin]

    if subsample_interpolation
        # Neighbouring columns must stay within the SAME rotation block
        # (different blocks correspond to different secondary-code hypotheses
        # and aren't physically adjacent in cp).
        num_code_bins = samples_per_code
        col_left  = mod(scrambled_col - 1, num_code_bins) + rotation_block * samples_per_code
        col_right = mod(scrambled_col + 1, num_code_bins) + rotation_block * samples_per_code
        dop_row_left  = peak_doppler_bin == 1 ? num_doppler_bins : peak_doppler_bin - 1
        dop_row_right = peak_doppler_bin == num_doppler_bins ? 1 : peak_doppler_bin + 1
        local power_left::Float32, power_right::Float32
        local dop_left::Float32, dop_right::Float32
        if store_buf !== nothing
            power_left  = store_buf[peak_doppler_bin, col_left + 1]
            power_right = store_buf[peak_doppler_bin, col_right + 1]
            dop_left  = store_buf[dop_row_left, peak_col]
            dop_right = store_buf[dop_row_right, peak_col]
        else
            col_power = scratch.col_power_buf
            _recompute_column_power!(col_power, plan, scratch, prn, peak_col)
            dop_left  = col_power[dop_row_left]
            dop_right = col_power[dop_row_right]
            power_left  = _recompute_column_power!(col_power, plan, scratch, prn, col_left + 1)[peak_doppler_bin]
            power_right = _recompute_column_power!(col_power, plan, scratch, prn, col_right + 1)[peak_doppler_bin]
        end
        if max(power_left, power_right) > sqrt(noise_power)
            fractional_col_offset = _parabolic_interp(power_left, peak_power, power_right)
            delay_samples_interp = delay_samples + fractional_col_offset
            code_phase = mod(-delay_samples_interp * code_freq_hz / sampling_freq_hz, code_length)
        end
        if max(dop_left, dop_right) > sqrt(noise_power)
            fractional_doppler_offset = _parabolic_interp(dop_left, peak_power, dop_right)
            doppler = doppler + fractional_doppler_offset * doppler_step
        end
    end

    # Exact secondary-code phase — same gate and estimator as `_extract_result!`.
    secondary_code_phase = if plan.num_secondary_rotations > 1
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
        store_buf,
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
