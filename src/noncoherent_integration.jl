# src/noncoherent_integration.jl

"""
    _accumulate_noncoherent_integration_pilot!(noncoherent_integration_matrix, coherent_integration_matrix, col_buf, col_fft_plan, samples_per_code)

Column-wise FFT path for pilot channels and sub-bit integration (num_data_bits = 1,
no data bit combination search). Applies fftshift so row indices map to sorted
`doppler_freqs`, accumulates `|FFT|²` into `noncoherent_integration_matrix` (shape `(num_doppler_bins, samples_per_code)`).
"""
function _accumulate_noncoherent_integration_pilot!(
    noncoherent_integration_matrix::Matrix{Float32},
    coherent_integration_matrix::Matrix{ComplexF32},
    col_buf::Vector{ComplexF32},
    col_fft_plan,
    samples_per_code::Int,
)
    num_doppler_bins = size(coherent_integration_matrix, 1)
    for col_idx in 1:samples_per_code
        copyto!(col_buf, 1, coherent_integration_matrix, (col_idx - 1) * num_doppler_bins + 1, num_doppler_bins)
        mul!(col_buf, col_fft_plan, col_buf)
        @inbounds for doppler_bin in 1:num_doppler_bins
            noncoherent_integration_matrix[doppler_bin, col_idx] += abs2(col_buf[doppler_bin])
        end
    end
end

"""
    _sign_search_step!(noncoherent_integration_buf, coherent_integration_matrix, col_buf, col_fft_plan,
                       samples_per_code, num_doppler_bins, num_data_bits, row_offset, sub_block_ffts, patterns)

Sign-pattern search step for one bit-edge alignment of one non-coherent block. Does *not*
perform non-coherent accumulation across blocks — that happens at the outer level in
`_accumulate_noncoherent_integration_step!`. The patterns to enumerate are supplied via
`patterns` (shape `(num_coh_periods, num_patterns)`), built by [`sign_patterns`](@ref).
Today's patterns enumerate data-bit polarities only; the same mechanism is intended to
also enumerate secondary-code rotations in the future (see issue #55).

FM-DBZP column-wise FFT for data channels (paper: N_db > 1). For each of the
samples_per_code columns:
1. Splits the column (with circular row offset `row_offset`) into num_data_bits sub-blocks
   of length num_doppler_bins÷num_data_bits, zero-pads each to num_doppler_bins, and FFTs it.
2. For each sign pattern, sums ±sub-block FFTs (sign per data bit looked up from the
   first row of that bit's segment in `patterns`), and accumulates the cell-wise maximum
   of |result|² into `noncoherent_integration_buf`.

`sub_block_ffts` is a pre-allocated (num_doppler_bins, num_data_bits) scratch matrix.
"""
function _sign_search_step!(
    noncoherent_integration_buf::Matrix{Float32},
    coherent_integration_matrix::Matrix{ComplexF32},
    col_buf::Vector{ComplexF32},
    col_fft_plan,
    samples_per_code::Int,
    num_doppler_bins::Int,
    num_data_bits::Int,
    row_offset::Int,
    sub_block_ffts::Matrix{ComplexF32},  # (num_doppler_bins, num_data_bits)
    patterns::Matrix{Float32},           # (num_coh_periods, num_patterns)
)
    sub_len = num_doppler_bins ÷ num_data_bits   # rows per data bit sub-block
    num_sign_patterns = size(patterns, 2)
    coh_per_bit = size(patterns, 1) ÷ num_data_bits

    for col_idx in 1:samples_per_code
        # Compute num_data_bits sub-block FFTs for column col_idx
        for bit_idx in 0:num_data_bits-1
            fill!(col_buf, zero(ComplexF32))
            for sub_row in 0:sub_len-1
                src_row = ((row_offset + bit_idx * sub_len + sub_row) % num_doppler_bins) + 1  # 1-indexed
                col_buf[sub_row + 1] = coherent_integration_matrix[src_row, col_idx]
            end
            mul!(col_buf, col_fft_plan, col_buf)
            sub_block_ffts[:, bit_idx + 1] .= col_buf   # raw FFT order; output fftshift happens once in `_accumulate_noncoherent_integration_step!`
        end

        # Iterate over precomputed sign patterns; sign per data bit from the first row
        # of that bit's segment in `patterns`.
        for p in 1:num_sign_patterns
            fill!(col_buf, zero(ComplexF32))
            for bit_idx in 0:num_data_bits-1
                bit_sign = patterns[bit_idx * coh_per_bit + 1, p]
                col_buf .+= bit_sign .* view(sub_block_ffts, :, bit_idx + 1)
            end
            @inbounds for doppler_row in 1:num_doppler_bins
                cell_power = abs2(col_buf[doppler_row])
                if cell_power > noncoherent_integration_buf[doppler_row, col_idx]
                    noncoherent_integration_buf[doppler_row, col_idx] = cell_power
                end
            end
        end
    end
end

"""
    _sign_search_step_with_rotations!(noncoherent_integration_buf, coherent_integration_matrix, col_buf,
                                      col_fft_plan, samples_per_code, num_doppler_bins, num_coh_periods, num_blocks,
                                      row_offset, sub_block_ffts, patterns)

Sign-pattern search step for one bit-edge alignment when per-coherent-period signs vary
(secondary-code rotation search). Splits each column into `num_coh_periods` sub-blocks
of `num_blocks` rows each (one per coherent code period) and computes one column FFT per
sub-block. Then, for each pattern column, sums `sign[k] * sub_block_FFT[k]` over the
`num_coh_periods` sub-blocks and accumulates the cell-wise max of `|result|²` into
`noncoherent_integration_buf`.

This is the generalisation of [`_sign_search_step!`](@ref) — that kernel groups consecutive
coherent periods into one sub-block per data bit (`num_data_bits` sub-blocks total) and is
correct only when the sign is constant within each data bit. When the secondary code varies
the sign per coherent period, we need one sub-block per period; `patterns` may have
arbitrary `±1` entries per row.

`sub_block_ffts` must have at least `num_coh_periods` columns.
"""
function _sign_search_step_with_rotations!(
    noncoherent_integration_buf::Matrix{Float32},
    coherent_integration_matrix::Matrix{ComplexF32},
    col_buf::Vector{ComplexF32},
    col_fft_plan,
    samples_per_code::Int,
    num_doppler_bins::Int,
    num_coh_periods::Int,
    num_blocks::Int,
    row_offset::Int,
    sub_block_ffts::Matrix{ComplexF32},
    patterns::Matrix{Float32},
)
    num_sign_patterns = size(patterns, 2)
    for col_idx in 1:samples_per_code
        # One sub-block FFT per coherent period (sub-block size = num_blocks).
        for k in 0:num_coh_periods-1
            fill!(col_buf, zero(ComplexF32))
            for sub_row in 0:num_blocks-1
                src_row = ((row_offset + k * num_blocks + sub_row) % num_doppler_bins) + 1
                col_buf[sub_row + 1] = coherent_integration_matrix[src_row, col_idx]
            end
            mul!(col_buf, col_fft_plan, col_buf)
            sub_block_ffts[:, k + 1] .= col_buf   # raw FFT order; output fftshift happens once in `_accumulate_noncoherent_integration_step!`
        end

        for p in 1:num_sign_patterns
            fill!(col_buf, zero(ComplexF32))
            for k in 0:num_coh_periods-1
                sign = patterns[k + 1, p]
                col_buf .+= sign .* view(sub_block_ffts, :, k + 1)
            end
            @inbounds for doppler_row in 1:num_doppler_bins
                cell_power = abs2(col_buf[doppler_row])
                if cell_power > noncoherent_integration_buf[doppler_row, col_idx]
                    noncoherent_integration_buf[doppler_row, col_idx] = cell_power
                end
            end
        end
    end
end

"""
    _apply_code_drift!(noncoherent_integration_buf, plan, accumulation_step_index)

Circularly shift each row `w` of `noncoherent_integration_buf` by `S_CD[w]` samples to correct for
code-phase walk-off caused by carrier Doppler during coherent integration.

`accumulation_step_index` is the 0-based incoherent step index (0 → no shift).
`S_CD[w] = round(doppler_hz[w] * accumulation_step_index * T_coherent * sampling_freq_hz / carrier_freq_hz)`

The buffer is in **raw FFT-bin order** (matches the sign-search kernels' direct
FFT output). Raw bin `w` corresponds to sorted-Doppler position `fftshift_perm[w]`,
so the per-row shift uses `doppler_freqs[fftshift_perm[w]]`. The single output
fftshift in `_accumulate_noncoherent_integration_step!` then re-maps the
corrected raw buffer into sorted-Doppler order on the way out.
"""
function _apply_code_drift!(noncoherent_integration_buf::Matrix{Float32}, plan::AcquisitionPlan, scratch, accumulation_step_index::Int)
    accumulation_step_index == 0 && return  # no drift on first step
    num_doppler_bins = size(noncoherent_integration_buf, 1)
    sampling_freq_hz = ustrip(Hz, plan.sampling_freq)
    carrier_freq_hz = ustrip(Hz, get_center_frequency(plan.system))
    coherent_duration_s = plan.num_coherently_integrated_code_periods * plan.samples_per_code / sampling_freq_hz
    row_buf = scratch.row_buf
    row_shift_buf = scratch.row_shift_buf
    fftshift_perm = plan.fftshift_perm
    for doppler_row in 1:num_doppler_bins
        doppler_hz = ustrip(Hz, plan.doppler_freqs[fftshift_perm[doppler_row]])
        shift = round(Int, doppler_hz * accumulation_step_index * coherent_duration_s * sampling_freq_hz / carrier_freq_hz)
        shift == 0 && continue
        row_buf .= view(noncoherent_integration_buf, doppler_row, :)
        circshift!(row_shift_buf, row_buf, shift)
        noncoherent_integration_buf[doppler_row, :] .= row_shift_buf
    end
end
# Single-argument scratch convenience: used by tests and code-drift testset directly
_apply_code_drift!(buf::Matrix{Float32}, plan::AcquisitionPlan, accumulation_step_index::Int) =
    _apply_code_drift!(buf, plan, _default_scratch(plan), accumulation_step_index)

# Per-column FFT followed by `|x|²` accumulation into `accumulator`, with the
# row written at its fftshift-permuted position. Folds two passes
# (`_accumulate_noncoherent_integration_pilot!` + `_scatter_fftshift_accumulate!`)
# into one, dropping the full-matrix-size `noncoherent_integration_buf`
# intermediate. Valid only at num_noncoherent_accumulations == 1 simple/pilot
# path: code-drift correction is a no-op at step 0 and there is no bit-edge max
# search to reuse the buf across alignments.
#
# The fftshift permutation is a fixed rotation by `shift = N ÷ 2` rows:
#   src rows 1..N-shift    → dst rows 1+shift..N      (advance)
#   src rows N-shift+1..N  → dst rows 1..shift        (wrap)
# Two contiguous loops express both halves so the compiler can SIMD.
function _accumulate_fftshifted_power_pilot!(
    accumulator::Matrix{Float32},
    coherent_integration_matrix::Matrix{ComplexF32},
    col_buf::Vector{ComplexF32},
    col_fft_plan,
    samples_per_code::Int,
    num_doppler_bins::Int,
)
    shift = num_doppler_bins ÷ 2
    front = num_doppler_bins - shift
    @inbounds for col_idx in 1:samples_per_code
        copyto!(col_buf, 1, coherent_integration_matrix, (col_idx - 1) * num_doppler_bins + 1, num_doppler_bins)
        mul!(col_buf, col_fft_plan, col_buf)
        @simd for doppler_bin in 1:front
            accumulator[doppler_bin + shift, col_idx] += abs2(col_buf[doppler_bin])
        end
        @simd for doppler_bin in 1:shift
            accumulator[doppler_bin, col_idx] += abs2(col_buf[doppler_bin + front])
        end
    end
end

# Batched-FFT counterpart: FFT all columns at once in-place on the CIM, then
# accumulate `|x|²` into `accumulator` with the fftshift row permutation.
function _accumulate_fftshifted_power_pilot_batched!(
    accumulator::Matrix{Float32},
    coherent_integration_matrix::Matrix{ComplexF32},
    col_batch_fft_plan,
    samples_per_code::Int,
    num_doppler_bins::Int,
)
    mul!(coherent_integration_matrix, col_batch_fft_plan, coherent_integration_matrix)
    shift = num_doppler_bins ÷ 2
    front = num_doppler_bins - shift
    @inbounds for col_idx in 1:samples_per_code
        @simd for doppler_bin in 1:front
            accumulator[doppler_bin + shift, col_idx] += abs2(coherent_integration_matrix[doppler_bin, col_idx])
        end
        @simd for doppler_bin in 1:shift
            accumulator[doppler_bin, col_idx] += abs2(coherent_integration_matrix[doppler_bin + front, col_idx])
        end
    end
end

# Per-column FFT followed by `|x|²` accumulation with per-row code-drift column
# shift AND fftshift row permutation, folded into one pass. Extends the slice-5
# fusion (`_accumulate_fftshifted_power_pilot!`) to the multistep simple path
# (Issue #62): drops the `noncoherent_integration_buf` intermediate that the
# unfused pipeline (`pilot! → _apply_code_drift! → _scatter_fftshift_accumulate!`)
# materialises and traverses twice per step.
#
# `code_drift_shifts[r]` is the row-r column shift, pre-normalised to
# `[0, samples_per_code)` by `_fill_code_drift_shifts!`. That lets the inner
# loop replace `mod(c-1+shift_r, spc)+1` with a single conditional subtract.
# The caller must dispatch to the non-drift slice-5 kernel when every row's
# drift rounds to 0 (step 0, or any step whose rounded drift is zero — at the
# L1CA / 5 MHz / N_coh=1 grid that's true for every step up to N_nc≈8, so the
# early-exit covers the dominant simple-path case).
function _accumulate_fftshifted_power_drift_pilot!(
    accumulator::Matrix{Float32},
    coherent_integration_matrix::Matrix{ComplexF32},
    col_buf::Vector{ComplexF32},
    col_fft_plan,
    samples_per_code::Int,
    num_doppler_bins::Int,
    fftshift_perm::Vector{Int},
    code_drift_shifts::Vector{Int},
)
    @inbounds for col_idx in 1:samples_per_code
        copyto!(col_buf, 1, coherent_integration_matrix, (col_idx - 1) * num_doppler_bins + 1, num_doppler_bins)
        mul!(col_buf, col_fft_plan, col_buf)
        c_minus_1 = col_idx - 1
        for r in 1:num_doppler_bins
            sum = c_minus_1 + code_drift_shifts[r]
            dst_col = (sum >= samples_per_code ? sum - samples_per_code : sum) + 1
            accumulator[fftshift_perm[r], dst_col] += abs2(col_buf[r])
        end
    end
end

# Batched-FFT counterpart of `_accumulate_fftshifted_power_drift_pilot!`.
function _accumulate_fftshifted_power_drift_pilot_batched!(
    accumulator::Matrix{Float32},
    coherent_integration_matrix::Matrix{ComplexF32},
    col_batch_fft_plan,
    samples_per_code::Int,
    num_doppler_bins::Int,
    fftshift_perm::Vector{Int},
    code_drift_shifts::Vector{Int},
)
    mul!(coherent_integration_matrix, col_batch_fft_plan, coherent_integration_matrix)
    @inbounds for col_idx in 1:samples_per_code
        c_minus_1 = col_idx - 1
        for r in 1:num_doppler_bins
            sum = c_minus_1 + code_drift_shifts[r]
            dst_col = (sum >= samples_per_code ? sum - samples_per_code : sum) + 1
            accumulator[fftshift_perm[r], dst_col] += abs2(coherent_integration_matrix[r, col_idx])
        end
    end
end

# Fill `shifts[r]` with the row-r column shift from `_apply_code_drift!`,
# normalised to `[0, samples_per_code)` so the hot loop in the fused kernel
# can drop `mod` for a single conditional subtract. Returns `true` if any
# shift rounded to non-zero (caller dispatches to the fused-drift kernel) or
# `false` if every row's drift rounds to 0 (caller uses the cheaper non-drift
# slice-5 kernel — the typical case at low-fs / short-T_coh grids).
function _fill_code_drift_shifts!(shifts::Vector{Int}, plan::AcquisitionPlan, accumulation_step_index::Int)
    if accumulation_step_index == 0
        fill!(shifts, 0)
        return false
    end
    sampling_freq_hz = ustrip(Hz, plan.sampling_freq)
    carrier_freq_hz = ustrip(Hz, get_center_frequency(plan.system))
    coherent_duration_s = plan.num_coherently_integrated_code_periods * plan.samples_per_code / sampling_freq_hz
    samples_per_code = plan.samples_per_code
    has_drift = false
    @inbounds for r in 1:length(shifts)
        doppler_hz = ustrip(Hz, plan.doppler_freqs[r])
        raw = round(Int, doppler_hz * accumulation_step_index * coherent_duration_s * sampling_freq_hz / carrier_freq_hz)
        if raw != 0
            has_drift = true
        end
        shifts[r] = mod(raw, samples_per_code)
    end
    return has_drift
end

# Scatter `src` into `dst` applying fftshift row permutation (accumulating).
# For even num_doppler_bins — the only case in practice — this is a top/bottom
# half swap, which SIMDs. For odd N, fall back to the indexed scatter (the
# permutation isn't a pure swap in that case).
function _scatter_fftshift_accumulate!(dst::Matrix{Float32}, src::Matrix{Float32}, fftshift_perm::Vector{Int}, samples_per_code::Int)
    num_doppler_bins = size(src, 1)
    if iseven(num_doppler_bins)
        half = num_doppler_bins ÷ 2
        @inbounds for c in 1:samples_per_code
            @simd for r in 1:half
                dst[r, c]        += src[r + half, c]
            end
            @simd for r in 1:half
                dst[r + half, c] += src[r, c]
            end
        end
    else
        @inbounds for c in 1:samples_per_code
            for r in 1:num_doppler_bins
                dst[fftshift_perm[r], c] += src[r, c]
            end
        end
    end
    return dst
end

"""
    _accumulate_noncoherent_integration_step!(noncoherent_integration_matrix, coherent_integration_matrix, plan, scratch, prn, m)

For incoherent step `m` (0-based): run the column FFT (pilot/sub-bit or data/rotation),
apply bit edge search and code drift correction, then accumulate into
`noncoherent_integration_matrix`.

For pilot channels and sub-bit integration with no rotation search
(num_data_bits == 1, bit_edge_search_steps == 1, num_secondary_rotations == 1 or N == 1):
uses the fast pilot path (_accumulate_noncoherent_integration_pilot!) — one column FFT per
column, no sign search.

When secondary-code rotation search is active (num_secondary_rotations > 1 and N > 1):
uses [`_sign_search_step_with_rotations!`](@ref), which splits the column into N
coherent-period sub-blocks (size num_blocks each) and searches across the Cartesian
product of data-bit polarities and secondary-code rotations.

For data-bit / bit-edge search without rotation
(num_data_bits > 1 or bit_edge_search_steps > 1, num_secondary_rotations == 1):
uses [`_sign_search_step!`](@ref) with bit_edge_search_steps candidate alignments and
2^(num_data_bits − 1) data-bit polarity combinations.
"""
function _accumulate_noncoherent_integration_step!(
    noncoherent_integration_matrix::Matrix{Float32},
    coherent_integration_matrix::Matrix{ComplexF32},
    plan::AcquisitionPlan,
    scratch,
    prn::Int,
    accumulation_step_index::Int,
)
    num_doppler_bins = plan.num_coherently_integrated_code_periods * plan.num_blocks
    rotation_search_active = plan.num_secondary_rotations > 1 &&
                             plan.num_coherently_integrated_code_periods > 1

    if plan.num_data_bits == 1 && plan.bit_edge_search_steps == 1 && !rotation_search_active
        # Simple/pilot path: fused FFT+|x|²+(code-drift)+fftshift kernel writes
        # straight into `noncoherent_integration_matrix`. The drift kernel is
        # taken only when at least one row's rounded drift is non-zero — at
        # step 0, and at every step where the rounded drift is 0 for all rows
        # (typical for short coherent durations / small doppler grids), the
        # cheaper SIMD-friendly slice-5 non-drift kernel runs instead.
        has_drift = _fill_code_drift_shifts!(scratch.code_drift_shifts, plan, accumulation_step_index)
        if !has_drift
            if num_doppler_bins <= BATCH_FFT_THRESHOLD
                _accumulate_fftshifted_power_pilot_batched!(
                    noncoherent_integration_matrix, coherent_integration_matrix,
                    plan.col_batch_fft_plan, plan.samples_per_code, num_doppler_bins)
            else
                _accumulate_fftshifted_power_pilot!(
                    noncoherent_integration_matrix, coherent_integration_matrix,
                    scratch.col_buf, plan.col_fft_plan,
                    plan.samples_per_code, num_doppler_bins)
            end
        else
            if num_doppler_bins <= BATCH_FFT_THRESHOLD
                _accumulate_fftshifted_power_drift_pilot_batched!(
                    noncoherent_integration_matrix, coherent_integration_matrix,
                    plan.col_batch_fft_plan, plan.samples_per_code, num_doppler_bins,
                    plan.fftshift_perm, scratch.code_drift_shifts)
            else
                _accumulate_fftshifted_power_drift_pilot!(
                    noncoherent_integration_matrix, coherent_integration_matrix,
                    scratch.col_buf, plan.col_fft_plan,
                    plan.samples_per_code, num_doppler_bins,
                    plan.fftshift_perm, scratch.code_drift_shifts)
            end
        end
    else
        sign_search_max_buf = scratch.sign_search_max_buf
        noncoherent_integration_buf = scratch.noncoherent_integration_buf
        # Sign-pattern search path. The plan pre-computed the pattern matrix per
        # PRN at `plan_acquire` time; the kernel selection depends only on the
        # rotation axis. The kernel reads `patterns` to decide what to search.
        patterns = plan.sign_patterns_by_prn[prn]
        fill!(sign_search_max_buf, 0.0f0)
        bit_period_codes = plan.num_coherently_integrated_code_periods ÷ plan.num_data_bits
        edge_step = bit_period_codes * plan.num_blocks ÷ plan.bit_edge_search_steps
        for bit_edge_search_idx in 0:plan.bit_edge_search_steps-1
            fill!(noncoherent_integration_buf, 0.0f0)
            row_offset = bit_edge_search_idx * edge_step
            if rotation_search_active
                _sign_search_step_with_rotations!(noncoherent_integration_buf, coherent_integration_matrix,
                    scratch.col_buf, plan.col_fft_plan,
                    plan.samples_per_code, num_doppler_bins,
                    plan.num_coherently_integrated_code_periods, plan.num_blocks,
                    row_offset, scratch.sub_block_ffts, patterns)
            else
                _sign_search_step!(noncoherent_integration_buf, coherent_integration_matrix, scratch.col_buf,
                    plan.col_fft_plan, plan.samples_per_code, num_doppler_bins,
                    plan.num_data_bits, row_offset, scratch.sub_block_ffts, patterns)
            end
            _apply_code_drift!(noncoherent_integration_buf, plan, scratch, accumulation_step_index)
            @. sign_search_max_buf = max(sign_search_max_buf, noncoherent_integration_buf)
        end
        # Unified convention: the sign-search kernels emit raw FFT-bin order
        # (no internal circshift), `_apply_code_drift!` shifts in raw order using
        # `doppler_freqs[fftshift_perm[w]]`, and the final scatter below performs
        # the single fftshift from raw to sorted-Doppler order on the way into
        # `noncoherent_integration_matrix`. This matches the pilot path's
        # "shift exactly once, at the output" convention and removes the
        # per-column circshift that previously ran 120 000× per dwell.
        _scatter_fftshift_accumulate!(noncoherent_integration_matrix,
            sign_search_max_buf, plan.fftshift_perm, plan.samples_per_code)
    end
end
# Convenience wrapper for tests and single-threaded callers — routes through
# thread 1's scratch (see `_default_scratch`) and uses the plan's first avail PRN.
_accumulate_noncoherent_integration_step!(nim, cim, plan::AcquisitionPlan, accumulation_step_index::Int) =
    _accumulate_noncoherent_integration_step!(nim, cim, plan, _default_scratch(plan),
        first(plan.avail_prns), accumulation_step_index)
