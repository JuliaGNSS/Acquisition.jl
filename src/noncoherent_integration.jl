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
    _accumulate_noncoherent_integration_data_bits!(noncoherent_integration_buf, coherent_integration_matrix, col_buf, col_fftshift_buf, col_fft_plan,
                               samples_per_code, num_doppler_bins, num_data_bits, row_offset, sub_block_ffts)

FM-DBZP column-wise FFT for data channels (paper: N_db > 1). For each of the samples_per_code columns:
1. Splits the column (with circular row offset `row_offset`) into num_data_bits sub-blocks of
   length num_doppler_bins÷num_data_bits, zero-pads each to num_doppler_bins, and FFTs it.
2. For each of 2^(num_data_bits−1) bit sign patterns (d[0]=+1 always fixed), sums ±sub-block
   FFTs, and accumulates the cell-wise maximum of |result|² into `noncoherent_integration_buf`.

`sub_block_ffts` is a pre-allocated (num_doppler_bins, num_data_bits) scratch matrix.
"""
function _accumulate_noncoherent_integration_data_bits!(
    noncoherent_integration_buf::Matrix{Float32},
    coherent_integration_matrix::Matrix{ComplexF32},
    col_buf::Vector{ComplexF32},
    col_fftshift_buf::Vector{ComplexF32},
    col_fft_plan,
    samples_per_code::Int,
    num_doppler_bins::Int,
    num_data_bits::Int,
    row_offset::Int,
    sub_block_ffts::Matrix{ComplexF32},  # (num_doppler_bins, num_data_bits)
)
    sub_len = num_doppler_bins ÷ num_data_bits   # rows per data bit sub-block

    for col_idx in 1:samples_per_code
        # Compute num_data_bits sub-block FFTs for column col_idx
        for bit_idx in 0:num_data_bits-1
            fill!(col_buf, zero(ComplexF32))
            for sub_row in 0:sub_len-1
                src_row = ((row_offset + bit_idx * sub_len + sub_row) % num_doppler_bins) + 1  # 1-indexed
                col_buf[sub_row + 1] = coherent_integration_matrix[src_row, col_idx]
            end
            mul!(col_buf, col_fft_plan, col_buf)
            circshift!(col_fftshift_buf, col_buf, num_doppler_bins ÷ 2)
            sub_block_ffts[:, bit_idx + 1] .= col_fftshift_buf
        end

        # Iterate over 2^(num_data_bits−1) bit sign patterns; d[0] = +1 fixed
        num_sign_patterns = 1 << (num_data_bits - 1)
        for bits in 0:num_sign_patterns-1
            fill!(col_buf, zero(ComplexF32))
            for bit_idx in 0:num_data_bits-1
                bit_sign = bit_idx == 0 ? 1.0f0 : (((bits >> (bit_idx - 1)) & 1) == 0 ? 1.0f0 : -1.0f0)
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
    _apply_code_drift!(noncoherent_integration_buf, plan, accumulation_step_index)

Circularly shift each row `w` of `noncoherent_integration_buf` by `S_CD[w]` samples to correct for
code-phase walk-off caused by carrier Doppler during coherent integration.

`accumulation_step_index` is the 0-based incoherent step index (0 → no shift).
`S_CD[w] = round(doppler_hz[w] * accumulation_step_index * T_coherent * sampling_freq_hz / carrier_freq_hz)`
"""
function _apply_code_drift!(noncoherent_integration_buf::Matrix{Float32}, plan::AcquisitionPlan, scratch, accumulation_step_index::Int)
    accumulation_step_index == 0 && return  # no drift on first step
    num_doppler_bins = size(noncoherent_integration_buf, 1)
    sampling_freq_hz = ustrip(Hz, plan.sampling_freq)
    carrier_freq_hz = ustrip(Hz, get_center_frequency(plan.system))
    coherent_duration_s = plan.num_coherently_integrated_code_periods * plan.samples_per_code / sampling_freq_hz
    row_buf = scratch.row_buf
    row_shift_buf = scratch.row_shift_buf
    for doppler_row in 1:num_doppler_bins
        doppler_hz = ustrip(Hz, plan.doppler_freqs[doppler_row])
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
    _accumulate_noncoherent_integration_step!(noncoherent_integration_matrix, coherent_integration_matrix, plan, m)

For incoherent step `m` (0-based): run the column FFT (pilot/sub-bit or data), apply
bit edge search and code drift correction, then accumulate into `noncoherent_integration_matrix`.

For pilot channels and sub-bit integration (num_data_bits == 1, bit_edge_search_steps == 1):
uses the fast pilot path (_accumulate_noncoherent_integration_pilot!) — one column FFT per column, no bit search.

For sub-bit integration with bit edge search (num_data_bits == 1, bit_edge_search_steps > 1):
uses _accumulate_noncoherent_integration_data_bits! with bit_edge_search_steps candidate alignments to find the one
that avoids a bit transition within the window.

For multi-bit integration (num_data_bits > 1): uses _accumulate_noncoherent_integration_data_bits! with both
bit edge search and 2^(num_data_bits-1) bit sign combination search.
"""
function _accumulate_noncoherent_integration_step!(
    noncoherent_integration_matrix::Matrix{Float32},
    coherent_integration_matrix::Matrix{ComplexF32},
    plan::AcquisitionPlan,
    scratch,
    accumulation_step_index::Int,
)
    num_doppler_bins = plan.num_coherently_integrated_code_periods * plan.num_blocks

    if plan.num_data_bits == 1 && plan.bit_edge_search_steps == 1
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
        noncoherent_integration_max_buf = scratch.noncoherent_integration_max_buf
        noncoherent_integration_buf = scratch.noncoherent_integration_buf
        # Data channel or sub-bit with bit edge search: iterate over bit_edge_search_steps candidate
        # alignments. For sub-bit (num_data_bits==1), only one bit sign pattern exists so
        # _accumulate_noncoherent_integration_data_bits! just picks the best-aligned window. For multi-bit
        # (num_data_bits>1) it also searches 2^(num_data_bits-1) bit sign combinations.
        fill!(noncoherent_integration_max_buf, 0.0f0)
        bit_period_codes = plan.num_coherently_integrated_code_periods ÷ plan.num_data_bits
        edge_step = bit_period_codes * plan.num_blocks ÷ plan.bit_edge_search_steps
        for bit_edge_search_idx in 0:plan.bit_edge_search_steps-1
            fill!(noncoherent_integration_buf, 0.0f0)
            row_offset = bit_edge_search_idx * edge_step
            _accumulate_noncoherent_integration_data_bits!(noncoherent_integration_buf, coherent_integration_matrix, scratch.col_buf, scratch.col_fftshift_buf,
                plan.col_fft_plan, plan.samples_per_code, num_doppler_bins,
                plan.num_data_bits, row_offset, scratch.sub_block_ffts)
            _apply_code_drift!(noncoherent_integration_buf, plan, scratch, accumulation_step_index)
            @. noncoherent_integration_max_buf = max(noncoherent_integration_max_buf, noncoherent_integration_buf)
        end
        # `_accumulate_noncoherent_integration_data_bits!` already fftshifts each
        # column FFT internally (circshift by N÷2), so `noncoherent_integration_max_buf`
        # is in sorted-Doppler order — the same order `_apply_code_drift!` assumes
        # and the result extraction reads. Accumulate directly; applying
        # `fftshift_perm` here would shift a second time (the two compose to the
        # identity), leaving the Doppler axis in raw FFT order and biasing every
        # reported Doppler by half the searched band. The pilot reference path
        # (raw-order kernel) is the one that pairs with `_scatter_fftshift_accumulate!`.
        # Both arrays are (num_doppler_bins, samples_per_code), so a whole-array
        # broadcast covers the same extent as the former scatter.
        noncoherent_integration_matrix .+= noncoherent_integration_max_buf
    end
end
# Convenience wrapper for tests and single-threaded callers — routes through
# thread 1's scratch (see `_default_scratch`).
_accumulate_noncoherent_integration_step!(nim, cim, plan::AcquisitionPlan, accumulation_step_index::Int) =
    _accumulate_noncoherent_integration_step!(nim, cim, plan, _default_scratch(plan), accumulation_step_index)
