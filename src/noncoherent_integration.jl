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
    _apply_code_drift!(buf, plan, plan, accumulation_step_index)

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
    noncoherent_integration_max_buf = scratch.noncoherent_integration_max_buf
    noncoherent_integration_buf = scratch.noncoherent_integration_buf

    if plan.num_data_bits == 1 && plan.bit_edge_search_steps == 1
        # Pilot channel or no bit edge search requested: fast path
        fill!(noncoherent_integration_buf, 0.0f0)
        if num_doppler_bins <= 320
            # Batched column FFT: FFT all columns at once along dim 1 (in-place on CIM).
            # Faster than individual column FFTs when num_doppler_bins is small because
            # it amortises FFTW plan-dispatch overhead across all columns.
            mul!(coherent_integration_matrix, plan.col_batch_fft_plan, coherent_integration_matrix)
            @inbounds for col_idx in 1:plan.samples_per_code
                for doppler_bin in 1:num_doppler_bins
                    noncoherent_integration_buf[doppler_bin, col_idx] += abs2(coherent_integration_matrix[doppler_bin, col_idx])
                end
            end
        else
            # Individual column FFTs: better cache utilisation for large num_doppler_bins.
            _accumulate_noncoherent_integration_pilot!(noncoherent_integration_buf, coherent_integration_matrix, scratch.col_buf,
                plan.col_fft_plan, plan.samples_per_code)
        end
        _apply_code_drift!(noncoherent_integration_buf, plan, scratch, accumulation_step_index)
        _scatter_fftshift_accumulate!(noncoherent_integration_matrix, noncoherent_integration_buf, plan.fftshift_perm, plan.samples_per_code)
    else
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
        _scatter_fftshift_accumulate!(noncoherent_integration_matrix, noncoherent_integration_max_buf, plan.fftshift_perm, plan.samples_per_code)
    end
end
# Convenience wrapper for tests and single-threaded callers (scratch = plan itself)
_accumulate_noncoherent_integration_step!(nim, cim, plan::AcquisitionPlan, accumulation_step_index::Int) =
    _accumulate_noncoherent_integration_step!(nim, cim, plan, plan, accumulation_step_index)
