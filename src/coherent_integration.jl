# src/coherent_integration.jl

"""
    _precompute_signal_block_ffts!(signal_block_ffts, signal_f32, samples_per_code, num_blocks, block_size, num_coherently_integrated_code_periods, double_block_buf, fft_plan)

Precompute the FFT of every signal double-block once per `acquire!` call.
Each column `global_block_idx + 1` of `signal_block_ffts` (size
`(double_block_size, num_coh*num_blocks)`) holds the FFT of the
`global_block_idx`-th double-block of `signal_f32`. Hoisting this out of the
PRN loop avoids redoing the same FFT for each of the (typically 32) PRNs.
"""
function _precompute_signal_block_ffts!(
    signal_block_ffts::Matrix{ComplexF32},  # (double_block_size, num_coh * num_blocks)
    signal_f32::Vector{ComplexF32},
    samples_per_code::Int,
    num_blocks::Int,
    block_size::Int,
    num_coherently_integrated_code_periods::Int,
    double_block_buf::Vector{ComplexF32},
    fft_plan,
)
    double_block_size = 2 * block_size
    segment_length = num_coherently_integrated_code_periods * samples_per_code
    length(signal_f32) >= segment_length || throw(ArgumentError(
        "signal_f32 length $(length(signal_f32)) < num_coherently_integrated_code_periods*samples_per_code = $segment_length"))

    @inbounds for global_block_idx in 0:num_coherently_integrated_code_periods*num_blocks-1
        block_start = global_block_idx * block_size + 1
        next_start  = (global_block_idx + 1) * block_size + 1
        if next_start + block_size - 1 <= segment_length
            copyto!(double_block_buf, 1,             signal_f32, block_start, block_size)
            copyto!(double_block_buf, block_size+1,  signal_f32, next_start,  block_size)
        else
            # last double-block of segment wraps to the start
            remaining = segment_length - (next_start - 1)
            copyto!(double_block_buf, 1,                       signal_f32, block_start, block_size)
            copyto!(double_block_buf, block_size+1,            signal_f32, next_start,  remaining)
            copyto!(double_block_buf, block_size+remaining+1,  signal_f32, 1,           block_size - remaining)
        end
        mul!(double_block_buf, fft_plan, double_block_buf)
        copyto!(signal_block_ffts, global_block_idx * double_block_size + 1,
                double_block_buf, 1, double_block_size)
    end
    return signal_block_ffts
end

"""
    _build_coherent_integration_matrix!(coherent_integration_matrix, signal_block_ffts, prn_conj_fft_matrix, samples_per_code, num_blocks, block_size, num_coherently_integrated_code_periods, corr_buf, bfft_plan)

Fill `coherent_integration_matrix` (size `(num_coherently_integrated_code_periods*num_blocks, samples_per_code)`) with double-block correlation results, given precomputed signal-block FFTs.

This is the per-PRN inner kernel. Signal FFTs do not depend on PRN, so they
are computed once per `acquire!` call by [`_precompute_signal_block_ffts!`](@ref)
and reused across all PRNs.

## Structure

The coherent integration matrix is partitioned into `num_blocks` column blocks of `block_size` columns each.
Column block `r` (columns `r*block_size+1 .. (r+1)*block_size`) is filled by correlating each
signal double-block `k` with PRN sub-block `(k + r) mod num_blocks`:

    coherent_integration_matrix[p*num_blocks + k + 1, r*block_size+1 : (r+1)*block_size] =
        IFFT( signal_block_ffts[:, p*num_blocks+k+1] * conj(FFT(prn_sub_block_{(k+r) mod num_blocks})) )[1:block_size]

This ensures that for the true code delay `tau_samples = k_true*block_size + fine`:
- Every signal block `k` has exactly one column block `r = (num_blocks - k_true) mod num_blocks` where
  the correlation peaks at lag `fine` with full block_size amplitude.
- All num_blocks rows within each code period contribute coherently → full samples_per_code-chip correlation gain.

## Delay recovery

After the column-wise FFT (`_accumulate_noncoherent_integration_pilot!` or `_accumulate_noncoherent_integration_data_bits!`), the peak
column index `pc` (0-indexed) satisfies:

    tau_samples = (num_blocks - pc ÷ block_size) mod num_blocks * block_size + pc mod block_size
"""
function _build_coherent_integration_matrix!(
    coherent_integration_matrix::Matrix{ComplexF32},
    signal_block_ffts::Matrix{ComplexF32},  # (double_block_size, num_coh*num_blocks) — precomputed
    prn_conj_fft_matrix::Matrix{ComplexF32},  # (double_block_size, num_blocks), already conjugated
    samples_per_code::Int,
    num_blocks::Int,
    block_size::Int,
    num_coherently_integrated_code_periods::Int,
    corr_buf::Vector{ComplexF32},
    bfft_plan,
)
    double_block_size = 2 * block_size
    inverse_double_block_size = 1f0 / double_block_size

    # No fill!: every cell of `coherent_integration_matrix` is written exactly
    # once. Outer loops cover all `num_coh × num_blocks` rows, the column loop
    # tiles all `num_blocks × block_size = samples_per_code` columns.

    for code_period_idx in 0:num_coherently_integrated_code_periods-1
        for block_idx in 0:num_blocks-1
            global_block_idx = code_period_idx * num_blocks + block_idx   # 0-based
            sig_fft_col = view(signal_block_ffts, :, global_block_idx + 1)

            # For each column block: multiply precomputed signal FFT with
            # conj(PRN sub-block (block_idx+col_block_idx) mod num_blocks).
            for col_block_idx in 0:num_blocks-1
                prn_block = mod(block_idx + col_block_idx, num_blocks)

                # Multiply precomputed signal FFT with the conjugated PRN FFT.
                corr_buf .= sig_fft_col .* view(prn_conj_fft_matrix, :, prn_block + 1)

                # Inverse FFT (unnormalised), normalise, keep first block_size.
                mul!(corr_buf, bfft_plan, corr_buf)
                col_start = col_block_idx * block_size + 1
                row = global_block_idx + 1
                coherent_integration_matrix[row, col_start:col_start + block_size - 1] .=
                    view(corr_buf, 1:block_size) .* inverse_double_block_size
            end
        end
    end
end
