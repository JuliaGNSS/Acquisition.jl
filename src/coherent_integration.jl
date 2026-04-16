# src/coherent_integration.jl

"""
    _build_coherent_integration_matrix!(coherent_integration_matrix, signal_f32, prn_fft_matrix, samples_per_code, num_blocks, block_size, num_coherently_integrated_code_periods, double_block_buf, fft_plan, bfft_plan)

Fill `coherent_integration_matrix` (size `(num_coherently_integrated_code_periods*num_blocks, samples_per_code)`) with double-block correlation results.

## Structure

The coherent integration matrix is partitioned into `num_blocks` column blocks of `block_size` columns each.
Column block `r` (columns `r*block_size+1 .. (r+1)*block_size`) is filled by correlating each
signal double-block `k` with PRN sub-block `(k + r) mod num_blocks`:

    coherent_integration_matrix[p*num_blocks + k + 1, r*block_size+1 : (r+1)*block_size] =
        IFFT( FFT(signal_double_block_{p*num_blocks+k}) * conj(FFT(prn_sub_block_{(k+r) mod num_blocks})) )[1:block_size]

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
    signal_f32::Vector{ComplexF32},
    prn_conj_fft_matrix::Matrix{ComplexF32},  # (double_block_size, num_blocks), already conjugated
    samples_per_code::Int,
    num_blocks::Int,
    block_size::Int,
    num_coherently_integrated_code_periods::Int,
    double_block_buf::Vector{ComplexF32},
    corr_buf::Vector{ComplexF32},
    fft_plan,
    bfft_plan,
)
    double_block_size = 2 * block_size
    inverse_double_block_size = 1f0 / double_block_size

    fill!(coherent_integration_matrix, zero(ComplexF32))
    length(signal_f32) >= num_coherently_integrated_code_periods * samples_per_code || throw(ArgumentError(
        "signal_f32 length $(length(signal_f32)) < num_coherently_integrated_code_periods*samples_per_code = $(num_coherently_integrated_code_periods * samples_per_code)"))

    for code_period_idx in 0:num_coherently_integrated_code_periods-1
        for block_idx in 0:num_blocks-1
            global_block_idx = code_period_idx * num_blocks + block_idx   # global block index (0-based)

            # Signal double-block: two consecutive block_size blocks
            block_start  = global_block_idx * block_size + 1
            next_start = (global_block_idx + 1) * block_size + 1  # consecutive, no modular wrap needed
            # (signal_f32 has length num_coherently_integrated_code_periods*samples_per_code, last block wraps to start of segment)
            if next_start + block_size - 1 <= num_coherently_integrated_code_periods * samples_per_code
                copyto!(double_block_buf, 1,             signal_f32, block_start,  block_size)
                copyto!(double_block_buf, block_size+1,  signal_f32, next_start, block_size)
            else
                # last double-block of segment wraps
                remaining = num_coherently_integrated_code_periods * samples_per_code - (next_start - 1)
                copyto!(double_block_buf, 1,                    signal_f32, block_start,  block_size)
                copyto!(double_block_buf, block_size+1,          signal_f32, next_start, remaining)
                copyto!(double_block_buf, block_size+remaining+1, signal_f32, 1,         block_size-remaining)
            end

            # Forward FFT of double-block (in-place)
            mul!(double_block_buf, fft_plan, double_block_buf)

            # For each column block: multiply with conj(PRN sub-block (block_idx+col_block_idx) mod num_blocks)
            for col_block_idx in 0:num_blocks-1
                prn_block = mod(block_idx + col_block_idx, num_blocks)

                # Multiply double_block_buf with precomputed conj(PRN FFT) into corr_buf
                corr_buf .= double_block_buf .* view(prn_conj_fft_matrix, :, prn_block + 1)

                # Inverse FFT (unnormalised), normalise, keep first block_size
                mul!(corr_buf, bfft_plan, corr_buf)
                col_start = col_block_idx * block_size + 1
                row = global_block_idx + 1
                coherent_integration_matrix[row, col_start:col_start + block_size - 1] .=
                    view(corr_buf, 1:block_size) .* inverse_double_block_size
            end
        end
    end
end
