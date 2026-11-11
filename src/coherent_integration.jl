# src/coherent_integration.jl

"""
    _precompute_signal_block_ffts!(signal_block_ffts, signal_f32, samples_per_code, num_blocks, block_size, num_coherently_integrated_code_periods, double_block_buf, fft_plan, code_doppler_shift_per_period=0.0)

Precompute the FFT of every signal double-block once per `acquire!` call.
Each column `global_block_idx + 1` of `signal_block_ffts` (size
`(double_block_size, num_coh*num_blocks)`) holds the FFT of the
`global_block_idx`-th double-block of `signal_f32`. Hoisting this out of the
PRN loop avoids redoing the same FFT for each of the (typically 32) PRNs.
When `code_doppler_shift_per_period` is nonzero, period `p` is delayed by
`p * code_doppler_shift_per_period` samples with a fractional-delay phase ramp.
This compensates the pass centre's code advance while retaining the original
sample grid and PRN FFTs. Drift within one code period remains uncompensated.
"""
function _precompute_signal_block_ffts!(
    signal_block_ffts::AbstractMatrix{ComplexF32},  # (double_block_size, num_coh * num_blocks) — may be one segment slice of the plan's all-segment cache
    signal_f32::Vector{ComplexF32},
    samples_per_code::Int,
    num_blocks::Int,
    block_size::Int,
    num_coherently_integrated_code_periods::Int,
    double_block_buf::Vector{ComplexF32},
    fft_plan,
    code_doppler_shift_per_period::Float64 = 0.0,
)
    double_block_size = 2 * block_size
    # The inner FFT is run at `fft_size` (>= double_block_size) — the padded,
    # fast length chosen in `plan_acquire` (`_inner_fft_size`). The output cache defines it; the
    # buffer and FFT plan match it. The pad tail [double_block_size+1 : fft_size]
    # carries only zeros, so the first `block_size` correlation lags recovered in
    # `_build_coherent_tile!` are unchanged (linear-corr length 2·bs-1 <= fft_size).
    fft_size = size(signal_block_ffts, 1)
    segment_length = num_coherently_integrated_code_periods * samples_per_code
    length(signal_f32) >= segment_length || throw(ArgumentError(
        "signal_f32 length $(length(signal_f32)) < num_coherently_integrated_code_periods*samples_per_code = $segment_length"))

    @inbounds for global_block_idx in 0:num_coherently_integrated_code_periods*num_blocks-1
        # Clear the pad tail — the previous iteration's in-place FFT overwrote it.
        # No-op when fft_size == double_block_size (unpadded / smooth size).
        for i in double_block_size+1:fft_size
            double_block_buf[i] = zero(ComplexF32)
        end
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
        # Delay period p by p * delta samples to remove the pass centre's
        # code advance. Use signed FFT frequencies for a fractional delay;
        # padding is left in place and only the retained linear-correlation
        # lags are consumed. The zero-delay path is exactly the original FFT.
        delay = (global_block_idx ÷ num_blocks) * code_doppler_shift_per_period
        if !iszero(delay)
            phase_step = cis(-2π * delay / fft_size)
            phase = 1.0 + 0.0im
            for i in 1:fft_size÷2+1
                double_block_buf[i] *= ComplexF32(phase)
                phase *= phase_step
            end
            phase = cis(2π * (fft_size - (fft_size ÷ 2 + 1)) * delay / fft_size)
            for i in fft_size÷2+2:fft_size
                double_block_buf[i] *= ComplexF32(phase)
                phase *= phase_step
            end
        end
        copyto!(signal_block_ffts, global_block_idx * fft_size + 1,
                double_block_buf, 1, fft_size)
    end
    return signal_block_ffts
end

"""
    _build_coherent_integration_matrix!(coherent_integration_matrix, signal_block_ffts, prn_conj_fft_matrix, samples_per_code, num_blocks, block_size, num_coherently_integrated_code_periods, corr_buf, bfft_plan)

Fill `coherent_integration_matrix` (size `(num_coherently_integrated_code_periods*num_blocks, samples_per_code)`) with double-block correlation results, given precomputed signal-block FFTs.

Reference/test form of the per-PRN build: the production pipeline builds and
consumes one column-block tile at a time via [`_build_coherent_tile!`](@ref)
instead of materialising this matrix. Signal FFTs do not depend on PRN, so they
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

After the column-wise FFT (`_accumulate_noncoherent_integration_pilot!` or `_sign_search_step!`), the peak
column index `pc` (0-indexed) satisfies:

    tau_samples = (num_blocks - pc ÷ block_size) mod num_blocks * block_size + pc mod block_size
"""
function _build_coherent_integration_matrix!(
    coherent_integration_matrix::AbstractMatrix{ComplexF32},
    signal_block_ffts::AbstractMatrix{ComplexF32},  # (double_block_size, num_coh*num_blocks) — precomputed
    prn_conj_fft_matrix::Matrix{ComplexF32},  # (double_block_size, num_blocks), already conjugated
    samples_per_code::Int,
    num_blocks::Int,
    block_size::Int,
    num_coherently_integrated_code_periods::Int,
    corr_buf::Vector{ComplexF32},
    bfft_plan,
)
    # Reference/test entry point: materialises the full matrix by looping the
    # tiled builder over every column block. The production pipeline never
    # calls this — it builds one (num_doppler_bins, block_size) tile at a time
    # via `_build_coherent_tile!` and reduces it immediately.
    for col_block_idx in 0:num_blocks-1
        col_start = col_block_idx * block_size + 1
        tile = view(coherent_integration_matrix, :, col_start:col_start + block_size - 1)
        _build_coherent_tile!(tile, signal_block_ffts, prn_conj_fft_matrix,
            col_block_idx, num_blocks, block_size,
            num_coherently_integrated_code_periods, corr_buf, bfft_plan)
    end
    return coherent_integration_matrix
end

"""
    _build_coherent_tile!(tile, signal_block_ffts, prn_conj_fft_matrix, col_block_idx,
                          num_blocks, block_size, num_coherently_integrated_code_periods,
                          corr_buf, bfft_plan)

Fill `tile` (size `(num_coh*num_blocks, block_size)`) with column block
`col_block_idx` (0-based) of the coherent integration matrix — i.e. columns
`col_block_idx*block_size+1 .. (col_block_idx+1)*block_size` of the full
surface. Row `global_block_idx + 1` holds the first `block_size` lags of the
correlation of signal double-block `global_block_idx` with PRN sub-block
`(block_idx + col_block_idx) mod num_blocks` (see
[`_build_coherent_integration_matrix!`](@ref) for the full structure and delay
recovery).

This is the unit of work of the tiled pipeline: every downstream consumer
(column FFT, sign search, rotation search) is column-oriented, so the full
`(num_doppler_bins, samples_per_code)` matrix never needs to exist — each tile
is built, reduced, and overwritten by the next one. The tile is `num_blocks`×
smaller than the full matrix and typically L2-cache resident, which also spares
the column FFTs a round-trip through DRAM.
"""
function _build_coherent_tile!(
    tile::AbstractMatrix{ComplexF32},         # (num_coh*num_blocks, block_size)
    signal_block_ffts::AbstractMatrix{ComplexF32},  # (double_block_size, num_coh*num_blocks) — precomputed
    prn_conj_fft_matrix::Matrix{ComplexF32},  # (double_block_size, num_blocks), already conjugated
    col_block_idx::Int,                       # 0-based column block index
    num_blocks::Int,
    block_size::Int,
    num_coherently_integrated_code_periods::Int,
    corr_buf::Vector{ComplexF32},
    bfft_plan,
)
    # Normalise by the ACTUAL inverse-FFT length. corr_buf may be zero-padded to
    # the fast `fft_size` >= 2*block_size chosen in `plan_acquire`; dividing
    # by that length recovers correlation values identical to the unpadded case
    # (the standard IFFT 1/N and the N summed terms cancel to the true lag).
    inv_fft_size = 1f0 / length(corr_buf)

    # No fill!: every cell of `tile` is written exactly once — the row loops
    # cover all `num_coh × num_blocks` rows and each IFFT fills a whole row.
    for code_period_idx in 0:num_coherently_integrated_code_periods-1
        for block_idx in 0:num_blocks-1
            global_block_idx = code_period_idx * num_blocks + block_idx   # 0-based
            sig_fft_col = view(signal_block_ffts, :, global_block_idx + 1)
            prn_block = mod(block_idx + col_block_idx, num_blocks)

            # Multiply precomputed signal FFT with the conjugated PRN FFT.
            corr_buf .= sig_fft_col .* view(prn_conj_fft_matrix, :, prn_block + 1)

            # Inverse FFT (unnormalised), normalise, keep first block_size lags.
            mul!(corr_buf, bfft_plan, corr_buf)
            row = global_block_idx + 1
            tile[row, 1:block_size] .= view(corr_buf, 1:block_size) .* inv_fft_size
        end
    end
    return tile
end

"""
    _build_coherent_tile_t!(tile_t, signal_block_ffts, prn_conj_fft_matrix, col_block_idx, num_blocks, block_size, num_coherently_integrated_code_periods, corr_buf, bfft_plan)

[`_build_coherent_tile!`](@ref) in the TRANSPOSED layout: `tile_t` is
`(block_size, num_doppler_bins)`, column `m` holding coherent-integration row
`m` (double block `m - 1`) for the tile's `block_size` code phases. Each
double-block IFFT result is written as a contiguous column instead of a
strided row.

The simple path follows this with one Doppler FFT along dim 2
([`_build_doppler_tile_t!`](@ref)); the secondary-code rotation search first
sign-modulates the Doppler axis once per pattern (see
[`_rotation_tile_power!`](@ref)).
"""
function _build_coherent_tile_t!(
    tile_t::AbstractMatrix{ComplexF32},       # (block_size, num_coh*num_blocks)
    signal_block_ffts::AbstractMatrix{ComplexF32},
    prn_conj_fft_matrix::Matrix{ComplexF32},
    col_block_idx::Int,
    num_blocks::Int,
    block_size::Int,
    num_coherently_integrated_code_periods::Int,
    corr_buf::Vector{ComplexF32},
    bfft_plan,
)
    inv_fft_size = 1f0 / length(corr_buf)
    for code_period_idx in 0:num_coherently_integrated_code_periods-1
        for block_idx in 0:num_blocks-1
            global_block_idx = code_period_idx * num_blocks + block_idx   # 0-based
            prn_block = mod(block_idx + col_block_idx, num_blocks)
            corr_buf .= view(signal_block_ffts, :, global_block_idx + 1) .*
                        view(prn_conj_fft_matrix, :, prn_block + 1)
            mul!(corr_buf, bfft_plan, corr_buf)
            col = global_block_idx + 1
            @inbounds @simd for i in 1:block_size
                tile_t[i, col] = corr_buf[i] * inv_fft_size
            end
        end
    end
    return tile_t
end

"""
    _build_doppler_tile_t!(tile_t, signal_block_ffts, prn_conj_fft_matrix, col_block_idx, num_blocks, block_size, num_coherently_integrated_code_periods, corr_buf, bfft_plan, doppler_fft_plan)

Simple-path (pilot / sub-bit) tile: [`_build_coherent_tile_t!`](@ref) followed
by the Doppler FFT along dim 2, so column `r` of `tile_t` holds raw
(un-fftshifted) Doppler bin `r` for the tile's `block_size` code phases.

The Doppler FFT runs one FFT per code phase, batched over contiguous
`block_size`-long vectors, which vectorises across the code-phase axis. With
FFTL this is faster than both the batched dim-1 FFT and the per-column FFTs of
the `(num_doppler_bins, block_size)` layout at every tile shape, so no
batched/per-column crossover is needed.
"""
function _build_doppler_tile_t!(
    tile_t::AbstractMatrix{ComplexF32},       # (block_size, num_coh*num_blocks)
    signal_block_ffts::AbstractMatrix{ComplexF32},
    prn_conj_fft_matrix::Matrix{ComplexF32},
    col_block_idx::Int,
    num_blocks::Int,
    block_size::Int,
    num_coherently_integrated_code_periods::Int,
    corr_buf::Vector{ComplexF32},
    bfft_plan,
    doppler_fft_plan,
)
    _build_coherent_tile_t!(tile_t, signal_block_ffts, prn_conj_fft_matrix,
        col_block_idx, num_blocks, block_size,
        num_coherently_integrated_code_periods, corr_buf, bfft_plan)
    mul!(tile_t, doppler_fft_plan, tile_t)
    return tile_t
end

# Raw (FFT-order) Doppler bin feeding sorted (fftshifted) row `dst`:
# sorted rows 1..shift come from raw bins front+1..N, rows shift+1..N from 1..front.
@inline _raw_doppler_bin(dst::Int, shift::Int, front::Int) =
    dst <= shift ? dst + front : dst - shift

# Power of one Doppler-domain tile cell: a complex FFT output (`abs2`) or a cell
# that already holds power (the max-reduced rotation-search tile).
@inline _cell_power(x::ComplexF32) = abs2(x)
@inline _cell_power(x::Float32) = x
