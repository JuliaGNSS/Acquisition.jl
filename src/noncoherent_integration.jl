# src/noncoherent_integration.jl
#
# The FM-DBZP reduction stage. Production runs TILED: the per-PRN driver
# (`_accumulate_prn_step_tiled!` at N_nc > 1, `_acquire_prn_streamed!` at
# N_nc == 1) builds one (num_doppler_bins, block_size) column-block tile of the
# coherent integration surface at a time and reduces it immediately, so no
# buffer proportional to `samples_per_code` lives in the per-thread scratch.
# The kernels here therefore come in two granularities:
#
#   - per-column primitives (`_sign_search_column!`, `_rotation_column!`) and
#     tile consumers (the fused pilot kernels, `_accumulate_sign_tile!`) — the
#     production hot path;
#   - full-matrix wrappers (`_sign_search_step!`,
#     `_sign_search_step_with_rotations!`, `_accumulate_noncoherent_integration_step!`)
#     that delegate to the primitives column by column. They preserve the
#     historical single-call semantics for the test suite's kernel-parity
#     checks and are NOT called from the production pipeline.

"""
    _accumulate_noncoherent_integration_pilot!(noncoherent_integration_matrix, coherent_integration_matrix, col_buf, col_fft_plan, samples_per_code)

Column-wise FFT path for pilot channels and sub-bit integration (num_data_bits = 1,
no data bit combination search). Applies fftshift so row indices map to sorted
`doppler_freqs`, accumulates `|FFT|²` into `noncoherent_integration_matrix` (shape `(num_doppler_bins, samples_per_code)`).

Reference implementation used by tests; production uses the fused kernels below.
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
    _sign_search_column!(stage_col, cim, src_col, col_buf, col_fft_plan,
                         num_doppler_bins, num_data_bits, row_offset, sub_block_ffts, patterns)

Sign-pattern search for ONE code-phase column: computes the `num_data_bits`
sub-block FFTs of column `src_col` of `cim` (a tile or a full coherent
integration matrix — either way the column carries all `num_doppler_bins`
rows), combines them per sign pattern, and max-reduces `|result|²` into
`stage_col` in **sorted-Doppler row order** (fftshift folded as a top/bottom
half swap). The caller zeroes `stage_col` once per column; calling again with a
different `row_offset` folds further bit-edge alignments into the same max.

This is the per-column core of [`_sign_search_step!`](@ref); see that docstring
for the algorithm.
"""
function _sign_search_column!(
    stage_col::AbstractVector{Float32},
    cim::AbstractMatrix{ComplexF32},
    src_col::Int,
    col_buf::Vector{ComplexF32},
    col_fft_plan,
    num_doppler_bins::Int,
    num_data_bits::Int,
    row_offset::Int,
    sub_block_ffts::Matrix{ComplexF32},  # (num_doppler_bins, ≥ num_data_bits)
    patterns::Matrix{Float32},           # (num_coh_periods, num_patterns)
)
    sub_len = num_doppler_bins ÷ num_data_bits   # rows per data bit sub-block
    num_sign_patterns = size(patterns, 2)
    coh_per_bit = size(patterns, 1) ÷ num_data_bits

    # Compute num_data_bits sub-block FFTs for this column
    for bit_idx in 0:num_data_bits-1
        fill!(col_buf, zero(ComplexF32))
        for sub_row in 0:sub_len-1
            src_row = ((row_offset + bit_idx * sub_len + sub_row) % num_doppler_bins) + 1  # 1-indexed
            col_buf[sub_row + 1] = cim[src_row, src_col]
        end
        mul!(col_buf, col_fft_plan, col_buf)
        sub_block_ffts[:, bit_idx + 1] .= col_buf   # raw FFT order; fftshift folds into the stage write below
    end

    # Iterate over precomputed sign patterns; sign per data bit from the first row
    # of that bit's segment in `patterns`. We write directly in sorted-Doppler row
    # order: fftshift is the top/bottom half swap, so the raw-bin range [1, half]
    # maps to dst rows [half+1, N] and [half+1, N] maps to [1, half]. Two contiguous
    # SIMD-friendly half-loops replace what used to be a separate scatter pass.
    half = num_doppler_bins ÷ 2
    for p in 1:num_sign_patterns
        fill!(col_buf, zero(ComplexF32))
        for bit_idx in 0:num_data_bits-1
            bit_sign = patterns[bit_idx * coh_per_bit + 1, p]
            col_buf .+= bit_sign .* view(sub_block_ffts, :, bit_idx + 1)
        end
        @inbounds for ω in 1:half
            cell_power = abs2(col_buf[ω])
            dst_row = ω + half
            if cell_power > stage_col[dst_row]
                stage_col[dst_row] = cell_power
            end
        end
        @inbounds for ω in half+1:num_doppler_bins
            cell_power = abs2(col_buf[ω])
            dst_row = ω - half
            if cell_power > stage_col[dst_row]
                stage_col[dst_row] = cell_power
            end
        end
    end
    return stage_col
end

"""
    _sign_search_step!(noncoherent_integration_buf, coherent_integration_matrix, col_buf, col_fft_plan,
                       samples_per_code, num_doppler_bins, num_data_bits, row_offset, sub_block_ffts, patterns)

Sign-pattern search step for one bit-edge alignment of one non-coherent block over a
fully materialised coherent integration matrix. Does *not* perform non-coherent
accumulation across blocks. The patterns to enumerate are supplied via `patterns`
(shape `(num_coh_periods, num_patterns)`), built by [`sign_patterns`](@ref).

FM-DBZP column-wise FFT for data channels (paper: N_db > 1). For each of the
samples_per_code columns:
1. Splits the column (with circular row offset `row_offset`) into num_data_bits sub-blocks
   of length num_doppler_bins÷num_data_bits, zero-pads each to num_doppler_bins, and FFTs it.
2. For each sign pattern, sums ±sub-block FFTs (sign per data bit looked up from the
   first row of that bit's segment in `patterns`), and accumulates the cell-wise maximum
   of |result|² into `noncoherent_integration_buf` **in sorted-Doppler row order**
   (fftshift folded inline as a top/bottom half-row swap).

`sub_block_ffts` is a pre-allocated (num_doppler_bins, num_data_bits) scratch matrix.

Reference/test wrapper: delegates to [`_sign_search_column!`](@ref) column by
column. Production consumes tiles instead and never materialises the full CIM.
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
    for col_idx in 1:samples_per_code
        _sign_search_column!(view(noncoherent_integration_buf, :, col_idx),
            coherent_integration_matrix, col_idx, col_buf, col_fft_plan,
            num_doppler_bins, num_data_bits, row_offset, sub_block_ffts, patterns)
    end
end

"""
    _rotation_column!(stage, cim, src_col, cp_col, col_buf, combine_buf_re, combine_buf_im,
                      col_fft_plan, num_doppler_bins, num_coh_periods, num_data_combos,
                      num_blocks, block_size, row_offset, sub_block_ffts,
                      tiled_phase_patterns_re, tiled_phase_patterns_im, use_max)

Secondary-code rotation search for ONE code-phase column: computes one sub-block
FFT per coherent period from column `src_col` of `cim`, combines them with the
pre-tiled complex phasors, and writes `|result|²` of each pattern into its
rotation's column of `stage` (shape `(num_doppler_bins, ≥ num_rotations)`), in
sorted-Doppler row order (fftshift folded).

`cp_col` is the 1-based code-phase column in `[1, samples_per_code]` that
`src_col` corresponds to — the per-column block alignment (see
[`_sign_search_step_with_rotations!`](@ref)) depends on the code-phase, not on
the position within the tile. With `use_max = false` (single alignment, single
data-bit combo — the dominant case) each cell is a plain SIMD store; otherwise
patterns max-reduce into the stage, which also folds bit-edge alignments across
repeated calls. The caller zeroes `stage` once per column.

This is the per-column core of [`_sign_search_step_with_rotations!`](@ref); see
that docstring for the algorithm, the inter-sub-block phase ramp, and the
per-column block alignment.
"""
function _rotation_column!(
    stage::AbstractMatrix{Float32},          # (num_doppler_bins, ≥ num_rotation_blocks)
    cim::AbstractMatrix{ComplexF32},
    src_col::Int,
    cp_col::Int,
    col_buf::Vector{ComplexF32},
    combine_buf_re::Vector{Float32},
    combine_buf_im::Vector{Float32},
    col_fft_plan,
    num_doppler_bins::Int,
    num_coh_periods::Int,
    num_data_combos::Int,
    num_blocks::Int,
    block_size::Int,
    row_offset::Int,
    sub_block_ffts::Matrix{ComplexF32},
    tiled_phase_patterns_re::Array{Float32,3},
    tiled_phase_patterns_im::Array{Float32,3},
    use_max::Bool,
)
    num_sign_patterns = size(tiled_phase_patterns_re, 3)
    half_block = block_size >> 1

    # Snap the sub-block partition to the primary-code wrap for this column.
    # `wrap_sample` is the FM-DBZP delay associated with `cp_col`; rounding
    # to the nearest multiple of block_size aligns sub-block 0 (and hence
    # every sub-block, by uniform shift) to within ±half_block of an NH10
    # chip boundary. Cheap (four integer ops per column) and recovers ~6 dB
    # of coherent gain at the worst code phase.
    pc = cp_col - 1
    wrap_sample = ((num_blocks - div(pc, block_size)) % num_blocks) * block_size + (pc % block_size)
    col_row_offset = div(wrap_sample + half_block, block_size) % num_blocks
    eff_row_offset = (row_offset + col_row_offset) % num_doppler_bins

    # One sub-block FFT per coherent period (sub-block size = num_blocks).
    for k in 0:num_coh_periods-1
        fill!(col_buf, zero(ComplexF32))
        for sub_row in 0:num_blocks-1
            src_row = ((eff_row_offset + k * num_blocks + sub_row) % num_doppler_bins) + 1
            col_buf[sub_row + 1] = cim[src_row, src_col]
        end
        mul!(col_buf, col_fft_plan, col_buf)
        sub_block_ffts[:, k + 1] .= col_buf   # raw FFT order; fftshift folds into the stage write below
    end

    # Combine sub-block FFTs with the precomputed (ω, p, q)-tiled phasors,
    # then WRITE each pattern to its ROTATION's stage column in sorted-Doppler
    # row order. Loop ordering: outer pattern q, middle sub-block p, inner ω.
    # The complex MAC is unfused into four real `muladd`s so the compiler packs
    # the inner loop into Float32 SIMD lanes — measured ~3× faster than the
    # equivalent ComplexF32-storage form.
    #
    # fftshift is folded into the row-write here: raw bin ω∈[1, half] maps
    # to sorted-Doppler row ω+half, and ω∈[half+1, N] maps to ω-half. Two
    # contiguous SIMD-friendly half-loops.
    half = num_doppler_bins ÷ 2
    for q in 1:num_sign_patterns
        fill!(combine_buf_re, 0f0)
        fill!(combine_buf_im, 0f0)
        for p in 1:num_coh_periods
            @inbounds @simd for ω in 1:num_doppler_bins
                ar = tiled_phase_patterns_re[ω, p, q]
                ai = tiled_phase_patterns_im[ω, p, q]
                br, bi = reim(sub_block_ffts[ω, p])
                combine_buf_re[ω] = muladd(ar, br, muladd(-ai, bi, combine_buf_re[ω]))
                combine_buf_im[ω] = muladd(ar, bi, muladd( ai, br, combine_buf_im[ω]))
            end
        end
        # Patterns are laid out rotation-major, data-bit-minor (see
        # `sign_patterns`): pattern q decodes to rotation
        # `(q-1) ÷ num_data_combos` and data-bit combo `(q-1) % num_data_combos`.
        # ROTATIONS are distinct secondary-code phases, so each gets its own
        # stage column (no collapse; preserves the matched-filter CFAR layout).
        # DATA-BIT polarity is a nuisance hypothesis, so the `num_data_combos`
        # patterns sharing a rotation are collapsed by a cell-wise MAX.
        #
        # `use_max == false` (single alignment, no multi-bit search) is the
        # dominant case: each pattern is its own rotation and only one alignment
        # writes, so cells never collide and a plain SIMD store applies. The
        # read-compare-store max can't vectorise like the pure store, so gating
        # it keeps the common path at full speed.
        rotation_idx = (q - 1) ÷ num_data_combos
        dst_col = rotation_idx + 1
        if !use_max
            @inbounds @simd for ω in 1:half
                stage[ω + half, dst_col] =
                    muladd(combine_buf_re[ω], combine_buf_re[ω],
                           combine_buf_im[ω] * combine_buf_im[ω])
            end
            @inbounds @simd for ω in half+1:num_doppler_bins
                stage[ω - half, dst_col] =
                    muladd(combine_buf_re[ω], combine_buf_re[ω],
                           combine_buf_im[ω] * combine_buf_im[ω])
            end
        else
            @inbounds for ω in 1:half
                cell_power = muladd(combine_buf_re[ω], combine_buf_re[ω],
                                    combine_buf_im[ω] * combine_buf_im[ω])
                dst = ω + half
                if cell_power > stage[dst, dst_col]
                    stage[dst, dst_col] = cell_power
                end
            end
            @inbounds for ω in half+1:num_doppler_bins
                cell_power = muladd(combine_buf_re[ω], combine_buf_re[ω],
                                    combine_buf_im[ω] * combine_buf_im[ω])
                dst = ω - half
                if cell_power > stage[dst, dst_col]
                    stage[dst, dst_col] = cell_power
                end
            end
        end
    end
    return stage
end

"""
    _sign_search_step_with_rotations!(noncoherent_integration_buf, coherent_integration_matrix, col_buf,
                                      combine_buf_re, combine_buf_im, col_fft_plan,
                                      samples_per_code, num_doppler_bins, num_coh_periods, num_data_combos,
                                      num_blocks, block_size, row_offset, sub_block_ffts,
                                      tiled_phase_patterns_re, tiled_phase_patterns_im)

Sign-pattern search step for one bit-edge alignment when per-coherent-period signs vary
(secondary-code rotation search), over a fully materialised coherent integration matrix.
Splits each column into `num_coh_periods` sub-blocks of `num_blocks` rows each (one per
coherent code period) and computes one column FFT per sub-block. Then, for each pattern
column, accumulates `tiled_phase_patterns[ω, p, q] * sub_block_FFT_p[ω]` over the
`num_coh_periods` sub-blocks and writes `|result|²` into `noncoherent_integration_buf`.

The `num_sign_patterns` pattern columns are the Cartesian product of
`num_secondary_rotations` secondary-code rotations and `num_data_combos = 2^(num_data_bits−1)`
data-bit polarities (rotation-major, data-minor; see [`sign_patterns`](@ref)). Each rotation
is a distinct code phase and gets its own `samples_per_code`-wide cp slice; the
`num_data_combos` polarities sharing a rotation are collapsed into that slice by a cell-wise
max (the data sign is a nuisance hypothesis). The buffer's cp axis must therefore be at least
`(num_sign_patterns ÷ num_data_combos) * samples_per_code` wide.

This is the generalisation of [`_sign_search_step!`](@ref) — that kernel groups consecutive
coherent periods into one sub-block per data bit (`num_data_bits` sub-blocks total) and is
correct only when the sign is constant within each data bit. When the secondary code varies
the sign per coherent period, we need one sub-block per period; `tiled_phase_patterns`
holds the per-(ω, p, q) complex coefficient.

## Inter-sub-block phase ramp (LongL5I equivalence)

A naive `Σ_p sign[p] · sub_block_FFT_p[ω]` combination (the previous form of this kernel)
is matched-filter optimal only when the true Doppler lands on the coarse `1 / (T_code)`
sub-block FFT grid. Between grid points the inter-sub-block phase rotates by some
`δ ∈ (0, 2π)` per code period that ±1 cannot fit, losing 3–8 dB of coherent gain. The fix
is to recognise that a length-`num_doppler_bins` column FFT decomposes as

    FFT_total(x)[ω] = Σ_p exp(-2πi · p · s / num_coh_periods) · sub_block_FFT_p[ω],
                                                                       s = ω mod num_coh_periods

so the optimal NH10-rotation-aware combination at fine-bin class `s` is

    Σ_p NH10[(p+r) mod L] · exp(-2πi · p · s / num_coh_periods) · sub_block_FFT_p[ω].

`tiled_phase_patterns_re/im[ω, p, q]` are the compact `(num_coh_periods, num_patterns,
num_coh_periods)` phasor of [`combined_phase_patterns`](@ref) pre-tiled along ω AND
split into Float32 real/imag arrays so the kernel inner loop drives Float32 SIMD;
see [`tile_phase_patterns`](@ref) for the layout. This buys the LongL5I full-FFT
sensitivity (≈5 dB) with a combine loop that runs slightly faster than the previous
±1 ComplexF32 kernel on commodity x86, because complex MACs pack better as Float32
than as ComplexF32 in Julia today. `combine_buf_re/im` are length-`num_doppler_bins`
Float32 scratch vectors holding the running per-ω complex accumulator in split form;
they live in `AcquisitionScratch` alongside `col_buf`.

## Per-column block alignment

A fixed sub-block partition (`row_offset` constant across columns) makes one NH10 chip
correspond to one sub-block *only* when the primary-code wrap lands at sample 0 of every
sub-block — i.e. at `code_phase == 0`. At any other code phase the wrap drifts inside the
sub-block, each sub-block carries a fractional mix of two consecutive NH10 chips, and no
discrete ±1 rotation hypothesis matches the mix. Worst case (wrap at 50% of the block)
burns ~6.5 dB of coherent gain and flattens the Doppler spectrum.

The wrap sample for code-phase column `c` is exactly
`W(c) = _fmdbzp_column_to_tau(c-1, num_blocks, block_size)`, so we can pick a per-column
row shift `s(c) = round(W(c)/block_size) mod num_blocks` that snaps each sub-block to the
nearest NH10 chip boundary. Residual misalignment is at most half a block_size
(≈ `1/(2*num_blocks)` of a primary-code period) regardless of code phase. The shift is
applied uniformly to every sub-block in the column, so the cross-sub-block Doppler phase
relationship is preserved (the overall phase factor — even with the complex combined
phasor — vanishes in `|·|²` because it factors out of the sum over `p`).

`sub_block_ffts` must have at least `num_coh_periods` columns.

Reference/test wrapper: delegates to [`_rotation_column!`](@ref) column by
column. Production consumes tiles instead and never materialises the full CIM.
"""
function _sign_search_step_with_rotations!(
    noncoherent_integration_buf::Matrix{Float32},
    coherent_integration_matrix::Matrix{ComplexF32},
    col_buf::Vector{ComplexF32},
    combine_buf_re::Vector{Float32},
    combine_buf_im::Vector{Float32},
    col_fft_plan,
    samples_per_code::Int,
    num_doppler_bins::Int,
    num_coh_periods::Int,
    num_data_combos::Int,
    num_blocks::Int,
    block_size::Int,
    row_offset::Int,
    sub_block_ffts::Matrix{ComplexF32},
    tiled_phase_patterns_re::Array{Float32,3},
    tiled_phase_patterns_im::Array{Float32,3},
)
    num_sign_patterns = size(tiled_phase_patterns_re, 3)
    num_data_combos >= 1 || throw(ArgumentError("num_data_combos must be >= 1, got $num_data_combos"))
    num_sign_patterns % num_data_combos == 0 || throw(ArgumentError(
        "num_sign_patterns=$num_sign_patterns must be a multiple of num_data_combos=$num_data_combos"))
    num_rotation_blocks = num_sign_patterns ÷ num_data_combos
    size(noncoherent_integration_buf, 2) >= num_rotation_blocks * samples_per_code || throw(ArgumentError(
        "noncoherent_integration_buf has $(size(noncoherent_integration_buf, 2)) cp columns, " *
        "need num_rotation_blocks*samples_per_code=$(num_rotation_blocks * samples_per_code) " *
        "(num_sign_patterns=$num_sign_patterns ÷ num_data_combos=$num_data_combos rotations × samples_per_code=$samples_per_code)"))
    size(tiled_phase_patterns_re, 1) == num_doppler_bins || throw(ArgumentError(
        "tiled_phase_patterns_re has $(size(tiled_phase_patterns_re, 1)) rows, expected num_doppler_bins=$num_doppler_bins"))
    size(tiled_phase_patterns_re, 2) == num_coh_periods || throw(ArgumentError(
        "tiled_phase_patterns_re has $(size(tiled_phase_patterns_re, 2)) sub-block columns, expected num_coh_periods=$num_coh_periods"))
    size(tiled_phase_patterns_im) == size(tiled_phase_patterns_re) || throw(ArgumentError(
        "tiled_phase_patterns_im shape $(size(tiled_phase_patterns_im)) ≠ tiled_phase_patterns_re shape $(size(tiled_phase_patterns_re))"))
    length(combine_buf_re) == num_doppler_bins && length(combine_buf_im) == num_doppler_bins ||
        throw(ArgumentError("combine_buf_re/im must each have length num_doppler_bins=$num_doppler_bins"))
    for col_idx in 1:samples_per_code
        # Each rotation's cp slice column for this code phase: col_idx + r*spc.
        stage = view(noncoherent_integration_buf, :,
            range(col_idx, step = samples_per_code, length = num_rotation_blocks))
        _rotation_column!(stage, coherent_integration_matrix, col_idx, col_idx,
            col_buf, combine_buf_re, combine_buf_im, col_fft_plan,
            num_doppler_bins, num_coh_periods, num_data_combos,
            num_blocks, block_size, row_offset, sub_block_ffts,
            tiled_phase_patterns_re, tiled_phase_patterns_im,
            num_data_combos > 1)
    end
end

"""
    _apply_code_drift!(noncoherent_integration_buf, plan, accumulation_step_index)

Circularly shift each row `w` of `noncoherent_integration_buf` by `S_CD[w]` samples to correct for
code-phase walk-off caused by carrier Doppler during coherent integration.

`accumulation_step_index` is the 0-based incoherent step index (0 → no shift).
`S_CD[w] = round(doppler_hz[w] * accumulation_step_index * T_coherent * sampling_freq_hz / carrier_freq_hz)`

The buffer is in **sorted-Doppler row order**, so `doppler_freqs[w]` directly
gives the frequency for row `w`.

Reference implementation (allocates its row scratch locally): production folds
the drift shift into the per-column destination scatter — see
[`_accumulate_sign_tile!`](@ref) and the fused drift pilot kernels — so this
function is only exercised by tests and the full-matrix
[`_accumulate_noncoherent_integration_step!`](@ref) shim.
"""
function _apply_code_drift!(noncoherent_integration_buf::Matrix{Float32}, plan::AcquisitionPlan, accumulation_step_index::Int)
    accumulation_step_index == 0 && return  # no drift on first step
    num_doppler_bins = size(noncoherent_integration_buf, 1)
    sampling_freq_hz = ustrip(Hz, plan.sampling_freq)
    carrier_freq_hz = ustrip(Hz, get_center_frequency(plan.system))
    coherent_duration_s = plan.num_coherently_integrated_code_periods * plan.samples_per_code / sampling_freq_hz
    # On the rotation path the cp axis is `samples_per_code_eff` =
    # `samples_per_code * num_secondary_rotations` wide, with each rotation
    # hypothesis occupying its own `samples_per_code`-wide block. Drift is
    # phase-walk of the primary code within ONE rotation hypothesis, so the
    # circshift wraps inside each block — not across rotation-block boundaries
    # (which would wrongly mix hypotheses). On the non-rotation path
    # `num_rotation_blocks == 1` and the body collapses to the original
    # single-circshift form.
    samples_per_code = plan.samples_per_code
    num_rotation_blocks = size(noncoherent_integration_buf, 2) ÷ samples_per_code
    row_buf = zeros(Float32, samples_per_code)
    row_shift_buf = zeros(Float32, samples_per_code)
    for doppler_row in 1:num_doppler_bins
        doppler_hz = ustrip(Hz, plan.doppler_freqs[doppler_row])
        shift = round(Int, doppler_hz * accumulation_step_index * coherent_duration_s * sampling_freq_hz / carrier_freq_hz)
        shift == 0 && continue
        for r in 0:num_rotation_blocks-1
            base = r * samples_per_code
            row_buf .= view(noncoherent_integration_buf, doppler_row, (base+1):(base+samples_per_code))
            circshift!(row_shift_buf, row_buf, shift)
            view(noncoherent_integration_buf, doppler_row, (base+1):(base+samples_per_code)) .= row_shift_buf
        end
    end
end

# Per-column FFT followed by `|x|²` accumulation into `accumulator`, with the
# row written at its fftshift-permuted position. `cim` may be one column-block
# tile of the surface; `dest_col_offset` places its columns at their global
# code-phase positions in the accumulator. Valid only when code drift rounds
# to zero for every row (step 0, or any step whose rounded drift is zero).
#
# The fftshift permutation is a fixed rotation by `shift = N ÷ 2` rows:
#   src rows 1..N-shift    → dst rows 1+shift..N      (advance)
#   src rows N-shift+1..N  → dst rows 1..shift        (wrap)
# Two contiguous loops express both halves so the compiler can SIMD.
function _accumulate_fftshifted_power_pilot!(
    accumulator::Matrix{Float32},
    cim::Matrix{ComplexF32},
    col_buf::Vector{ComplexF32},
    col_fft_plan,
    samples_per_code::Int,
    num_doppler_bins::Int,
    dest_col_offset::Int = 0,
)
    shift = num_doppler_bins ÷ 2
    front = num_doppler_bins - shift
    @inbounds for src_col in 1:size(cim, 2)
        copyto!(col_buf, 1, cim, (src_col - 1) * num_doppler_bins + 1, num_doppler_bins)
        mul!(col_buf, col_fft_plan, col_buf)
        dst_col = dest_col_offset + src_col
        @simd for doppler_bin in 1:front
            accumulator[doppler_bin + shift, dst_col] += abs2(col_buf[doppler_bin])
        end
        @simd for doppler_bin in 1:shift
            accumulator[doppler_bin, dst_col] += abs2(col_buf[doppler_bin + front])
        end
    end
end

# Batched-FFT counterpart: FFT all of `cim`'s columns at once in-place, then
# accumulate `|x|²` into `accumulator` with the fftshift row permutation.
# In production `cim` is the (num_doppler_bins, block_size) tile and the plan
# is tile-shaped.
function _accumulate_fftshifted_power_pilot_batched!(
    accumulator::Matrix{Float32},
    cim::Matrix{ComplexF32},
    col_batch_fft_plan,
    samples_per_code::Int,
    num_doppler_bins::Int,
    dest_col_offset::Int = 0,
)
    mul!(cim, col_batch_fft_plan, cim)
    shift = num_doppler_bins ÷ 2
    front = num_doppler_bins - shift
    @inbounds for src_col in 1:size(cim, 2)
        dst_col = dest_col_offset + src_col
        @simd for doppler_bin in 1:front
            accumulator[doppler_bin + shift, dst_col] += abs2(cim[doppler_bin, src_col])
        end
        @simd for doppler_bin in 1:shift
            accumulator[doppler_bin, dst_col] += abs2(cim[doppler_bin + front, src_col])
        end
    end
end

# Per-column FFT followed by `|x|²` accumulation with per-row code-drift column
# shift AND fftshift row permutation, folded into one pass (Issue #62).
#
# `code_drift_shifts[r]` is the row-r column shift, pre-normalised to
# `[0, samples_per_code)` by `_fill_code_drift_shifts!`. That lets the inner
# loop replace `mod(c-1+shift_r, spc)+1` with a single conditional subtract.
# The caller must dispatch to the non-drift kernel when every row's drift
# rounds to 0 (step 0, or any step whose rounded drift is zero — at the
# L1CA / 5 MHz / N_coh=1 grid that's true for every step up to N_nc≈8, so the
# early-exit covers the dominant simple-path case).
function _accumulate_fftshifted_power_drift_pilot!(
    accumulator::Matrix{Float32},
    cim::Matrix{ComplexF32},
    col_buf::Vector{ComplexF32},
    col_fft_plan,
    samples_per_code::Int,
    num_doppler_bins::Int,
    fftshift_perm::Vector{Int},
    code_drift_shifts::Vector{Int},
    dest_col_offset::Int = 0,
)
    @inbounds for src_col in 1:size(cim, 2)
        copyto!(col_buf, 1, cim, (src_col - 1) * num_doppler_bins + 1, num_doppler_bins)
        mul!(col_buf, col_fft_plan, col_buf)
        c_minus_1 = dest_col_offset + src_col - 1
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
    cim::Matrix{ComplexF32},
    col_batch_fft_plan,
    samples_per_code::Int,
    num_doppler_bins::Int,
    fftshift_perm::Vector{Int},
    code_drift_shifts::Vector{Int},
    dest_col_offset::Int = 0,
)
    mul!(cim, col_batch_fft_plan, cim)
    @inbounds for src_col in 1:size(cim, 2)
        c_minus_1 = dest_col_offset + src_col - 1
        for r in 1:num_doppler_bins
            sum = c_minus_1 + code_drift_shifts[r]
            dst_col = (sum >= samples_per_code ? sum - samples_per_code : sum) + 1
            accumulator[fftshift_perm[r], dst_col] += abs2(cim[r, src_col])
        end
    end
end

# Fill `shifts[r]` with the row-r column shift from `_apply_code_drift!`,
# normalised to `[0, samples_per_code)` so the hot loops can drop `mod` for a
# single conditional subtract. Returns `true` if any shift rounded to non-zero
# (caller dispatches to the drift-folded kernels) or `false` if every row's
# drift rounds to 0 (caller uses the cheaper non-drift kernels — the typical
# case at low-fs / short-T_coh grids).
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

# Fill `stage` (num_doppler_bins × num_stage_slices) with the sign-search power
# of ONE code-phase column, max-reduced across bit-edge alignments (and, within
# each rotation slice, across data-bit polarities). `src_col` indexes into
# `cim` (a tile or a full matrix); `cp_col` is the 1-based code-phase column in
# [1, samples_per_code] it corresponds to. The caller zeroes `stage` first.
# Pattern matrices are passed in (hoisted out of the per-column loop) so no
# Dict lookup happens per column.
@inline function _fill_sign_stage_column!(
    stage::AbstractMatrix{Float32},
    cim::AbstractMatrix{ComplexF32},
    src_col::Int,
    cp_col::Int,
    plan::AcquisitionPlan,
    scratch,
    patterns::Matrix{Float32},
    tiled_re::Array{Float32,3},
    tiled_im::Array{Float32,3},
    rotation_active::Bool,
    num_data_combos::Int,
    edge_step::Int,
    use_max::Bool,
)
    num_doppler_bins = plan.num_coherently_integrated_code_periods * plan.num_blocks
    for bit_edge_search_idx in 0:plan.bit_edge_search_steps-1
        row_offset = bit_edge_search_idx * edge_step
        if rotation_active
            _rotation_column!(stage, cim, src_col, cp_col,
                scratch.col_buf, scratch.combine_buf_re, scratch.combine_buf_im,
                plan.col_fft_plan, num_doppler_bins,
                plan.num_coherently_integrated_code_periods, num_data_combos,
                plan.num_blocks, plan.block_size, row_offset,
                scratch.sub_block_ffts, tiled_re, tiled_im, use_max)
        else
            _sign_search_column!(view(stage, :, 1), cim, src_col,
                scratch.col_buf, plan.col_fft_plan, num_doppler_bins,
                plan.num_data_bits, row_offset, scratch.sub_block_ffts, patterns)
        end
    end
    return stage
end

# Sign-search consumer for one tile on the multistep (N_nc > 1) path: for each
# tile column, fill the per-column stage and scatter it into the per-PRN
# accumulator with the code-drift column shift folded in. Replaces the
# full-surface `noncoherent_integration_buf` → `_apply_code_drift!` →
# max/accumulate pipeline with a column-local one. The drift shift is identical
# across bit-edge alignments and data combos (it depends only on Doppler row
# and step), so max-then-shift equals shift-then-max.
function _accumulate_sign_tile!(
    nim::Matrix{Float32},
    tile::Matrix{ComplexF32},
    col_block_idx::Int,
    plan::AcquisitionPlan,
    scratch,
    prn::Int,
    has_drift::Bool,
)
    block_size = plan.block_size
    samples_per_code = plan.samples_per_code
    num_doppler_bins = size(nim, 1)
    stage = scratch.stage_buf
    num_slices = size(stage, 2)
    shifts = scratch.code_drift_shifts
    rotation_active = plan.num_secondary_rotations > 1 &&
                      plan.num_coherently_integrated_code_periods > 1
    num_data_combos = 1 << (plan.num_data_bits - 1)
    bit_period_codes = plan.num_coherently_integrated_code_periods ÷ plan.num_data_bits
    edge_step = bit_period_codes * plan.num_blocks ÷ plan.bit_edge_search_steps
    use_max = num_data_combos > 1 || plan.bit_edge_search_steps > 1
    patterns = plan.sign_patterns_by_prn[prn]
    tiled_re = rotation_active ? plan.tiled_phase_patterns_re_by_prn[prn] :
        _EMPTY_TILED_PATTERNS
    tiled_im = rotation_active ? plan.tiled_phase_patterns_im_by_prn[prn] :
        _EMPTY_TILED_PATTERNS

    for local_col in 1:block_size
        cp_col = col_block_idx * block_size + local_col
        fill!(stage, 0f0)
        _fill_sign_stage_column!(stage, tile, local_col, cp_col, plan, scratch,
            patterns, tiled_re, tiled_im, rotation_active, num_data_combos,
            edge_step, use_max)
        if has_drift
            @inbounds for s in 0:num_slices-1
                base = s * samples_per_code
                for r in 1:num_doppler_bins
                    shifted = (cp_col - 1) + shifts[r]
                    shifted >= samples_per_code && (shifted -= samples_per_code)
                    nim[r, base + shifted + 1] += stage[r, s + 1]
                end
            end
        else
            @inbounds for s in 0:num_slices-1
                base = s * samples_per_code
                @simd for r in 1:num_doppler_bins
                    nim[r, base + cp_col] += stage[r, s + 1]
                end
            end
        end
    end
end

const _EMPTY_TILED_PATTERNS = Array{Float32,3}(undef, 0, 0, 0)

"""
    _accumulate_prn_step_tiled!(nim, plan, scratch, prn, accumulation_step_index)

One PRN × one non-coherent step of the multistep (N_nc > 1) pipeline, tiled:
for each FM-DBZP column block, build the coherent tile and accumulate its power
into the per-PRN matrix `nim` — via the fused pilot kernels on the simple path
or the per-column sign-search/rotation kernels otherwise. Code drift is folded
into the destination columns; no intermediate surface is materialised.
"""
function _accumulate_prn_step_tiled!(
    nim::Matrix{Float32},
    plan::AcquisitionPlan,
    scratch,
    prn::Int,
    accumulation_step_index::Int,
)
    num_doppler_bins = plan.num_coherently_integrated_code_periods * plan.num_blocks
    rotation_active = plan.num_secondary_rotations > 1 &&
                      plan.num_coherently_integrated_code_periods > 1
    simple_path = plan.num_data_bits == 1 && plan.bit_edge_search_steps == 1 &&
                  !rotation_active
    has_drift = _fill_code_drift_shifts!(scratch.code_drift_shifts, plan, accumulation_step_index)
    prn_conj_fft = plan.prn_conj_ffts[prn]
    tile = scratch.coherent_tile

    for col_block_idx in 0:plan.num_blocks-1
        _build_coherent_tile!(tile, plan.signal_block_ffts, prn_conj_fft,
            col_block_idx, plan.num_blocks, plan.block_size,
            plan.num_coherently_integrated_code_periods,
            scratch.corr_buf, plan.double_block_bfft_plan)
        dest_col_offset = col_block_idx * plan.block_size
        if simple_path
            if !has_drift
                if num_doppler_bins <= BATCH_FFT_THRESHOLD
                    _accumulate_fftshifted_power_pilot_batched!(
                        nim, tile, plan.col_batch_fft_plan,
                        plan.samples_per_code, num_doppler_bins, dest_col_offset)
                else
                    _accumulate_fftshifted_power_pilot!(
                        nim, tile, scratch.col_buf, plan.col_fft_plan,
                        plan.samples_per_code, num_doppler_bins, dest_col_offset)
                end
            else
                if num_doppler_bins <= BATCH_FFT_THRESHOLD
                    _accumulate_fftshifted_power_drift_pilot_batched!(
                        nim, tile, plan.col_batch_fft_plan,
                        plan.samples_per_code, num_doppler_bins,
                        plan.fftshift_perm, scratch.code_drift_shifts, dest_col_offset)
                else
                    _accumulate_fftshifted_power_drift_pilot!(
                        nim, tile, scratch.col_buf, plan.col_fft_plan,
                        plan.samples_per_code, num_doppler_bins,
                        plan.fftshift_perm, scratch.code_drift_shifts, dest_col_offset)
                end
            end
        else
            _accumulate_sign_tile!(nim, tile, col_block_idx, plan, scratch, prn, has_drift)
        end
    end
    return nim
end

"""
    _accumulate_noncoherent_integration_step!(noncoherent_integration_matrix, coherent_integration_matrix, plan, scratch, prn, m)

For incoherent step `m` (0-based): run the column FFT (pilot/sub-bit or data/rotation),
apply bit edge search and code drift correction, then accumulate into
`noncoherent_integration_matrix`.

Reference/test dispatcher operating on a fully materialised coherent
integration matrix; it mirrors the routing of the tiled production driver
[`_accumulate_prn_step_tiled!`](@ref) so kernel parity can be validated from
tests. Sign-search staging buffers are allocated locally (this entry point is
not on the hot path). The simple path always uses the per-column fused kernels
— the plan's batched FFT plan is tile-shaped and does not fit a full matrix.
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
        # straight into `noncoherent_integration_matrix`.
        drift_shifts = length(scratch.code_drift_shifts) == num_doppler_bins ?
            scratch.code_drift_shifts : zeros(Int, num_doppler_bins)
        has_drift = _fill_code_drift_shifts!(drift_shifts, plan, accumulation_step_index)
        if !has_drift
            _accumulate_fftshifted_power_pilot!(
                noncoherent_integration_matrix, coherent_integration_matrix,
                scratch.col_buf, plan.col_fft_plan,
                plan.samples_per_code, num_doppler_bins)
        else
            _accumulate_fftshifted_power_drift_pilot!(
                noncoherent_integration_matrix, coherent_integration_matrix,
                scratch.col_buf, plan.col_fft_plan,
                plan.samples_per_code, num_doppler_bins,
                plan.fftshift_perm, drift_shifts)
        end
    else
        # Sign-pattern search path, materialised form: kernel → code drift →
        # (max across bit-edge alignments) → accumulate. Buffers are allocated
        # locally — production streams this per column instead.
        noncoherent_integration_buf =
            zeros(Float32, num_doppler_bins, size(noncoherent_integration_matrix, 2))
        bit_period_codes = plan.num_coherently_integrated_code_periods ÷ plan.num_data_bits
        edge_step = bit_period_codes * plan.num_blocks ÷ plan.bit_edge_search_steps
        num_data_combos = 1 << (plan.num_data_bits - 1)
        if plan.bit_edge_search_steps == 1
            if rotation_search_active
                _sign_search_step_with_rotations!(noncoherent_integration_buf, coherent_integration_matrix,
                    scratch.col_buf, scratch.combine_buf_re, scratch.combine_buf_im,
                    plan.col_fft_plan,
                    plan.samples_per_code, num_doppler_bins,
                    plan.num_coherently_integrated_code_periods, num_data_combos, plan.num_blocks,
                    plan.block_size,
                    0, scratch.sub_block_ffts,
                    plan.tiled_phase_patterns_re_by_prn[prn],
                    plan.tiled_phase_patterns_im_by_prn[prn])
            else
                _sign_search_step!(noncoherent_integration_buf, coherent_integration_matrix, scratch.col_buf,
                    plan.col_fft_plan, plan.samples_per_code, num_doppler_bins,
                    plan.num_data_bits, 0, scratch.sub_block_ffts,
                    plan.sign_patterns_by_prn[prn])
            end
            _apply_code_drift!(noncoherent_integration_buf, plan, accumulation_step_index)
            noncoherent_integration_matrix .+= noncoherent_integration_buf
        else
            sign_search_max_buf =
                zeros(Float32, num_doppler_bins, size(noncoherent_integration_matrix, 2))
            for bit_edge_search_idx in 0:plan.bit_edge_search_steps-1
                fill!(noncoherent_integration_buf, 0.0f0)
                row_offset = bit_edge_search_idx * edge_step
                if rotation_search_active
                    _sign_search_step_with_rotations!(noncoherent_integration_buf, coherent_integration_matrix,
                        scratch.col_buf, scratch.combine_buf_re, scratch.combine_buf_im,
                        plan.col_fft_plan,
                        plan.samples_per_code, num_doppler_bins,
                        plan.num_coherently_integrated_code_periods, num_data_combos, plan.num_blocks,
                        plan.block_size,
                        row_offset, scratch.sub_block_ffts,
                        plan.tiled_phase_patterns_re_by_prn[prn],
                        plan.tiled_phase_patterns_im_by_prn[prn])
                else
                    _sign_search_step!(noncoherent_integration_buf, coherent_integration_matrix, scratch.col_buf,
                        plan.col_fft_plan, plan.samples_per_code, num_doppler_bins,
                        plan.num_data_bits, row_offset, scratch.sub_block_ffts,
                        plan.sign_patterns_by_prn[prn])
                end
                _apply_code_drift!(noncoherent_integration_buf, plan, accumulation_step_index)
                @. sign_search_max_buf = max(sign_search_max_buf, noncoherent_integration_buf)
            end
            noncoherent_integration_matrix .+= sign_search_max_buf
        end
    end
end
# Convenience wrapper for tests and single-threaded callers — routes through
# thread 1's scratch (see `_default_scratch`) and uses the plan's first avail PRN.
_accumulate_noncoherent_integration_step!(nim, cim, plan::AcquisitionPlan, accumulation_step_index::Int) =
    _accumulate_noncoherent_integration_step!(nim, cim, plan, _default_scratch(plan),
        first(plan.avail_prns), accumulation_step_index)
