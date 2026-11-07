# src/doppler_passes.jl
#
# Doppler search layout: how many FM-DBZP blocks per code period, and whether
# the Doppler range is covered by one wide FM-DBZP grid or by several narrow
# passes (issue #101).
#
# One FM-DBZP pass with `num_blocks` blocks per code period covers
# `num_blocks × bin_width` of Doppler (`bin_width = fs / samples_per_code`), and
# `num_blocks` must divide `samples_per_code`. When `samples_per_code` has
# sparse divisors (1.542 MHz → 1542 = 2·3·257) the smallest admissible divisor
# can sit far above what the coverage needs: ±7 kHz forces 257 blocks, i.e. 257
# Doppler bins with 6-sample blocks.
#
# Instead, a small divisor `B` can cover the range in `P` passes. Pass `k`
# downconverts with `interm_freq + f_k`, which moves the band centred at `f_k`
# to 0, and runs the unchanged FM-DBZP pipeline with `B` blocks. It keeps the
# `K` bins nearest its centre, and the passes' kept bins tile one uniform
# Doppler grid of `P × K` bins:
#
#     global bin j  (frequency j·spacing, j = -(n÷2) .. n-1-n÷2, n = P·K)
#     pass k keeps j ∈ j0 + k·K .. j0 + k·K + K-1                 (j0 = -(n÷2))
#     pass k is centred on bin c_k = j0 + k·K + K÷2, i.e. f_k = c_k · spacing
#
# so its kept bins sit at local offsets -(K÷2) .. K-1-K÷2 — exactly the bins a
# single pass of `M = N_coh · B` bins has around 0 (its fftshifted grid spans
# -(M÷2) .. M-1-M÷2). With `P = 1` and `K = M`, `c_0 = 0` and this reduces to
# the classic single-grid search.
#
# Why trim to K < M? FM-DBZP correlates per block, so a Doppler `f` away from
# the pass centre loses `|sinc(f · T_block)|²` (`T_block = 1 / (B · bin_width)`):
# measured on GPS L1 C/A within 0.05 dB of the model. At a pass edge
# (`f = B·bin_width/2`) that is -3.9 dB for every `B`. A single-grid search at
# a rate whose `samples_per_code` has the ideal divisor `B_min` (the smallest
# that reaches the coverage) loses `sinc²(f_max / (B_min·bin_width))` at the
# coverage edge — -2.9 dB for ±7 kHz at 1 ms (`B_min = 16`). The passes keep
# only bins whose whole bin (centre ± spacing/2) stays within that same loss, so
# no Doppler in the searched range is worse off than on a well-chosen rate.

struct DopplerLayout
    num_blocks::Int       # FM-DBZP blocks per code period (divides samples_per_code)
    num_passes::Int       # P — narrow passes covering the Doppler range
    bins_per_pass::Int    # K — Doppler bins each pass contributes to the grid
end

# Doppler bins one pass's FM-DBZP grid has (the Doppler FFT length).
_pass_doppler_bins(layout::DopplerLayout, num_coherently_integrated_code_periods::Int) =
    num_coherently_integrated_code_periods * layout.num_blocks

# Smallest divisor of `samples_per_code` that is >= `min_num_blocks`.
# Returns `nothing` if no such divisor exists.
function _smallest_valid_num_blocks(samples_per_code::Integer, min_num_blocks::Integer)
    for d in max(1, min_num_blocks):samples_per_code
        samples_per_code % d == 0 && return Int(d)
    end
    nothing
end

# Smallest number of blocks a single FM-DBZP grid needs so that its half-open
# Doppler grid reaches ±min_doppler_coverage at both ends. See `plan_acquire`.
_min_num_blocks(min_doppler_coverage_hz, bin_width_hz, num_coherently_integrated_code_periods) =
    ceil(Int, 2 * min_doppler_coverage_hz / bin_width_hz +
              2 / num_coherently_integrated_code_periods)

"""
    _search_cost(samples_per_code, num_blocks, num_coherently_integrated_code_periods, num_passes) -> Float64

Relative per-PRN cost of one coherent FM-DBZP search, used to choose between
Doppler layouts and to rank sampling frequencies.

Per pass, the pipeline runs `N_coh·B²` inner IFFTs of the padded length
`F = _inner_fft_size(2·samples_per_code/B)` and `B` Doppler FFTs of length
`M = N_coh·B` over `block_size` code phases each:

```
cost = P × ( N_coh·B² × (561 + F·log₂F) + samples_per_code·M·log₂M × w(M) )
w(M) = 0.75 if M is 7-smooth, else 1.18
```

The per-call constant is the fixed overhead of a small inner transform, which
dominates when blocks are short. The constants are a weighted least-squares
fit to single-threaded 32-PRN `acquire!` timings over 117 layouts (GPS L1 C/A,
1.024–16.368 MHz, `N_coh ∈ {1, 10}`, FFTL, AMD Zen 5): median error 7%, and
the cheapest layout by this model is within 12% of the measured fastest
everywhere. Only the ranking matters.
"""
function _search_cost(samples_per_code::Integer, num_blocks::Integer,
                      num_coherently_integrated_code_periods::Integer, num_passes::Integer)
    B = num_blocks
    block_size = samples_per_code ÷ B
    F = _inner_fft_size(2 * block_size)
    M = num_coherently_integrated_code_periods * B
    inner = num_coherently_integrated_code_periods * B^2 *
            (_INNER_FFT_CALL_COST + F * log2(F))
    doppler_fft = M <= 1 ? 0.0 :
        float(samples_per_code) * M * log2(M) *
        (_largest_prime_factor(M) > 7 ? _ROUGH_DOPPLER_FFT_WEIGHT : _SMOOTH_DOPPLER_FFT_WEIGHT)
    return num_passes * (inner + doppler_fft)
end

const _INNER_FFT_CALL_COST = 561.0
const _SMOOTH_DOPPLER_FFT_WEIGHT = 0.75
const _ROUGH_DOPPLER_FFT_WEIGHT = 1.18

# A multi-pass layout replaces the single grid only when the model expects it
# to be at least this much cheaper. Passes keep every Doppler within the loss of
# an ideal rate's grid, but a single grid with spare blocks is better than that
# at its edges — so a gain within the model's error is not worth taking.
const _MULTI_PASS_MIN_GAIN = 0.9

"""
    _doppler_search_layout(samples_per_code, bin_width_hz, min_doppler_coverage_hz,
                           num_coherently_integrated_code_periods, max_doppler_passes) -> DopplerLayout

Choose `num_blocks` and the pass layout for a search reaching
`±min_doppler_coverage_hz`. The single-grid layout (smallest divisor of
`samples_per_code` that reaches the coverage, one pass) is always a candidate;
with `max_doppler_passes > 1` every smaller divisor is tried as a multi-pass
layout that holds the sensitivity bound described at the top of this file, and
the cheapest by [`_search_cost`](@ref) wins — a multi-pass layout only if it
is expected to be at least 10% cheaper than the single grid. Returns `nothing` if no layout
exists (only when `samples_per_code` is smaller than the blocks the coverage
needs).
"""
function _doppler_search_layout(samples_per_code::Integer, bin_width_hz::Real,
                                min_doppler_coverage_hz::Real,
                                num_coherently_integrated_code_periods::Integer,
                                max_doppler_passes::Integer)
    N_coh = num_coherently_integrated_code_periods
    min_num_blocks = _min_num_blocks(min_doppler_coverage_hz, bin_width_hz, N_coh)
    single_num_blocks = _smallest_valid_num_blocks(samples_per_code, min_num_blocks)
    best = single_num_blocks === nothing ? nothing :
        DopplerLayout(single_num_blocks, 1, N_coh * single_num_blocks)
    best_cost = best === nothing ? Inf :
        _MULTI_PASS_MIN_GAIN * _search_cost(samples_per_code, single_num_blocks, N_coh, 1)
    max_doppler_passes > 1 || return best

    # Grid the passes must fill: the fewest bins (odd, so symmetric about 0)
    # that reach ±min_doppler_coverage. `spacing = bin_width / N_coh`.
    coverage_in_bins = min_doppler_coverage_hz * N_coh / bin_width_hz
    min_grid_bins = 2 * max(0, ceil(Int, coverage_in_bins - 1e-9)) + 1
    # The grid must not span more than the sampling frequency: beyond that the
    # passes would search aliases of frequencies already on the grid.
    min_grid_bins <= N_coh * samples_per_code || return best
    # Loss bound: a kept bin's far edge may sit at most `max_offset` of the
    # pass's block-loss scale `B · bin_width` from the pass centre — the offset
    # of the coverage edge on an ideal rate (`min_num_blocks` blocks).
    max_offset = min_doppler_coverage_hz / (min_num_blocks * bin_width_hz)
    last_divisor = something(single_num_blocks, samples_per_code + 1) - 1
    for B in 1:last_divisor
        samples_per_code % B == 0 || continue
        M = N_coh * B
        # Kept offsets -(K÷2) .. K-1-K÷2: the farthest edge is (K÷2 + 1/2)
        # bins out, which must stay within max_offset · B · N_coh bins.
        half = floor(Int, max_offset * B * N_coh - 0.5 + 1e-9)
        half < 0 && continue
        max_bins_per_pass = min(M, 2 * half + 1)
        num_passes = cld(min_grid_bins, max_bins_per_pass)
        num_passes <= max_doppler_passes || continue
        cost = _search_cost(samples_per_code, B, N_coh, num_passes)
        if cost < best_cost
            best_cost = cost
            best = DopplerLayout(B, num_passes, cld(min_grid_bins, num_passes))
        end
    end
    return best
end

# Doppler grid of a layout, in Hz: `n = P·K` bins at multiples of `spacing`,
# from `-(n÷2)·spacing`. This is the fftshifted FFT-bin order, so with one pass
# row `r` is exactly the frequency of the Doppler FFT bin that lands there.
function _doppler_grid_hz(layout::DopplerLayout, bin_width_hz::Real,
                          num_coherently_integrated_code_periods::Integer)
    spacing = bin_width_hz / num_coherently_integrated_code_periods
    n = layout.num_passes * layout.bins_per_pass
    return range(-(n ÷ 2) * spacing, step = spacing, length = n)
end

# Centre frequency (Hz) of each pass: bin `c_k = -(n÷2) + k·K + K÷2`.
function _doppler_pass_offsets_hz(layout::DopplerLayout, bin_width_hz::Real,
                                  num_coherently_integrated_code_periods::Integer)
    spacing = bin_width_hz / num_coherently_integrated_code_periods
    P, K = layout.num_passes, layout.bins_per_pass
    n = P * K
    return [(-(n ÷ 2) + k * K + K ÷ 2) * spacing for k in 0:P-1]
end
