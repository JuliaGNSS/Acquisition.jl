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

# Bins of the stitched Doppler grid: `P · K`.
_grid_bins(layout::DopplerLayout) = layout.num_passes * layout.bins_per_pass

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
    _search_cost(samples_per_code, num_blocks, num_coherently_integrated_code_periods, num_passes, num_prns) -> Float64

Relative per-PRN cost of one coherent FM-DBZP search over `num_prns` PRNs,
used to choose between Doppler layouts and to rank sampling frequencies.

Per pass, the pipeline runs `N_coh·B²` inner IFFTs of the padded length
`F = _inner_fft_size(2·samples_per_code/B)` and `B` Doppler FFTs of length
`M = N_coh·B` over `block_size` code phases each:

```
inner   = N_coh·B² × (_INNER_FFT_CALL_COST + F·log₂F)
doppler = samples_per_code·M·log₂M × w(M)
prep    = N_coh·B × (_INNER_FFT_CALL_COST + F·log₂F) + _DOWNCONVERT_COST × N_coh·samples_per_code
cost    = P × (inner + doppler + prep / num_prns)
w(M)    = _SMOOTH_DOPPLER_FFT_WEIGHT if M is 7-smooth, else _ROUGH_DOPPLER_FFT_WEIGHT
```

The per-call constant is the fixed overhead of a small inner transform, which
dominates when blocks are short. `prep` is the per-pass signal preparation that
all PRNs share — downconverting the segment and its `N_coh·B` forward
double-block FFTs — so it matters when few PRNs are searched: many passes over
a long coherent window cost little per PRN at 32 PRNs but can dominate at one.
The constants are a weighted least-squares fit to single-threaded 32-PRN
`acquire!` timings over 117 layouts (GPS L1 C/A, 1.024–16.368 MHz,
`N_coh ∈ {1, 10}`, FFTL, AMD Zen 5): median error 7%, and the cheapest layout
by this model is within 12% of the measured fastest everywhere. Only the
ranking matters. When code-Doppler compensation is active, an additional
`_CODE_DOPPLER_COST * N_coh * B * F` models phase-ramp work in `prep`.
"""
function _search_cost(samples_per_code::Integer, num_blocks::Integer,
                      num_coherently_integrated_code_periods::Integer, num_passes::Integer,
                      num_prns::Integer, compensate_code_doppler::Bool = false)
    B = num_blocks
    block_size = samples_per_code ÷ B
    F = _inner_fft_size(2 * block_size)
    M = num_coherently_integrated_code_periods * B
    inner_fft = _INNER_FFT_CALL_COST + F * log2(F)
    inner = num_coherently_integrated_code_periods * B^2 * inner_fft
    doppler_fft = M <= 1 ? 0.0 :
        float(samples_per_code) * M * log2(M) *
        (_largest_prime_factor(M) > 7 ? _ROUGH_DOPPLER_FFT_WEIGHT : _SMOOTH_DOPPLER_FFT_WEIGHT)
    prep = num_coherently_integrated_code_periods * B * inner_fft +
           _DOWNCONVERT_COST * num_coherently_integrated_code_periods * samples_per_code
    # Fractional-delay phase ramps touch every padded signal FFT once. This
    # work, like the forward FFTs, is shared across PRNs in a pass.
    if compensate_code_doppler && num_passes > 1
        prep += _CODE_DOPPLER_COST * num_coherently_integrated_code_periods * B * F
    end
    return num_passes * (inner + doppler_fft + prep / max(1, num_prns))
end

const _INNER_FFT_CALL_COST = 561.0
const _SMOOTH_DOPPLER_FFT_WEIGHT = 0.75
const _ROUGH_DOPPLER_FFT_WEIGHT = 1.18
const _DOWNCONVERT_COST = 23.0
# Estimated cost of a phase recurrence and complex multiply per FFT element.
const _CODE_DOPPLER_COST = 6.0

# A multi-pass layout replaces the single grid only when the model expects it
# to cost at most this fraction of the single grid. Passes keep every Doppler
# within the loss of an ideal rate's grid, but a single grid with spare blocks
# is better than that at its edges — so a gain within the model's error is not
# worth taking.
const _MULTI_PASS_MIN_GAIN = 0.9

# Without code-Doppler compensation, passes are only considered when
# `samples_per_code`'s divisors are sparse: the single grid needs at least this many times the blocks the coverage
# asks for. That is where the divisor gap costs real time (1.542 MHz: 257 blocks
# for 16 needed; 1.023 and 3.069 MHz: 31 for 16). Rates with a divisor closer
# to the need (2.048 MHz at ±10 kHz: 32 for 22; 4 MHz at ±25 kHz: 80 for 52;
# 5 MHz at ±5 kHz, 5 ms: 20 for 11; 1.5 MHz: 20 for 16) keep their single
# grid, and with it the lower loss its spare blocks give near 0 Hz.
const _SPARSE_DIVISOR_MIN_EXCESS = 1.9

# Maximum uncompensated drift across a coherent window, in chips. Below this
# threshold the original pipeline is used, including for narrow L1 searches.
const _MAX_CODE_DOPPLER_DRIFT = 0.1

_code_doppler_compensation(coverage, ratio, N_coh, bin_width) =
    coverage * ratio * N_coh / bin_width > _MAX_CODE_DOPPLER_DRIFT

"""
    _doppler_search_layout(samples_per_code, bin_width_hz, min_doppler_coverage_hz,
                           num_coherently_integrated_code_periods, max_doppler_passes,
                           num_prns) -> Union{DopplerLayout,Nothing}

Choose `num_blocks` and the pass layout for a search reaching
`±min_doppler_coverage_hz`. The single-grid layout (smallest divisor of
`samples_per_code` that reaches the coverage, one pass) is the default. When
that divisor is sparse (at least `_SPARSE_DIVISOR_MIN_EXCESS` times the blocks
the coverage needs) and `max_doppler_passes > 1`, every smaller divisor is tried
as a multi-pass layout that holds the sensitivity bound described at the top of
this file; the cheapest by `_search_cost` wins, a multi-pass layout only if it
costs at most `_MULTI_PASS_MIN_GAIN` times the single grid. Returns `nothing` if
no layout exists (only when `samples_per_code` is smaller than the blocks the
coverage needs). With a nonzero `code_center_frequency_ratio`, drift above
`_MAX_CODE_DOPPLER_DRIFT` also enables passes at ordinary rates and constrains
their kept width. The cheapest compensated layout wins without a speedup
requirement; if the pass cap prevents this, the original layout is retained.
"""
function _doppler_search_layout(samples_per_code::Integer, bin_width_hz::Real,
                                min_doppler_coverage_hz::Real,
                                num_coherently_integrated_code_periods::Integer,
                                max_doppler_passes::Integer, num_prns::Integer,
                                code_center_frequency_ratio::Real = 0.0)
    N_coh = num_coherently_integrated_code_periods
    min_num_blocks = _min_num_blocks(min_doppler_coverage_hz, bin_width_hz, N_coh)
    single_num_blocks = _smallest_valid_num_blocks(samples_per_code, min_num_blocks)
    best = single_num_blocks === nothing ? nothing :
        DopplerLayout(single_num_blocks, 1, N_coh * single_num_blocks)
    max_doppler_passes > 1 || return best
    sparse_divisors = single_num_blocks === nothing ||
        single_num_blocks >= _SPARSE_DIVISOR_MIN_EXCESS * min_num_blocks
    compensate = _code_doppler_compensation(min_doppler_coverage_hz,
        code_center_frequency_ratio, N_coh, bin_width_hz)
    (sparse_divisors || compensate) || return best
    # When code drift requires narrower passes, compare all layouts at their
    # full cost rather than requiring a speedup over the uncompensated grid.
    fallback = best
    compensate && (best = nothing)
    best_cost = best === nothing ? Inf :
        _MULTI_PASS_MIN_GAIN * _search_cost(samples_per_code, single_num_blocks, N_coh, 1, num_prns)

    # Grid the passes must fill: the fewest bins (odd, so symmetric about 0)
    # that reach ±min_doppler_coverage. `spacing = bin_width / N_coh`.
    coverage_in_bins = min_doppler_coverage_hz * N_coh / bin_width_hz
    min_grid_bins = 2 * max(0, ceil(Int, coverage_in_bins - 1e-9)) + 1
    # Loss bound: a kept bin's far edge may sit at most `max_offset` of the
    # pass's block-loss scale `B · bin_width` from the pass centre — the offset
    # of the coverage edge on an ideal rate (`min_num_blocks` blocks).
    max_offset = min_doppler_coverage_hz / (min_num_blocks * bin_width_hz)
    # Local bin edges must drift by at most the tolerance over N_coh periods.
    # spacing * T_coh = 1, so the limit in bins is tolerance / ratio.
    max_code_bins = compensate ?
        2 * floor(Int, _MAX_CODE_DOPPLER_DRIFT / code_center_frequency_ratio - 0.5) + 1 :
        typemax(Int)
    last_divisor = something(single_num_blocks, samples_per_code + 1) - 1
    compensate && (last_divisor = something(single_num_blocks, samples_per_code))
    for B in 1:last_divisor
        samples_per_code % B == 0 || continue
        M = N_coh * B
        # Kept offsets -(K÷2) .. K-1-K÷2: the farthest edge is (K÷2 + 1/2)
        # bins out, which must stay within max_offset · B · N_coh bins.
        half = floor(Int, max_offset * B * N_coh - 0.5 + 1e-9)
        half < 0 && continue
        max_bins_per_pass = min(M, 2 * half + 1, max_code_bins)
        max_bins_per_pass >= 1 || continue
        num_passes = cld(min_grid_bins, max_bins_per_pass)
        num_passes <= max_doppler_passes || continue
        layout = DopplerLayout(B, num_passes, cld(min_grid_bins, num_passes))
        # The stitched grid (rounded up to whole passes) must not span more
        # than the sampling frequency: beyond that the passes would search
        # aliases of frequencies already on the grid.
        _grid_bins(layout) <= N_coh * samples_per_code || continue
        cost = _search_cost(samples_per_code, B, N_coh, num_passes, num_prns, compensate)
        if cost < best_cost
            best_cost = cost
            best = layout
        end
    end
    # Honour the user's pass cap even if it cannot meet the drift tolerance.
    return best === nothing ? fallback : best
end

# Doppler grid of a layout, in Hz: `n = P·K` bins at multiples of `spacing`,
# from `-(n÷2)·spacing`. This is the fftshifted FFT-bin order, so with one pass
# row `r` is exactly the frequency of the Doppler FFT bin that lands there.
function _doppler_grid_hz(layout::DopplerLayout, bin_width_hz::Real,
                          num_coherently_integrated_code_periods::Integer)
    spacing = bin_width_hz / num_coherently_integrated_code_periods
    n = _grid_bins(layout)
    return range(-(n ÷ 2) * spacing, step = spacing, length = n)
end

# Centre frequency (Hz) of each pass: bin `c_k = -(n÷2) + k·K + K÷2`.
function _doppler_pass_offsets_hz(layout::DopplerLayout, bin_width_hz::Real,
                                  num_coherently_integrated_code_periods::Integer)
    spacing = bin_width_hz / num_coherently_integrated_code_periods
    P, K = layout.num_passes, layout.bins_per_pass
    n = _grid_bins(layout)
    return [(-(n ÷ 2) + k * K + K ÷ 2) * spacing for k in 0:P-1]
end

# The rows one Doppler pass contributes: sorted rows
# `src_offset+1 : src_offset+num_rows` of the pass's own FM-DBZP grid land on
# rows `dest_offset+1 : dest_offset+num_rows` of the stitched grid. A single
# untrimmed pass keeps every row in place (both offsets 0).
struct PassRows
    src_offset::Int
    num_rows::Int
    dest_offset::Int
end

PassRows(num_rows::Int) = PassRows(0, num_rows, 0)

@inline _pass_rows(plan::AcquisitionPlan, pass_idx::Int) =
    PassRows(plan.pass_row_offset, plan.doppler_bins_per_pass,
        (pass_idx - 1) * plan.doppler_bins_per_pass)

# Map a row of the stitched Doppler grid to `(pass_idx, pass_row)`: the Doppler
# pass that produced it and its sorted row within that pass's own grid. The
# inverse of `_pass_rows`.
@inline function _doppler_row_to_pass(plan::AcquisitionPlan, row::Int)
    K = plan.doppler_bins_per_pass
    pass_idx = (row - 1) ÷ K + 1
    return pass_idx, plan.pass_row_offset + (row - 1) % K + 1
end

# Rows on either side of `row` in the stitched Doppler grid, for subsample
# interpolation, or `0` where there is none. A grid that is one whole FM-DBZP
# pass spans its full alias period, so its ends are neighbours (row 1 follows
# the last row). A stitched or trimmed grid stops at the coverage edge: the row
# past it is not on the grid, and wrapping would pair the edge with a Doppler
# on the far side of the range.
@inline function _doppler_neighbour_rows(plan::AcquisitionPlan, row::Int)
    n = length(plan.doppler_freqs)
    wraps = plan.num_doppler_passes == 1 && plan.doppler_bins_per_pass == plan.pass_fft_bins
    left = row > 1 ? row - 1 : (wraps ? n : 0)
    right = row < n ? row + 1 : (wraps ? 1 : 0)
    return left, right
end
