# src/sign_search_layout.jl
#
# The sign hypotheses of a sign search, in one place: which data-bit segments
# each bit-edge alignment splits the coherent window into, which sign
# combinations it searches, and the per-row signs of each hypothesis. The tile
# kernel (`_sign_search_tile_power!`), the plan's scratch sizing and the CFAR
# null model (`_sign_search_null`) all enumerate hypotheses through these
# helpers, so they cannot disagree.

# Shape of a sign search: a coherent window of `num_coh` code periods
# (`num_coh · num_blocks` Doppler rows) holding `num_data_bits` data bits,
# searched at `num_edges` bit-edge alignments.
struct SignSearchLayout
    num_data_bits::Int
    num_coh::Int
    num_edges::Int
    num_blocks::Int
end

_sign_search_layout(plan) = SignSearchLayout(plan.num_data_bits,
    plan.num_coherently_integrated_code_periods, plan.bit_edge_search_steps,
    plan.num_blocks)

"""
    _edge_alignment(layout, edge) -> (; first_edge, bit_period, num_segments, combos)

Data-bit segments of bit-edge alignment `edge` (0-based) and the sign
combinations searched over them.

The window holds `num_data_bits` bit periods of `bit_period = num_coh ÷
num_data_bits` code periods; alignment `edge` puts the first bit edge
`first_edge = edge · bit_period ÷ num_edges` code periods into the window.
Aligned (`first_edge == 0`), the window is `num_data_bits` whole bits;
otherwise a partial bit, the `num_data_bits − 1` whole bits and another partial
bit, each with its own sign. Segment boundaries are code-period boundaries
(`_segment_rows`).

`combos` are the sign combinations, each an `Int` bitmask read by
`_negates_segment`: segment 0 is always `+1` (only `|·|²` is used, so the global
sign is free). A one-bit window at a nonzero alignment searches only the
transition: its no-transition combination repeats alignment 0.

Approximation: in a code-phase column whose first code-period boundary is row
`off > 0`, the leading `off` rows (less than one code period) belong to segment
0. At a nonzero alignment that is exact. Aligned, those rows belong to the bit
before the window; when that bit's sign differs from the first bit's, the
hypothesis loses up to `2 · off / (num_coh · num_blocks)` of its amplitude (at
most 0.9 dB at 20 code periods, 0.45 dB on average over columns).
"""
@inline function _edge_alignment(layout::SignSearchLayout, edge::Int)
    (; num_data_bits, num_coh, num_edges) = layout
    bit_period = num_coh ÷ num_data_bits
    first_edge = edge * bit_period ÷ num_edges
    num_segments = first_edge == 0 ? num_data_bits : num_data_bits + 1
    combos = num_data_bits == 1 && num_segments == 2 ? (1:1) :
             (0:(1 << (num_segments - 1)) - 1)
    return (; first_edge, bit_period, num_segments, combos)
end

# Whether sign combination `combo` negates segment `seg ≥ 1`: bit `seg − 1`.
@inline _negates_segment(combo::Int, seg::Int) = (combo >> (seg - 1)) & 1 == 1

# Whether an alignment's sign combinations are each transformed directly, or
# its segments are transformed once and combined by Gray code. A Gray step is
# one fused pass over the tile; an FFTL tile FFT costs a few such passes plus
# the sign modulation, so segments pay off from twice as many combinations as
# segments on (measured: 8 over 4 segments 1.3× faster, 4 over 3 1.1–1.35×
# slower).
@inline _direct_hypotheses(alignment) =
    length(alignment.combos) < 2 * alignment.num_segments

# Sign hypotheses per rotation, over all alignments.
_num_sign_hypotheses(layout::SignSearchLayout) =
    sum(edge -> length(_edge_alignment(layout, edge).combos), 0:layout.num_edges-1)

# Segment tiles the kernel needs: the most segments of any alignment it
# combines by Gray code, 0 when it transforms every combination directly.
_num_segment_tiles(layout::SignSearchLayout) =
    maximum(0:layout.num_edges-1) do edge
        alignment = _edge_alignment(layout, edge)
        _direct_hypotheses(alignment) ? 0 : alignment.num_segments
    end

# Rows (1-based) of segment `seg` for a column whose first code-period boundary
# is row `off` (0-based): segment boundaries are code-period boundaries, the
# leading rows before `off` belong to segment 0, and the last segment ends at
# the window's last row.
@inline function _segment_rows(alignment, seg::Int, off::Int, num_blocks::Int,
        num_doppler_bins::Int)
    lo = _segment_start_row(alignment, seg, off, num_blocks, num_doppler_bins)
    hi = _segment_start_row(alignment, seg + 1, off, num_blocks, num_doppler_bins)
    return lo+1:hi
end

@inline function _segment_start_row(alignment, seg::Int, off::Int, num_blocks::Int,
        num_doppler_bins::Int)
    (; first_edge, bit_period, num_segments) = alignment
    seg == 0 && return 0
    seg >= num_segments && return num_doppler_bins
    period = first_edge == 0 ? seg * bit_period : first_edge + (seg - 1) * bit_period
    return off + period * num_blocks
end

"""
    _fill_hypothesis_signs!(signs, patterns, q, layout, alignment, combo, off)

Per-row signs of one sign hypothesis, for a code-phase column whose first
code-period boundary is row `off`: secondary-code pattern `q` (cyclic, see
`_fill_rotation_signs!`) times the data-bit sign of each row's segment at
`alignment` (`_edge_alignment`), with segment signs from `combo`. Shared by
[`_sign_search_tile_power!`](@ref) and the CFAR null model
(`_hypothesis_signs`).
"""
function _fill_hypothesis_signs!(signs::AbstractVector{Float32}, patterns::Matrix{Float32},
        q::Int, layout::SignSearchLayout, alignment, combo::Int, off::Int)
    _fill_rotation_signs!(signs, patterns, q, off, layout.num_blocks)
    for seg in 1:alignment.num_segments-1
        _negates_segment(combo, seg) || continue
        @inbounds @simd for m in _segment_rows(alignment, seg, off, layout.num_blocks,
            length(signs))
            signs[m] = -signs[m]
        end
    end
    return signs
end

# Secondary-code signs of pattern `q` on the rows of segment `seg` at
# `alignment`, zero elsewhere: the mask the segment-wise FFTs of
# `_sign_search_tile_power!` apply.
function _fill_segment_signs!(signs::AbstractVector{Float32}, patterns::Matrix{Float32},
        q::Int, layout::SignSearchLayout, alignment, seg::Int, off::Int)
    _fill_rotation_signs!(signs, patterns, q, off, layout.num_blocks)
    rows = _segment_rows(alignment, seg, off, layout.num_blocks, length(signs))
    @inbounds for m in eachindex(signs)
        m in rows || (signs[m] = 0f0)
    end
    return signs
end
