# src/sign_patterns.jl

"""
    sign_patterns(secondary_code, prn, num_data_bits, num_secondary_rotations, num_coh_periods, use_secondary_code)
        -> Matrix{Float32}  # (num_coh_periods, num_patterns)

Pure enumeration of ±1 sign patterns to search inside `_sign_search_step!`. Each column is
one pattern over `num_coh_periods` coherent code periods.

Two independent search axes:
- Data-bit polarities: `2^(num_data_bits − 1)` patterns with `d[0] = +1` fixed.
- Secondary-code rotations: `num_secondary_rotations` cyclic start phases, where rotation
  `r ∈ 0..L−1` produces signs `secondary_value(sec, prn, (k + r) mod L)` for `k = 0..N−1`.

When `use_secondary_code = false` or `num_secondary_rotations == 1`, the secondary axis
collapses to a single all-`+1` pattern and the result is exactly the data-bit polarity set.

The two axes are combined as a Cartesian product: the resulting matrix has
`2^(num_data_bits − 1) × num_secondary_rotations` columns. Within each column, the per-row
sign is the product of the data-bit sign for that row's bit segment and the secondary-code
sign for that row's coherent period.
"""
function sign_patterns(secondary_code, prn::Int, num_data_bits::Int,
                       num_secondary_rotations::Int, num_coh_periods::Int,
                       use_secondary_code::Bool)
    num_data_combos = 1 << (num_data_bits - 1)
    sec_active = use_secondary_code && num_secondary_rotations > 1
    num_sec = sec_active ? num_secondary_rotations : 1
    coh_per_bit = num_coh_periods ÷ num_data_bits
    L = sec_active ? num_secondary_rotations : 1
    patterns = Matrix{Float32}(undef, num_coh_periods, num_data_combos * num_sec)
    col = 0
    for r in 0:num_sec-1
        for d in 0:num_data_combos-1
            col += 1
            for k in 0:num_coh_periods-1
                bit_idx = k ÷ coh_per_bit
                data_sign = bit_idx == 0 ? 1.0f0 :
                    (((d >> (bit_idx - 1)) & 1) == 0 ? 1.0f0 : -1.0f0)
                sec_sign = sec_active ?
                    Float32(GNSSSignals.secondary_value(secondary_code, prn, mod(k + r, L))) : 1.0f0
                @inbounds patterns[k + 1, col] = data_sign * sec_sign
            end
        end
    end
    return patterns
end
