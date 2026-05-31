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

"""
    combined_phase_patterns(patterns, num_coh_periods)
        -> Array{ComplexF32, 3}  # (num_coh_periods, num_patterns, num_coh_periods)

Pre-multiply each per-period ±1 sign pattern by the inter-sub-block DFT phase
ramp so the rotation kernel's combination across sub-blocks becomes a full
length-`num_doppler_bins` column FFT (LongL5I-equivalent). Without this ramp,
the kernel's `Σ_p sign[p] · sub_block_FFT_p[ω]` combination is matched-filter
optimal only when the true Doppler lands on the coarse 1-kHz sub-block FFT
grid — between grid points the inter-sub-block phase rotates by some
δ ∈ (0, 2π) per code period that ±1 cannot fit, losing ≈3–8 dB of coherent
gain.

A length-N column FFT decomposes exactly as

    FFT_N(x)[ω] = Σ_p exp(-2πi · p · s / N_coh) · sub_block_FFT_p[ω],  s = ω mod N_coh

so the optimal NH10-rotation-aware combination at fine-bin class `s` is

    Σ_p NH10[(p+r) mod L] · exp(-2πi · p · s / N_coh) · sub_block_FFT_p[ω].

We pre-compute `out[p+1, q, s+1] = patterns[p+1, q] · exp(-2πi · p · s / N_coh)`
once per plan, so the kernel inner loop just multiplies `out[:, q, s+1]` against
the sub-block FFT column at each ω. The phase ramp depends only on (p, s) and
is independent of (data-bit polarity, rotation), so the same `phase_ramp[p, s]`
table is reused across every pattern column.

`s = ω mod num_coh_periods` indexes the `num_coh_periods` distinct fine
classes — every ω in raw FFT-bin order shares the phase ramp with all other ω
that have the same residue mod `num_coh_periods`.
"""
function combined_phase_patterns(patterns::Matrix{Float32}, num_coh_periods::Int)
    num_patterns = size(patterns, 2)
    size(patterns, 1) == num_coh_periods || throw(ArgumentError(
        "patterns has $(size(patterns, 1)) rows but num_coh_periods=$num_coh_periods"))
    out = Array{ComplexF32, 3}(undef, num_coh_periods, num_patterns, num_coh_periods)
    # Compute the phase factor with the divide last (same op order the user can
    # reproduce in their head and the @test below uses): `cispi(-2 · p · s / N)`.
    # Done once at plan_acquire time so the per-step kernel pays no cost.
    @inbounds for s in 0:num_coh_periods-1, q in 1:num_patterns, p in 0:num_coh_periods-1
        phase = cispi(-2.0f0 * p * s / num_coh_periods)   # exp(-2πi · p · s / N_coh)
        out[p + 1, q, s + 1] = patterns[p + 1, q] * phase
    end
    return out
end

"""
    tile_phase_patterns(combined_phase_patterns, num_doppler_bins)
        -> Array{ComplexF32, 3}  # (num_doppler_bins, num_coh_periods, num_patterns)

Expand the compact `(num_coh_periods, num_patterns, num_coh_periods)` phasor
table from [`combined_phase_patterns`](@ref) into the kernel-consumed tiled
form indexed by `(ω, p, q)`. Each row block of `num_coh_periods` ω-entries
shares the same fine-bin class `s = (ω-1) mod num_coh_periods`, so the tile
is just `num_blocks = num_doppler_bins / num_coh_periods` copies of the
compact phasor.

The rotation kernel's inner loop wants contiguous access to both
`tiled[ω, p, q]` and `sub_block_ffts[ω, p]` for fixed `(p, q)` while ω
sweeps across the column. The compact form requires looking up
`combined[p+1, q, ((ω-1) mod num_coh_periods)+1]` per ω, which is
SIMD-hostile (per-iteration modulo + non-contiguous coefficient). Pre-tiling
trades a small amount of storage (`num_blocks ×` blow-up; ~120 kB per PRN at
L5I/12 MHz) for a fully contiguous SIMD-friendly inner loop.
"""
function tile_phase_patterns(combined::Array{ComplexF32, 3}, num_doppler_bins::Int)
    num_coh_periods = size(combined, 1)
    num_patterns    = size(combined, 2)
    size(combined, 3) == num_coh_periods || throw(ArgumentError(
        "combined has $(size(combined, 3)) fine-bin classes, expected num_coh_periods=$num_coh_periods"))
    num_doppler_bins % num_coh_periods == 0 || throw(ArgumentError(
        "num_doppler_bins=$num_doppler_bins must be a multiple of num_coh_periods=$num_coh_periods"))
    out = Array{ComplexF32, 3}(undef, num_doppler_bins, num_coh_periods, num_patterns)
    @inbounds for q in 1:num_patterns, p in 1:num_coh_periods
        for ω in 1:num_doppler_bins
            s = (ω - 1) % num_coh_periods
            out[ω, p, q] = combined[p, q, s + 1]
        end
    end
    return out
end
