# power_bins layout: (num_doppler_bins, samples_per_code)
#   rows = Doppler bins, columns = code phase bins
#
# Single-pass strategy:
#   Pass 1 — scan column-major: find global max, accumulate per-column sums into col_sums.
#   Pass 2 — scan col_sums: total minus exclusion zone around peak column.
# This avoids two separate partial-range scans over the full matrix.
#
# Returns: (signal_power, noise_power, code_bin_idx, doppler_bin_idx)

function _findmax_and_colsums!(col_sums::Vector{T}, power_bins::AbstractMatrix{T}) where {T<:AbstractFloat}
    nrows, ncols = size(power_bins)
    max_val  = power_bins[1, 1]
    max_drow = 1
    max_ccol = 1
    @inbounds for c in 1:ncols
        cs = zero(T)
        for r in 1:nrows
            v = power_bins[r, c]
            cs += v
            if v > max_val
                max_val  = v
                max_drow = r
                max_ccol = c
            end
        end
        col_sums[c] = cs
    end
    max_val, max_drow, max_ccol
end

function _noise_from_colsums(col_sums::Vector{T}, nrows::Int, max_ccol::Int, samples_per_chip::Int) where {T<:AbstractFloat}
    ncols   = length(col_sums)
    excl_lo = max(1, max_ccol - samples_per_chip)
    excl_hi = min(ncols, max_ccol + samples_per_chip)
    n_excl  = excl_hi - excl_lo + 1
    total = zero(T)
    excl  = zero(T)
    @inbounds for c in 1:ncols
        cs = col_sums[c]
        total += cs
        if excl_lo <= c <= excl_hi
            excl += cs
        end
    end
    (total - excl) / T((ncols - n_excl) * nrows)
end

# Hot path: called from acquire! with pre-allocated col_sums buffer and noise_power=nothing
function est_signal_noise_power(
    power_bins::AbstractMatrix{T},
    sampling_freq,
    code_freq,
    ::Nothing,
    col_sums::Vector{T},
) where {T<:AbstractFloat}
    samples_per_chip = floor(Int, sampling_freq / code_freq)
    nrows = size(power_bins, 1)
    max_val, max_drow, max_ccol = _findmax_and_colsums!(col_sums, power_bins)
    noise_power  = _noise_from_colsums(col_sums, nrows, max_ccol, samples_per_chip)
    signal_power = max_val - noise_power
    signal_power, noise_power, max_ccol, max_drow
end

# Convenience fallback: allocates col_sums (used in tests and one-off calls)
function est_signal_noise_power(power_bins, sampling_freq, code_freq, noise_power_in)
    T = eltype(power_bins)
    col_sums = zeros(T, size(power_bins, 2))
    if isnothing(noise_power_in)
        est_signal_noise_power(power_bins, sampling_freq, code_freq, nothing, col_sums)
    else
        max_val, max_drow, max_ccol = _findmax_and_colsums!(col_sums, power_bins)
        noise_power  = T(noise_power_in)
        signal_power = max_val - noise_power
        signal_power, noise_power, max_ccol, max_drow
    end
end
