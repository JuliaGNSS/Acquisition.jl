# power_bins layout: (num_doppler_bins, num_code_phase_bins)
#   rows = Doppler bins, columns = code phase bins
#
# Single-pass strategy:
#   Pass 1 — scan column-major: find global max, accumulate per-column sums into col_sums.
#   Pass 2 — scan col_sums: total minus exclusion zone around peak column.
# This avoids two separate partial-range scans over the full matrix.
#
# Returns: (signal_power, noise_power, code_phase_bin, doppler_bin)

function _findmax_and_colsums!(col_sums::Vector{T}, power_bins::AbstractMatrix{T}) where {T<:AbstractFloat}
    num_doppler_bins, num_code_phase_bins = size(power_bins)
    peak_power           = power_bins[1, 1]
    peak_doppler_bin     = 1
    peak_code_phase_bin  = 1
    @inbounds for c in 1:num_code_phase_bins
        col_sum = zero(T)
        for r in 1:num_doppler_bins
            v = power_bins[r, c]
            col_sum += v
            if v > peak_power
                peak_power          = v
                peak_doppler_bin    = r
                peak_code_phase_bin = c
            end
        end
        col_sums[c] = col_sum
    end
    peak_power, peak_doppler_bin, peak_code_phase_bin
end

function _noise_from_colsums(
    col_sums::Vector{T},
    num_doppler_bins::Int,
    peak_code_phase_bin::Int,
    samples_per_chip::Int,
) where {T<:AbstractFloat}
    num_code_phase_bins = length(col_sums)
    excl_col_lo   = max(1, peak_code_phase_bin - samples_per_chip)
    excl_col_hi   = min(num_code_phase_bins, peak_code_phase_bin + samples_per_chip)
    num_excl_cols = excl_col_hi - excl_col_lo + 1
    total_power = zero(T)
    excl_power  = zero(T)
    @inbounds for c in 1:num_code_phase_bins
        col_sum = col_sums[c]
        total_power += col_sum
        if excl_col_lo <= c <= excl_col_hi
            excl_power += col_sum
        end
    end
    (total_power - excl_power) / T((num_code_phase_bins - num_excl_cols) * num_doppler_bins)
end

function est_signal_noise_power(
    power_bins::AbstractMatrix{T},
    sampling_freq,
    code_freq,
    col_sums::Vector{T},
) where {T<:AbstractFloat}
    samples_per_chip = floor(Int, sampling_freq / code_freq)
    num_doppler_bins = size(power_bins, 1)
    peak_power, peak_doppler_bin, peak_code_phase_bin =
        _findmax_and_colsums!(col_sums, power_bins)
    noise_power  = _noise_from_colsums(col_sums, num_doppler_bins, peak_code_phase_bin, samples_per_chip)
    signal_power = peak_power - noise_power
    signal_power, noise_power, peak_code_phase_bin, peak_doppler_bin
end
