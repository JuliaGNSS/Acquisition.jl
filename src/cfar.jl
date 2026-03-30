"""
    cfar_threshold(pfa, num_cells; num_noncoherent_integrations=1)

Compute the CFAR (Constant False Alarm Rate) detection threshold for a given
probability of false alarm.

The threshold is computed so that the probability of any noise-only cell in the
search grid exceeding the threshold equals `pfa`. This accounts for the multiple
hypothesis testing across all `num_cells` search cells (code phases × Doppler bins)
using a Bonferroni-like correction.

The test statistic is `peak_power / noise_power` (the `peak_to_noise_ratio` field
of [`AcquisitionResults`](@ref)).

# Arguments

  - `pfa`: Target probability of false alarm (e.g., `0.01`). Must be in `(0, 1)`.
  - `num_cells`: Total number of search cells (typically `num_code_phases * num_doppler_bins`).

# Keyword Arguments

  - `num_noncoherent_integrations`: Number of non-coherent integration dwells (default: `1`).
    More dwells increase the degrees of freedom of the chi-squared distribution.

# Returns

Threshold value to compare against `result.peak_to_noise_ratio`.

# Example

```julia
using Acquisition, GNSSSignals

results = acquire(GPSL1(), signal, 5e6Hz, 1:32)
num_cells = size(results[1].power_bins, 1) * size(results[1].power_bins, 2)
threshold = cfar_threshold(0.01, num_cells)
detected = filter(r -> r.peak_to_noise_ratio > threshold, results)
```

# Mathematical Background

Under the null hypothesis (noise only), the test statistic `peak_to_noise_ratio`
(computed as `peak_power / (noise_power / 2)`) follows a chi-squared distribution
with `2 * num_noncoherent_integrations` degrees of freedom. The factor-of-2
normalization in the noise estimate places the statistic on the χ²(2M) scale
(mean = 2M under H0), matching the threshold returned by this function.

The per-cell false alarm probability is adjusted for multiple testing:
`pfa_per_cell = 1 - (1 - pfa)^(1/num_cells)`.

The threshold is then the inverse CDF (quantile) of the chi-squared distribution
at `1 - pfa_per_cell`.

# References

  - Springer Handbook of GNSS, Section 14.3.1 "Test Statistics", Eq. 14.28–14.30
  - GNSS-SDR PCPS Acquisition: `pcps_acquisition::calculate_threshold()`
  - Kay Borre et al., "A Software-Defined GPS and Galileo Receiver", Birkhäuser, 2007

# See also

[`AcquisitionResults`](@ref), [`acquire`](@ref)
"""
function cfar_threshold(pfa, num_cells; num_noncoherent_integrations = 1)
    0 < pfa < 1 || throw(ArgumentError("pfa must be in (0, 1), got $pfa"))
    num_cells > 0 || throw(ArgumentError("num_cells must be positive, got $num_cells"))
    num_noncoherent_integrations > 0 ||
        throw(ArgumentError("num_noncoherent_integrations must be positive"))
    # Per-cell false alarm probability (Bonferroni-like correction)
    pfa_per_cell = 1 - (1 - pfa)^(1 / num_cells)
    # Degrees of freedom: 2M for M non-coherent integrations of complex samples
    # Under H0, peak_power/noise_power ~ Gamma(M, 1) (after normalization)
    # chi2(2M) = 2 * Gamma(M, 1), so threshold = 2 * gamma_quantile
    # gamma_inc_inv(a, p, q) returns x such that gamma_inc(a, x) = (p, q)
    # We want P(X > threshold) = pfa_per_cell, i.e., q = pfa_per_cell
    a = Float64(num_noncoherent_integrations)
    p_lower = 1.0 - pfa_per_cell
    q_upper = pfa_per_cell
    x = gamma_inc_inv(a, p_lower, q_upper)
    Float64(2 * x)
end
