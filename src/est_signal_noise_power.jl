function est_signal_noise_power(power_bins, sampling_freq, code_freq, noise_power)
    samples_per_chip = floor(Int, sampling_freq / code_freq)
    signal_noise_power, index = findmax(power_bins)
    lower_code_phases = 1:index[1]-samples_per_chip
    upper_code_phases = index[1]+samples_per_chip:size(power_bins, 1)
    samples = (length(lower_code_phases) + length(upper_code_phases)) * size(power_bins, 2)
    noise_power =
        isnothing(noise_power) ?
        (
            sum(view(power_bins, lower_code_phases, :)) +
            sum(view(power_bins, upper_code_phases, :))
        ) / samples : noise_power
    signal_power = signal_noise_power - noise_power
    # Divide noise_power by 2 so peak_to_noise_ratio is on the chi-squared(2M) scale.
    # Under H0, each power bin is |I|² + |Q|² ~ χ²(2M) with mean 2M.
    # The CFAR threshold (from cfar_threshold()) is computed on this scale,
    # so the test statistic must match: peak / (mean/2) ≈ χ²(2M) under H0.
    # This is consistent with GNSS-SDR's pcps_acquisition::max_to_input_power_statistic().
    peak_to_noise_ratio = signal_noise_power / (noise_power / 2)
    signal_power, noise_power, peak_to_noise_ratio, index[1], index[2]
end