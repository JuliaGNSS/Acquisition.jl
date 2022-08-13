function est_signal_noise_power(power_bins, sampling_freq, code_freq)
    samples_per_chip = floor(Int, sampling_freq / code_freq)
    signal_noise_power, index = findmax(power_bins)
    lower_code_phases = 1:index[1] - samples_per_chip
    upper_code_phases = index[1] + samples_per_chip:size(power_bins, 1)
    samples = (length(lower_code_phases) + length(upper_code_phases)) * size(power_bins, 2)
    noise_power = (sum(power_bins[lower_code_phases,:]) + sum(power_bins[upper_code_phases,:])) / samples
    signal_power = signal_noise_power - noise_power
    signal_power, noise_power, index[1], index[2]
end