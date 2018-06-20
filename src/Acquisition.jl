module Acquisition

    using DocStringExtensions, GNSSSignals

    export acquire

    struct AcquisitionResults
        acquired::Bool                # Sat acquired?
        f_d::Float64                  # Doppler frequency
        φ_c::Float64                  # Code phase
        C╱N₀::Float64                 # C╱N₀ in dB
        power_bins::Array{Float64, 2} # Cross corr powers in code_bins x doppler_bins
    end
    
    function acquire(signal, sample_freq, interm_freq, code_freq, code, max_doppler, threshold)
        code_period = length(code) / code_freq
        integration_time = length(signal) / sample_freq
        doppler_step = 2 / 3 / integration_time
        doppler_steps = -max_doppler:doppler_step:max_doppler
        cross_corr_powers = power_over_doppler_and_code(signal, code, doppler_steps, sample_freq, interm_freq, code_freq)
        signal_power, noise_power, signal_index = est_signal_noise_power(cross_corr_powers, doppler_steps, integration_time, sample_freq, code_freq)
        C╱N₀ = 10 * log10(signal_power / noise_power / code_period)
        if C╱N₀ >= threshold
            c_idx, d_idx = ind2sub(cross_corr_powers, signal_index)
            doppler = (d_idx - 1) * doppler_step - max_doppler
            AcquisitionResults(true, doppler, (c_idx - 1) / (sample_freq / code_freq), C╱N₀, cross_corr_powers)
        else
            AcquisitionResults(false, NaN, NaN, C╱N₀, cross_corr_powers)
        end
    end
    
    function power_over_doppler_and_code(signal, code, doppler_steps, sample_freq, interm_freq, code_freq)
        code_freq_domain = fft(gen_sat_code(1:length(signal), code_freq, 0, 0, sample_freq, code))
        return mapreduce(doppler -> power_over_code(signal, code_freq_domain, doppler, sample_freq, interm_freq), hcat, doppler_steps)
    end
    
    function power_over_code(signal, code_freq_domain, doppler, sample_freq, interm_freq)
        replica_carrier = gen_carrier(1:length(signal), interm_freq, doppler, 0, sample_freq)
        signal_baseband_freq_domain = fft(signal .* conj(replica_carrier))
        powers = abs2.(ifft(code_freq_domain .* conj(signal_baseband_freq_domain)))
        return powers[1:Int(sample_freq * 1e-3)]
    end
    
    function est_signal_noise_power(power_bins, doppler_steps, integration_time, sample_freq, code_freq)
        samples_per_chip = floor(Int, sample_freq / code_freq)
        signal_noise_power, index = findmax(power_bins)
        c_idx, d_idx = ind2sub(power_bins, index)
        max_index_self_interf_doppler = 1 / integration_time / step(doppler_steps)
        max_index_self_intercode_freqode = samples_per_chip
        code_circ_shift = ((c_idx - max_index_self_intercode_freqode < 1) - (c_idx + max_index_self_intercode_freqode > size(power_bins, 1))) * max_index_self_intercode_freqode
        shifted_power_bins = circshift(power_bins, (code_circ_shift, 0))
        c_lower_idx = Int(c_idx - max_index_self_intercode_freqode + code_circ_shift)
        c_upper_idx = Int(c_idx + max_index_self_intercode_freqode + code_circ_shift)
        d_lower_idx = floor(Int, max(1, d_idx - max_index_self_interf_doppler))
        d_upper_idx = floor(Int, min(size(power_bins, 2), d_idx + max_index_self_interf_doppler))
        signal_self_interf = shifted_power_bins[c_lower_idx:c_upper_idx, d_lower_idx:d_upper_idx]
        noise_power = (sum(power_bins) - sum(signal_self_interf)) / (length(power_bins) - length(signal_self_interf))
        signal_power = signal_noise_power - noise_power
        return signal_power, noise_power, index
    end
    

end