module Acquisition

    using DocStringExtensions, GNSSSignals, PyPlot, FFTW, Statistics
    import Unitful: s, Hz

    include("plots.jl")

    export acquire, plot_acquisition_results

    struct AcquisitionResults
        carrier_doppler::typeof(1.0Hz)
        code_phase::Float64
        CN0::Float64
        power_bins::Array{Float64, 2}
        doppler_steps::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}
    end

    function acquire(gnss_system::T, signal, sample_freq, interm_freq, sat_prn, max_doppler) where T <: AbstractGNSSSystem
        code_period = gnss_system.code_length / gnss_system.code_freq
        integration_time = length(signal) / sample_freq
        doppler_step = 1 / 10 / integration_time
        doppler_steps = -max_doppler:doppler_step:max_doppler
        cross_corr_powers = power_over_doppler_and_code(gnss_system, signal, sat_prn, doppler_steps, sample_freq, interm_freq)
        signal_power, noise_power, code_index, doppler_index = est_signal_noise_power(cross_corr_powers, doppler_steps, integration_time, sample_freq, gnss_system.code_freq)
        CN0 = 10 * log10(signal_power / noise_power / code_period / 1.0Hz)
        doppler = (doppler_index - 1) * doppler_step - max_doppler
        AcquisitionResults(doppler, (code_index - 1) / (sample_freq / gnss_system.code_freq), CN0, cross_corr_powers, first(doppler_steps) / 1.0Hz:step(doppler_steps) / 1.0Hz:last(doppler_steps) / 1.0Hz)
    end

    function power_over_doppler_and_code(gnss_system, signal, sat_prn, doppler_steps, sample_freq, interm_freq)
        code_freq_domain = fft(gen_code.(Ref(gnss_system), 1:length(signal), gnss_system.code_freq, 0, sample_freq, sat_prn))
        mapreduce(doppler -> power_over_code(gnss_system, signal, code_freq_domain, doppler, sample_freq, interm_freq), hcat, doppler_steps)
    end

    function power_over_code(gnss_system, signal, code_freq_domain, doppler, sample_freq, interm_freq)
        Δt = length(signal) / sample_freq
        code_interval = gnss_system.code_length / gnss_system.code_freq
        replica_carrier = gen_carrier.(1:length(signal), interm_freq + doppler, 0.0, sample_freq)
        signal_baseband_freq_domain = fft(signal .* conj(replica_carrier))
        powers = abs2.(ifft(code_freq_domain .* conj(signal_baseband_freq_domain)))
        powers[1:convert(Int, sample_freq * min(Δt, code_interval))]
    end

    function est_signal_noise_power(power_bins, doppler_steps, integration_time, sample_freq, code_freq)
        samples_per_chip = floor(Int, sample_freq / code_freq)
        signal_noise_power, index = findmax(power_bins)
        linear_index = size(power_bins, 1) * (index[2] - 1) + index[1]
        noise_power = median(power_bins[[1:linear_index - 1; linear_index + 1:length(power_bins)]])
        signal_power = signal_noise_power - noise_power
        signal_power, noise_power, index[1], index[2]
    end
end
