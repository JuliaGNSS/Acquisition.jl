module Acquisition

    using DocStringExtensions, GNSSSignals, PyPlot, FFTW, Statistics, LinearAlgebra
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

    function acquire(::Type{S}, signal, sample_freq, interm_freq, sat_prn, max_doppler) where S <: AbstractGNSSSystem
        code_period = get_code_length(S) / get_code_frequency(S)
        integration_time = length(signal) / sample_freq
        doppler_step = 1 / 3 / integration_time
        doppler_steps = -max_doppler:doppler_step:max_doppler
        powers = power_over_doppler_and_code(S, signal, sat_prn, doppler_steps, sample_freq, interm_freq)
        signal_power, noise_power, code_index, doppler_index = est_signal_noise_power(powers, doppler_steps, integration_time, sample_freq, get_code_frequency(S))
        CN0 = 10 * log10(signal_power / noise_power / code_period / 1.0Hz)
        doppler = (doppler_index - 1) * doppler_step - max_doppler
        AcquisitionResults(doppler, (code_index - 1) / (sample_freq / get_code_frequency(S)), CN0, powers, first(doppler_steps) / 1.0Hz:step(doppler_steps) / 1.0Hz:last(doppler_steps) / 1.0Hz)
    end

    function power_over_doppler_and_code(::Type{S}, signal, sat_prn, doppler_steps, sample_freq, interm_freq) where S <: AbstractGNSSSystem
        code = get_code.(S, (1:length(signal)) .* get_code_frequency(S) ./ sample_freq, sat_prn)
        fft_plan = plan_fft(code)
        code_freq_domain = fft_plan * code
        mapreduce(doppler -> power_over_code(S, signal, fft_plan, code_freq_domain, doppler, sample_freq, interm_freq), hcat, doppler_steps)
    end

    function power_over_code(::Type{S}, signal, fft_plan, code_freq_domain, doppler, sample_freq, interm_freq) where S <: AbstractGNSSSystem
        Δt = length(signal) / sample_freq
        code_interval = get_code_length(S) / get_code_frequency(S)
        signal_baseband = signal .* cis.(-2π .* (1:length(signal)) .* (interm_freq + doppler) ./ sample_freq)
        signal_baseband_freq_domain = fft_plan * signal_baseband
        code_freq_baseband_freq_domain = code_freq_domain .* conj(signal_baseband_freq_domain)
        powers = abs2.(fft_plan \ code_freq_baseband_freq_domain)
        powers[1:convert(Int, sample_freq * min(Δt, code_interval))]
    end

    function est_signal_noise_power(power_bins, doppler_steps, integration_time, sample_freq, code_freq)
        samples_per_chip = floor(Int, sample_freq / code_freq)
        signal_noise_power, index = findmax(power_bins)
        lower_code_phases = 1:index[1] - samples_per_chip
        upper_code_phases = index[1] + samples_per_chip:size(power_bins, 1)
        samples = (length(lower_code_phases) + length(upper_code_phases)) * size(power_bins, 2)
        noise_power = (sum(power_bins[lower_code_phases,:]) + sum(power_bins[upper_code_phases,:])) / samples
        signal_power = signal_noise_power - noise_power
        signal_power, noise_power, index[1], index[2]
    end
end
