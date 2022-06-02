module Acquisition

    using DocStringExtensions
    using FFTW
    using GNSSSignals
    using LinearAlgebra
    using RecipesBase
    using Statistics

    import Unitful: s, Hz

    export acquire, plot_acquisition_results

    struct AcquisitionResults{S<:AbstractGNSS}
        system::S
        prn::Int
        sampling_frequency::typeof(1.0Hz)
        carrier_doppler::typeof(1.0Hz)
        code_phase::Float64
        CN0::Float64
        power_bins::Array{Float64, 2}
        dopplers::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}
    end

    @recipe function f(acq_res::AcquisitionResults, log_scale::Bool = false;)
        seriestype := :surface
        seriescolor --> :viridis
        yguide --> "Code phase"
        xguide --> "Dopplers [Hz]"
        zguide --> (log_scale ? "Magnitude [dB]" : "Magnitude")
        y = (1:size(acq_res.power_bins, 1)) ./ acq_res.sampling_frequency .* get_code_frequency(acq_res.system)
        acq_res.dopplers, y, log_scale ? 10 * log10.(acq_res.power_bins) : acq_res.power_bins
    end

    """
    $(SIGNATURES)
    Perform the aquisition of the satellite `sat_prn` in System `S` in signal `signal`
    sampled at rate `sampling_freq`. The aquisition is performed as parallel code phase
    search using the doppler frequencies `dopplers`.
    """
    function acquire(S::AbstractGNSS, signal, sampling_freq, sat_prn; interm_freq = 0.0Hz, dopplers = -5000.0Hz:1 / 3 / (length(signal) / sampling_freq):5000.0Hz)
        code_period = get_code_length(S) / get_code_frequency(S)
        integration_time = length(signal) / sampling_freq
        powers = power_over_doppler_and_code(S, signal, sat_prn, dopplers, sampling_freq, interm_freq)
        signal_power, noise_power, code_index, doppler_index = est_signal_noise_power(powers, sampling_freq, get_code_frequency(S))
        CN0 = 10 * log10(signal_power / noise_power / code_period / 1.0Hz)
        doppler = (doppler_index - 1) * step(dopplers) - last(dopplers)
        code_phase = (code_index - 1) / (sampling_freq / get_code_frequency(S))
        AcquisitionResults(S, sat_prn, sampling_freq, doppler, code_phase, CN0, powers, dopplers / 1.0Hz)
    end

    function power_over_doppler_and_code(S::AbstractGNSS, signal, sat_prn, doppler_steps, sampling_freq, interm_freq)
        code = get_code.(S, (1:length(signal)) .* get_code_frequency(S) ./ sampling_freq, sat_prn)
        fft_plan = plan_fft(code)
        code_freq_domain = fft_plan * code
        mapreduce(doppler -> power_over_code(S, signal, fft_plan, code_freq_domain, doppler, sampling_freq, interm_freq), hcat, doppler_steps)
    end

    function power_over_code(S::AbstractGNSS, signal, fft_plan, code_freq_domain, doppler, sampling_freq, interm_freq)
        Δt = length(signal) / sampling_freq
        code_interval = get_code_length(S) / get_code_frequency(S)
        signal_baseband = signal .* cis.(-2π .* (1:length(signal)) .* (interm_freq + doppler) ./ sampling_freq)
        signal_baseband_freq_domain = fft_plan * signal_baseband
        code_freq_baseband_freq_domain = code_freq_domain .* conj(signal_baseband_freq_domain)
        powers = abs2.(fft_plan \ code_freq_baseband_freq_domain)
        powers[1:convert(Int, sampling_freq * min(Δt, code_interval))]
    end

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
end
