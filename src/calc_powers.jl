function power_over_doppler_and_codes(S::AbstractGNSS, signal, sat_prns, dopplers, sampling_freq, interm_freq)
    codes = [gen_code(length(signal), S, sat_prn, sampling_freq) for sat_prn in sat_prns]
    signal_baseband = Vector{ComplexF32}(undef, length(signal))
    signal_baseband_freq_domain = similar(signal_baseband)
    code_freq_baseband_freq_domain = similar(signal_baseband)
    code_baseband = similar(signal_baseband)
    Δt = length(signal) / sampling_freq
    code_interval = get_code_length(S) / get_code_frequency(S)
    signal_powers = [Matrix{Float32}(undef, convert(Int, sampling_freq * min(Δt, code_interval)), length(dopplers)) for _ in sat_prns]
    fft_plan = plan_fft(signal_baseband)
    codes_freq_domain = map(code -> fft_plan * code, codes)
    foreach(enumerate(dopplers)) do (doppler_idx, doppler)
        power_over_code!(
            signal_powers,
            doppler_idx,
            signal_baseband,
            signal_baseband_freq_domain,
            code_freq_baseband_freq_domain,
            code_baseband,
            signal,
            fft_plan,
            codes_freq_domain,
            doppler,
            sampling_freq,
            interm_freq
        )
    end
    signal_powers
end

function power_over_code!(
    signal_powers,
    doppler_idx,
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    signal,
    fft_plan,
    codes_freq_domain,
    doppler,
    sampling_freq,
    interm_freq
)
    downconvert!(signal_baseband, signal, interm_freq + doppler, sampling_freq)
    mul!(signal_baseband_freq_domain, fft_plan, signal_baseband)
    foreach(codes_freq_domain, signal_powers) do code_freq_domain, signal_power
        code_freq_baseband_freq_domain .= code_freq_domain .* conj.(signal_baseband_freq_domain)
        ldiv!(code_baseband, fft_plan, code_freq_baseband_freq_domain)
        signal_power[:,doppler_idx] .= abs2.(view(code_baseband, 1:size(signal_power, 1)))
    end
end