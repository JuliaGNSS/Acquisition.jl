function power_over_doppler_and_codes!(
    acq_plan::AcquisitionPlan,
    signal,
    sat_prns,
    interm_freq,
    doppler_offset,
)
    prn_channels = findall(x -> x in sat_prns, acq_plan.avail_prn_channels)
    foreach(enumerate(acq_plan.dopplers)) do (doppler_idx, doppler)
        power_over_code!(
            acq_plan.system,
            prn_channels,
            view(acq_plan.signal_powers, prn_channels),
            doppler_idx,
            acq_plan.signal_baseband,
            acq_plan.signal_baseband_freq_domain,
            acq_plan.code_freq_baseband_freq_domain,
            acq_plan.code_baseband,
            signal,
            acq_plan.fft_plan,
            acq_plan.code_freq_domain,
            doppler + doppler_offset,
            acq_plan.sampling_freq,
            interm_freq,
        )
    end
    view(acq_plan.signal_powers, prn_channels)
end

function power_over_code!(
    system,
    prns,
    signal_powers,
    doppler_idx,
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    signal,
    fft_plan,
    code_freq_domain,
    doppler,
    sampling_freq,
    interm_freq,
)
    downconvert!(signal_baseband, signal, interm_freq + doppler, sampling_freq)
    mul!(signal_baseband_freq_domain, fft_plan, signal_baseband)
    code_frequency = get_code_frequency(system) + doppler * get_code_center_frequency_ratio(system)
    foreach(signal_powers, prns) do signal_power, prn
        gen_code!(code_baseband, system, prn, sampling_freq, code_frequency)
        mul!(code_freq_domain, fft_plan, code_baseband)
        code_freq_baseband_freq_domain .=
            code_freq_domain .* conj.(signal_baseband_freq_domain)
        ldiv!(code_baseband, fft_plan, code_freq_baseband_freq_domain)
        signal_power[:, doppler_idx] .= abs2.(view(code_baseband, 1:size(signal_power, 1)))
    end
end