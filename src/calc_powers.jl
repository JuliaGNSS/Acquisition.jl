function power_over_doppler_and_codes!(
    acq_plan::AcquisitionPlan,
    signal,
    sat_prns,
    interm_freq,
    doppler_offset,
)
    prn_channels = findall(x -> x in sat_prns, acq_plan.avail_prn_channels)
    signal_powers_view = view(acq_plan.signal_powers, prn_channels)
    codes_freq_domain_view = view(acq_plan.codes_freq_domain, prn_channels)
    @inbounds for (doppler_idx, doppler) in enumerate(acq_plan.dopplers)
        power_over_code!(
            signal_powers_view,
            doppler_idx,
            acq_plan.signal_baseband,
            acq_plan.signal_baseband_freq_domain,
            acq_plan.code_freq_baseband_freq_domain,
            acq_plan.code_baseband,
            signal,
            acq_plan.fft_plan,
            acq_plan.bfft_plan,
            codes_freq_domain_view,
            doppler + doppler_offset,
            acq_plan.sampling_freq,
            interm_freq,
        )
    end
    signal_powers_view
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
    bfft_plan,
    codes_freq_domain,
    doppler,
    sampling_freq,
    interm_freq,
)
    downconvert!(signal_baseband, signal, interm_freq + doppler, sampling_freq)
    mul!(signal_baseband_freq_domain, fft_plan, signal_baseband)
    @inbounds for (code_freq_domain, signal_power) in zip(codes_freq_domain, signal_powers)
        code_freq_baseband_freq_domain .=
            code_freq_domain .* conj.(signal_baseband_freq_domain)
        mul!(code_baseband, bfft_plan, code_freq_baseband_freq_domain)
        signal_power[:, doppler_idx] .= abs2.(view(code_baseband, 1:size(signal_power, 1)))
    end
end