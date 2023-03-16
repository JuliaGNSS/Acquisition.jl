function power_over_doppler_and_codes!(
    acq_plan::AcquisitionPlan,
    signal,
    sat_prns,
    interm_freq,
    doppler_offset;
)
    prn_channels = findall(x -> x in sat_prns, acq_plan.avail_prn_channels)
    foreach(enumerate(acq_plan.dopplers)) do (doppler_idx, doppler)
        power_over_code!(
            view(acq_plan.signal_powers, prn_channels),
            view(acq_plan.complex_signal, prn_channels),
            doppler_idx,
            acq_plan.signal_baseband,
            acq_plan.signal_baseband_freq_domain,
            acq_plan.code_freq_baseband_freq_domain,
            acq_plan.code_baseband,
            signal,
            acq_plan.fft_plan,
            view(acq_plan.codes_freq_domain, prn_channels),
            doppler + doppler_offset,
            acq_plan.sampling_freq,
            interm_freq;
        )
    end
    (view(acq_plan.signal_powers, prn_channels),view(acq_plan.complex_signal, prn_channels))
end

function power_over_doppler_and_codes!(
    acq_plan::AcquisitionPlan,
    signal,
    sat_prns,
    interm_freq,
    doppler_offset,
    time_shift_amt
)
    prn_channels = findall(x -> x in sat_prns, acq_plan.avail_prn_channels)
    foreach(enumerate(acq_plan.dopplers)) do (doppler_idx, doppler)
        power_over_code!(
            view(acq_plan.signal_powers, prn_channels),
            view(acq_plan.complex_signal, prn_channels),
            doppler_idx,
            acq_plan.signal_baseband,
            acq_plan.signal_baseband_freq_domain,
            acq_plan.code_freq_baseband_freq_domain,
            acq_plan.code_baseband,
            signal,
            acq_plan.fft_plan,
            view(acq_plan.codes_freq_domain, prn_channels),
            doppler + doppler_offset,
            acq_plan.sampling_freq,
            interm_freq,
            time_shift_amt,
            acq_plan.compensate_doppler_code
        )
    end
    (view(acq_plan.signal_powers, prn_channels),view(acq_plan.complex_signal, prn_channels))
end


function power_over_code!(
    signal_powers,
    complex_signal,
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
    interm_freq,
)
    downconvert!(signal_baseband, signal, interm_freq + doppler, sampling_freq)
    mul!(signal_baseband_freq_domain, fft_plan, signal_baseband)
    foreach(codes_freq_domain, signal_powers, complex_signal) do code_freq_domain, signal_power, complex_sig
        code_freq_baseband_freq_domain .=
            code_freq_domain .* conj.(signal_baseband_freq_domain)
        ldiv!(code_baseband, fft_plan, code_freq_baseband_freq_domain)
        signal_power[:, doppler_idx] .= abs2.(view(code_baseband, 1:size(signal_power, 1)))
        complex_sig[:, doppler_idx] .= view(code_baseband, 1:size(signal_power, 1))
    end
end

function power_over_code!(
    signal_powers,
    complex_signal,
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
    interm_freq,
    time_shift_amt,
    compensate_doppler_code
)
    downconvert!(signal_baseband, signal, interm_freq + doppler, sampling_freq,samplestep)
    mul!(signal_baseband_freq_domain, fft_plan, signal_baseband)
    timestep = samplestep ./ Float32(ustrip(sampling_freq))
    #println(time_shift_amt) 
    doppler_correction = exp.(1im .* 2pi .* time_shift_amt .*Float32(ustrip(sampling_freq)) .* collect(0:(length(signal_baseband_freq_domain)-1)) .* (ustrip(doppler) / 1575.42e6))
    #println(doppler_correction)
    foreach(codes_freq_domain, signal_powers, complex_signal) do code_freq_domain, signal_power, complex_sig
        if compensate_doppler_code
            code_freq_baseband_freq_domain .=
                code_freq_domain .* conj.(signal_baseband_freq_domain) .* doppler_correction
        else
            code_freq_baseband_freq_domain .=
                code_freq_domain .* conj.(signal_baseband_freq_domain)
        end
        ldiv!(code_baseband, fft_plan, code_freq_baseband_freq_domain)
        signal_power[:, doppler_idx] .= abs2.(view(code_baseband, 1:size(signal_power, 1)))
        complex_sig[:, doppler_idx] .= view(code_baseband, 1:size(signal_power, 1))
    end
end