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
        #complex_sig[:, doppler_idx] .= view(code_baseband, 1:size(signal_power, 1))
    end
end

function power_over_dopplers_code_mt!(
    acq_plan::AcquisitionPlan,
    signal,
    sat_prns,
    interm_freq,
    signal_baseband_buffer,
    signal_baseband_freq_domain_buffer
)
#Loop over codes
    foreach(sat_prns) do prn
        #println(prn)
        power_over_dopplers_mt!(
            acq_plan.signal_powers[prn],
            signal_baseband_buffer,
            signal_baseband_freq_domain_buffer,
            signal,
            acq_plan.fft_plan,
            acq_plan.dopplers,
            acq_plan.sampling_freq,
            interm_freq,
            acq_plan.codes_freq_domain[prn]
        )
    end
    return view(acq_plan.signal_powers, sat_prns)
end



function power_over_dopplers_code_mt!(
    acq_plan::AcquisitionPlan,
    signal,
    sat_prns,
    interm_freq,
    center_freqs,
    signal_baseband_buffer,
    signal_baseband_freq_domain_buffer
)
#Loop over codes
    ThreadsX.foreach(sat_prns, center_freqs) do  prn, center_freq
        #println(prn)
        power_over_dopplers_mt!(
            acq_plan.signal_powers[prn],
            signal_baseband_buffer,
            signal_baseband_freq_domain_buffer,
            signal,
            acq_plan.fft_plan,
            acq_plan.dopplers,
            acq_plan.sampling_freq,
            interm_freq + center_freq,
            acq_plan.codes_freq_domain[prn]
        )
    end
    return view(acq_plan.signal_powers, sat_prns)
end

function power_over_dopplers_mt!(
    signal_powers,
    signal_baseband_buffer,
    signal_baseband_freq_domain_buffer,
    signal,
    fft_plan,
    dopplers,
    sampling_freq,
    interm_freq,
    code_freq_domain
)
    #= 
    read only shared variables:
    signal
    interm_freq
    sampling_freq
    fft_plan
    code_freq_domain

    read only thread local:
    doppler

    read/write buffers:
    signal_baseband
    =#
#=     Threads.@threads for (   signal_power, 
            signal_baseband, 
            signal_baseband_freq_domain,
            doppler
             ) in zip(
            eachcol(signal_powers),
            eachcol(signal_baseband_buffer),
            eachcol(signal_baseband_freq_domain_buffer),
            ustrip(dopplers)
        ) =#

    ThreadsX.foreach(eachcol(signal_powers),
            eachcol(signal_baseband_buffer),
            eachcol(signal_baseband_freq_domain_buffer),
            dopplers) do signal_power,signal_baseband,signal_baseband_freq_domain,doppler

            signal_baseband = view(signal_baseband,1:length(signal))
            signal_baseband_freq_domain = view(signal_baseband_freq_domain,1:length(signal))


            downconvert!(signal_baseband, signal, interm_freq + doppler, sampling_freq)
            mul!(signal_baseband_freq_domain, fft_plan, signal_baseband) #forward FFT 
            #signal_baseband_freq_domain .= code_freq_domain .* conj.(signal_baseband_freq_domain) #multiply conjugate
            signal_baseband_freq_domain .= conj.(code_freq_domain) .* signal_baseband_freq_domain #flip the conjugate
            
            #mulconj_reinterpret!(signal_baseband_freq_domain,code_freq_domain,signal_baseband_freq_domain)
            ldiv!(signal_baseband, fft_plan,signal_baseband_freq_domain) #inverse FFT
            signal_power .= abs2.(signal_baseband)
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
    downconvert!(signal_baseband, signal, interm_freq + doppler, sampling_freq)
    mul!(signal_baseband_freq_domain, fft_plan, signal_baseband)
    #timestep = samplestep ./ Float32(ustrip(sampling_freq))
    doppler_correction = exp.(-1im .* 2pi .* time_shift_amt .* ustrip(doppler) ./ length(signal_baseband_freq_domain) .* collect(0:(length(signal_baseband_freq_domain)-1)))
    #println(time_shift_amt)
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