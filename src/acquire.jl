"""
$(SIGNATURES)
Perform the aquisition of multiple satellites with `prns` in system `system` with signal `signal`
sampled at rate `sampling_freq`. Optional arguments are the intermediate frequency `interm_freq`
(default 0Hz), the maximum expected Doppler `max_doppler` (default 7000Hz). If the maximum Doppler
is too unspecific you can instead pass a Doppler range with with your individual step size using
the argument `dopplers`.
"""
function acquire(
    system::AbstractGNSS,
    signal,
    sampling_freq,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    dopplers = -max_doppler:1/3/(0.001s):max_doppler,
    noncoherent_rounds = 1,
    compensate_doppler_code = :positive,
    coherent_integration_length_samples=4096
)
    acq_plan = AcquisitionPlan(
        system,
        coherent_integration_length_samples,
        sampling_freq;
        dopplers,
        prns,
        fft_flag = FFTW.ESTIMATE,
        compensate_doppler_code = compensate_doppler_code,
        noncoherent_rounds= noncoherent_rounds
    )
    acquire!(acq_plan, signal, prns; interm_freq)
end

#= function acquire(
    system::AbstractGNSS,
    signal,
    sampling_freq,
    prns::AbstractVector{<:Integer},
    time_shift_amt;
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    dopplers = -max_doppler:1/3/(length(signal)/sampling_freq):max_doppler,
    compensate_doppler_code = false
)
    acq_plan = AcquisitionPlan(
        system,
        length(signal),
        sampling_freq;
        dopplers,
        prns,
        fft_flag = FFTW.ESTIMATE,
        compensate_doppler_code = compensate_doppler_code
    )
    acquire!(acq_plan, signal, prns,time_shift_amt; interm_freq)
end =#


#= """
$(SIGNATURES)
This acquisition function uses a predefined acquisition plan to accelerate the computing time.
This will be useful, if you have to calculate the acquisition multiple time in a row.
"""
function acquire!(
    acq_plan::AcquisitionPlan,
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
    noise_power = nothing,
)
    all(map(prn -> prn in acq_plan.avail_prn_channels, prns)) ||
        throw(ArgumentError("You'll need to plan every PRN"))
    code_period = get_code_length(acq_plan.system) / get_code_frequency(acq_plan.system)
    powers_per_sats, complex_sigs_per_sats =
        power_over_doppler_and_codes!(acq_plan, signal, prns, interm_freq, doppler_offset)
    map(powers_per_sats, complex_sigs_per_sats, prns) do powers, complex_sig, prn
        signal_power, noise_power, code_index, doppler_index = est_signal_noise_power(
            powers,
            acq_plan.sampling_freq,
            get_code_frequency(acq_plan.system), 
            noise_power,
        )
        CN0 = 10 * log10(signal_power / noise_power / code_period / 1.0Hz)
        doppler =
            (doppler_index - 1) * step(acq_plan.dopplers) +
            first(acq_plan.dopplers) +
            doppler_offset
        code_phase =
            (code_index - 1) /
            (acq_plan.sampling_freq / get_code_frequency(acq_plan.system))
        AcquisitionResults(
            acq_plan.system,
            prn,
            acq_plan.sampling_freq,
            doppler,
            code_phase,
            CN0,
            noise_power,
            powers,
            complex_sig,
            (acq_plan.dopplers .+ doppler_offset) / 1.0Hz,
        )
    end
end  =#

"""
$(SIGNATURES)
This acquisition function uses a predefined acquisition plan to accelerate the computing time.
This will be useful, if you have to calculate the acquisition multiple time in a row.
"""
function acquire!(
    acq_plan::AcquisitionPlan,
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
    noise_power = nothing,    
)
    all(map(prn -> prn in acq_plan.avail_prn_channels, prns)) ||
        throw(ArgumentError("You'll need to plan every PRN"))
    code_period = get_code_length(acq_plan.system) / get_code_frequency(acq_plan.system)


#=     powers_per_sats, complex_sigs_per_sats =
        power_over_doppler_and_codes!(acq_plan, signal, prns, interm_freq, doppler_offset) =#

    #Currently we parallelize over doppler taps 
    #It might be faster if we parallelize over PRNs

    #preallocate noncoherent integration workbuffer
    # we want a 3d array with dimensions of (1ms signal * doppler taps * prns)
    plan = acq_plan
    powers_per_sats = [zeros(Float32,plan.signal_length,length(plan.dopplers)) for _ in eachindex(prns)]
    intermediate_freq = interm_freq
    center_frequency = 0.0Hz
    noncoherent_rounds = plan.noncoherent_rounds
    compensate_doppler_code = plan.compensate_doppler_code
    for (noncoherent_power_buffer_idx, prn) in enumerate(prns)
        noncoherent_power_accumulator_buffer = powers_per_sats[noncoherent_power_buffer_idx]
        for (chunk,round_idx) in zip(Iterators.partition(signal[1:(noncoherent_rounds*plan.signal_length)],plan.signal_length), 0:noncoherent_rounds-1)
            acq = acquire_mt!(plan,chunk, [prn]; interm_freq=(intermediate_freq+center_frequency))
            #power_bin_ncoh3 += acq.power_bins
              #for each dopplers compensate for code drift
              #for idx1 in 1:length(plan.dopplers)
              for (acc,power_doppler,dop) in zip(eachcol(noncoherent_power_accumulator_buffer),eachcol(acq[1]),plan.dopplers)
                  #CM-ABS algo 3 step 6 eqn 7
                  doppler1 = ustrip(dop)
                  if compensate_doppler_code == :negative
                      Ns = sign(ustrip(doppler1) - ustrip(center_frequency)) * round(round_idx*0.001* ustrip(plan.sampling_freq) * abs(ustrip(doppler1) - ustrip(center_frequency))/ustrip(get_center_frequency(plan.system)), RoundNearest)
                      acc .= acc .+ circshift(power_doppler,-Ns)
                  elseif compensate_doppler_code == :positive
                      Ns = sign(ustrip(doppler1) - ustrip(center_frequency)) * round(round_idx*0.001* ustrip(plan.sampling_freq) * abs(ustrip(doppler1) - ustrip(center_frequency))/ustrip(get_center_frequency(plan.system)), RoundNearest)
                      acc .= acc .+ circshift(power_doppler,Ns)
                  else
                      acc .= acc .+ power_doppler
                  end
              end
        end
    end

    map(powers_per_sats, prns) do powers, prn
        signal_power, noise_power, code_index, doppler_index = est_signal_noise_power(
            powers,
            acq_plan.sampling_freq,
            get_code_frequency(acq_plan.system), 
            noise_power,
        )
        CN0 = 10 * log10(signal_power / noise_power / code_period / 1.0Hz)
        doppler =
            (doppler_index - 1) * step(acq_plan.dopplers) +
            first(acq_plan.dopplers) +
            doppler_offset
        code_phase =
            (code_index - 1) /
            (acq_plan.sampling_freq / get_code_frequency(acq_plan.system))
        AcquisitionResults(
            acq_plan.system,
            prn,
            acq_plan.sampling_freq,
            doppler,
            code_phase,
            CN0,
            noise_power,
            powers,
            (acq_plan.dopplers .+ doppler_offset) / 1.0Hz,
        )
    end 
end

function acquire_mt!(
    acq_plan::AcquisitionPlan,
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
    noise_power = nothing,
)
    all(map(prn -> prn in acq_plan.avail_prn_channels, prns)) ||
        throw(ArgumentError("You'll need to plan every PRN"))
    code_period = get_code_length(acq_plan.system) / get_code_frequency(acq_plan.system)

    signal_baseband_buffer, signal_baseband_freq_domain_buffer = preallocate_thread_local_buffer(acq_plan.signal_length, length(acq_plan.dopplers))
    #println(prns)

    powers_per_sats = power_over_dopplers_code_mt!(acq_plan,signal,prns,interm_freq,signal_baseband_buffer,signal_baseband_freq_domain_buffer)
    return powers_per_sats
end


#= function acquire!(
    acq_plan::AcquisitionPlan,
    signal,
    prns::AbstractVector{<:Integer},
    time_shift_amt;
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
    noise_power = nothing,
)
    all(map(prn -> prn in acq_plan.avail_prn_channels, prns)) ||
        throw(ArgumentError("You'll need to plan every PRN"))
    code_period = get_code_length(acq_plan.system) / get_code_frequency(acq_plan.system)
    powers_per_sats, complex_sigs_per_sats =
        power_over_doppler_and_codes!(acq_plan, signal, prns, interm_freq, doppler_offset,time_shift_amt)
    map(powers_per_sats, complex_sigs_per_sats, prns) do powers, complex_sig, prn
        signal_power, noise_power, code_index, doppler_index = est_signal_noise_power(
            powers,
            acq_plan.sampling_freq,
            get_code_frequency(acq_plan.system),
            noise_power,
        )
        CN0 = 10 * log10(signal_power / noise_power / code_period / 1.0Hz)
        doppler =
            (doppler_index - 1) * step(acq_plan.dopplers) +
            first(acq_plan.dopplers) +
            doppler_offset
        code_phase =
            (code_index - 1) /
            (acq_plan.sampling_freq / get_code_frequency(acq_plan.system))
        AcquisitionResults(
            acq_plan.system,
            prn,
            acq_plan.sampling_freq,
            doppler,
            code_phase,
            CN0,
            noise_power,
            powers,
            complex_sig,
            (acq_plan.dopplers .+ doppler_offset) / 1.0Hz,
        )
    end
  end =#


  function noncoherent_integrate!(fp, prn, noncoherent_rounds, center_frequency, plan::AcquisitionPlan; intermediate_freq=0, compensate_doppler_code=false)
    rate = ustrip(plan.sampling_freq)
  
    samples_1ms = Int(round(rate * 0.001))
    power_bin_ncoh3 = zeros(Float32,plan.signal_length,length(plan.dopplers))
    for (chunk,round_idx) in zip(Iterators.partition(fp[1:(noncoherent_rounds*plan.signal_length)],plan.signal_length), 0:noncoherent_rounds-1)
      acq = acquire_mt!(plan,chunk, [prn]; interm_freq=(intermediate_freq+center_frequency)*Hz)
      #power_bin_ncoh3 += acq.power_bins
        #for each dopplers compensate for code drift
        #for idx1 in 1:length(plan.dopplers)
        for (acc,power_doppler,dop) in zip(eachcol(power_bin_ncoh3),eachcol(acq[1]),plan.dopplers)
            #CM-ABS algo 3 step 6 eqn 7
            #doppler1 = ustrip(plan.dopplers[idx1])
            doppler1 = ustrip(dop)
            if compensate_doppler_code == :negative
                Ns = sign(doppler1 - center_frequency) * round(round_idx*0.001* ustrip(plan.sampling_freq) * abs(doppler1 - center_frequency)/ustrip(get_center_frequency(plan.system)), RoundNearest)
                #power_bin_ncoh3[:,idx1] = power_bin_ncoh3[:,idx1] .+ circshift(acq.power_bins[:,idx1],Ns)
                acc .= acc .+ circshift(power_doppler,-Ns)
            elseif compensate_doppler_code == :positive
                Ns = sign(doppler1 - center_frequency) * round(round_idx*0.001* ustrip(plan.sampling_freq) * abs(doppler1 - center_frequency)/ustrip(get_center_frequency(plan.system)), RoundNearest)
                #power_bin_ncoh3[:,idx1] = power_bin_ncoh3[:,idx1] .+ circshift(acq.power_bins[:,idx1],Ns)
                acc .= acc .+ circshift(power_doppler,Ns)
            else
                acc .= acc .+ power_doppler
            end
        end

    end
    return power_bin_ncoh3  
  end

function acquire!(
    acq_plan::CoarseFineAcquisitionPlan,
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
)
    acq_res = acquire!(acq_plan.coarse_plan, signal, prns; interm_freq)
    map(acq_res, prns) do res, prn
        acquire!(
            acq_plan.fine_plan,
            signal,
            prn;
            interm_freq,
            doppler_offset = res.carrier_doppler,
            noise_power = res.noise_power,
        )
    end
end

"""
$(SIGNATURES)
Perform the aquisition of a single satellite `prn` in system `system` with signal `signal`
sampled at rate `sampling_freq`. Optional arguments are the intermediate frequency `interm_freq`
(default 0Hz), the maximum expected Doppler `max_doppler` (default 7000Hz). If the maximum Doppler
is too unspecific you can instead pass a Doppler range with with your individual step size using
the argument `dopplers`.
"""
function acquire(
    system::AbstractGNSS,
    signal,
    sampling_freq,
    prn::Integer;
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    dopplers = -max_doppler:1/3/(length(signal)/sampling_freq):max_doppler,
    noncoherent_rounds=1
)
    only(acquire(system, signal, sampling_freq, [prn]; interm_freq, dopplers, noncoherent_rounds))
end

function acquire!(
    acq_plan::AcquisitionPlan,
    signal,
    prn::Integer;
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
    noise_power = nothing,
)
    only(acquire!(acq_plan, signal, [prn]; interm_freq, doppler_offset, noise_power))
end

function acquire!(acq_plan::CoarseFineAcquisitionPlan, signal, prn::Integer; interm_freq = 0.0Hz)
    only(acquire!(acq_plan, signal, [prn]; interm_freq))
end

"""
$(SIGNATURES)
Performs a coarse aquisition and fine acquisition of multiple satellites `prns` in system `system` with signal `signal`
sampled at rate `sampling_freq`. The aquisition is performed as parallel code phase
search using the Doppler frequencies with coarse step size `coarse_step` and fine step size `fine_step`.
"""
function coarse_fine_acquire(
    system::AbstractGNSS,
    signal,
    sampling_freq,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    coarse_step = 1 / 3 / (length(signal) / sampling_freq),
    fine_step = 1 / 12 / (length(signal) / sampling_freq),
)
    acq_plan = CoarseFineAcquisitionPlan(
        system,
        length(signal),
        sampling_freq;
        max_doppler,
        coarse_step,
        fine_step,
        prns,
        fft_flag = FFTW.ESTIMATE,
    )
    acquire!(acq_plan, signal, prns; interm_freq)
end

"""
$(SIGNATURES)
Performs a coarse aquisition and fine acquisition of a single satellite with PRN `prn` in system `system` with signal `signal`
sampled at rate `sampling_freq`. The aquisition is performed as parallel code phase
search using the Doppler frequencies with coarse step size `coarse_step` and fine step size `fine_step`.
"""
function coarse_fine_acquire(
    system::AbstractGNSS,
    signal,
    sampling_freq,
    prn::Integer;
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    coarse_step = 1 / 3 / (length(signal) / sampling_freq),
    fine_step = 1 / 12 / (length(signal) / sampling_freq),
)
    only(
        coarse_fine_acquire(
            system,
            signal,
            sampling_freq,
            [prn];
            interm_freq,
            max_doppler,
            coarse_step,
            fine_step,
        ),
    )
end