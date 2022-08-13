"""
$(SIGNATURES)
Perform the aquisition of the satellite `sat_prn` in system `S` with signal `signal`
sampled at rate `sampling_freq`. The aquisition is performed as parallel code phase
search using the doppler frequencies `dopplers`.
"""
function acquire(
    S::AbstractGNSS,
    signal,
    sampling_freq,
    sat_prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    dopplers = -max_doppler:1 / 3 / (length(signal) / sampling_freq):max_doppler
)
    code_period = get_code_length(S) / get_code_frequency(S)
    powers_per_sats = power_over_doppler_and_codes(S, signal, sat_prns, dopplers, sampling_freq, interm_freq)  
    map(powers_per_sats, sat_prns) do powers, sat_prn
        signal_power, noise_power, code_index, doppler_index = est_signal_noise_power(powers, sampling_freq, get_code_frequency(S))
        CN0 = 10 * log10(signal_power / noise_power / code_period / 1.0Hz)
        doppler = (doppler_index - 1) * step(dopplers) + first(dopplers)
        code_phase = (code_index - 1) / (sampling_freq / get_code_frequency(S))
        AcquisitionResults(S, sat_prn, sampling_freq, doppler, code_phase, CN0, powers, dopplers / 1.0Hz)
    end
end

"""
$(SIGNATURES)
Perform the aquisition of the satellite `sat_prn` in system `S` with signal `signal`
sampled at rate `sampling_freq`. The aquisition is performed as parallel code phase
search using the doppler frequencies `dopplers`.
"""
function acquire(
    S::AbstractGNSS,
    signal,
    sampling_freq,
    sat_prn::Integer;
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    dopplers = -max_doppler:1 / 3 / (length(signal) / sampling_freq):max_doppler
)
    only(acquire(S, signal, sampling_freq, [sat_prn]; interm_freq, dopplers))
end

"""
$(SIGNATURES)
Performs a coarse aquisition and fine acquisition of multiple satellites `sat_prns` in system `S` with signal `signal`
sampled at rate `sampling_freq`. The aquisition is performed as parallel code phase
search using the doppler frequencies with coarse step size `coarse_step` and fine step size `fine_step`.
"""
function coarse_fine_acquire(
    S::AbstractGNSS,
    signal,
    sampling_freq,
    sat_prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    coarse_step = 1 / 3 / (length(signal) / sampling_freq),
    fine_step = 1 / 12 / (length(signal) / sampling_freq)
)
    coarse_dopplers = -max_doppler:coarse_step:max_doppler
    acq_res = acquire(S, signal, sampling_freq, sat_prns; interm_freq, dopplers = coarse_dopplers)
    map(acq_res, sat_prns) do res, sat_prn
        fine_dopplers = res.carrier_doppler - 2 * coarse_step:fine_step:res.carrier_doppler + 2 * coarse_step
        acquire(S, signal, sampling_freq, sat_prn; interm_freq, dopplers = fine_dopplers)
    end
end

"""
$(SIGNATURES)
Performs a coarse aquisition and fine acquisition of the satellite `sat_prn` in system `S` with signal `signal`
sampled at rate `sampling_freq`. The aquisition is performed as parallel code phase
search using the doppler frequencies with coarse step size `coarse_step` and fine step size `fine_step`.
"""
function coarse_fine_acquire(
    S::AbstractGNSS,
    signal,
    sampling_freq,
    sat_prn::Integer;
    interm_freq = 0.0Hz,
    max_doppler = 7000Hz,
    coarse_step = 1 / 3 / (length(signal) / sampling_freq),
    fine_step = 1 / 12 / (length(signal) / sampling_freq)
)
    only(coarse_fine_acquire(
        S,
        signal,
        sampling_freq,
        [sat_prn];
        interm_freq,
        max_doppler,
        coarse_step,
        fine_step
    ))
end