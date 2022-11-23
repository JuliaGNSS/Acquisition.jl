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
    dopplers = -max_doppler:1/3/(length(signal)/sampling_freq):max_doppler,
)
    acq_plan = AcquisitionPlan(
        system,
        length(signal),
        sampling_freq;
        dopplers,
        prns,
        fft_flag = FFTW.ESTIMATE,
    )
    acquire!(acq_plan, signal, prns; interm_freq)
end

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
    powers_per_sats =
        power_over_doppler_and_codes!(acq_plan, signal, prns, interm_freq, doppler_offset)
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
)
    only(acquire(system, signal, sampling_freq, [prn]; interm_freq, dopplers))
end

function acquire!(
    acq_plan::AcquisitionPlan,
    signal,
    prn;
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
    noise_power = nothing,
)
    only(acquire!(acq_plan, signal, [prn]; interm_freq, doppler_offset, noise_power))
end

function acquire!(acq_plan::CoarseFineAcquisitionPlan, signal, prn; interm_freq = 0.0Hz)
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