@testset "Power over code for $system" for system in [GPSL1(), GalileoE1B()]
    Random.seed!(2345)
    num_samples = 60000
    doppler = 1234Hz
    code_phase = 110.613261
    prn = 1
    sampling_freq = 15e6Hz - 1Hz # Allow num_samples * sampling_freq to be non integer of ms
    interm_freq = 0.0Hz
    CN0 = 45

    code = gen_code(
        num_samples,
        system,
        prn,
        sampling_freq,
        get_code_frequency(system) + doppler * get_code_center_frequency_ratio(system),
        code_phase,
    )

    carrier =
        cis.(2π * (0:num_samples-1) * (interm_freq + doppler) / sampling_freq .+ π / 8)

    noise_power = 1
    signal_power = CN0 - 10 * log10(sampling_freq / 1.0Hz)
    noise = randn(ComplexF64, num_samples)
    signal = (carrier .* code) * 10^(signal_power / 20) + noise * 10^(noise_power / 20)

    signal_powers = [Matrix{Float32}(undef, 5000, 1)]

    signal_baseband = Vector{ComplexF32}(undef, length(signal))
    signal_baseband_freq_domain = similar(signal_baseband)
    code_freq_baseband_freq_domain = similar(signal_baseband)
    code_freq_domain = similar(signal_baseband)
    code_baseband = similar(signal_baseband)
    fft_plan = plan_fft(signal_baseband)

    Acquisition.power_over_code!(
        system,
        [prn],
        signal_powers,
        1,
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

    maxval, maxidx = findmax(signal_powers[1][:, 1])

    @test (maxidx - 1) * get_code_frequency(system) / sampling_freq ≈ code_phase atol = 0.15
end

@testset "Power over code and Doppler for $system" for system in [GPSL1(), GalileoE1B()]
    Random.seed!(2345)
    num_samples = 60000
    doppler = 1234Hz
    code_phase = 110.613261
    prn = 1
    sampling_freq = 15e6Hz - 1Hz # Allow num_samples * sampling_freq to be non integer of ms
    interm_freq = 243.0Hz
    CN0 = 45

    code = gen_code(
        num_samples,
        system,
        prn,
        sampling_freq,
        get_code_frequency(system) + doppler * get_code_center_frequency_ratio(system),
        code_phase,
    )

    carrier =
        cis.(2π * (0:num_samples-1) * (interm_freq + doppler) / sampling_freq .+ π / 8)

    noise_power = 1
    signal_power = CN0 - 10 * log10(sampling_freq / 1.0Hz)
    noise = randn(ComplexF64, num_samples)
    signal = (carrier .* code) * 10^(signal_power / 20) + noise * 10^(noise_power / 20)

    max_doppler = 7000Hz
    dopplers = -max_doppler:250Hz:max_doppler

    acq_plan = AcquisitionPlan(system, length(signal), sampling_freq; dopplers, prns = 1:1)

    powers_per_sats =
        Acquisition.power_over_doppler_and_codes!(acq_plan, signal, [1], interm_freq, 0.0Hz)

    maxval, maxidx = findmax(powers_per_sats[1])
    @test (maxidx[1] - 1) * get_code_frequency(system) / sampling_freq ≈ code_phase atol =
        0.08

    est_doppler = (maxidx[2] - 1) * step(dopplers) + first(dopplers)
    @test abs(est_doppler - doppler) < step(dopplers) / 2
end