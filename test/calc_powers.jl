@testset "Power over code for $system" for system in [GPSL1(), GalileoE1B()]
    (; signal, doppler, code_phase, prn, sampling_freq, interm_freq, num_samples) =
        generate_test_signal(
            system, 1; interm_freq = 0.0Hz, unit_noise_power = true,
        )

    code_freq_domain = fft(
        get_code.(
            system,
            (0:(num_samples-1)) .* get_code_frequency(system) ./ sampling_freq,
            prn,
        ),
    )
    # Nested format: codes_freq_domain[prn][code_doppler_idx]
    codes_freq_domain = [[code_freq_domain]]
    signal_powers = [Matrix{Float32}(undef, 5000, 1)]

    signal_baseband = Vector{ComplexF32}(undef, length(signal))
    signal_baseband_freq_domain = similar(signal_baseband)
    code_freq_baseband_freq_domain = similar(signal_baseband)
    code_baseband = similar(signal_baseband)
    fft_plan = plan_fft(signal_baseband)
    bfft_plan = plan_bfft(signal_baseband)

    @inferred Acquisition.power_over_code!(
        signal_powers,
        1,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        signal,
        fft_plan,
        bfft_plan,
        codes_freq_domain,
        1,
        doppler,
        sampling_freq,
        interm_freq,
    )

    maxval, maxidx = findmax(signal_powers[1][:, 1])

    @test (maxidx - 1) * get_code_frequency(system) / sampling_freq ≈ code_phase atol = 0.15
end

@testset "Power over code with zero-padding for $system" for system in [GPSL1(), GalileoE1B()]
    # 16368 = 2^4 × 3 × 11 × 31 — not FFTW-friendly, pads to 16384
    (; signal, doppler, code_phase, prn, sampling_freq, interm_freq, num_samples) =
        generate_test_signal(
            system, 1;
            num_samples = 16368, code_phase = 50.5, sampling_freq = 16.368e6Hz,
            interm_freq = 0.0Hz, unit_noise_power = true,
        )

    bfft_size = Acquisition.fftw_friendly_size(num_samples)
    @test bfft_size > num_samples  # Verify padding actually occurs

    code_freq_domain = fft(
        get_code.(
            system,
            (0:(num_samples-1)) .* get_code_frequency(system) ./ sampling_freq,
            prn,
        ),
    )
    codes_freq_domain = [[code_freq_domain]]
    effective_sampling_freq = sampling_freq * bfft_size / num_samples
    code_interval = get_code_length(system) / get_code_frequency(system)
    num_code_samples =
        ceil(Int, effective_sampling_freq * min(num_samples / sampling_freq, code_interval))
    signal_powers = [Matrix{Float32}(undef, num_code_samples, 1)]

    signal_baseband = Vector{ComplexF32}(undef, num_samples)
    signal_baseband_freq_domain = similar(signal_baseband)
    code_freq_baseband_freq_domain = Vector{ComplexF32}(undef, bfft_size)
    code_baseband = similar(code_freq_baseband_freq_domain)
    fft_plan = plan_fft(signal_baseband)
    bfft_plan = plan_bfft(code_freq_baseband_freq_domain)

    Acquisition.power_over_code!(
        signal_powers,
        1,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        signal,
        fft_plan,
        bfft_plan,
        codes_freq_domain,
        1,
        doppler,
        sampling_freq,
        interm_freq,
    )

    maxval, maxidx = findmax(signal_powers[1][:, 1])

    @test (maxidx - 1) * get_code_frequency(system) / effective_sampling_freq ≈
          code_phase atol = 0.15
end

@testset "Power over code and Doppler for $system" for system in [GPSL1(), GalileoE1B()]
    (; signal, doppler, code_phase, prn, sampling_freq, interm_freq, num_samples) =
        generate_test_signal(system, 1; unit_noise_power = true)

    max_doppler = 7000Hz
    dopplers = (-max_doppler):250Hz:max_doppler

    acq_plan = AcquisitionPlan(system, length(signal), sampling_freq; dopplers, prns = 1:1)
    effective_sampling_freq = sampling_freq * acq_plan.bfft_size / acq_plan.linear_fft_size

    powers_per_sats = @inferred Acquisition.power_over_doppler_and_codes!(
        acq_plan,
        signal,
        [1],
        interm_freq,
        0.0Hz,
    )

    maxval, maxidx = findmax(powers_per_sats[1])
    # Max quantization error is code_phase_step/2 ≈ 0.031 chips for these parameters
    @test (maxidx[1] - 1) * get_code_frequency(system) / effective_sampling_freq ≈
          code_phase atol = 0.04

    est_doppler = (maxidx[2] - 1) * step(dopplers) + first(dopplers)
    @test abs(est_doppler - doppler) < step(dopplers) / 2
end
