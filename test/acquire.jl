@testset "Acquire signal $(get_system_string(system)) and signal type $type" for system in [GPSL1(), GalileoE1B()], type in [Float64, Float32, Int16, Int32]

    Random.seed!(2345)
    num_samples = 20000
    doppler = 1234Hz
    code_phase = 110.613261
    prn = 1
    sampling_freq = 5e6Hz - 1Hz # Allow num_samples * sampling_freq to be non integer of ms
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

    noise_power = 10 * log10(sampling_freq / 1.0Hz)
    signal_power = CN0
    noise = randn(ComplexF64, num_samples)
    signal = (carrier .* code) * 10^(signal_power / 20) + noise * 10^(noise_power / 20)
    if type <: Integer
        signal_typed = complex.(floor.(type, real.(signal)), floor.(type, imag.(signal)))
    else
        signal_typed = Complex{type}.(signal)
    end

    max_doppler = 7000Hz
    dopplers = -max_doppler:1/3/(length(signal)/sampling_freq):max_doppler

    acq_res = acquire(system, signal_typed, sampling_freq, prn; interm_freq, dopplers)

    acq_plan =
        AcquisitionPlan(system, length(signal_typed), sampling_freq; dopplers, prns = 1:34)

    inplace_acq_res = acquire!(acq_plan, signal_typed, prn; interm_freq)

    @test acq_res.code_phase ≈ code_phase atol = 0.08
    @test abs(acq_res.carrier_doppler - doppler) < step(dopplers) / 2
    @test acq_res.prn == prn
    @test acq_res.CN0 ≈ CN0 atol = 6

    @test inplace_acq_res.code_phase ≈ code_phase atol = 0.08
    @test abs(inplace_acq_res.carrier_doppler - doppler) < step(dopplers) / 2
    @test inplace_acq_res.prn == prn
    @test inplace_acq_res.CN0 ≈ CN0 atol = 6

    coarse_fine_acq_res =
        coarse_fine_acquire(system, signal_typed, sampling_freq, prn; interm_freq)

    coarse_fine_acq_plan =
        CoarseFineAcquisitionPlan(system, length(signal_typed), sampling_freq; prns = 1:34)

    inplace_inplace_acq_res = acquire!(coarse_fine_acq_plan, signal_typed, prn; interm_freq)

    @test coarse_fine_acq_res.code_phase ≈ code_phase atol = 0.08
    @test abs(coarse_fine_acq_res.carrier_doppler - doppler) < step(dopplers) / 2
    @test coarse_fine_acq_res.prn == prn
    @test coarse_fine_acq_res.CN0 ≈ CN0 atol = 6

    @test inplace_inplace_acq_res.code_phase ≈ code_phase atol = 0.08
    @test abs(inplace_inplace_acq_res.carrier_doppler - doppler) < step(dopplers) / 2
    @test inplace_inplace_acq_res.prn == prn
    @test inplace_inplace_acq_res.CN0 ≈ CN0 atol = 6
end