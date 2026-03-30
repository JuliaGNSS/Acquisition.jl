using KernelAbstractions

# Test with CPU backend (KernelAbstractions works on regular Arrays)
@testset "KAAcquisitionPlan CPU backend" begin
    @testset "Acquire signal $system" for system in [GPSL1(), GalileoE1B()]
        (; signal, doppler, code_phase, prn, sampling_freq, interm_freq, CN0, num_samples) =
            generate_test_signal(system, 1)
        signal_f32 = ComplexF32.(signal)
        coherent_samples = num_samples ÷ 2

        max_doppler = 7000Hz
        dopplers = (-max_doppler):250Hz:max_doppler

        # Create KAAcquisitionPlan with Array (CPU backend)
        ka_plan = KAAcquisitionPlan(
            system,
            coherent_samples,
            sampling_freq,
            Array;
            dopplers,
            prns = 1:34,
        )

        # Test single PRN acquisition
        ka_res = acquire!(ka_plan, signal_f32, prn; interm_freq)

        @test ka_res.code_phase ≈ code_phase atol = 0.08
        @test abs(ka_res.carrier_doppler - doppler) < step(dopplers) / 2
        @test ka_res.prn == prn
        @test ka_res.CN0 ≈ CN0 atol = 7

        # Test multiple PRN acquisition
        ka_res_multi = acquire!(ka_plan, signal_f32, [1, 2, 3]; interm_freq)
        @test length(ka_res_multi) == 3
        @test ka_res_multi[1].prn == 1
        @test ka_res_multi[2].prn == 2
        @test ka_res_multi[3].prn == 3

        # Compare with CPU AcquisitionPlan results
        cpu_plan =
            AcquisitionPlan(system, coherent_samples, sampling_freq; dopplers, prns = 1:34)
        cpu_res = acquire!(cpu_plan, signal_f32, prn; interm_freq)

        # Results should be very similar (not identical due to different FFT implementations)
        @test abs(ka_res.carrier_doppler - cpu_res.carrier_doppler) < step(dopplers)
        @test abs(ka_res.code_phase - cpu_res.code_phase) < 0.5
        @test abs(ka_res.CN0 - cpu_res.CN0) < 3
    end
end

@testset "KAAcquisitionPlan eltype parameter" begin
    system = GPSL1()
    num_samples = 10000
    sampling_freq = 5e6Hz

    # Test default eltype (Float32)
    plan_default = KAAcquisitionPlan(system, num_samples, sampling_freq, Array; prns = 1:1)
    @test eltype(plan_default.signal_baseband) == ComplexF32
    @test eltype(plan_default.codes_freq_domain) == ComplexF32

    # Test explicit Float32
    plan_f32 = KAAcquisitionPlan(
        system,
        num_samples,
        sampling_freq,
        Array;
        prns = 1:1,
        eltype = Float32,
    )
    @test eltype(plan_f32.signal_baseband) == ComplexF32

    # Test Float64
    plan_f64 = KAAcquisitionPlan(
        system,
        num_samples,
        sampling_freq,
        Array;
        prns = 1:1,
        eltype = Float64,
    )
    @test eltype(plan_f64.signal_baseband) == ComplexF64
    @test eltype(plan_f64.codes_freq_domain) == ComplexF64

    # Verify acquisition works with Float64 buffers
    Random.seed!(1234)
    signal = randn(ComplexF64, 2 * num_samples)
    result_f64 = acquire!(plan_f64, signal, 1)
    @test result_f64 isa AcquisitionResults
end

@testset "KAAcquisitionPlan PRN validation" begin
    system = GPSL1()
    num_samples = 10000
    sampling_freq = 5e6Hz

    plan = KAAcquisitionPlan(system, num_samples, sampling_freq, Array; prns = 1:10)

    signal = randn(ComplexF32, 2 * num_samples)

    # Valid PRNs should work
    @test length(acquire!(plan, signal, [1, 5, 10])) == 3

    # Invalid PRN should throw
    @test_throws ArgumentError acquire!(plan, signal, [11])
    @test_throws ArgumentError acquire!(plan, signal, [1, 15])
end

@testset "KAAcquisitionPlan empty PRNs" begin
    system = GPSL1()
    num_samples = 10000
    sampling_freq = 5e6Hz

    plan = KAAcquisitionPlan(system, num_samples, sampling_freq, Array; prns = 1:10)
    signal = randn(ComplexF32, 2 * num_samples)

    # Empty PRNs should return empty vector
    result = acquire!(plan, signal, Int[])
    @test isempty(result)
    @test result isa Vector{<:AcquisitionResults}
end

@testset "KAAcquisitionPlan with doppler_offset" begin
    system = GPSL1()
    (; signal, doppler, code_phase, prn, sampling_freq, num_samples) =
        generate_test_signal(
            system, 1;
            seed = 4567, code_phase = 50.0, sampling_freq = 15e6Hz,
            interm_freq = 0.0Hz, phase_offset = 0.0,
        )
    signal = ComplexF32.(signal)

    # Use a narrow Doppler range with offset
    base_doppler = 1000Hz
    plan = KAAcquisitionPlan(
        system,
        num_samples ÷ 2,
        sampling_freq,
        Array;
        min_doppler = -500Hz,
        max_doppler = 500Hz,
        prns = 1:10,
    )

    result = acquire!(plan, signal, prn; doppler_offset = base_doppler)

    @test result.code_phase ≈ code_phase atol = 0.1
    @test abs(result.carrier_doppler - doppler) < step(plan.dopplers) / 2 * 1.0Hz
end

@testset "KAAcquisitionPlan power_bins always empty (no store_powers)" begin
    system = GPSL1()
    sampling_freq = 4e6Hz
    num_samples = 4000
    prn = 1

    signal = randn(ComplexF32, 2 * num_samples)
    plan = KAAcquisitionPlan(system, num_samples, sampling_freq, Array; prns = 1:5)

    result = acquire!(plan, signal, prn)
    @test size(result.power_bins) == (0, 0)
end

@testset "KAAcquisitionPlan non-coherent integration" begin
    system = GPSL1()
    sampling_freq = 5e6Hz

    # Calculate bit period samples (20ms for GPS L1 at 50Hz data rate)
    bit_period_samples = ceil(Int, sampling_freq / get_data_frequency(system))

    # Create signal spanning 2.5 bit periods (50ms)
    num_samples = ceil(Int, 2.5 * bit_period_samples)

    (; signal, doppler, code_phase, prn, CN0) = generate_test_signal(
        system, 1;
        seed = 4567, num_samples, doppler = 500Hz, code_phase = 50.5,
        sampling_freq, interm_freq = 0.0Hz, phase_offset = 0.0,
    )
    signal_typed = ComplexF32.(signal)

    # Test with plan using default bit period chunk size (uses convenience constructor)
    ka_plan = KAAcquisitionPlan(system, sampling_freq, Array; prns = [prn])
    @test ka_plan.num_samples_to_integrate_coherently == bit_period_samples

    result = acquire!(ka_plan, signal_typed, prn)

    # Verify acquisition still finds the signal correctly
    @test result.code_phase ≈ code_phase atol = 0.08
    @test abs(result.carrier_doppler - doppler) < step(ka_plan.dopplers) / 2 * 1.0Hz
    @test result.prn == prn

    # CN0 includes coherent integration gain only (no explicit non-coherent gain)
    codes_per_chunk = 20  # 1 bit period = 20 code periods
    coherent_gain = 10 * log10(codes_per_chunk)
    expected_CN0 = CN0 + coherent_gain
    @test result.CN0 ≈ expected_CN0 atol = 5

    # Compare with CPU AcquisitionPlan - results should be similar
    cpu_plan =
        AcquisitionPlan(system, sampling_freq; prns = [prn], fft_flag = FFTW.ESTIMATE)
    cpu_result = acquire!(cpu_plan, signal_typed, prn)

    @test abs(result.carrier_doppler - cpu_result.carrier_doppler) < step(ka_plan.dopplers) / 2 * 1.0Hz
    @test abs(result.code_phase - cpu_result.code_phase) < 0.02
    @test abs(result.CN0 - cpu_result.CN0) < 3
end

@testset "KAAcquisitionPlan code Doppler compensation" begin
    system = GPSL1()
    sampling_freq = 5e6Hz

    T_coh_ms = 10
    coherent_samples = ceil(Int, T_coh_ms * 1e-3 * (sampling_freq / 1.0Hz))
    num_samples = 2 * coherent_samples

    (; signal, doppler, code_phase, prn) = generate_test_signal(
        system, 1;
        seed = 7890, num_samples, doppler = 5000Hz, code_phase = 50.5,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 55, phase_offset = 0.0,
    )
    signal_typed = ComplexF32.(signal)

    ka_plan = KAAcquisitionPlan(system, coherent_samples, sampling_freq, Array; prns = [prn])

    @test ka_plan.num_code_dopplers > 1

    result = acquire!(ka_plan, signal_typed, prn)

    @test result.code_phase ≈ code_phase atol = 0.08
    @test abs(result.carrier_doppler - doppler) < step(ka_plan.dopplers) / 2 * 1.0Hz
end

@testset "KAAcquisitionPlan zero_pad_power=0 (no padding)" begin
    system = GPSL1()
    (; signal, doppler, code_phase, prn, sampling_freq, interm_freq, CN0, num_samples) =
        generate_test_signal(system, 1)
    signal_f32 = ComplexF32.(signal)
    coherent_samples = num_samples ÷ 2

    ka_plan = KAAcquisitionPlan(
        system,
        coherent_samples,
        sampling_freq,
        Array;
        prns = 1:34,
        zero_pad_power = 0,
    )

    @test ka_plan.bfft_size == Acquisition.fftw_friendly_size(2 * coherent_samples)

    result = acquire!(ka_plan, signal_f32, prn; interm_freq)

    @test result.code_phase ≈ code_phase atol = 0.08
    @test abs(result.carrier_doppler - doppler) < step(ka_plan.dopplers) / 2 * 1.0Hz
    @test result.CN0 ≈ CN0 atol = 7
end

@testset "KAAcquisitionPlan zero_pad_power=1 (frequency-domain zero-padding)" begin
    system = GPSL1()
    coherent_samples = 16368
    (; signal, doppler, code_phase, prn, sampling_freq, num_samples, CN0) =
        generate_test_signal(
            system, 1;
            num_samples = 2 * coherent_samples, code_phase = 50.5, sampling_freq = 16.368e6Hz,
            CN0 = 50, interm_freq = 0.0Hz, phase_offset = 0.0,
        )
    signal_f32 = ComplexF32.(signal)

    ka_plan = KAAcquisitionPlan(
        system,
        coherent_samples,
        sampling_freq,
        Array;
        prns = 1:34,
        zero_pad_power = 1,
    )

    linear_fft_size = Acquisition.fftw_friendly_size(2 * coherent_samples)
    @test ka_plan.bfft_size == Acquisition.fftw_friendly_size(linear_fft_size << 1)
    @test ka_plan.bfft_size > ka_plan.linear_fft_size  # needs_padding == true

    result = acquire!(ka_plan, signal_f32, prn)

    @test result.code_phase ≈ code_phase atol = 0.001
    @test abs(result.carrier_doppler - doppler) < step(ka_plan.dopplers) / 2 * 1.0Hz
    @test result.CN0 ≈ CN0 atol = 7
end

@testset "KAAcquisitionPlan with zero-padding (non-smooth sample count)" begin
    system = GPSL1()
    # 16368 = 2^4 × 3 × 11 × 31 — not FFTW-friendly, pads to 16384
    coherent_samples = 16368
    (; signal, doppler, code_phase, prn, sampling_freq, num_samples, CN0) =
        generate_test_signal(
            system, 1;
            num_samples = 2 * coherent_samples, code_phase = 50.5, sampling_freq = 16.368e6Hz,
            interm_freq = 0.0Hz, phase_offset = 0.0,
        )
    signal_f32 = ComplexF32.(signal)

    ka_plan = KAAcquisitionPlan(
        system,
        coherent_samples,
        sampling_freq,
        Array;
        prns = 1:34,
    )

    @test ka_plan.bfft_size > coherent_samples

    result = acquire!(ka_plan, signal_f32, prn)

    @test result.code_phase ≈ code_phase atol = 0.08
    @test abs(result.carrier_doppler - doppler) < step(ka_plan.dopplers) / 2 * 1.0Hz
    @test result.CN0 ≈ CN0 atol = 7
end

@testset "KAAcquisitionPlan max_gpu_memory=nothing (no batching)" begin
    system = GPSL1()
    (; signal, doppler, code_phase, prn, sampling_freq, interm_freq, num_samples) =
        generate_test_signal(system, 1)
    signal_f32 = ComplexF32.(signal)
    coherent_samples = num_samples ÷ 2

    ka_plan = KAAcquisitionPlan(
        system,
        coherent_samples,
        sampling_freq,
        Array;
        prns = 1:34,
        max_gpu_memory = nothing,
    )

    # With max_gpu_memory=nothing, batch_size should equal num_dopplers (no batching)
    @test ka_plan.doppler_batch_size == length(ka_plan.dopplers)

    result = acquire!(ka_plan, signal_f32, prn; interm_freq)
    @test result.code_phase ≈ code_phase atol = 0.08
    @test abs(result.carrier_doppler - doppler) < step(ka_plan.dopplers) / 2 * 1.0Hz
end

@testset "KAAcquisitionPlan with fft_flag=FFTW.ESTIMATE" begin
    system = GPSL1()
    (; signal, doppler, code_phase, prn, sampling_freq, interm_freq, num_samples) =
        generate_test_signal(system, 1)
    signal_f32 = ComplexF32.(signal)
    coherent_samples = num_samples ÷ 2

    ka_plan = KAAcquisitionPlan(
        system,
        coherent_samples,
        sampling_freq,
        Array;
        prns = 1:34,
        fft_flag = FFTW.ESTIMATE,
    )

    result = acquire!(ka_plan, signal_f32, prn; interm_freq)
    @test result.code_phase ≈ code_phase atol = 0.08
    @test abs(result.carrier_doppler - doppler) < step(ka_plan.dopplers) / 2 * 1.0Hz
end
