@testset "Acquire signal $system and signal type $type" for system in
                                                            [GPSL1(), GalileoE1B()],
    type in [Float64, Float32, Int16, Int32]

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
    dopplers = -max_doppler:250Hz:max_doppler

    acq_res =
        @inferred acquire(system, signal_typed, sampling_freq, prn; interm_freq, dopplers)

    acq_plan = @inferred AcquisitionPlan(
        system,
        length(signal_typed),
        sampling_freq;
        dopplers,
        prns = 1:34,
    )

    inplace_acq_res = @inferred acquire!(acq_plan, signal_typed, prn; interm_freq)

    @test acq_res.code_phase ≈ code_phase atol = 0.08
    @test abs(acq_res.carrier_doppler - doppler) < step(dopplers) / 2
    @test acq_res.prn == prn
    @test acq_res.CN0 ≈ CN0 atol = 7

    @test inplace_acq_res.code_phase ≈ code_phase atol = 0.08
    @test abs(inplace_acq_res.carrier_doppler - doppler) < step(dopplers) / 2
    @test inplace_acq_res.prn == prn
    @test inplace_acq_res.CN0 ≈ CN0 atol = 7

    coarse_fine_acq_res =
        @inferred coarse_fine_acquire(system, signal_typed, sampling_freq, prn; interm_freq)

    coarse_fine_acq_plan = @inferred CoarseFineAcquisitionPlan(
        system,
        length(signal_typed),
        sampling_freq;
        prns = 1:34,
    )

    inplace_inplace_acq_res =
        @inferred acquire!(coarse_fine_acq_plan, signal_typed, prn; interm_freq)

    @test coarse_fine_acq_res.code_phase ≈ code_phase atol = 0.08
    @test abs(coarse_fine_acq_res.carrier_doppler - doppler) < step(dopplers) / 2
    @test coarse_fine_acq_res.prn == prn
    @test coarse_fine_acq_res.CN0 ≈ CN0 atol = 7

    @test inplace_inplace_acq_res.code_phase ≈ code_phase atol = 0.08
    @test abs(inplace_inplace_acq_res.carrier_doppler - doppler) < step(dopplers) / 2
    @test inplace_inplace_acq_res.prn == prn
    @test inplace_inplace_acq_res.CN0 ≈ CN0 atol = 7
end

@testset "AcquisitionPlan eltype parameter" begin
    system = GPSL1()
    num_samples = 10000
    sampling_freq = 5e6Hz

    # Test default eltype (Float32)
    plan_default = AcquisitionPlan(system, num_samples, sampling_freq; prns = 1:1)
    @test eltype(plan_default.signal_baseband) == ComplexF32
    @test eltype(plan_default.signal_baseband_freq_domain) == ComplexF32

    # Test explicit Float32
    plan_f32 = AcquisitionPlan(system, num_samples, sampling_freq; prns = 1:1, eltype = Float32)
    @test eltype(plan_f32.signal_baseband) == ComplexF32
    @test eltype(plan_f32.signal_baseband_freq_domain) == ComplexF32

    # Test Float64
    plan_f64 = AcquisitionPlan(system, num_samples, sampling_freq; prns = 1:1, eltype = Float64)
    @test eltype(plan_f64.signal_baseband) == ComplexF64
    @test eltype(plan_f64.signal_baseband_freq_domain) == ComplexF64

    # Test CoarseFineAcquisitionPlan with eltype
    cf_plan_f64 = CoarseFineAcquisitionPlan(system, num_samples, sampling_freq; prns = 1:1, eltype = Float64)
    @test eltype(cf_plan_f64.coarse_plan.signal_baseband) == ComplexF64
    @test eltype(cf_plan_f64.fine_plan.signal_baseband) == ComplexF64

    # Verify acquisition still works correctly with Float64 buffers
    Random.seed!(1234)
    signal = randn(ComplexF64, num_samples)
    result_f64 = acquire!(plan_f64, signal, 1)
    @test result_f64 isa AcquisitionResults
end

@testset "Acquire with asymmetric Doppler range" begin
    Random.seed!(2345)
    system = GPSL1()
    num_samples = 60000
    doppler = 1234Hz
    code_phase = 110.613261
    prn = 1
    sampling_freq = 15e6Hz - 1Hz
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
    signal_typed = Complex{Float64}.(signal)

    # Test asymmetric Doppler range: 0Hz to 7000Hz (positive only)
    min_doppler = 0Hz
    max_doppler = 7000Hz
    dopplers = min_doppler:250Hz:max_doppler

    # Test acquire with min_doppler
    acq_res = @inferred acquire(
        system,
        signal_typed,
        sampling_freq,
        prn;
        interm_freq,
        min_doppler,
        max_doppler,
    )

    @test acq_res.code_phase ≈ code_phase atol = 0.08
    @test abs(acq_res.carrier_doppler - doppler) < step(dopplers) / 2
    @test acq_res.prn == prn
    @test acq_res.CN0 ≈ CN0 atol = 7

    # Test AcquisitionPlan with min_doppler
    acq_plan = @inferred AcquisitionPlan(
        system,
        length(signal_typed),
        sampling_freq;
        min_doppler,
        max_doppler,
        prns = 1:34,
    )

    @test first(acq_plan.dopplers) ≈ min_doppler
    @test last(acq_plan.dopplers) >= max_doppler - step(acq_plan.dopplers)

    inplace_acq_res = @inferred acquire!(acq_plan, signal_typed, prn; interm_freq)

    @test inplace_acq_res.code_phase ≈ code_phase atol = 0.08
    @test abs(inplace_acq_res.carrier_doppler - doppler) < step(dopplers) / 2
    @test inplace_acq_res.prn == prn
    @test inplace_acq_res.CN0 ≈ CN0 atol = 7

    # Test coarse_fine_acquire with min_doppler
    coarse_fine_acq_res = @inferred coarse_fine_acquire(
        system,
        signal_typed,
        sampling_freq,
        prn;
        interm_freq,
        min_doppler,
        max_doppler,
    )

    @test coarse_fine_acq_res.code_phase ≈ code_phase atol = 0.08
    @test abs(coarse_fine_acq_res.carrier_doppler - doppler) < step(dopplers) / 2
    @test coarse_fine_acq_res.prn == prn
    @test coarse_fine_acq_res.CN0 ≈ CN0 atol = 7

    # Test CoarseFineAcquisitionPlan with min_doppler
    coarse_fine_acq_plan = @inferred CoarseFineAcquisitionPlan(
        system,
        length(signal_typed),
        sampling_freq;
        min_doppler,
        max_doppler,
        prns = 1:34,
    )

    @test first(coarse_fine_acq_plan.coarse_plan.dopplers) ≈ min_doppler
    @test last(coarse_fine_acq_plan.coarse_plan.dopplers) >=
          max_doppler - step(coarse_fine_acq_plan.coarse_plan.dopplers)

    inplace_coarse_fine_acq_res =
        @inferred acquire!(coarse_fine_acq_plan, signal_typed, prn; interm_freq)

    @test inplace_coarse_fine_acq_res.code_phase ≈ code_phase atol = 0.08
    @test abs(inplace_coarse_fine_acq_res.carrier_doppler - doppler) < step(dopplers) / 2
    @test inplace_coarse_fine_acq_res.prn == prn
    @test inplace_coarse_fine_acq_res.CN0 ≈ CN0 atol = 7
end

@testset "Base.show for AcquisitionResults" begin
    Random.seed!(1234)
    system = GPSL1()
    num_samples = 10000
    sampling_freq = 5e6Hz
    signal = randn(ComplexF64, num_samples)

    plan = AcquisitionPlan(system, num_samples, sampling_freq; prns = 1:3)

    # Test show for single AcquisitionResults
    result = acquire!(plan, signal, 1)
    io = IOBuffer()
    show(io, MIME"text/plain"(), result)
    output = String(take!(io))
    @test contains(output, "AcquisitionResults")
    @test contains(output, "PRN 1")
    @test contains(output, "CN0")
    @test contains(output, "dB-Hz")
    @test contains(output, "Doppler")
    @test contains(output, "Code phase")
    @test contains(output, "chips")

    # Test show for Vector{AcquisitionResults}
    results = [acquire!(plan, signal, prn) for prn in 1:3]
    io = IOBuffer()
    show(io, MIME"text/plain"(), results)
    output = String(take!(io))
    @test contains(output, "PRN")
    @test contains(output, "CN0")
    @test contains(output, "Carrier Doppler")
    @test contains(output, "Code phase")

    # Test CN0 highlighting (green for CN0 > 42, red for CN0 < 42)
    high_cn0_result = Acquisition.AcquisitionResults(
        system, 1, sampling_freq, 0.0Hz, 0.0, 50.0, 0.0, zeros(1, 1), 0.0Hz:1.0Hz:0.0Hz
    )
    low_cn0_result = Acquisition.AcquisitionResults(
        system, 2, sampling_freq, 0.0Hz, 0.0, 30.0, 0.0, zeros(1, 1), 0.0Hz:1.0Hz:0.0Hz
    )
    io = IOBuffer()
    ioc = IOContext(io, :color => true)
    show(ioc, MIME"text/plain"(), [high_cn0_result, low_cn0_result])
    output = String(take!(io))
    # Verify green is applied to the high CN0 value with reset after
    @test contains(output, "\e[32m50.0\e[0m")  # Green for CN0 > 42
    # Verify red is applied to the low CN0 value with reset after
    @test contains(output, "\e[31m30.0\e[0m")  # Red for CN0 < 42
end

@testset "Non-allocating result handling in acquire!" begin
    Random.seed!(2345)
    system = GPSL1()
    num_samples = 60000
    sampling_freq = 15e6Hz - 1Hz
    # Use ComplexF32 signal to match default Float32 buffers (avoids type conversion allocations)
    signal = randn(ComplexF32, num_samples)

    acq_plan = AcquisitionPlan(
        system,
        num_samples,
        sampling_freq;
        prns = 1:5,
    )

    # Verify plan.results field exists and is populated
    @test hasfield(typeof(acq_plan), :results)
    @test length(acq_plan.results) == 5
    @test hasfield(typeof(acq_plan), :prn_indices)

    # Pre-allocate PRN vector to avoid measuring its allocation
    prns = [1, 2]

    # Warmup to trigger compilation
    acquire!(acq_plan, signal, prns)
    acquire!(acq_plan, signal, prns)

    # Measure allocations - should be zero when types match
    allocs = @allocated acquire!(acq_plan, signal, prns)
    @test allocs == 0

    # Verify results are returned as a Vector (non-allocating via resize!)
    results = acquire!(acq_plan, signal, [1, 3])
    @test results isa Vector
    @test length(results) == 2
    @test results[1].prn == 1
    @test results[2].prn == 3

    # Verify results point to the same power_bins as signal_powers
    @test results[1].power_bins === acq_plan.signal_powers[1]
    @test results[2].power_bins === acq_plan.signal_powers[3]

    # Test CoarseFineAcquisitionPlan is also allocation-free
    cf_plan = CoarseFineAcquisitionPlan(
        system,
        num_samples,
        sampling_freq;
        prns = 1:5,
    )

    # Warmup
    acquire!(cf_plan, signal, prns)
    acquire!(cf_plan, signal, prns)

    # Measure allocations
    cf_allocs = @allocated acquire!(cf_plan, signal, prns)
    @test cf_allocs == 0

    # Verify CoarseFine results are also returned as a Vector
    cf_results = acquire!(cf_plan, signal, [1, 3])
    @test cf_results isa Vector
    @test cf_results[1].power_bins === cf_plan.fine_plan.signal_powers[1]
end

@testset "Non-coherent integration for signals longer than bit period" begin
    Random.seed!(4567)
    system = GPSL1()
    sampling_freq = 5e6Hz
    doppler = 500Hz
    code_phase = 50.5
    prn = 1
    CN0 = 45

    # Calculate bit period samples (20ms for GPS L1 at 50Hz data rate)
    bit_period_samples = ceil(Int, sampling_freq / get_data_frequency(system))

    # Create signal spanning 2.5 bit periods (50ms)
    num_samples = ceil(Int, 2.5 * bit_period_samples)

    code = gen_code(
        num_samples,
        system,
        prn,
        sampling_freq,
        get_code_frequency(system) + doppler * get_code_center_frequency_ratio(system),
        code_phase,
    )

    carrier = cis.(2π * (0:num_samples-1) * doppler / sampling_freq)

    noise_power = 10 * log10(sampling_freq / 1.0Hz)
    signal_power = CN0
    noise = randn(ComplexF64, num_samples)
    signal = (carrier .* code) * 10^(signal_power / 20) + noise * 10^(noise_power / 20)
    signal_typed = ComplexF32.(signal)

    # Test with plan using default bit period chunk size
    acq_plan = AcquisitionPlan(system, sampling_freq; prns = [prn], fft_flag = FFTW.ESTIMATE)
    @test acq_plan.num_samples_to_integrate_coherently == bit_period_samples

    result = acquire!(acq_plan, signal_typed, prn)

    # Verify acquisition still finds the signal correctly
    @test result.code_phase ≈ code_phase atol = 0.5
    @test abs(result.carrier_doppler - doppler) < 250Hz  # Within one Doppler bin
    @test result.prn == prn
    # CN0 estimate includes coherent integration gain only:
    # - Coherent gain from 1 bit period (20 code periods): 10·log10(20) ≈ 13 dB
    # Non-coherent integration improves detection probability but doesn't increase
    # the measured SNR ratio, so no explicit gain is added to avoid false detections.
    # Expected CN0 ≈ input (45) + coherent (13) ≈ 58 dBHz
    codes_per_chunk = 20  # 1 bit period = 20 code periods
    coherent_gain = 10 * log10(codes_per_chunk)
    expected_CN0 = CN0 + coherent_gain
    @test result.CN0 ≈ expected_CN0 atol = 5

    # Test CoarseFineAcquisitionPlan with long signal
    cf_plan = CoarseFineAcquisitionPlan(system, sampling_freq; prns = [prn], fft_flag = FFTW.ESTIMATE)
    cf_result = acquire!(cf_plan, signal_typed, prn)

    @test cf_result.code_phase ≈ code_phase atol = 0.5
    @test abs(cf_result.carrier_doppler - doppler) < 50Hz  # Fine search should be more accurate
    @test cf_result.prn == prn
    @test cf_result.CN0 ≈ expected_CN0 atol = 5
end

@testset "Non-coherent integration is allocation-free" begin
    Random.seed!(5678)
    system = GPSL1()
    sampling_freq = 5e6Hz

    # Calculate bit period samples
    bit_period_samples = ceil(Int, sampling_freq / get_data_frequency(system))

    # Create plan with bit period chunk size
    acq_plan = AcquisitionPlan(system, sampling_freq; prns = 1:5, fft_flag = FFTW.ESTIMATE)

    # Create signal spanning multiple bit periods (2 chunks)
    num_samples = ceil(Int, 1.5 * bit_period_samples)
    signal = randn(ComplexF32, num_samples)

    prns = [1, 2]

    # Warmup
    acquire!(acq_plan, signal, prns)
    acquire!(acq_plan, signal, prns)

    # Measure allocations - should be zero even with chunking
    allocs = @allocated acquire!(acq_plan, signal, prns)
    @test allocs == 0

    # Also test CoarseFineAcquisitionPlan
    cf_plan = CoarseFineAcquisitionPlan(system, sampling_freq; prns = 1:5, fft_flag = FFTW.ESTIMATE)

    # Warmup
    acquire!(cf_plan, signal, prns)
    acquire!(cf_plan, signal, prns)

    cf_allocs = @allocated acquire!(cf_plan, signal, prns)
    @test cf_allocs == 0
end

@testset "Code Doppler compensation with long coherent integration" begin
    Random.seed!(7890)
    system = GPSL1()
    sampling_freq = 5e6Hz
    doppler = 5000Hz
    code_phase = 50.5
    prn = 1
    CN0 = 55  # High CN0 for clean detection

    # Long coherent integration: 10ms (10x the default 1ms code period)
    T_coh_ms = 10
    num_samples = ceil(Int, T_coh_ms * 1e-3 * (sampling_freq / 1.0Hz))

    # Generate signal with realistic code Doppler
    code_doppler = doppler * get_code_center_frequency_ratio(system)
    code = gen_code(
        num_samples,
        system,
        prn,
        sampling_freq,
        get_code_frequency(system) + code_doppler,
        code_phase,
    )

    carrier = cis.(2π * (0:num_samples-1) * doppler / sampling_freq)

    noise_power = 10 * log10(sampling_freq / 1.0Hz)
    signal_power = CN0
    noise = randn(ComplexF64, num_samples)
    signal = (carrier .* code) * 10^(signal_power / 20) + noise * 10^(noise_power / 20)
    signal_typed = ComplexF32.(signal)

    # Acquire with code Doppler compensation (default tolerance)
    acq_plan = AcquisitionPlan(
        system,
        num_samples,
        sampling_freq;
        prns = [prn],
        fft_flag = FFTW.ESTIMATE,
    )

    # Verify multiple code Doppler groups were created
    @test acq_plan.num_code_dopplers > 1

    result = acquire!(acq_plan, signal_typed, prn)

    @test result.code_phase ≈ code_phase atol = 0.5
    @test abs(result.carrier_doppler - doppler) < step(acq_plan.dopplers)

    # Also test via coarse_fine_acquire
    cf_plan = CoarseFineAcquisitionPlan(
        system,
        num_samples,
        sampling_freq;
        prns = [prn],
        fft_flag = FFTW.ESTIMATE,
    )
    cf_result = acquire!(cf_plan, signal_typed, prn)

    @test cf_result.code_phase ≈ code_phase atol = 0.5
    @test abs(cf_result.carrier_doppler - doppler) < step(cf_plan.coarse_plan.dopplers)
end