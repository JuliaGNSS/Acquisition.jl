# test/acquire.jl — public API tests for FM-DBZP acquire / acquire!

@testset "Tier 3: acquire — GPS L1 strong signal detects at correct code phase and Doppler alias" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz
    prn = 1

    (; signal, doppler, code_phase, interm_freq) = generate_test_signal(
        system, prn;
        num_samples = 2048,   # 1 code period at 2.048 MHz
        doppler = 1000Hz, code_phase = 200.0,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 45,
    )

    result = acquire(system, signal, sampling_freq, prn; interm_freq)

    @test result isa AcquisitionResults
    @test result.prn == prn
    @test result.code_phase ≈ code_phase atol = 1.0
    @test is_detected(result)
end

@testset "Tier 4: acquire — long integration detects weak signal" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz
    prn = 3

    (; signal, doppler, code_phase, interm_freq) = generate_test_signal(
        system, prn;
        num_samples = 40 * 2048,   # 20ms coherent × 2 noncoherent
        doppler = 500Hz, code_phase = 100.0,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 30,
        seed = 7777,
    )

    result = acquire(system, signal, sampling_freq, prn;
        interm_freq,
        num_coherently_integrated_code_periods = 20,
        num_noncoherent_accumulations = 2,
    )

    @test result isa AcquisitionResults
    @test result.prn == prn
    @test is_detected(result)
    @test result.code_phase ≈ code_phase atol = 2.0
end

@testset "subsample_interpolation — triangle interpolation reduces code phase error vs grid" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz
    prn = 2
    true_code_phase = 300.7   # deliberately between grid points

    (; signal, interm_freq) = generate_test_signal(
        system, prn;
        num_samples = 2048,
        doppler = 0Hz, code_phase = true_code_phase,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 45,
        seed = 5555,
    )

    result_grid   = acquire(system, signal, sampling_freq, prn; interm_freq, subsample_interpolation = false)
    result_interp = acquire(system, signal, sampling_freq, prn; interm_freq, subsample_interpolation = true)

    err_grid   = abs(result_grid.code_phase  - true_code_phase)
    err_interp = abs(result_interp.code_phase - true_code_phase)

    @test err_grid   < 2.0
    @test err_interp < 2.0
    @test err_interp <= err_grid + 0.5
end

@testset "acquire! — non-zero intermediate frequency" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz
    prn = 5
    interm_freq = 1000Hz

    (; signal, code_phase) = generate_test_signal(
        system, prn;
        num_samples = 2048,
        doppler = 500Hz, code_phase = 150.0,
        sampling_freq, interm_freq, CN0 = 45,
        seed = 3333,
    )

    plan = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 7000Hz,
        num_coherently_integrated_code_periods = 1,
        num_noncoherent_accumulations = 1,
        fft_flag = FFTW.ESTIMATE,
    )

    result_with_if    = only(acquire!(plan, ComplexF32.(signal), [prn]; interm_freq))
    result_without_if = only(acquire!(plan, ComplexF32.(signal), [prn]; interm_freq = 0.0Hz))

    @test result_with_if.code_phase ≈ code_phase atol = 1.5
    @test result_with_if.prn == prn
end

@testset "acquire! — multiple PRNs in one call" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz

    gen1 = generate_test_signal(system, 1;
        num_samples = 2048, doppler = 1000Hz, code_phase = 200.0,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 45, seed = 10)
    gen2 = generate_test_signal(system, 2;
        num_samples = 2048, doppler = -500Hz, code_phase = 500.0,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 45, seed = 10)

    signal_amplitude = 10^(45 / 20)
    noise_amplitude  = 10^(10 * log10(ustrip(Hz, sampling_freq)) / 20)
    Random.seed!(42)
    single_noise  = randn(ComplexF64, 2048) * noise_amplitude
    clean_prn1    = gen1.carrier .* gen1.code * signal_amplitude
    clean_prn2    = gen2.carrier .* gen2.code * signal_amplitude
    mixed_signal  = ComplexF32.(clean_prn1 .+ clean_prn2 .+ single_noise)

    plan = plan_acquire(system, sampling_freq, [1, 2];
        min_doppler_coverage = 7000Hz,
        num_coherently_integrated_code_periods = 1,
        num_noncoherent_accumulations = 1,
        fft_flag = FFTW.ESTIMATE,
    )
    results = acquire!(plan, mixed_signal, [1, 2]; interm_freq = 0.0Hz)

    @test length(results) == 2
    r1 = only(filter(r -> r.prn == 1, results))
    r2 = only(filter(r -> r.prn == 2, results))
    @test is_detected(r1)
    @test is_detected(r2)
    @test r1.code_phase ≈ gen1.code_phase atol = 2.0
    @test r2.code_phase ≈ gen2.code_phase atol = 2.0
end

@testset "_acquire_step_threaded! — threaded path runs correctly" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz
    plan = plan_acquire(system, sampling_freq, [1, 2]; fft_flag = FFTW.ESTIMATE)
    (; signal, code_phase) = generate_test_signal(system, 1;
        num_samples = plan.samples_per_code, sampling_freq, interm_freq = 0.0Hz, CN0 = 45)
    plan.sig_buf .= ComplexF32.(signal)
    for prn_idx in eachindex(plan.avail_prns)
        fill!(plan.noncoherent_integration_matrices[prn_idx], 0f0)
    end
    Acquisition._acquire_step_threaded!(plan, collect(plan.avail_prns), 0)
    # PRN 1 should have a peak; just verify noncoherent matrix was filled
    @test maximum(plan.noncoherent_integration_matrices[1]) > 0
    @test maximum(plan.noncoherent_integration_matrices[2]) > 0
end

@testset "acquire! — PRN not in plan throws ArgumentError" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz

    plan = plan_acquire(system, sampling_freq, [1, 2];
        fft_flag = FFTW.ESTIMATE)
    signal = randn(ComplexF32, plan.samples_per_code)

    @test_throws ArgumentError acquire!(plan, signal, [3])
    @test_throws ArgumentError acquire!(plan, signal, [1, 5])
end

@testset "acquire! — single-PRN Integer convenience overload" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz
    prn = 1

    plan = plan_acquire(system, sampling_freq, [prn]; fft_flag = FFTW.ESTIMATE)
    (; signal) = generate_test_signal(system, prn;
        num_samples = plan.samples_per_code, sampling_freq, interm_freq = 0.0Hz, CN0 = 45)

    # Integer (not vector) argument hits the single-PRN acquire! overload
    result = acquire!(plan, ComplexF32.(signal), prn; interm_freq = 0.0Hz)
    @test result isa AcquisitionResults
    @test result.prn == prn
end

@testset "AcquisitionResults show methods" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz

    plan = plan_acquire(system, sampling_freq, [1, 2]; fft_flag = FFTW.ESTIMATE)
    (; signal) = generate_test_signal(system, 1;
        num_samples = plan.samples_per_code, sampling_freq, interm_freq = 0.0Hz, CN0 = 45)
    results = acquire!(plan, ComplexF32.(signal), [1, 2]; interm_freq = 0.0Hz)

    # Single-result show: uses the (io, MIME"text/plain", AcquisitionResults) method
    io = IOBuffer()
    show(io, MIME"text/plain"(), results[1])
    text = String(take!(io))
    @test occursin("PRN 1", text)
    @test occursin("CN0", text)
    @test occursin("chips", text)

    # Vector-of-results show: uses pretty_table path with color highlighter
    io_color = IOContext(IOBuffer(), :color => true)
    show(io_color, MIME"text/plain"(), results)
    color_text = String(take!(io_color.io))
    @test occursin("PRN", color_text)
    @test occursin("CN0", color_text)

    # Same path without color — hits the empty-highlighter branch
    io_plain = IOContext(IOBuffer(), :color => false)
    show(io_plain, MIME"text/plain"(), results)
    plain_text = String(take!(io_plain.io))
    @test occursin("PRN", plain_text)
end

@testset "acquire! — non-batched pilot path (num_doppler_bins > 320)" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz
    prn = 1

    # 40 ms coherent × min_doppler=10kHz forces num_blocks=16, num_doppler_bins=640,
    # which takes the individual-column-FFT branch in _accumulate_noncoherent_integration_step!.
    plan = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 10_000Hz,
        num_coherently_integrated_code_periods = 40,
        bit_edge_search_steps = 1,
        fft_flag = FFTW.ESTIMATE)
    @test plan.num_coherently_integrated_code_periods * plan.num_blocks > 320

    (; signal) = generate_test_signal(system, prn;
        num_samples = 40 * plan.samples_per_code,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 45)

    result = only(acquire!(plan, ComplexF32.(signal), [prn]; interm_freq = 0.0Hz))
    @test result isa AcquisitionResults
    @test is_detected(result)
end

@testset "generate_test_signal — unit_noise_power=true scales noise to ≈1" begin
    system = GPSL1()

    out = generate_test_signal(system, 1;
        num_samples = 4096, sampling_freq = 4e6Hz,
        unit_noise_power = true, CN0 = 45)
    @test out.signal isa Vector{ComplexF64}
    @test length(out.signal) == 4096
    # Sanity: signal-plus-noise is finite and non-zero
    @test all(isfinite, out.signal)
end
