@testset "CFAR threshold" begin
    # For single dwell (M=1), peak_to_noise_ratio ~ Exp(1)
    # P(X > t) = exp(-t), so threshold = -log(pfa_per_cell)
    # With 1 cell: pfa_per_cell = pfa
    @test cfar_threshold(0.01, 1) ≈ -log(0.01)
    @test cfar_threshold(0.001, 1) ≈ -log(0.001)

    # Threshold increases with more cells (Bonferroni correction)
    t1 = cfar_threshold(0.01, 100)
    t2 = cfar_threshold(0.01, 10000)
    @test t2 > t1

    # Threshold increases with stricter pfa
    t_loose = cfar_threshold(0.1, 1000)
    t_strict = cfar_threshold(0.001, 1000)
    @test t_strict > t_loose

    # More non-coherent integrations concentrate the noise distribution around mean=1,
    # so the threshold (on peak_to_noise_ratio scale) decreases for the same pfa.
    t_m1 = cfar_threshold(0.01, 1000; num_noncoherent_integrations = 1)
    t_m3 = cfar_threshold(0.01, 1000; num_noncoherent_integrations = 3)
    @test t_m3 < t_m1

    # Argument validation
    @test_throws ArgumentError cfar_threshold(0.0, 100)
    @test_throws ArgumentError cfar_threshold(1.0, 100)
    @test_throws ArgumentError cfar_threshold(-0.1, 100)
    @test_throws ArgumentError cfar_threshold(0.01, 0)
end

@testset "CFAR detection with synthetic signal" begin
    system = GPSL1CA()
    prn = 1

    # Generate signal with known satellite
    (; signal, sampling_freq, interm_freq) = generate_test_signal(system, prn)

    # Acquire all PRNs
    results = acquire(system, signal, sampling_freq, collect(1:32); interm_freq)

    threshold = cfar_threshold(0.01, get_num_cells(results[1]))

    # The signal PRN should be detected
    signal_result = only(filter(r -> r.prn == prn, results))
    @test signal_result.peak_to_noise_ratio > threshold

    # Most noise-only PRNs should NOT exceed the threshold
    noise_results = filter(r -> r.prn != prn, results)
    num_false_alarms = count(r -> r.peak_to_noise_ratio > threshold, noise_results)
    # pfa=0.01 is per-PRN (each has its own search grid), so expect very few false alarms
    @test num_false_alarms <= 3
end

@testset "CFAR detection with non-coherent integration (M > 1)" begin
    # End-to-end check that the CFAR threshold's nominal false-alarm rate is
    # honoured by the actual `peak_to_noise_ratio` distribution when
    # num_noncoherent_accumulations > 1. This covers the gap left by the
    # single-dwell synthetic test above: the analytic derivation in
    # `cfar_threshold` assumes per-cell χ²(2M) distribution, and we want the
    # end-to-end `acquire!` pipeline to reproduce that on synthetic AWGN.
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    samples_per_code = 2048
    pfa = 0.01
    n_trials = 50

    for (M, num_code_cycles) in [(2, 1), (4, 1), (8, 1), (4, 4)]
        # All-noise input: no satellite signal injected.
        seg_len = num_code_cycles * samples_per_code * M
        false_alarms = 0
        Random.seed!(0xC0FFEE + M * 1000)
        plan = plan_acquire(system, float(sampling_freq), collect(1:32);
            num_coherently_integrated_code_periods = num_code_cycles,
            num_noncoherent_accumulations = M)
        num_cells = length(plan.doppler_freqs) * plan.num_blocks * plan.block_size
        threshold = cfar_threshold(pfa, num_cells; num_noncoherent_integrations = M)
        for _ in 1:n_trials
            noise = randn(ComplexF64, seg_len)
            results = acquire!(plan, noise, collect(1:32); interm_freq = 0.0Hz)
            false_alarms += count(r -> r.peak_to_noise_ratio > threshold, results)
        end
        # Across n_trials × 32 PRNs we expect ≈ n_trials * 32 * pfa false alarms.
        # Allow a generous Poisson-style margin (3× expected) — the test should
        # only fire when the empirical rate is materially higher than nominal.
        expected = n_trials * 32 * pfa
        @test false_alarms ≤ ceil(Int, 3 * expected)
        @info "CFAR FP rate vs nominal" M expected false_alarms threshold
    end
end

@testset "CFAR detection with non-coherent integration: signal still passes threshold" begin
    # Counterpart to the FP-rate test above: verify a real signal is still
    # detected when num_noncoherent_accumulations > 1. This guards against a
    # threshold that's so loose it admits false alarms but also against one
    # that's so tight it rejects real signals.
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    samples_per_code = 2048
    pfa = 0.01

    for (M, num_code_cycles, CN0) in [(2, 4, 40), (4, 4, 38), (8, 1, 42)]
        seg_len = num_code_cycles * samples_per_code * M
        prn = 7
        (; signal, interm_freq) = generate_test_signal(system, prn;
            num_samples = seg_len,
            sampling_freq, interm_freq = 0.0Hz,
            CN0, doppler = 500Hz, code_phase = 100.0,
            seed = 9999 + M)
        plan = plan_acquire(system, float(sampling_freq), collect(1:32);
            num_coherently_integrated_code_periods = num_code_cycles,
            num_noncoherent_accumulations = M)
        result = only(acquire!(plan, signal, [prn]; interm_freq))
        @test is_detected(result; pfa)
    end
end

@testset "peak_to_noise_ratio is consistent with CN0" begin
    system = GPSL1CA()

    # Strong signal
    (; signal, prn, sampling_freq, interm_freq) =
        generate_test_signal(system, 1; CN0 = 55)
    strong = acquire(system, signal, sampling_freq, prn; interm_freq)

    # Weak signal
    (; signal, prn, sampling_freq, interm_freq) =
        generate_test_signal(system, 1; CN0 = 35)
    weak = acquire(system, signal, sampling_freq, prn; interm_freq)

    # Strong signal should have higher peak_to_noise_ratio
    @test strong.peak_to_noise_ratio > weak.peak_to_noise_ratio
    @test strong.CN0 > weak.CN0
end

@testset "Sign-search CFAR null distribution" begin
    # One hypothesis per cell: the null model reproduces the χ² (Gamma) threshold.
    for M in (1, 3, 20)
        null = Acquisition._sign_search_null(ones(Float32, 400, 1), M)
        @test null.num_hypotheses == 1
        for cells in (10^3, 655_360)
            @test Acquisition._cfar_threshold(0.01, cells, M, null) ≈
                  cfar_threshold(0.01, cells; num_noncoherent_integrations = M) rtol = 6e-4
        end
    end
    # Two identical hypotheses are still one exponential.
    dup = Acquisition._sign_search_null(ones(Float32, 400, 2), 1)
    @test Acquisition._cfar_threshold(0.01, 1000, 1, dup) ≈ cfar_threshold(0.01, 1000) rtol = 1e-2
    # Two orthogonal hypotheses: W = max of two iid Exp(1), E[W] = 3/2, so
    # P(W / E[W] > t) = 1 - (1 - e^{-3t/2})².
    orth = Acquisition._sign_search_null(Float32[ones(200); ones(200);; ones(200); -ones(200)], 1)
    pfa_cell = -expm1(log1p(-0.01) / 1000)
    exact = -log1p(-sqrt(1 - pfa_cell)) / 1.5
    @test Acquisition._cfar_threshold(0.01, 1000, 1, orth) ≈ exact rtol = 1e-2
    # More hypotheses per cell → lower threshold on the mean-normalised statistic.
    @test Acquisition._cfar_threshold(0.01, 1000, 1, orth) < cfar_threshold(0.01, 1000)

    # Plans: a sign search carries the null model, the simple path does not.
    simple = plan_acquire(GPSL1CA(), 2.048e6Hz, [1]; num_coherently_integrated_code_periods = 20,
        fft_flag = FFTL.ESTIMATE)
    @test simple.sign_search_null === nothing
    # 40 ms, N_be = 4: 2 aligned polarities + 3 alignments × 4 (partial bits).
    for (ncoh, nbe, K) in ((20, 5, 5), (40, 1, 2), (40, 4, 14), (60, 1, 4))
        plan = plan_acquire(GPSL1CA(), 2.048e6Hz, [1]; min_doppler_coverage = 5000Hz,
            num_coherently_integrated_code_periods = ncoh, bit_edge_search_steps = nbe,
            fft_flag = FFTL.ESTIMATE)
        @test plan.sign_search_null.num_hypotheses == K
    end
    # The null model averages the column offsets the kernel's partial segments
    # move with (`_tile_row_offsets` takes every value in 0:num_blocks-1).
    @test Acquisition._null_model_offsets(16) == [0, 4, 8, 12]
    @test Acquisition._null_model_offsets(3) == [0, 1, 2]
    @test Acquisition._null_model_offsets(1) == [0]
    # L5I rotation search: one hypothesis per rotation at N_coh = 10 …
    @test plan_acquire(GPSL5I(), 10.23e6Hz, [1]; num_coherently_integrated_code_periods = 10,
        fft_flag = FFTL.ESTIMATE).sign_search_null === nothing
    # … two data-bit polarities per rotation at N_coh = 20.
    @test plan_acquire(GPSL5I(), 10.23e6Hz, [1]; num_coherently_integrated_code_periods = 20,
        fft_flag = FFTL.ESTIMATE).sign_search_null.num_hypotheses == 2
end

@testset "Sign-search CFAR holds its false-alarm rate on noise" begin
    # Noise only through the full pipeline: with the null model the realised
    # false-alarm rate matches the requested one (the χ² threshold gave 0%).
    # pfa = 0.1 keeps the trial count small; Binomial(trials, 0.1) ± 3σ.
    function false_alarms(system, fs, trials; ncoh, edges, kwargs...)
        plan = plan_acquire(system, fs, [1]; num_coherently_integrated_code_periods = ncoh,
            bit_edge_search_steps = edges, fft_flag = FFTL.ESTIMATE, kwargs...)
        @test plan.sign_search_null !== nothing
        rng = Xoshiro(42)
        num_samples = ncoh * plan.num_noncoherent_accumulations * plan.samples_per_code
        alarms = count(1:trials) do _
            noise = randn(rng, ComplexF32, num_samples)
            is_detected(only(acquire!(plan, noise, [1]; interm_freq = 0.0Hz)); pfa = 0.1)
        end
        σ = sqrt(trials * 0.1 * 0.9)
        @test trials * 0.1 - 3σ <= alarms <= trials * 0.1 + 3σ
        return plan
    end
    @testset "one-bit window, 5 transitions" begin
        plan = false_alarms(GPSL1CA(), 2.048e6Hz, 400; ncoh = 20, edges = 5,
            min_doppler_coverage = 5000Hz)
        r = only(acquire!(plan, randn(Xoshiro(1), ComplexF32, 20 * plan.samples_per_code), [1];
            interm_freq = 0.0Hz))
        @test cfar_threshold(r; pfa = 0.01) < cfar_threshold(0.01, get_num_cells(r))
    end
    @testset "two bits, 4 alignments (partial bits)" false_alarms(GPSL1CA(), 2.048e6Hz, 300;
        ncoh = 40, edges = 4, min_doppler_coverage = 5000Hz)
    # The Lugannani–Rice sum over steps, on a two-pass Doppler grid.
    @testset "N_nc = 3, two Doppler passes" begin
        plan = false_alarms(GPSL1CA(), 1.023e6Hz, 300; ncoh = 40, edges = 4,
            num_noncoherent_accumulations = 3, min_doppler_coverage = 2000Hz)
        @test plan.num_doppler_passes > 1
    end
    # Secondary-code rotations × data bits × bit edges.
    @testset "L5I rotations × 2 alignments" false_alarms(GPSL5I(), 10.23e6Hz, 300;
        ncoh = 20, edges = 2, min_doppler_coverage = 1000Hz)
end
