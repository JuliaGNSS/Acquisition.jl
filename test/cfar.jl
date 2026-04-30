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
    system = GPSL1()
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
    system = GPSL1()
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
    system = GPSL1()
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
    system = GPSL1()

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
