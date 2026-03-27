@testset "CFAR threshold" begin
    # For single dwell (M=1), chi2(2) = exponential(1/2)
    # P(X > t) = exp(-t/2), so threshold = -2*log(pfa_per_cell)
    # With 1 cell: pfa_per_cell = pfa
    @test cfar_threshold(0.01, 1) ≈ -2 * log(0.01)
    @test cfar_threshold(0.001, 1) ≈ -2 * log(0.001)

    # Threshold increases with more cells (Bonferroni correction)
    t1 = cfar_threshold(0.01, 100)
    t2 = cfar_threshold(0.01, 10000)
    @test t2 > t1

    # Threshold increases with stricter pfa
    t_loose = cfar_threshold(0.1, 1000)
    t_strict = cfar_threshold(0.001, 1000)
    @test t_strict > t_loose

    # Multiple non-coherent integrations increase threshold
    t_m1 = cfar_threshold(0.01, 1000; num_noncoherent_integrations = 1)
    t_m3 = cfar_threshold(0.01, 1000; num_noncoherent_integrations = 3)
    @test t_m3 > t_m1

    # Argument validation
    @test_throws ArgumentError cfar_threshold(0.0, 100)
    @test_throws ArgumentError cfar_threshold(1.0, 100)
    @test_throws ArgumentError cfar_threshold(-0.1, 100)
    @test_throws ArgumentError cfar_threshold(0.01, 0)
end

@testset "CFAR detection with synthetic signal" begin
    system = GPSL1()

    # Generate signal with known satellite
    (; signal, doppler, code_phase, prn, sampling_freq, interm_freq, CN0) =
        generate_test_signal(system, 1)

    # Acquire all PRNs
    results = acquire(system, signal, sampling_freq, 1:32; interm_freq)

    # Compute CFAR threshold
    num_cells = size(results[1].power_bins, 1) * size(results[1].power_bins, 2)
    threshold = cfar_threshold(0.01, num_cells)

    # The signal PRN should be detected
    signal_result = results[prn]
    @test signal_result.peak_to_noise_ratio > threshold

    # Most noise-only PRNs should NOT exceed the threshold
    noise_prns = filter(r -> r.prn != prn, results)
    num_false_alarms = count(r -> r.peak_to_noise_ratio > threshold, noise_prns)
    # pfa=0.01 is per-PRN (each has its own search grid), so expect very few false alarms
    @test num_false_alarms <= 3
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
