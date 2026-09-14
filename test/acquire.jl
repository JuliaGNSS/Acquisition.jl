# test/acquire.jl — public API tests for FM-DBZP acquire / acquire!

@testset "Tier 3: acquire — GPS L1 strong signal detects at correct code phase and Doppler alias" begin
    system = GPSL1CA()
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
    system = GPSL1CA()
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
    system = GPSL1CA()
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
    system = GPSL1CA()
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
    system = GPSL1CA()
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

@testset "multistep acquire! — PRN-outer parallel layout" begin
    # The multistep (N_nc>1) path is PRN-outer: each parallel chunk carries one
    # PRN through all accumulation steps against its claimed scratch slot's
    # accumulator. Drive it through the public API with two PRNs and verify
    # both results are produced, the planted PRN is found, and every scratch
    # slot returns to the pool.
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    plan = plan_acquire(system, sampling_freq, [1, 2];
        num_noncoherent_accumulations = 2, fft_flag = FFTW.ESTIMATE)
    (; signal, code_phase) = generate_test_signal(system, 1;
        num_samples = 2 * plan.samples_per_code, doppler = 800Hz, code_phase = 150.0,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 45, seed = 13)
    results = acquire!(plan, ComplexF32.(signal), [1, 2]; interm_freq = 0.0Hz)
    @test length(results) == 2
    @test is_detected(results[1])
    @test results[1].code_phase ≈ code_phase atol = 1.5
    @test !is_detected(results[2])   # PRN 2 is absent
    @test length(plan.scratch_free) == length(plan.thread_scratch)
end

@testset "acquire! N_nc=1 sequential vs N_nc=2 multistep parity" begin
    # Concatenating the same coherent segment twice and running with N_nc=2
    # should detect the same PRN at the same code phase as a single segment at
    # N_nc=1. CN0 differs (more integration), but doppler / code_phase / detection
    # outcome must match — confirms the new sequential path is semantically
    # equivalent to the multistep path on shared work.
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    prn = 1
    code_phase_true = 100.0
    doppler_true = 1500Hz
    CN0_dbhz = 45
    seed = 123

    (; signal) = generate_test_signal(system, prn;
        num_samples = 2 * 2048,                  # two full code periods
        doppler = doppler_true, code_phase = code_phase_true,
        sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = CN0_dbhz, seed = seed)

    plan_seq = plan_acquire(system, sampling_freq, [prn];
        num_coherently_integrated_code_periods = 1, num_noncoherent_accumulations = 1,
        fft_flag = FFTW.ESTIMATE)
    plan_multi = plan_acquire(system, sampling_freq, [prn];
        num_coherently_integrated_code_periods = 1, num_noncoherent_accumulations = 2,
        fft_flag = FFTW.ESTIMATE)

    # Sequential path consumes the first segment.
    res_seq = only(acquire!(plan_seq, ComplexF32.(signal[1:2048]), [prn]; interm_freq = 0.0Hz))
    # Multistep path consumes both segments.
    res_multi = only(acquire!(plan_multi, ComplexF32.(signal), [prn]; interm_freq = 0.0Hz))

    @test is_detected(res_seq)
    @test is_detected(res_multi)
    @test abs(res_seq.code_phase   - code_phase_true) < 1.0
    @test abs(res_multi.code_phase - code_phase_true) < 1.0
    @test abs(res_seq.carrier_doppler   / 1Hz - ustrip(Hz, doppler_true)) < ustrip(Hz, step(plan_seq.doppler_freqs))
    @test abs(res_multi.carrier_doppler / 1Hz - ustrip(Hz, doppler_true)) < ustrip(Hz, step(plan_multi.doppler_freqs))
end

@testset "acquire! — sign-search path reports correct Doppler (no half-band fftshift offset)" begin
    # Regression for the double-fftshift bug in the sign-search kernel: when
    # num_data_bits > 1 the column sub-block FFT applied fftshift (circshift by
    # N/2) AND the result was scattered through fftshift_perm a second time.
    # Two fftshifts compose to the identity, leaving the Doppler axis in raw FFT
    # order — every reported Doppler was off by exactly half the searched band,
    # while code phase (the column axis) stayed correct. The pilot path shifts
    # once and is unaffected, so existing tests (all num_data_bits == 1) missed it.
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    prn = 1
    doppler_true = 500Hz      # on the 25 Hz Doppler grid
    code_phase_true = 100.0

    # N_coh = 40 code periods = 2 GPS data-bit periods (20 each) → num_data_bits = 2,
    # which routes through the sign-search kernel rather than the simple pilot path.
    (; signal) = generate_test_signal(system, prn;
        num_samples = 40 * 2048,
        doppler = doppler_true, code_phase = code_phase_true,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 45)

    plan = plan_acquire(system, sampling_freq, [prn];
        num_coherently_integrated_code_periods = 40, num_noncoherent_accumulations = 1,
        fft_flag = FFTW.ESTIMATE)
    @test plan.num_data_bits == 2   # confirm the sign-search path is exercised

    result = acquire!(plan, signal, prn; interm_freq = 0.0Hz)

    @test is_detected(result)
    @test result.code_phase ≈ code_phase_true atol = 1.0
    # The bug shifted Doppler by half the band (8000 Hz here); a correct result
    # lands within one Doppler bin of the truth.
    @test abs(result.carrier_doppler / 1Hz - ustrip(Hz, doppler_true)) < ustrip(Hz, step(plan.doppler_freqs))
end

@testset "acquire! — PRN not in plan throws ArgumentError" begin
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz

    plan = plan_acquire(system, sampling_freq, [1, 2];
        fft_flag = FFTW.ESTIMATE)
    signal = randn(ComplexF32, plan.samples_per_code)

    @test_throws ArgumentError acquire!(plan, signal, [3])
    @test_throws ArgumentError acquire!(plan, signal, [1, 5])
end

@testset "acquire! — single-PRN Integer convenience overload" begin
    system = GPSL1CA()
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
    system = GPSL1CA()
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
    system = GPSL1CA()
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

@testset "acquire! — fused simple-path kernel above batch threshold (num_doppler_bins > 320)" begin
    # Forces the per-column fused kernel `_accumulate_fftshifted_power_pilot!`
    # (the only fused-kernel path no other test hits): simple/pilot routing +
    # num_doppler_bins > BATCH_FFT_THRESHOLD. N_coh=10 keeps num_data_bits=1
    # (GPS L1CA bit_period_codes=20), bit_edge_search_steps=1 keeps the simple
    # path, and a 20 kHz doppler request inflates num_blocks past 32 →
    # num_doppler_bins = 10 × num_blocks > 320.
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    prn = 1

    plan = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 20_000Hz,
        num_coherently_integrated_code_periods = 10,
        bit_edge_search_steps = 1,
        fft_flag = FFTW.ESTIMATE)
    @test plan.num_data_bits == 1
    @test plan.bit_edge_search_steps == 1
    @test length(plan.doppler_freqs) > Acquisition.BATCH_FFT_THRESHOLD

    (; signal) = generate_test_signal(system, prn;
        num_samples = 10 * plan.samples_per_code,
        doppler = 1500Hz, code_phase = 100.0,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 45)

    result = only(acquire!(plan, ComplexF32.(signal), [prn]; interm_freq = 0.0Hz))
    @test is_detected(result)
    @test abs(result.code_phase - 100.0) < 1.0
end

@testset "generate_test_signal — unit_noise_power=true scales noise to ≈1" begin
    system = GPSL1CA()

    out = generate_test_signal(system, 1;
        num_samples = 4096, sampling_freq = 4e6Hz,
        unit_noise_power = true, CN0 = 45)
    @test out.signal isa Vector{ComplexF64}
    @test length(out.signal) == 4096
    # Sanity: signal-plus-noise is finite and non-zero
    @test all(isfinite, out.signal)
end

@testset "acquire! — GlobalMean noise estimator on the streamed N_nc=1 path" begin
    # The streamed path computes GlobalMean noise from column sums collected on
    # the fly (col_sums_buf is only allocated for this estimator at N_nc=1).
    # Detection and peak location must agree with the default estimator.
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    prn = 4
    (; signal) = generate_test_signal(system, prn;
        num_samples = 2048, doppler = 750Hz, code_phase = 512.0,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 45, seed = 21)

    plan_gm = plan_acquire(system, sampling_freq, [prn];
        noise_estimator = GlobalMeanNoiseEstimator(), fft_flag = FFTW.ESTIMATE)
    plan_or = plan_acquire(system, sampling_freq, [prn]; fft_flag = FFTW.ESTIMATE)
    res_gm = acquire!(plan_gm, ComplexF32.(signal), prn; interm_freq = 0.0Hz)
    res_or = acquire!(plan_or, ComplexF32.(signal), prn; interm_freq = 0.0Hz)

    @test is_detected(res_gm)
    @test res_gm.code_phase == res_or.code_phase           # same peak cell
    @test res_gm.carrier_doppler == res_or.carrier_doppler
    @test res_gm.noise_power != res_or.noise_power          # different estimator

    # And the GlobalMean noise must match the reference computed from the
    # stored surface via est_signal_noise_power. The two sum the same cells in
    # the same order, but only one of the loops is vectorised, so the Float32
    # column sums can reassociate and land one ULP apart — on which surfaces
    # that happens shifts with the FFTW plan, so an exact `==` here fails a few
    # runs in ten. One ULP of headroom still pins the estimator; anything
    # actually wrong is orders of magnitude away.
    res_stored = acquire!(plan_gm, ComplexF32.(signal), prn;
        interm_freq = 0.0Hz, store_power_bins = true)
    col_sums = zeros(Float32, plan_gm.samples_per_code_eff)
    sp, np, _, _ = Acquisition.est_signal_noise_power(res_stored.power_bins,
        ustrip(Hz, sampling_freq), ustrip(Hz, get_code_frequency(system)),
        col_sums, GlobalMeanNoiseEstimator())
    @test res_stored.noise_power ≈ Float32(np) rtol = 4eps(Float32)
end

@testset "subsample_interpolation without stored bins — on-demand column recompute" begin
    # With store_power_bins=false the streamed path recomputes the up-to-three
    # peak-neighbour columns on demand; the interpolated result must be
    # identical to the store_power_bins=true run (which reads the neighbours
    # back from the stored surface). Exercised on the simple path AND the
    # secondary-code rotation path (stage-based recompute).
    @testset "simple path" begin
        system = GPSL1CA()
        sampling_freq = 2.048e6Hz
        prn = 2
        (; signal) = generate_test_signal(system, prn;
            num_samples = 2048, doppler = 250Hz, code_phase = 300.7,
            sampling_freq, interm_freq = 0.0Hz, CN0 = 45, seed = 5)
        plan = plan_acquire(system, sampling_freq, [prn]; fft_flag = FFTW.ESTIMATE)
        r_nostore = acquire!(plan, ComplexF32.(signal), prn;
            interm_freq = 0.0Hz, subsample_interpolation = true)
        r_store = acquire!(plan, ComplexF32.(signal), prn;
            interm_freq = 0.0Hz, subsample_interpolation = true, store_power_bins = true)
        # The recompute uses the per-column FFT plan while the stored surface
        # came from the batched tile FFT; FFTW's two algorithms differ in the
        # last bits, so the interpolated estimates match to ~1e-7 relative,
        # not exactly. Tolerances far below a bin still catch logic errors.
        @test r_nostore.code_phase ≈ r_store.code_phase atol = 1e-3
        @test abs(r_nostore.carrier_doppler - r_store.carrier_doppler) < 0.01Hz
    end
    @testset "rotation path (L5I NH10)" begin
        system = GPSL5I()
        sampling_freq = 10.24e6Hz
        prn = 1
        (; signal) = generate_test_signal(system, prn;
            num_samples = 10 * 10240, doppler = 850Hz, code_phase = 5115.3,
            sampling_freq, interm_freq = 0.0Hz, CN0 = 45, seed = 6)
        plan = plan_acquire(system, sampling_freq, [prn];
            num_coherently_integrated_code_periods = 10, fft_flag = FFTW.ESTIMATE)
        @test plan.num_secondary_rotations == 10   # rotation search active
        r_nostore = acquire!(plan, ComplexF32.(signal), prn;
            interm_freq = 0.0Hz, subsample_interpolation = true)
        r_store = acquire!(plan, ComplexF32.(signal), prn;
            interm_freq = 0.0Hz, subsample_interpolation = true, store_power_bins = true)
        @test is_detected(r_nostore)
        # Same last-bit caveat as the simple path above.
        @test r_nostore.code_phase ≈ r_store.code_phase atol = 1e-3
        @test abs(r_nostore.carrier_doppler - r_store.carrier_doppler) < 0.01Hz
        @test r_nostore.secondary_code_phase == r_store.secondary_code_phase
    end
end

@testset "acquire! multistep — subsample_interpolation and store_power_bins" begin
    # Covers the N_nc>1 extraction: interpolation neighbours read from the
    # per-PRN accumulation matrix, and the stored copy of that matrix.
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    prn = 1
    true_cp = 700.4
    (; signal) = generate_test_signal(system, prn;
        num_samples = 4 * 2048, doppler = 500Hz, code_phase = true_cp,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 42, seed = 8)
    plan = plan_acquire(system, sampling_freq, [prn];
        num_noncoherent_accumulations = 4, fft_flag = FFTW.ESTIMATE)
    result = acquire!(plan, ComplexF32.(signal), prn;
        interm_freq = 0.0Hz, subsample_interpolation = true, store_power_bins = true)
    @test is_detected(result)
    @test result.code_phase ≈ true_cp atol = 1.0
    @test result.power_bins isa Matrix{Float32}
    @test size(result.power_bins) ==
        (length(plan.doppler_freqs), plan.samples_per_code_eff)
    # The stored surface is reproducible: a second identical run stores the
    # same accumulated power.
    result2 = acquire!(plan, ComplexF32.(signal), prn;
        interm_freq = 0.0Hz, subsample_interpolation = true, store_power_bins = true)
    @test result2.power_bins == result.power_bins
    @test result2.power_bins === plan.result_buffers[1]   # cached buffer reused
end

@testset "CN0 is a per-Hz quantity: independent of the coherent integration length" begin
    # C/N₀ = C / N₀ has units of Hz and describes the signal and the front end,
    # not the search. The peak search measures an SNR, and coherent integration
    # over `T` makes that SNR `(C/N₀)·T`, so `T` must be divided back out. It had
    # been divided by one code period regardless of `N =
    # num_coherently_integrated_code_periods`, which reported `10·log₁₀(N)` dB too
    # much — 10 dB at the N = 10 a receiver uses for weak signals, enough to make
    # a fixed dB-Hz gate mean something different at every search setting.
    #
    # What is asserted is INVARIANCE across `N`, not accuracy: the estimate is a
    # lower bound on the truth, low by the residual coherent loss (frequency
    # error inside the Doppler bin, phase noise), which is a few dB here.
    system = GPSL1CA()
    prn = 1
    sampling_freq = 4e6Hz
    true_cn0 = 45
    lengths = (1, 2, 4, 5, 10)

    reported = map(lengths) do n
        # Average over code phases: one trial carries a dB or two of estimator
        # noise and the point here is the systematic offset. Undetected trials are
        # skipped rather than asserted on — at `N = 1` a 45 dBHz signal over a
        # 25 kHz search is genuinely marginal.
        estimates = Float64[]
        for k = 1:8
            (; signal) = generate_test_signal(
                system, prn;
                seed = 700 + k,
                num_samples = 4000 * n + 100,
                doppler = 1234Hz,
                code_phase = (k - 0.5) * 1023 / 8 + 0.137,
                sampling_freq,
                interm_freq = 0.0Hz,
                CN0 = true_cn0,
            )
            plan = plan_acquire(
                system, sampling_freq, [prn];
                num_coherently_integrated_code_periods = n,
                min_doppler_coverage = 25_000Hz,
                fft_flag = FFTW.ESTIMATE,
            )
            res = acquire!(plan, signal, prn; interm_freq = 0.0Hz,
                           subsample_interpolation = true)
            is_detected(res) && push!(estimates, res.CN0)
        end
        @test length(estimates) >= 5
        sum(estimates) / length(estimates)
    end

    # Every length lands near the truth from below, and — the property that
    # actually matters — they agree with each other instead of climbing with `N`.
    for cn0 in reported
        @test cn0 ≈ true_cn0 atol = 6.0
    end
    @test maximum(reported) - minimum(reported) < 4.0
    # Guard the specific regression: before the fix this climbed monotonically,
    # by ~8 dB from N = 1 to N = 10.
    @test reported[end] - reported[1] < 2.0
end
# A signal that throws the moment a sample is read: the simplest way to make the
# search fail on its own task, where a real failure would come from a bug in the
# correlation rather than from the caller.
struct FaultySignal <: AbstractVector{ComplexF32}
    inner::Vector{ComplexF32}
end
Base.size(signal::FaultySignal) = size(signal.inner)
Base.IndexStyle(::Type{FaultySignal}) = IndexLinear()
Base.getindex(::FaultySignal, ::Int) = error("faulty signal")

# `acquire_stream!` — the streaming API. The call returns a channel before any PRN is
# done and each result is `put!` onto it as soon as its own PRN is finished, instead of
# only being readable once every PRN is done. The channel belongs to the call: it is
# closed when the search finishes, or closed with the exception when it fails.
#
# The chunk count, i.e. the thread count of the test process, decides how many tasks
# publish: at one scratch slot the whole loop is a single inline chunk, above that the
# spawned chunks `put!` concurrently. CI runs the suite both ways.
@testset "acquire_stream!" begin
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    prns = collect(1:8)

    (; signal, code_phase, interm_freq) = generate_test_signal(
        system, 1;
        num_samples = 2048, doppler = 1000Hz, code_phase = 200.0,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 45, seed = 21,
    )
    plan = plan_acquire(system, sampling_freq, prns; fft_flag = FFTW.ESTIMATE)

    # Reference: the same search through the batch API.
    batch = acquire!(plan, signal, prns; interm_freq)
    reference = [(r.prn, r.code_phase, r.CN0, r.peak_to_noise_ratio) for r in batch]

    @testset "streams every result, and the same ones the batch API returns" begin
        streamed = collect(acquire_stream!(plan, signal, prns; interm_freq))

        @test length(streamed) == length(prns)
        # Completion order, not `prns` order — so compare as a set of PRNs. Sorting a
        # result per PRN with nothing left over is also what rules out a duplicate
        # publish, the invariant the concurrently publishing chunks have to keep.
        @test sort(getfield.(streamed, :prn)) == prns
        for r in streamed
            @test (r.prn, r.code_phase, r.CN0, r.peak_to_noise_ratio) ==
                reference[findfirst(==(r.prn), prns)]
        end
        @test is_detected(only(filter(r -> r.prn == 1, streamed)))
        @test only(filter(r -> r.prn == 1, streamed)).code_phase ≈ code_phase atol = 1.0
    end

    @testset "the channel is concretely typed, and the call is type stable" begin
        channel = @inferred acquire_stream!(plan, signal, prns; interm_freq)
        @test channel isa Channel{eltype(batch)}
        @test isconcretetype(eltype(channel))
        collect(channel)
    end

    @testset "closes itself when the search is done, and frees the plan" begin
        channel = acquire_stream!(plan, signal, prns; interm_freq)
        # No `close` from the caller, no counting of results: the loop ends because
        # the channel does.
        count = 0
        for _ in channel
            count += 1
        end
        @test count == length(prns)
        @test !isopen(channel)
        # Iteration ending means the search task is done, so the plan is free again
        # and can be handed straight to the next call.
        @test length(plan.scratch_free) == length(plan.thread_scratch)
        @test getfield.(acquire!(plan, signal, prns; interm_freq), :prn) == prns
    end

    @testset "results are consumable while the search is still running" begin
        # `buffer_size = 1` with no consumer blocks the second `put!`, so a search
        # that only published at the end could never hand out a result here: taking
        # one proves incremental delivery. And with one result taken at most two can
        # have been published, so ≥5 of the 8 PRNs are still outstanding — the search
        # cannot have finished, which is what an open channel then means.
        channel = acquire_stream!(plan, signal, prns; interm_freq, buffer_size = 1)
        first_result = take!(channel)
        @test first_result.prn in prns
        @test isopen(channel)
        rest = [take!(channel) for _ in 2:length(prns)]
        @test sort(getfield.(vcat(first_result, rest), :prn)) == prns
    end

    @testset "multistep path (N_nc > 1)" begin
        plan_ms = plan_acquire(system, sampling_freq, prns;
            num_noncoherent_accumulations = 2, fft_flag = FFTW.ESTIMATE)
        long_signal = generate_test_signal(system, 1;
            num_samples = 2 * 2048, doppler = 800Hz, code_phase = 150.0,
            sampling_freq, interm_freq = 0.0Hz, CN0 = 45, seed = 13).signal

        streamed = collect(acquire_stream!(plan_ms, ComplexF32.(long_signal), prns;
            interm_freq = 0.0Hz))

        @test sort(getfield.(streamed, :prn)) == prns
        @test is_detected(only(filter(r -> r.prn == 1, streamed)))
        @test length(plan_ms.scratch_free) == length(plan_ms.thread_scratch)
    end

    @testset "bad arguments throw at the call site" begin
        # Checked before the search is spawned, so the caller sees them where they
        # were made instead of out of the channel.
        @test_throws ArgumentError acquire_stream!(plan, signal, [1, 99]; interm_freq)
        @test_throws ArgumentError acquire_stream!(plan, signal[1:100], prns; interm_freq)
    end

    @testset "a failed search closes the channel with its exception" begin
        # Without this the consumer would wait forever on a channel nothing is ever
        # going to fill. `FaultySignal` fails inside the search task, the way a bug
        # in the correlation would.
        channel = acquire_stream!(plan, FaultySignal(ComplexF32.(signal)), prns;
            interm_freq)
        @test_throws Exception collect(channel)
        @test !isopen(channel)
        @test length(plan.scratch_free) == length(plan.thread_scratch)
        @test getfield.(acquire!(plan, signal, prns; interm_freq), :prn) == prns
    end

    @testset "closing the channel early unwinds the search" begin
        # Cancellation: the next `put!` throws, `@sync` joins every chunk before the
        # exception leaves the loop, and the scratch claim is released in a `finally`,
        # so the plan is left whole and usable. The search may still be unwinding when
        # `close` returns, hence the wait rather than an immediate check.
        #
        # The wait is on `plan.in_use`, NOT on the scratch pool: a slot is released in
        # the per-PRN `finally`, before the chunk publishes and long before `@sync`
        # joins, so a full pool does not mean the search is over. `in_use` is cleared
        # by the search task itself on its way out, so it does.
        channel = acquire_stream!(plan, signal, prns; interm_freq, buffer_size = 1)
        take!(channel)
        close(channel)
        @test timedwait(() -> !plan.in_use[], 10.0) === :ok
        @test length(plan.scratch_free) == length(plan.thread_scratch)
        @test getfield.(acquire!(plan, signal, prns; interm_freq), :prn) == prns
    end

    @testset "one plan serves any number of streams, one after another" begin
        # Sequential reuse is the normal way to acquire over a recording: the plan's
        # buffers are rewritten by each search, so what has to hold is that a stream
        # leaves the plan exactly as it found it — and that results already handed out
        # are snapshots, unaffected by the searches that follow. (Concurrent streams on
        # one plan are a different matter and are not supported: they would write the
        # same buffers at the same time.)
        segments = [
            (1, 200.0, 1000Hz),
            (5, 700.0, -500Hz),
            (3, 42.0, 250Hz),
        ]
        kept = eltype(batch)[]
        for (prn, expected_code_phase, doppler) in segments
            segment = generate_test_signal(
                system, prn;
                num_samples = 2048, doppler, code_phase = expected_code_phase,
                sampling_freq, interm_freq = 0.0Hz, CN0 = 45, seed = 21,
            ).signal
            expected = acquire!(plan, segment, prns; interm_freq)
            expected_fields = [(r.prn, r.code_phase, r.CN0) for r in expected]

            streamed = collect(acquire_stream!(plan, segment, prns; interm_freq))

            @test sort(getfield.(streamed, :prn)) == prns
            for r in streamed
                @test (r.prn, r.code_phase, r.CN0) ==
                    expected_fields[findfirst(==(r.prn), prns)]
            end
            @test is_detected(only(filter(r -> r.prn == prn, streamed)))
            @test length(plan.scratch_free) == length(plan.thread_scratch)
            append!(kept, streamed)
        end
        # Every result from every stream still reads as it did when it was published.
        for (i, (prn, expected_code_phase, _)) in enumerate(segments)
            window = kept[(i - 1) * length(prns) + 1:i * length(prns)]
            @test only(filter(r -> r.prn == prn, window)).code_phase ≈
                expected_code_phase atol = 1.0
        end
    end

    @testset "a finished but unread stream survives the next search" begin
        # `put!` copies the result struct into the channel, so a channel left unread is
        # a snapshot and not a view of the plan's results buffer — which is what lets
        # the next search start before the previous results have been looked at.
        other = generate_test_signal(
            system, 5;
            num_samples = 2048, doppler = -500Hz, code_phase = 700.0,
            sampling_freq, interm_freq = 0.0Hz, CN0 = 45, seed = 21,
        ).signal

        first_channel = acquire_stream!(plan, signal, prns; interm_freq)
        # Default `buffer_size` holds every result, so the search finishes — and closes
        # the channel — without anything being taken from it.
        @test timedwait(() -> !isopen(first_channel), 10.0) === :ok
        second = collect(acquire_stream!(plan, other, prns; interm_freq))
        first = collect(first_channel)   # read only now, one search later

        @test only(filter(r -> r.prn == 1, first)).code_phase ≈ code_phase atol = 1.0
        @test only(filter(r -> r.prn == 5, second)).code_phase ≈ 700.0 atol = 1.0
        @test sort(getfield.(first, :prn)) == prns
        @test sort(getfield.(second, :prn)) == prns
    end

    @testset "acquire_stream plans and streams in one call" begin
        streamed = collect(acquire_stream(system, signal, sampling_freq, [1, 2];
            interm_freq, fft_flag = FFTW.ESTIMATE))
        @test sort(getfield.(streamed, :prn)) == [1, 2]
        @test is_detected(only(filter(r -> r.prn == 1, streamed)))
    end

    # ------------------------------------------------------------------
    # Plan ownership. A stream keeps the plan for as long as its search runs — which
    # outlives the call that started it. These pin that the plan is never shared by
    # two searches, and that the `do`-block form always hands it back.
    # ------------------------------------------------------------------

    @testset "a second search on a plan still streaming throws" begin
        # Before this was enforced, the two searches wrote the same buffers and the
        # batch call returned results for the wrong PRNs — silently.
        channel = acquire_stream!(plan, signal, prns; interm_freq, buffer_size = 1)
        @test_throws PlanInUseError acquire!(plan, signal, prns; interm_freq)
        @test_throws PlanInUseError acquire_stream!(plan, signal, prns; interm_freq)
        for _ in channel
        end
        @test timedwait(() -> !plan.in_use[], 10.0) === :ok
        @test getfield.(acquire!(plan, signal, prns; interm_freq), :prn) == prns
    end

    @testset "an abandoned bare-channel stream is loud, not corrupting" begin
        # `break` out of `for result in acquire_stream!(...)` does NOT close the
        # channel, so the search is still running and still owns the plan. The point
        # is that the next call says so instead of interleaving with it.
        own_plan = plan_acquire(system, sampling_freq, prns; fft_flag = FFTW.ESTIMATE)
        channel = acquire_stream!(own_plan, signal, prns; interm_freq, buffer_size = 1)
        for _ in channel
            break
        end
        @test_throws PlanInUseError acquire!(own_plan, signal, prns; interm_freq)
    end

    @testset "the do-block form frees the plan however the block is left" begin
        # `break`, an early `return`, and a throw all have to reach the same teardown:
        # cancel, drain what is in flight, join the search.
        own_plan = plan_acquire(system, sampling_freq, prns; fft_flag = FFTW.ESTIMATE)

        acquire_stream!(own_plan, signal, prns; interm_freq, buffer_size = 1) do results
            for _ in results
                break
            end
        end
        @test !own_plan.in_use[]
        @test length(own_plan.scratch_free) == length(own_plan.thread_scratch)
        @test getfield.(acquire!(own_plan, signal, prns; interm_freq), :prn) == prns

        # Early return out of the consuming function.
        first_detected(p) = acquire_stream!(p, signal, prns; interm_freq,
                                            buffer_size = 1) do results
            for r in results
                is_detected(r) && return r.prn
            end
            return nothing
        end
        @test first_detected(own_plan) in prns
        @test !own_plan.in_use[]

        # A consumer that throws: its exception is the one that surfaces, not anything
        # from tearing the search down behind it.
        @test_throws "consumer failed" acquire_stream!(own_plan, signal, prns;
                                                       interm_freq, buffer_size = 1) do results
            for _ in results
                error("consumer failed")
            end
        end
        @test !own_plan.in_use[]
        @test getfield.(acquire!(own_plan, signal, prns; interm_freq), :prn) == prns
    end

    @testset "the do-block form returns the block's value and still streams fully" begin
        own_plan = plan_acquire(system, sampling_freq, prns; fft_flag = FFTW.ESTIMATE)
        got = acquire_stream!(own_plan, signal, prns; interm_freq) do results
            sort([r.prn for r in results])
        end
        @test got == prns
        @test !own_plan.in_use[]

        # And the convenience wrapper that plans for you.
        scoped = acquire_stream(system, signal, sampling_freq, [1, 2];
                                interm_freq, fft_flag = FFTW.ESTIMATE) do results
            sort([r.prn for r in results])
        end
        @test scoped == [1, 2]
    end

    @testset "a failed search still frees the plan" begin
        own_plan = plan_acquire(system, sampling_freq, prns; fft_flag = FFTW.ESTIMATE)
        channel = acquire_stream!(own_plan, FaultySignal(ComplexF32.(signal)), prns;
            interm_freq)
        @test_throws Exception collect(channel)
        @test timedwait(() -> !own_plan.in_use[], 10.0) === :ok
        @test getfield.(acquire!(own_plan, signal, prns; interm_freq), :prn) == prns
    end

    @testset "bad arguments leave the plan free" begin
        # The claim is taken before the channel is built, so anything that throws
        # between the two has to hand it back — otherwise one bad call would brick
        # the plan for good.
        own_plan = plan_acquire(system, sampling_freq, prns; fft_flag = FFTW.ESTIMATE)
        @test_throws ArgumentError acquire_stream!(own_plan, signal, [1, 99]; interm_freq)
        @test !own_plan.in_use[]
        @test_throws Exception acquire_stream!(own_plan, signal, prns; interm_freq,
            buffer_size = -1)
        @test !own_plan.in_use[]
        # A unitless `interm_freq` is an easy slip; it has to throw here, not later out
        # of the iteration on another task.
        @test_throws Unitful.DimensionError acquire_stream!(own_plan, signal, prns;
            interm_freq = 5.0)
        @test !own_plan.in_use[]
        @test getfield.(acquire!(own_plan, signal, prns; interm_freq), :prn) == prns
    end

    @testset "cancellation stops the search at the next PRN boundary" begin
        # The flag is checked before each PRN, so a cancelled stream does not run a
        # further full PRN search per chunk on its way out. With a capacity-1 channel
        # over 8 PRNs, stopping after the first result must leave most PRNs unsearched.
        own_plan = plan_acquire(system, sampling_freq, prns; fft_flag = FFTW.ESTIMATE)
        published = 0
        acquire_stream!(own_plan, signal, prns; interm_freq, buffer_size = 1) do results
            for _ in results
                published += 1
                break
            end
        end
        @test published == 1
        @test !own_plan.in_use[]
        # Whatever was in flight was discarded by the drain, and the plan is clean.
        @test length(own_plan.scratch_free) == length(own_plan.thread_scratch)
        @test getfield.(acquire!(own_plan, signal, prns; interm_freq), :prn) == prns
    end
end
