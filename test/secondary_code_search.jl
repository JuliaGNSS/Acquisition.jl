@testset "secondary_code_search" begin
    @testset "L5I rotation discovery — detection succeeds at every NH10 starting phase" begin
        # For each r ∈ 0..9, build an L5I signal whose first integrated primary period
        # carries NH10[r], second carries NH10[(r+1) mod 10], etc. With use_secondary_code
        # = true, the rotation search must find the matching rotation and the resulting
        # peak must clear the opt-out baseline by a wide margin at every starting phase.
        system        = GPSL5I()
        sampling_freq = 10.24e6Hz
        prn           = 1
        fc            = get_code_frequency(system)
        samples_per_code = ceil(Int, get_code_length(system) / fc * sampling_freq)
        N = 10
        L = 10
        sec = get_secondary_code(system)
        primary = ComplexF32.(gen_code(samples_per_code, system, prn, sampling_freq, fc, 0.0))

        plan = plan_acquire(system, sampling_freq, [prn];
            min_doppler_coverage = 1_000Hz, num_coherently_integrated_code_periods = N,
            num_noncoherent_accumulations = 1)
        # Run an opt-out reference once so we have a baseline peak to compare against.
        ref_signal = ComplexF32[]
        for k in 0:N-1
            append!(ref_signal, Float32(GNSSSignals.secondary_value(sec, prn, k)) .* primary)
        end
        plan_off = plan_acquire(system, sampling_freq, [prn];
            min_doppler_coverage = 1_000Hz, num_coherently_integrated_code_periods = N,
            num_noncoherent_accumulations = 1, use_secondary_code = false)
        results_off = acquire!(plan_off, ref_signal, [prn]; store_power_bins = true)
        peak_off = maximum(results_off[1].power_bins)

        @testset "rotation r = $r" for r in 0:L-1
            signal = ComplexF32[]
            for k in 0:N-1
                nh10_chip = Float32(GNSSSignals.secondary_value(sec, prn, mod(k + r, L)))
                append!(signal, nh10_chip .* primary)
            end
            results_on = acquire!(plan, signal, [prn]; store_power_bins = true)
            peak_on = maximum(results_on[1].power_bins)
            # Rotation discovery: with the search active, the peak should clear the opt-out
            # baseline by a substantial margin, regardless of NH10 starting phase.
            @test peak_on > 5 * peak_off
        end
    end

    @testset "Opt-out parity — use_secondary_code=false reproduces the pre-feature pilot fast-path output" begin
        # Pinned values captured by running this exact configuration with the pilot fast-path
        # before the rotation kernel was added. Since `use_secondary_code = false` makes the
        # dispatcher route through the unchanged pilot fast-path (no kernel restructuring),
        # output stays numerically equivalent to the pre-feature behaviour. Comparisons use
        # `≈` rather than `==` because the underlying FFT (FFTW) gives ULP-level
        # differences across Julia versions / FFTW builds — the parity guarantee is
        # "no behavioural drift in the opt-out path", not platform-exact bits. `size` and
        # `argmax` are integer-valued and stay exact.
        system        = GPSL5I()
        sampling_freq = 10.24e6Hz
        prn           = 1
        fc            = get_code_frequency(system)
        samples_per_code = ceil(Int, get_code_length(system) / fc * sampling_freq)
        N = 10
        sec = get_secondary_code(system)

        primary = ComplexF32.(gen_code(samples_per_code, system, prn, sampling_freq, fc, 0.0))
        signal = ComplexF32[]
        for k in 0:N-1
            nh10_chip = Float32(GNSSSignals.secondary_value(sec, prn, k))
            append!(signal, nh10_chip .* primary)
        end

        plan = plan_acquire(system, sampling_freq, [prn];
            min_doppler_coverage = 1_000Hz, num_coherently_integrated_code_periods = N,
            num_noncoherent_accumulations = 1, use_secondary_code = false)
        results = acquire!(plan, signal, [prn]; store_power_bins = true)
        nim = results[1].power_bins

        @test size(nim) == (40, 10240)
        @test maximum(nim)   ≈ 1.146687232f9   rtol = 1e-4
        @test sum(nim)       ≈ 5.2553412608f10 rtol = 1e-4
        @test nim[1, 100]    ≈ 8.2944015625f4  rtol = 1e-4
        @test nim[end, end]  ≈ 2.318815625f4   rtol = 1e-4
        @test argmax(nim)    == CartesianIndex(17, 1)
    end

    @testset "L5I gain recovery — use_secondary_code=true recovers ~10× peak over the opt-out" begin
        # Synthesise a clean L5I signal with NH10 modulation applied across 10 primary periods
        # (= 1 NH10 period at the primary code rate). Without secondary-code rotation search,
        # summing 10 primary periods coherently against the bare primary code adds up the
        # NH10 = [+,+,+,+,-,-,+,-,+,-] signs → inner product = 2 instead of 10. With the
        # rotation search active, the correct alignment is found and the full coherent gain is
        # recovered. The theoretical ratio is (10/2)² = 25× in coherent power, but the search
        # also picks up the next-best off-alignment rotation in the opt-out's denominator, so
        # the realised on/off ratio is closer to ~9× (issue PRD documents "~10×"). Assert
        # ratio > 8 so a real regression of the rotation kernel — which would collapse the
        # ratio toward 1× — is caught, without flapping on FFT shape/windowing noise.
        system        = GPSL5I()
        sampling_freq = 10.24e6Hz
        prn           = 1
        fc            = get_code_frequency(system)
        samples_per_code = ceil(Int, get_code_length(system) / fc * sampling_freq)
        N = 10
        sec = get_secondary_code(system)

        primary = ComplexF32.(gen_code(samples_per_code, system, prn, sampling_freq, fc, 0.0))
        signal = ComplexF32[]
        for k in 0:N-1
            nh10_chip = Float32(GNSSSignals.secondary_value(sec, prn, k))
            append!(signal, nh10_chip .* primary)
        end

        # Acquire with use_secondary_code = true (default) and with use_secondary_code = false.
        plan_on = plan_acquire(system, sampling_freq, [prn];
            min_doppler_coverage = 5_000Hz, num_coherently_integrated_code_periods = N,
            num_noncoherent_accumulations = 1)
        plan_off = plan_acquire(system, sampling_freq, [prn];
            min_doppler_coverage = 5_000Hz, num_coherently_integrated_code_periods = N,
            num_noncoherent_accumulations = 1, use_secondary_code = false)

        results_on  = acquire!(plan_on,  signal, [prn]; store_power_bins = true)
        results_off = acquire!(plan_off, signal, [prn]; store_power_bins = true)

        peak_on  = maximum(results_on[1].power_bins)
        peak_off = maximum(results_off[1].power_bins)
        ratio = peak_on / peak_off
        @info "L5I 10ms peak ratio (on/off) = $ratio"
        @test ratio > 8
    end

    @testset "L5I rotation search reports the correct Doppler (no half-band fftshift offset)" begin
        # Regression for the double-fftshift bug shared by all sign-search kernels:
        # the per-column sub-block FFT fftshifts internally (circshift by N/2) AND
        # the result was scattered through fftshift_perm a second time. The two
        # compose to the identity, leaving the Doppler axis in raw FFT order — the
        # rotation search then recovered full peak *power* but at a Doppler bin
        # offset by exactly half the searched band. Detection and the gain-recovery
        # tests above pass either way (they don't pin the Doppler), so this asserts
        # the absolute Doppler value, which the bug got wrong.
        system        = GPSL5I()
        sampling_freq = 12e6Hz
        prn           = 1
        true_doppler  = 1000Hz        # on the 100 Hz Doppler grid at N_coh=10
        true_cp       = 1234.5

        (; signal) = generate_test_signal(system, prn;
            num_samples = 10 * 12000,
            doppler = true_doppler, code_phase = true_cp,
            sampling_freq, interm_freq = 0.0Hz, CN0 = 45)

        plan = plan_acquire(system, sampling_freq, [prn];
            num_coherently_integrated_code_periods = 10, num_noncoherent_accumulations = 1)
        @test plan.num_secondary_rotations > 1   # confirm the rotation search is active

        result = acquire!(plan, signal, prn; interm_freq = 0.0Hz)

        @test is_detected(result)
        @test result.code_phase ≈ true_cp atol = 1.0
        @test abs(result.carrier_doppler / 1Hz - ustrip(Hz, true_doppler)) < ustrip(Hz, step(plan.doppler_freqs))
    end

    @testset "L5I rotation search — correct Doppler/code-phase with non-zero code drift (N_nc>1)" begin
        # Locks the sign-search path's drift correction. `_apply_code_drift!` is
        # currently called per non-coherent step with a sorted-row buf; any future
        # refactor that moves the buf to raw-row order (e.g. fusing the kernel's
        # internal fftshift into a single output shift) must keep the drift
        # mapping consistent or this test will catch the regression.
        #
        # Parameters chosen so the per-row drift `round(f_d·m·T_coh·fs/f_c)`
        # rounds to non-zero at later steps: at f_c=L5=1.176 GHz, T_coh=10 ms,
        # fs=12 MHz, f_d=5000 Hz, the step-1 drift is ≈0.51 samples → rounds to 1;
        # step-3 ≈1.53 → 2. So the drift path is genuinely exercised.
        system        = GPSL5I()
        sampling_freq = 12e6Hz
        prn           = 1
        true_doppler  = 5000Hz
        true_cp       = 1234.5
        Ncoh = 10; Nnc = 4

        (; signal) = generate_test_signal(system, prn;
            num_samples = Nnc * Ncoh * 12000,
            doppler = true_doppler, code_phase = true_cp,
            sampling_freq, interm_freq = 0.0Hz, CN0 = 45)

        plan = plan_acquire(system, sampling_freq, [prn];
            num_coherently_integrated_code_periods = Ncoh,
            num_noncoherent_accumulations = Nnc)
        @test plan.num_secondary_rotations > 1
        @test plan.num_noncoherent_accumulations > 1   # multistep path → drift active

        result = acquire!(plan, signal, prn; interm_freq = 0.0Hz)

        @test is_detected(result)
        @test result.code_phase ≈ true_cp atol = 1.5
        @test abs(result.carrier_doppler / 1Hz - ustrip(Hz, true_doppler)) < ustrip(Hz, step(plan.doppler_freqs))
    end

    @testset "L5I rotation search — KNOWN BIAS at worst-case code-phase (block / NH10 misalignment)" begin
        # ──────────────────────────────────────────────────────────────────────
        # @test_broken: this test is *expected to fail* with the current rotation
        # search and documents a structural limitation rather than a fixable bug
        # in this kernel. If a future change makes it pass, that's a real
        # improvement and the @test_broken will flag itself as Unexpected Pass.
        # ──────────────────────────────────────────────────────────────────────
        #
        # Setup. L5I has L = 10 NH10 chips per primary code period (1 ms / 10230
        # chips). The rotation search assigns ONE NH10 chip per coherent
        # integration sub-block (= one primary code period of `samples_per_code`
        # samples). That mapping is exact only when the primary-code wrap (where
        # NH10 increments) happens at the block boundary — i.e. when
        # `code_phase == 0`.
        #
        # At any other code phase the wrap drifts INSIDE the block. The sample
        # where the wrap lands is
        #     W(cp) = round((code_length − cp) · fs / code_freq)
        # so for L5I at fs = 12 MHz and cp = 5115 chips the wrap lands at
        # sample W ≈ 6000 of 12000 — exactly 50% into each block. Each block
        # then carries ~51% of one NH10 chip and ~49% of the next. No discrete
        # rotation hypothesis matches that 51/49 mix.
        #
        # Two stacked consequences at this worst-case `cp`:
        #   1. Mid-block misalignment burns ~6.5 dB of coherent gain — empirically
        #      peak/N drops from ≈ 110 at cp=0 to ≈ 25 at cp=5115.
        #   2. The single-rotation Doppler spectrum at the peak code-phase column
        #      is near-flat (top ~7 bins all within 2% of each other). The
        #      argmax flips on noise within that flat region, biasing the
        #      reported Doppler by up to 7 bins from truth.
        #
        # The "best-rotation winner" replacement for cell-wise max was tested
        # (claude_scratch/best_rot_proto.jl) and produces BIT-IDENTICAL Doppler
        # errors to today's cell-wise max — the flatness lives inside the single
        # rotation's spectrum, not in the cross-rotation envelope.
        #
        # The only fix verified empirically is the long-code path
        # (acquire as if NH10 were baked into a 102 300-chip primary code,
        # N_coh = 1 long period, simple pilot). That gives Doppler exact at
        # every `cp` and ~10× the peak/N at the worst alignment, at the cost of
        # a 10× larger code-phase axis (≈ 1.6× per-PRN runtime).
        # See claude_scratch/secondary-code-acquisition-plan.md.
        system        = GPSL5I()
        sampling_freq = 12e6Hz
        prn           = 1
        true_doppler  = 1000Hz
        # cp = 5115 chips → wrap at exactly 50% of the receive block. This is
        # the deterministic worst case across the cp sweep. dop = 1000 Hz lands
        # on the 100 Hz Doppler grid so the assertion has no half-bin slop.
        true_cp = 5115.0

        (; signal) = generate_test_signal(system, prn;
            num_samples = 10 * 12000,
            doppler = true_doppler, code_phase = true_cp,
            sampling_freq, interm_freq = 0.0Hz, CN0 = 45)

        plan = plan_acquire(system, sampling_freq, [prn];
            num_coherently_integrated_code_periods = 10, num_noncoherent_accumulations = 1)
        @test plan.num_secondary_rotations > 1   # confirm the rotation search is active

        result = acquire!(plan, signal, prn; interm_freq = 0.0Hz)

        # Detection still works — peak/N is degraded but well above threshold.
        @test is_detected(result)
        # Code phase is unaffected (the bias is in the Doppler axis only).
        @test result.code_phase ≈ true_cp atol = 1.0
        # Doppler is the failing axis: empirically off by ≈ −700 Hz (7 bins).
        @test_broken abs(result.carrier_doppler / 1Hz - ustrip(Hz, true_doppler)) < ustrip(Hz, step(plan.doppler_freqs))
    end
end
