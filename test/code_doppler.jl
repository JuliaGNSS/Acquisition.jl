# Sparse compensation of code Doppler inside a coherent integration window.
@testset "Code Doppler — plan and recommendation agree" begin
    system = GPSL5I()
    fs = 10.24e6Hz
    for N_coh in (10, 20)
        plan = plan_acquire(system, fs, [1];
            num_coherently_integrated_code_periods = N_coh, use_secondary_code = false)
        @test plan.num_doppler_passes > 1
        @test first(plan.doppler_freqs) <= -7000Hz
        @test last(plan.doppler_freqs) >= 7000Hz
        # Include half a bin beyond the most distant kept bin.
        residual = (plan.doppler_bins_per_pass ÷ 2 + 0.5) *
            get_code_center_frequency_ratio(system)
        @test residual <= Acquisition._MAX_CODE_DOPPLER_DRIFT
        @test any(!iszero, plan.code_doppler_shifts_per_period)
        rec = only(recommend_sampling_freqs(system; fs_min = fs, fs_max = fs,
            num_prns = 1, num_coherently_integrated_code_periods = N_coh))
        @test rec.num_blocks == plan.num_blocks
        @test rec.num_doppler_passes == plan.num_doppler_passes
        @test rec.doppler_bins_per_pass == plan.doppler_bins_per_pass
        @test rec.cost == Acquisition._search_cost(plan.samples_per_code,
            plan.num_blocks, N_coh, plan.num_doppler_passes, 1, true)
        @test rec.cost > Acquisition._search_cost(plan.samples_per_code,
            plan.num_blocks, N_coh, plan.num_doppler_passes, 1)
    end
    single = plan_acquire(system, fs, [1];
        num_coherently_integrated_code_periods = 20, max_doppler_passes = 1)
    @test single.num_doppler_passes == 1
    @test single.code_doppler_shifts_per_period == [0.0]
    @test Acquisition._search_cost(single.samples_per_code, single.num_blocks, 20, 1, 1, true) ==
        Acquisition._search_cost(single.samples_per_code, single.num_blocks, 20, 1, 1)
    # A cap too small for the tolerance keeps the original layout.
    limited = plan_acquire(system, fs, [1];
        num_coherently_integrated_code_periods = 20, max_doppler_passes = 2)
    @test limited.num_doppler_passes <= 2

    # A wide search must compensate automatically, without knowing its pass
    # count in advance. The old default cap of 16 disabled it at ±15 kHz.
    wide = plan_acquire(system, fs, [1]; min_doppler_coverage = 15000Hz,
        num_coherently_integrated_code_periods = 20, use_secondary_code = false)
    @test wide.num_doppler_passes > 16
    @test any(!iszero, wide.code_doppler_shifts_per_period)
    @test (wide.doppler_bins_per_pass ÷ 2 + 0.5) *
        get_code_center_frequency_ratio(system) <= Acquisition._MAX_CODE_DOPPLER_DRIFT
    wide_rec = only(recommend_sampling_freqs(system; fs_min = fs, fs_max = fs,
        min_doppler_coverage = 15000Hz, num_prns = 1,
        num_coherently_integrated_code_periods = 20))
    @test wide_rec.num_blocks == wide.num_blocks
    @test wide_rec.num_doppler_passes == wide.num_doppler_passes
    @test wide_rec.doppler_bins_per_pass == wide.doppler_bins_per_pass
    capped = plan_acquire(system, fs, [1]; min_doppler_coverage = 15000Hz,
        num_coherently_integrated_code_periods = 20, max_doppler_passes = 16,
        use_secondary_code = false)
    @test capped.num_doppler_passes <= 16
end

@testset "Code Doppler — low-drift L1 retains the original FFTs" begin
    for N_coh in (10, 20)
        plan = plan_acquire(GPSL1CA(), 5e6Hz, [1];
            num_coherently_integrated_code_periods = N_coh)
        legacy = Acquisition._doppler_search_layout(5000, 1000.0, 7000.0, N_coh, 16, 1)
        @test plan.num_blocks == legacy.num_blocks
        @test plan.num_doppler_passes == legacy.num_passes == 1
        @test plan.doppler_bins_per_pass == legacy.bins_per_pass
        @test plan.code_doppler_shifts_per_period == [0.0]
        signal = randn(ComplexF32, N_coh * 5000)
        Acquisition._fill_signal_block_ffts!(plan, signal, length(signal), 0.0, 5e6)
        expected = similar(plan.signal_block_ffts)
        scratch = Acquisition._default_scratch(plan)
        Acquisition._precompute_signal_block_ffts!(expected, signal, 5000,
            plan.num_blocks, plan.block_size, N_coh, scratch.double_block_buf,
            plan.double_block_fft_plan)
        @test plan.signal_block_ffts == expected
    end
end

@testset "Code Doppler — noise-free L5 sensitivity at both Doppler edges" for fs in
    (10.24e6Hz, 12e6Hz)
    # The first rate uses unpadded FFTs, the second pads 12000 to 12288.
    system = GPSL5I()
    plan = plan_acquire(system, fs, [1];
        num_coherently_integrated_code_periods = 20, use_secondary_code = false)
    shifts = copy(plan.code_doppler_shifts_per_period)
    n = 20 * plan.samples_per_code
    peak(result) = result.noise_power * result.peak_to_noise_ratio
    for doppler in (-7000Hz, 7000Hz)
        losses = Float64[]
        uncompensated_losses = Float64[]
        carrier = cis.(2π .* (0:n-1) .* (doppler / fs))
        for code_phase in 5115.0625:0.125:5115.9375
            # Sample the primary code directly to isolate code drift from
            # the secondary-code and data-bit hypotheses tested elsewhere.
            primary(rate) = [system.codes[mod(floor(Int,
                code_phase + t * ustrip(rate / fs)), get_code_length(system)) + 1, 1]
                for t in 0:n-1]
            reference = carrier .* primary(get_code_frequency(system))
            signal = carrier .* primary(get_code_frequency(system) +
                doppler * get_code_center_frequency_ratio(system))
            # Keep the carrier layout identical in all three measurements.
            fill!(plan.code_doppler_shifts_per_period, 0.0)
            reference_peak = peak(acquire!(plan, reference, 1))
            uncompensated = acquire!(plan, signal, 1)
            push!(uncompensated_losses, 10log10(peak(uncompensated) / reference_peak))
            plan.code_doppler_shifts_per_period .= shifts
            compensated = acquire!(plan, signal, 1)
            @test compensated.carrier_doppler == doppler
            push!(losses, 10log10(peak(compensated) / reference_peak))
        end
        # At ~one sample/chip, fractional delay cannot undo aliasing. Bound
        # both the average and worst phase loss, and require an improvement
        # over the same search without compensation (about 1.2 chips drift).
        @test minimum(losses) > -2.5
        @test sum(losses) / length(losses) > -0.7
        @test sum(losses) > sum(uncompensated_losses) + length(losses) * 0.5
        @test minimum(losses) > minimum(uncompensated_losses) + 1.0
    end
end

@testset "Cached and uncached code delays equal time-domain integer delays" begin
    # Check signed-frequency reconstruction independently of the recurrence:
    # an integer code advance must become a circular time-domain delay of the
    # double block, including when that double block has a zero-padded tail.
    signal = randn(ComplexF32, 48)
    for fft_size in (16, 24), shift in (-1.0, 1.0), use_cache in (false, true)
        fft_plan = plan_fft!(zeros(ComplexF32, fft_size); threads = 1)
        cache = zeros(ComplexF32, fft_size, 6)
        phasors = use_cache ? Acquisition._code_doppler_phasors(fft_size, 3, shift) : nothing
        Acquisition._precompute_signal_block_ffts!(cache, signal, 16, 2, 8, 3,
            zeros(ComplexF32, fft_size), fft_plan, shift, phasors)
        for block in 0:5
            period = block ÷ 2
            padded = zeros(ComplexF32, fft_size)
            for i in 0:15
                padded[i+1] = signal[mod(block * 8 + i, length(signal)) + 1]
            end
            expected = fft(circshift(padded, period * Int(shift)))
            @test cache[:, block+1] ≈ expected rtol = 2e-6
        end
    end
    @test Acquisition._code_doppler_phasors(16, 3, 0.0) === nothing
    @test Acquisition._code_doppler_phasors(16, 1, 1.0) === nothing
end

@testset "Parallel batches avoid excessive serial signal preparation" begin
    if Acquisition._num_prn_workers(32) >= 3
        # This batch-sized plan is also used to search single-PRN subsets.
        # Treating all 32 PRNs as serial work chose eleven passes, almost
        # doubling the subset search time. Account for parallel PRN workers.
        plan = plan_acquire(GPSL1CA(), 2.048e6Hz, collect(1:32);
            min_doppler_coverage = 10000Hz,
            num_coherently_integrated_code_periods = 20)
        @test 1 < plan.num_doppler_passes < 11
        @test any(!iszero, plan.code_doppler_shifts_per_period)
    end
end
