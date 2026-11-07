# test/doppler_passes.jl
# Doppler search split into narrow passes (issue #101): layout, grid stitching,
# sensitivity bound, and every search path running on a multi-pass plan.

# A single-grid plan with exactly `num_blocks` blocks: the smallest divisor
# reaching the coverage is `num_blocks` when the coverage asks for just under it.
function _single_grid_plan(system, sampling_freq, prns, num_blocks; kwargs...)
    N_coh = get(kwargs, :num_coherently_integrated_code_periods, 1)
    spc = ceil(Int, get_code_length(system) / get_code_frequency(system) * sampling_freq)
    bin_width = sampling_freq / spc
    coverage = ((num_blocks - 2 / N_coh) / 2 - 0.01) * bin_width
    plan = plan_acquire(system, sampling_freq, prns; kwargs...,
        min_doppler_coverage = coverage, max_doppler_passes = 1)
    @assert plan.num_blocks == num_blocks
    return plan
end

@testset "Doppler passes — layout and grid at 1.542 MHz" begin
    # 1542 = 2·3·257: the only divisor reaching ±7 kHz in one grid is 257.
    plan = plan_acquire(GPSL1CA(), 1.542e6Hz, [1])
    @test plan.num_blocks == 6
    @test plan.num_doppler_passes == 3
    @test plan.doppler_bins_per_pass == 5
    @test plan.pass_doppler_bins == 6
    # One uniform grid, symmetric about 0, reaching ±7 kHz.
    @test plan.doppler_freqs == (-7000.0:1000.0:7000.0) .* Hz
    # Pass k is centred on its middle bin and keeps its 5 central bins.
    @test plan.doppler_pass_offsets_hz == [-5000.0, 0.0, 5000.0]
    @test plan.pass_row_offset == 1     # sorted rows 2:6 of a pass's 6 (-3..2 kHz)

    # Opting out restores the single 257-bin grid.
    single = plan_acquire(GPSL1CA(), 1.542e6Hz, [1]; max_doppler_passes = 1)
    @test single.num_blocks == 257
    @test single.num_doppler_passes == 1
    @test single.doppler_bins_per_pass == length(single.doppler_freqs) == 257
    @test single.doppler_pass_offsets_hz == [0.0]

    # Common rates with a well-placed divisor keep their single grid.
    for fs in (2.048e6Hz, 5e6Hz, 16.368e6Hz)
        @test plan_acquire(GPSL1CA(), fs, [1]).num_doppler_passes == 1
    end

    @test_throws ArgumentError plan_acquire(GPSL1CA(), 1.542e6Hz, [1]; max_doppler_passes = 0)
end

@testset "Doppler grid labels match the FFT bins for an odd bin count" begin
    # A 257-bin grid used to be labelled from -cov/2 = -128.5 kHz, half a bin
    # below the FFT bins it holds: a signal at 0 Hz peaked loss-free but was
    # reported at -500 Hz.
    plan = plan_acquire(GPSL1CA(), 1.542e6Hz, [1]; max_doppler_passes = 1)
    @test first(plan.doppler_freqs) == -128_000.0Hz
    @test 0.0Hz in plan.doppler_freqs
    for doppler in (0Hz, 3000Hz, -2000Hz)
        (; signal) = generate_test_signal(GPSL1CA(), 1; sampling_freq = 1.542e6Hz,
            num_samples = 3084, doppler, code_phase = 400.3, CN0 = 90,
            interm_freq = 0.0Hz)
        @test acquire!(plan, signal, 1).carrier_doppler == doppler
    end
end

@testset "Doppler passes — each pass's rows equal a single grid shifted by its offset" begin
    # The stitched surface must be exactly the concatenation of single-grid
    # searches with the same num_blocks, each downconverted by its pass's
    # offset and trimmed to its central bins.
    function check_stitching(system, fs, prn, signal; kwargs...)
        plan = plan_acquire(system, fs, [prn]; kwargs...)
        @test plan.num_doppler_passes > 1
        stitched = acquire!(plan, signal, prn; store_power_bins = true).power_bins
        ref_plan = _single_grid_plan(system, fs, [prn], plan.num_blocks; kwargs...)
        K = plan.doppler_bins_per_pass
        rows = plan.pass_row_offset+1 : plan.pass_row_offset+K
        for (k, offset) in enumerate(plan.doppler_pass_offsets_hz)
            ref = acquire!(ref_plan, signal, prn; interm_freq = offset * Hz,
                store_power_bins = true)
            @test stitched[(k-1)*K+1 : k*K, :] == ref.power_bins[rows, :]
            @test ref_plan.doppler_freqs[rows] .+ offset * Hz ==
                  plan.doppler_freqs[(k-1)*K+1 : k*K]
        end
    end
    fs = 1.542e6Hz
    (; signal) = generate_test_signal(GPSL1CA(), 3; sampling_freq = fs,
        num_samples = 45 * 1542, doppler = 2300Hz, code_phase = 123.4, CN0 = 45,
        interm_freq = 0.0Hz, seed = 3)
    signal = ComplexF32.(signal)
    @testset "simple path" check_stitching(GPSL1CA(), fs, 3, signal)
    @testset "multistep" check_stitching(GPSL1CA(), fs, 3, signal;
        num_noncoherent_accumulations = 3)
    @testset "data-bit + bit-edge search" check_stitching(GPSL1CA(), fs, 3, signal;
        num_coherently_integrated_code_periods = 40, bit_edge_search_steps = 4)
    @testset "GlobalMean estimator" check_stitching(GPSL1CA(), fs, 3, signal;
        noise_estimator = GlobalMeanNoiseEstimator())

    # Secondary-code rotation search (GPS L5I NH10). 10.238 MHz → 10238 = 2·5119.
    fs5 = 10.238e6Hz
    (; signal) = generate_test_signal(GPSL5I(), 1; sampling_freq = fs5,
        num_samples = 11 * 10238, doppler = -1700Hz, code_phase = 5115.3, CN0 = 50,
        interm_freq = 0.0Hz, seed = 6)
    @testset "rotation search" check_stitching(GPSL5I(), fs5, 1, ComplexF32.(signal);
        num_coherently_integrated_code_periods = 10)
end

@testset "Doppler passes — sensitivity is bounded by an ideal rate's edge loss" begin
    # FM-DBZP loses sinc²(f·T_block) a Doppler f from the grid centre. An ideal
    # rate for ±7 kHz (16 blocks, e.g. 2.048 MHz) loses sinc²(7/16) = -2.9 dB at
    # its edge; the passes must not lose more anywhere in ±7 kHz, and must
    # report the nearest bin. Noise-free, at bin centres (no scalloping).
    fs = 1.542e6Hz
    plan = plan_acquire(GPSL1CA(), fs, [7])
    @test plan.num_doppler_passes > 1
    peak(doppler) = maximum(acquire!(plan, generate_test_signal(GPSL1CA(), 7;
            sampling_freq = fs, num_samples = 3084, doppler, code_phase = 400.3,
            CN0 = 90, interm_freq = 0.0Hz).signal, 7; store_power_bins = true).power_bins)
    p0 = peak(0Hz)
    reference_loss_db = 20 * log10(sinc(7 / 16))
    for doppler in (-7000:1000:7000) .* Hz
        @test 10 * log10(peak(doppler) / p0) >= reference_loss_db - 0.1
    end
    # Off-grid Dopplers land on the nearest bin, across pass boundaries too.
    for doppler in (-6400, -2600, -2400, 2400, 2600, 6900) .* Hz
        (; signal) = generate_test_signal(GPSL1CA(), 7; sampling_freq = fs,
            num_samples = 3084, doppler, code_phase = 400.3, CN0 = 90,
            interm_freq = 0.0Hz)
        @test acquire!(plan, signal, 7).carrier_doppler == round(doppler / 1000Hz) * 1000Hz
    end
end

@testset "Doppler passes — detection and estimates through the public API" begin
    fs = 1.542e6Hz
    prns = 1:8
    (; signal) = generate_test_signal(GPSL1CA(), 7; sampling_freq = fs,
        num_samples = 4 * 1542, doppler = 5300Hz, code_phase = 400.3, CN0 = 50,
        interm_freq = 120Hz, seed = 7)
    signal = ComplexF32.(signal)
    for N_nc in (1, 3)
        plan = plan_acquire(GPSL1CA(), fs, prns; num_noncoherent_accumulations = N_nc)
        @test plan.num_doppler_passes > 1
        results = acquire!(plan, signal, prns; interm_freq = 120Hz)
        r7 = results[findfirst(==(7), prns)]
        @test is_detected(r7)
        @test abs(r7.carrier_doppler - 5300Hz) <= 500Hz
        @test abs(r7.code_phase - 400.3) < 1
        # Same result whether the PRN is searched alone or alongside others
        # (different scratch slots / threads).
        alone = acquire!(plan, signal, 7; interm_freq = 120Hz)
        @test alone.carrier_doppler == r7.carrier_doppler
        @test alone.code_phase == r7.code_phase
        @test alone.peak_to_noise_ratio == r7.peak_to_noise_ratio
        @test get_num_cells(r7) == length(plan.doppler_freqs) * plan.samples_per_code
    end
end

@testset "Doppler passes — subsample interpolation reads neighbours across passes" begin
    # 2400 Hz peaks on 2 kHz, the last row of the middle pass; its upper
    # Doppler neighbour (3 kHz) comes from the next pass. The on-demand
    # recompute must read it from there and agree exactly with the stored run.
    fs = 1.542e6Hz
    plan = plan_acquire(GPSL1CA(), fs, [2])
    (; signal) = generate_test_signal(GPSL1CA(), 2; sampling_freq = fs,
        num_samples = 1542, doppler = 2400Hz, code_phase = 300.7, CN0 = 50,
        interm_freq = 0.0Hz, seed = 5)
    r_nostore = acquire!(plan, signal, 2; subsample_interpolation = true)
    r_store = acquire!(plan, signal, 2; subsample_interpolation = true,
        store_power_bins = true)
    peak_row = findfirst(==(2000.0Hz), plan.doppler_freqs)
    @test Acquisition._doppler_row_to_pass(plan, peak_row)[1] !=
          Acquisition._doppler_row_to_pass(plan, peak_row + 1)[1]
    @test r_nostore.code_phase == r_store.code_phase
    @test r_nostore.carrier_doppler == r_store.carrier_doppler
    @test 2000Hz < r_store.carrier_doppler < 3000Hz   # pulled toward the truth
end

@testset "Doppler passes — max_blocking_time bounds the stride across passes" begin
    plan = plan_acquire(GPSL1CA(), 1.542e6Hz, 1:4; max_blocking_time = 1u"s")
    @test plan.num_doppler_passes > 1
    @test plan.preempt_block_stride[] == plan.num_blocks * plan.num_doppler_passes
    (; signal) = generate_test_signal(GPSL1CA(), 1; sampling_freq = 1.542e6Hz,
        num_samples = 1542, CN0 = 45, interm_freq = 0.0Hz)
    yielding = plan_acquire(GPSL1CA(), 1.542e6Hz, 1:4; max_blocking_time = 1u"ns")
    @test yielding.preempt_block_stride[] == 1
    summary(r) = (r.prn, r.carrier_doppler, r.code_phase, r.peak_to_noise_ratio)
    @test summary.(acquire!(yielding, signal, 1:4)) == summary.(acquire!(plan, signal, 1:4))
end

@testset "downconversion keeps its phase accurate deep into a long signal" begin
    # Each Doppler pass downconverts the signal again, so `_downconvert!` uses a
    # phasor recurrence re-anchored with an exact sincos. It must match the exact
    # carrier far into the signal and at a large IF, where a Float32 phase
    # `index × step` (the former implementation) was off by up to 2e-2.
    fs = 2.048e6
    n = 40_960
    signal = ComplexF32.(cis.(range(0, 3, length = 50n)))
    buf = Vector{ComplexF32}(undef, n)
    for f in (1234.0, -6000.0, 250_000.0), seg_start in (1, 40n + 1)
        Acquisition._downconvert!(buf, signal, seg_start, n, f, fs)
        exact = [ComplexF64(signal[seg_start + i - 1]) *
                 cispi(-2 * mod(f / fs * (seg_start + i - 2), 1.0)) for i in 1:n]
        @test maximum(abs.(buf .- exact)) < 1e-6
    end
end
