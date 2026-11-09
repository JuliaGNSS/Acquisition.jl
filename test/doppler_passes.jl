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
    @test plan.pass_fft_bins == 6
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

@testset "Doppler passes — only on sparse-divisor rates" begin
    # Passes only replace a single grid that needs far more blocks than the
    # coverage asks for. A grid with a few spare blocks is kept even where
    # passes would be faster: its spare blocks make Dopplers near 0 Hz
    # nearly loss-free.
    for (fs, coverage) in ((1.5e6Hz, 7000Hz),      # 20 blocks for 16
                           (2.048e6Hz, 10_000Hz),  # 32 for 22
                           (4e6Hz, 25_000Hz),      # 80 for 52
                           (8.184e6Hz, 7000Hz)),   # 22 for 16
        N_coh in (1, 10)
        plan = plan_acquire(GPSL1CA(), fs, [1]; min_doppler_coverage = coverage,
            num_coherently_integrated_code_periods = N_coh)
        @test plan.num_doppler_passes == 1
        @test length(plan.doppler_freqs) == plan.pass_fft_bins
    end
    # Sparse divisors: 1023 = 3·11·31 and 3069 = 3²·11·31 need 31 blocks for 16.
    for fs in (1.023e6Hz, 3.069e6Hz)
        @test plan_acquire(GPSL1CA(), fs, [1]).num_doppler_passes > 1
    end
end

@testset "Doppler passes — the stitched grid never spans more than fs" begin
    # The pass count rounds the grid up to whole passes, which can push it past
    # `N_coh · samples_per_code` bins (the sampling frequency) even when the
    # coverage itself fits; such a layout would search aliases. At 5 samples
    # per code and ±2 kHz, 2 passes of 3 bins (6 > 5) were picked before.
    @test Acquisition._doppler_search_layout(5, 1000.0, 2000.0, 1, 16, 32) === nothing
    for samples_per_code in (5, 6, 7, 31, 62, 257, 1542), N_coh in (1, 5),
        coverage_frac in 0.30:0.02:0.5
        layout = Acquisition._doppler_search_layout(samples_per_code, 1000.0,
            coverage_frac * samples_per_code * 1000.0, N_coh, 16, 32)
        layout === nothing && continue
        @test Acquisition._grid_bins(layout) <= N_coh * samples_per_code
    end
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
    (; signal) = generate_test_signal(GPSL5I(), 1; sampling_freq = fs5,
        num_samples = 21 * 10238, doppler = -1700Hz, code_phase = 5115.3, CN0 = 50,
        interm_freq = 0.0Hz, seed = 6)
    @testset "rotation × data bits × bit edges" check_stitching(GPSL5I(), fs5, 1,
        ComplexF32.(signal); num_coherently_integrated_code_periods = 20,
        bit_edge_search_steps = 2)
end

@testset "Doppler passes — multistep code drift follows each bin's true Doppler" begin
    # Between noncoherent steps the code phase drifts by the code Doppler of the
    # bin's TRUE Doppler (pass offset + local bin). A single grid shifted by
    # `interm_freq` only knows the local Doppler, so at the outer passes it
    # under-shifts (L5I at 10.238 MHz, 10 ms steps: one sample from ~5.8 kHz).
    # The reference therefore runs each step as its own single-grid search and
    # applies the drift of the true Doppler itself: step m's row r moves by
    # `round(f_r · m · T_coh · fs / f_carrier)` samples, wrapping within each
    # rotation's `samples_per_code`-wide slice.
    function check_multistep_drift(system, fs, prn, signal; N_nc, kwargs...)
        plan = plan_acquire(system, fs, [prn]; num_noncoherent_accumulations = N_nc,
            kwargs...)
        @test plan.num_doppler_passes > 1
        stitched = acquire!(plan, signal, prn; store_power_bins = true).power_bins
        ref_plan = _single_grid_plan(system, fs, [prn], plan.num_blocks; kwargs...)
        spc = plan.samples_per_code
        step_len = plan.num_coherently_integrated_code_periods * spc
        K = plan.doppler_bins_per_pass
        rows = plan.pass_row_offset+1 : plan.pass_row_offset+K
        expected = zeros(Float32, size(stitched))
        max_shift = 0
        for (k, offset) in enumerate(plan.doppler_pass_offsets_hz)
            dest = (k-1)*K+1 : k*K
            shifts = zeros(Int, K)
            for m in 0:N_nc-1
                segment = signal[m*step_len+1 : end]
                step = acquire!(ref_plan, segment, prn; interm_freq = offset * Hz,
                    store_power_bins = true).power_bins[rows, :]
                Acquisition._fill_code_drift_shifts!(shifts, plan, m,
                    Acquisition._pass_rows(plan, k))
                for i in 1:K, s in 0:size(step, 2)÷spc-1
                    slice = s*spc+1 : (s+1)*spc
                    expected[dest[i], slice] .+= circshift(step[i, slice], shifts[i])
                end
                max_shift = max(max_shift, maximum(min.(shifts, spc .- shifts)))
            end
        end
        @test max_shift >= 1          # the outer passes really drift
        @test stitched ≈ expected rtol = 1e-5
    end
    fs5 = 10.238e6Hz
    (; signal) = generate_test_signal(GPSL5I(), 1; sampling_freq = fs5,
        num_samples = 21 * 10238, doppler = 6400Hz, code_phase = 5115.3, CN0 = 50,
        interm_freq = 0.0Hz, seed = 6)
    signal = ComplexF32.(signal)
    @testset "rotation search" check_multistep_drift(GPSL5I(), fs5, 1, signal;
        N_nc = 2, num_coherently_integrated_code_periods = 10)
    (; signal) = generate_test_signal(GPSL5I(), 1; sampling_freq = fs5,
        num_samples = 41 * 10238, doppler = 6400Hz, code_phase = 5115.3, CN0 = 50,
        interm_freq = 0.0Hz, seed = 6)
    @testset "rotation × data bits × bit edges" check_multistep_drift(GPSL5I(), fs5, 1,
        ComplexF32.(signal); N_nc = 2, num_coherently_integrated_code_periods = 20,
        bit_edge_search_steps = 2)
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

@testset "Doppler passes — no subsample interpolation across the grid edge" begin
    # A single untrimmed grid spans a full alias period, so its first and last
    # rows are neighbours. A stitched grid stops at the coverage edge: the row
    # past +7 kHz is not -7 kHz, so a peak on an edge row keeps its bin Doppler.
    fs = 1.542e6Hz
    plan = plan_acquire(GPSL1CA(), fs, [4])
    n = length(plan.doppler_freqs)
    @test Acquisition._doppler_neighbour_rows(plan, 1) == (0, 2)
    @test Acquisition._doppler_neighbour_rows(plan, n) == (n - 1, 0)
    @test Acquisition._doppler_neighbour_rows(plan, 5) == (4, 6)
    single = plan_acquire(GPSL1CA(), 2.048e6Hz, [4])
    m = length(single.doppler_freqs)
    @test Acquisition._doppler_neighbour_rows(single, 1) == (m, 2)
    @test Acquisition._doppler_neighbour_rows(single, m) == (m - 1, 1)

    for (doppler, edge) in ((7300Hz, 7000Hz), (-7300Hz, -7000Hz))
        (; signal) = generate_test_signal(GPSL1CA(), 4; sampling_freq = fs,
            num_samples = 1542, doppler, code_phase = 300.7, CN0 = 60,
            interm_freq = 0.0Hz, seed = 9)
        for store_power_bins in (false, true)
            r = acquire!(plan, signal, 4; subsample_interpolation = true, store_power_bins)
            @test r.carrier_doppler == edge
        end
    end
end

@testset "Doppler passes — the convenience wrappers forward max_doppler_passes" begin
    fs = 1.542e6Hz
    (; signal) = generate_test_signal(GPSL1CA(), 1; sampling_freq = fs,
        num_samples = 1542, doppler = 1000Hz, code_phase = 100.2, CN0 = 50,
        interm_freq = 0.0Hz, seed = 2)
    grid_rows(r) = size(r.power_bins, 1)
    @test grid_rows(acquire(GPSL1CA(), signal, fs, 1; store_power_bins = true)) == 15
    @test grid_rows(acquire(GPSL1CA(), signal, fs, 1; store_power_bins = true,
        max_doppler_passes = 1)) == 257
    @test grid_rows(only(collect(acquire_stream(GPSL1CA(), signal, fs, [1];
        store_power_bins = true, max_doppler_passes = 1)))) == 257
    rows = acquire_stream(GPSL1CA(), signal, fs, [1]; store_power_bins = true,
            max_doppler_passes = 1) do results
        grid_rows(only(collect(results)))
    end
    @test rows == 257
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
