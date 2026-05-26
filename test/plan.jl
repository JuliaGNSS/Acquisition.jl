# test/plan.jl
# Tests for plan_acquire and PRN FFT precomputation

@testset "plan_acquire parameters — GPS L1 at 2.048 MHz" begin
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    plan = plan_acquire(system, sampling_freq, [1];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 1, bit_edge_search_steps = 1, num_noncoherent_accumulations = 1)

    @test plan.samples_per_code == 2048      # ceil(1023/1.023e6 * 2.048e6)
    @test plan.num_blocks == 32              # smallest divisor of 2048 >= ceil(20000/1000 + 2/1)=22 → 32
    @test plan.block_size == 64              # 2048÷32
    @test plan.num_coherently_integrated_code_periods == 1
    @test plan.num_data_bits == 1            # pilot/short: no data bit search
    @test plan.bit_edge_search_steps == 1
    @test plan.num_noncoherent_accumulations == 1
    @test length(plan.doppler_freqs) == 32   # num_doppler_bins = 1*32
    @test step(plan.doppler_freqs) ≈ 1000Hz  # doppler_bin_spacing_hz = 32000/32
    @test first(plan.doppler_freqs) ≈ -16000Hz  # -doppler_coverage_hz/2
    # Both ends of the grid must reach the requested coverage
    @test last(plan.doppler_freqs)  >= 10_000Hz
    @test first(plan.doppler_freqs) <= -10_000Hz
end

@testset "plan_acquire parameters — GPS L1 multi-ms" begin
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    # num_coherently_integrated_code_periods=40 → num_data_bits = 40÷20 = 2 data bits
    plan = plan_acquire(system, sampling_freq, [1];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 40, bit_edge_search_steps = 4, num_noncoherent_accumulations = 5)

    @test plan.samples_per_code == 2048
    @test plan.num_blocks == 32              # smallest divisor of 2048 >= ceil(20000/1000 + 2/40)=21 → 32
    @test plan.num_coherently_integrated_code_periods == 40
    @test plan.num_data_bits == 2            # 40 code periods / 20 code periods per GPS bit
    @test plan.bit_edge_search_steps == 4
    @test plan.num_noncoherent_accumulations == 5
    @test length(plan.doppler_freqs) == 1280 # num_doppler_bins = 40*32
    @test step(plan.doppler_freqs) ≈ 25Hz    # 32000/1280
end

@testset "plan_acquire validation errors" begin
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    # num_coherently_integrated_code_periods >= bit_period_codes (20) but not divisible by it
    @test_throws ArgumentError plan_acquire(system, sampling_freq, [1];
        num_coherently_integrated_code_periods = 30)
    # bit_edge_search_steps does not divide bit_period_codes (20)
    @test_throws ArgumentError plan_acquire(system, sampling_freq, [1];
        num_coherently_integrated_code_periods = 20, bit_edge_search_steps = 3)
    # bit_edge_search_steps > 1 only makes sense when num_coherently_integrated_code_periods > 1
    @test_throws ArgumentError plan_acquire(system, sampling_freq, [1];
        num_coherently_integrated_code_periods = 1, bit_edge_search_steps = 2)
    # Other basic guard checks
    @test_throws ArgumentError plan_acquire(system, sampling_freq, [1];
        num_noncoherent_accumulations = 0)
    @test_throws ArgumentError plan_acquire(system, sampling_freq, [1];
        num_coherently_integrated_code_periods = 0)
    @test_throws ArgumentError plan_acquire(system, sampling_freq, [1];
        bit_edge_search_steps = 0)
    # min_doppler_coverage > sampling_freq/2 → no valid divisor → ArgumentError
    @test_throws ArgumentError plan_acquire(system, sampling_freq, [1];
        min_doppler_coverage = 1.1e6Hz)
end

@testset "PRN FFT precomputation" begin
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    plan = plan_acquire(system, sampling_freq, [1, 2];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 1)
    double_block_size = 2 * plan.block_size

    @test haskey(plan.prn_conj_ffts, 1)
    @test haskey(plan.prn_conj_ffts, 2)
    @test size(plan.prn_conj_ffts[1]) == (double_block_size, plan.num_blocks)  # (128, 32): double_block_size=2*64, num_blocks=32
    @test size(plan.prn_conj_ffts[2]) == (double_block_size, plan.num_blocks)

    # PRNs 1 and 2 have different codes → different FFTs
    @test plan.prn_conj_ffts[1] != plan.prn_conj_ffts[2]

    # Each column should be conj(FFT) of a zero-padded block_size-length block.
    # Verify by manually computing PRN 1 block 0 FFT and comparing.
    code = gen_code(plan.samples_per_code, system, 1, sampling_freq, get_code_frequency(system), 0.0)
    prn_block0 = zeros(ComplexF32, double_block_size)
    prn_block0[1:plan.block_size] .= ComplexF32.(code[1:plan.block_size])
    expected_fft = fft(prn_block0)
    @test plan.prn_conj_ffts[1][:, 1] ≈ conj.(expected_fft) atol = 1e-3
end

@testset "Sampling frequency compatibility — num_blocks uses smallest divisor of samples_per_code" begin
    # Previously failing frequencies where nextpow(2, min_blocks) did not divide samples_per_code.
    # Verifies that plan_acquire picks a valid num_blocks and that acquire
    # detects the signal with correct code phase at each sampling frequency.
    # min_num_blocks = ceil(2*5000/1000 + 2/5) = 11 (with N_coh=5, min_doppler_coverage=5000Hz).
    cases = [
        (2.048e6Hz, 16, 128),    # smallest divisor of 2048 >= 11
        (5e6Hz,     20, 250),    # divisors of 5000: …,10,20,…; smallest >= 11 is 20
        (10e6Hz,    16, 625),    # divisors of 10000: …,10,16,20,…; smallest >= 11 is 16
        (16.368e6Hz, 11, 1488),  # 11 itself is a divisor (16368 = 2⁴·3·11·31)
        (25e6Hz,    20, 1250),   # divisors of 25000: …,10,20,25,…; smallest >= 11 is 20
    ]
    system      = GPSL1CA()
    prn         = 1
    code_phase  = 110.613261
    doppler_hz  = 1000Hz
    num_periods = 5

    for (sampling_freq, expected_num_blocks, expected_block_size) in cases
        @testset "sampling_freq=$(ustrip(Hz, sampling_freq) / 1e6) MHz" begin
            samples_per_code = ceil(Int, get_code_length(system) / ustrip(Hz, get_code_frequency(system)) * ustrip(Hz, sampling_freq))
            plan = plan_acquire(system, sampling_freq, [prn];
                min_doppler_coverage = 5_000Hz, num_coherently_integrated_code_periods = num_periods, num_noncoherent_accumulations = 1)

            @test plan.num_blocks * plan.block_size == samples_per_code
            @test plan.num_blocks  == expected_num_blocks
            @test plan.block_size  == expected_block_size

            (; signal) = generate_test_signal(system, prn;
                num_samples   = num_periods * samples_per_code,
                doppler       = doppler_hz,
                code_phase    = code_phase,
                sampling_freq = sampling_freq,
                interm_freq   = 0.0Hz,
                CN0           = 45,
                seed          = 42)

            results = acquire(system, ComplexF32.(signal), sampling_freq, [prn];
                interm_freq = 0.0Hz, min_doppler_coverage = 5_000Hz,
                num_coherently_integrated_code_periods = num_periods, num_noncoherent_accumulations = 1)

            @test is_detected(only(results))
            @test abs(only(results).code_phase - code_phase) < 1.0
        end
    end
end

@testset "plan_acquire does not hold plan-level scratch duplicates" begin
    # Regression-lock: AcquisitionScratch owns all per-thread scratch buffers.
    # Any field re-added to AcquisitionPlan that duplicates an AcquisitionScratch
    # field re-introduces the per-plan RAM waste #60 removed.
    plan = plan_acquire(GPSL1CA(), 2.048e6Hz, [1];
        num_coherently_integrated_code_periods = 1)
    duplicated_fields = (
        :coherent_integration_matrix,
        :noncoherent_integration_max_buf,
        :noncoherent_integration_buf,
        :sub_block_ffts,
        :col_buf,
        :col_fftshift_buf,
        :row_buf,
        :row_shift_buf,
        :double_block_buf,
        :corr_buf,
        :sig_buf,
        :col_sums_buf,
    )
    for f in duplicated_fields
        @test !hasfield(typeof(plan), f)
    end

    # _default_scratch names the "thread 1 is the ambient scratch" convention.
    scratch = Acquisition._default_scratch(plan)
    @test scratch === plan.thread_scratch[1]
    @test scratch isa Acquisition.AcquisitionScratch
    for f in duplicated_fields
        @test hasfield(typeof(scratch), f)
    end
end

@testset "result_buffers are lazy — nothing unless store_power_bins=true" begin
    # Default acquire! (store_power_bins=false) must not allocate any per-PRN
    # result matrix. Only when the caller opts in does the slot fill in, and once
    # filled it stays cached for the next call (reuse contract).
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    plan = plan_acquire(system, sampling_freq, [1, 2]; fft_flag = FFTW.ESTIMATE)
    @test all(b -> b === nothing, plan.result_buffers)

    (; signal) = generate_test_signal(system, 1;
        num_samples = plan.samples_per_code, sampling_freq = sampling_freq,
        interm_freq = 0.0Hz, CN0 = 45)
    sig = ComplexF32.(signal)

    # store_power_bins=false leaves the slots empty.
    acquire!(plan, sig, [1, 2]; interm_freq = 0.0Hz, store_power_bins = false)
    @test all(b -> b === nothing, plan.result_buffers)

    # First store_power_bins=true call allocates and caches.
    results = acquire!(plan, sig, [1, 2]; interm_freq = 0.0Hz, store_power_bins = true)
    @test all(b -> b isa Matrix{Float32}, plan.result_buffers)
    @test results[1].power_bins === plan.result_buffers[1]

    # Reuse contract: the second call must reuse the same buffer object.
    cached = plan.result_buffers[1]
    acquire!(plan, sig, [1, 2]; interm_freq = 0.0Hz, store_power_bins = true)
    @test plan.result_buffers[1] === cached
end

@testset "sign-search scratch is 0x0 when the simple path is provably taken" begin
    # The router in _accumulate_noncoherent_integration_step! takes the simple/pilot
    # path when num_data_bits == 1 && bit_edge_search_steps == 1. In that regime
    # noncoherent_integration_max_buf and sub_block_ffts are never read, so the
    # plan should allocate them as 0x0 sentinels per thread.
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz

    simple_plan = plan_acquire(system, sampling_freq, [1];
        num_coherently_integrated_code_periods = 1, bit_edge_search_steps = 1)
    simple_scratch = Acquisition._default_scratch(simple_plan)
    @test simple_plan.num_data_bits == 1
    @test simple_plan.bit_edge_search_steps == 1
    @test size(simple_scratch.noncoherent_integration_max_buf) == (0, 0)
    @test size(simple_scratch.sub_block_ffts) == (0, 0)
    # And every thread's slot matches — not just thread 1.
    for t_scratch in simple_plan.thread_scratch
        @test size(t_scratch.noncoherent_integration_max_buf) == (0, 0)
        @test size(t_scratch.sub_block_ffts) == (0, 0)
    end

    # Sign-search path: bit_edge_search_steps > 1 makes the kernel actually read
    # these buffers, so they must be full-sized.
    search_plan = plan_acquire(system, sampling_freq, [1];
        min_doppler_coverage = 10_000Hz,
        num_coherently_integrated_code_periods = 40, bit_edge_search_steps = 4)
    search_scratch = Acquisition._default_scratch(search_plan)
    expected = (length(search_plan.doppler_freqs), search_plan.samples_per_code)
    @test size(search_scratch.noncoherent_integration_max_buf) == expected
    @test size(search_scratch.sub_block_ffts) ==
        (length(search_plan.doppler_freqs), search_plan.num_data_bits)
end

@testset "noncoherent_integration_buf is 0x0 when fused FFT+abs2+fftshift kernel runs" begin
    # At N_nc==1 with the simple/pilot path, the fused kernel writes |FFT|²
    # straight into the accumulator with fftshift permutation and never touches
    # `noncoherent_integration_buf`. The buf can be dropped per-thread; sign-search
    # path and multistep (N_nc>1) still need it full-sized.
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz

    fused_plan = plan_acquire(system, sampling_freq, [1];
        num_coherently_integrated_code_periods = 1, bit_edge_search_steps = 1,
        num_noncoherent_accumulations = 1)
    fused_scratch = Acquisition._default_scratch(fused_plan)
    @test size(fused_scratch.noncoherent_integration_buf) == (0, 0)
    for t_scratch in fused_plan.thread_scratch
        @test size(t_scratch.noncoherent_integration_buf) == (0, 0)
    end

    # Sign-search at N_nc=1 still needs the buf.
    sign_search_plan = plan_acquire(system, sampling_freq, [1];
        min_doppler_coverage = 10_000Hz,
        num_coherently_integrated_code_periods = 40, bit_edge_search_steps = 4)
    @test size(Acquisition._default_scratch(sign_search_plan).noncoherent_integration_buf) ==
        (length(sign_search_plan.doppler_freqs), sign_search_plan.samples_per_code)

    # Multistep (N_nc>1) needs the buf too, even on the simple path.
    multi_plan = plan_acquire(system, sampling_freq, [1];
        num_coherently_integrated_code_periods = 1, num_noncoherent_accumulations = 2)
    @test size(Acquisition._default_scratch(multi_plan).noncoherent_integration_buf) ==
        (length(multi_plan.doppler_freqs), multi_plan.samples_per_code)
end

@testset "sequential N_nc=1 layout: empty per-PRN matrices, full per-thread accumulator" begin
    # When num_noncoherent_accumulations == 1 the per-PRN noncoherent matrices
    # collapse into a single per-thread accumulator reused across PRNs. Layout
    # should be: empty Vector{Matrix{Float32}} on the plan, full-sized
    # accumulator per thread.
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    prns = collect(1:4)

    seq_plan = plan_acquire(system, sampling_freq, prns;
        num_coherently_integrated_code_periods = 1, num_noncoherent_accumulations = 1)
    seq_scratch = Acquisition._default_scratch(seq_plan)
    @test length(seq_plan.noncoherent_integration_matrices) == 0
    expected = (length(seq_plan.doppler_freqs), seq_plan.samples_per_code)
    @test size(seq_scratch.noncoherent_integration_accumulator) == expected
    for t_scratch in seq_plan.thread_scratch
        @test size(t_scratch.noncoherent_integration_accumulator) == expected
    end

    # N_nc>1 path keeps the per-PRN matrices and uses no accumulator.
    multi_plan = plan_acquire(system, sampling_freq, prns;
        num_coherently_integrated_code_periods = 1, num_noncoherent_accumulations = 2)
    multi_scratch = Acquisition._default_scratch(multi_plan)
    @test length(multi_plan.noncoherent_integration_matrices) == length(prns)
    for nim in multi_plan.noncoherent_integration_matrices
        @test size(nim) == (length(multi_plan.doppler_freqs), multi_plan.samples_per_code)
    end
    @test size(multi_scratch.noncoherent_integration_accumulator) == (0, 0)
end
