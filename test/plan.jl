# test/plan.jl
# Tests for plan_acquire and PRN FFT precomputation

@testset "plan_acquire parameters — GPS L1 at 2.048 MHz" begin
    system = GPSL1()
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
    system = GPSL1()
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
    system = GPSL1()
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
    system = GPSL1()
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
    system      = GPSL1()
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
