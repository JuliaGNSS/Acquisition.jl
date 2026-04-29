# test/recommend_sampling_freq.jl
# Tests for recommend_sampling_freqs

@testset "recommend_sampling_freqs — code_length / code_freq form" begin
    # Worked example from the docs: fs=40 MHz, code 36 MHz × 10000 chips
    # has samples_per_code=11112 = 2³·3·463, which violates max_prime=7.
    # The recommender should skip 40 MHz and find smooth alternatives above it.
    rs = recommend_sampling_freqs(
        10_000, 36e6Hz;
        fs_min = 36e6Hz,
        fs_max = 40e6Hz,
        min_doppler_coverage = 10_000Hz,
        num_coherently_integrated_code_periods = 36,
        num_alternatives = 5,
        max_prime = 7,
        sort_by = :cost,
        fs_step = 1000Hz,
    )

    @test !isempty(rs)
    @test length(rs) <= 5
    @test eltype(rs) == SamplingFreqRecommendation

    # Every returned candidate must satisfy the smoothness budget on all three
    # FFT sizes.
    for r in rs
        @test r.inner_max_prime <= 7
        @test r.num_doppler_bins_max_prime <= 7
        # Sanity: divisibility, geometry, and coverage
        @test r.samples_per_code % r.num_blocks == 0
        @test r.block_size == r.samples_per_code ÷ r.num_blocks
        @test r.inner_fft_size == 2 * r.block_size
        @test r.num_doppler_bins == 36 * r.num_blocks
        @test ustrip(Hz, r.doppler_coverage) >= 2 * 10_000  # ±10 kHz minimum
    end

    # :cost sort must be non-decreasing in cost
    costs = [r.cost for r in rs]
    @test issorted(costs)
end

@testset "recommend_sampling_freqs — :smoothness sort" begin
    rs_smooth = recommend_sampling_freqs(
        10_000, 36e6Hz;
        fs_min = 36e6Hz,
        fs_max = 40e6Hz,
        min_doppler_coverage = 10_000Hz,
        num_coherently_integrated_code_periods = 36,
        num_alternatives = 5,
        sort_by = :smoothness,
    )
    @test !isempty(rs_smooth)
    primes = [r.inner_max_prime for r in rs_smooth]
    @test issorted(primes)
end

@testset "recommend_sampling_freqs — system convenience method" begin
    # GPS L1 C/A: code_length=1023, code_freq=1.023 MHz, T_code=1 ms
    rs = recommend_sampling_freqs(
        GPSL1();
        fs_min = 2e6Hz,
        fs_max = 5e6Hz,
        min_doppler_coverage = 7000Hz,
        num_coherently_integrated_code_periods = 1,
        num_alternatives = 3,
    )
    @test !isempty(rs)
    @test length(rs) <= 3
    # samples_per_code = ceil(fs/1000) for GPS L1 C/A — sanity check on at least one
    # candidate that the geometry agrees with plan_acquire's formula.
    r = first(rs)
    @test r.samples_per_code == ceil(Int, ustrip(Hz, r.sampling_freq) / 1000)
end

@testset "recommend_sampling_freqs — num_alternatives clamps result length" begin
    rs = recommend_sampling_freqs(
        GPSL1();
        fs_min = 2e6Hz,
        fs_max = 10e6Hz,
        num_alternatives = 2,
    )
    @test length(rs) <= 2
end

@testset "recommend_sampling_freqs — argument validation" begin
    @test_throws ArgumentError recommend_sampling_freqs(
        1023, 1.023e6Hz; sort_by = :nonsense)
    @test_throws ArgumentError recommend_sampling_freqs(
        1023, 1.023e6Hz; num_alternatives = 0)
    @test_throws ArgumentError recommend_sampling_freqs(
        1023, 1.023e6Hz; max_prime = 1)
    @test_throws ArgumentError recommend_sampling_freqs(
        1023, 1.023e6Hz; num_coherently_integrated_code_periods = 0)
    @test_throws ArgumentError recommend_sampling_freqs(
        1023, 1.023e6Hz; fs_min = 5e6Hz, fs_max = 1e6Hz)
    @test_throws ArgumentError recommend_sampling_freqs(
        1023, 1.023e6Hz; fs_step = 0Hz)
end

@testset "AD9361ClockPlan — basic validation" begin
    plan = AD9361ClockPlan()
    # Sample rates from the litex_m2sdr autotest list — all must be valid.
    for fs in (5e6, 10e6, 20e6, 30.72e6, 61.44e6)
        @test is_valid_sample_rate(plan, fs)
    end
    # Below the documented minimum (550 kHz) and above the maximum (61.44 MSPS) — invalid
    @test !is_valid_sample_rate(plan, 100e3)
    @test !is_valid_sample_rate(plan, 70e6)

    # Range envelope
    @test sample_rate_range(plan) == (0.55e6, 61.44e6)
end

@testset "AD9361ClockPlan — divider boundaries" begin
    plan = AD9361ClockPlan()
    # adc_clk_min = 25 MHz: at fs = 25 MHz (divider=1) ADC=25 MHz exactly — valid
    @test is_valid_sample_rate(plan, 25e6)
    # Right at fs_max = 61.44 MSPS — must be valid
    @test is_valid_sample_rate(plan, 61.44e6)
    # 1 MHz: with divider=12 → ADC=12 MHz < min; with d=8 → 8 MHz still <25;
    # need divider >= 25 → no divider in {1..12} works → invalid
    @test !is_valid_sample_rate(plan, 1e6)
    # 2.5 MHz: divider=12 → ADC=30 MHz, in [25,640] — valid
    @test is_valid_sample_rate(plan, 2.5e6)
end

@testset "recommend_sampling_freqs — sdr_clock_plan filters out unreachable" begin
    plan = AD9361ClockPlan()
    rs_unconstrained = recommend_sampling_freqs(
        GPSL1();
        fs_min = 2e6Hz, fs_max = 5e6Hz,
        num_alternatives = 5,
    )
    rs_constrained = recommend_sampling_freqs(
        GPSL1();
        fs_min = 2e6Hz, fs_max = 5e6Hz,
        num_alternatives = 5,
        sdr_clock_plan = plan,
    )
    # Every constrained candidate must be SDR-reachable
    for r in rs_constrained
        @test is_valid_sample_rate(plan, ustrip(Hz, r.sampling_freq))
    end
    # Constraint can only equal or shrink the candidate set
    @test length(rs_constrained) <= length(rs_unconstrained)
end

@testset "recommend_sampling_freqs — sdr_clock_plan clamps sweep range" begin
    plan = AD9361ClockPlan()
    # Ask for fs_max above hardware max — sweep should clamp to plan.fs_max
    rs = recommend_sampling_freqs(
        GPSL1();
        fs_min = 60e6Hz, fs_max = 80e6Hz,  # plan caps at 61.44 MHz
        num_alternatives = 5,
        sdr_clock_plan = plan,
    )
    for r in rs
        @test ustrip(Hz, r.sampling_freq) <= 61.44e6
    end
end

@testset "AD9361ClockPlan — show method" begin
    io = IOBuffer()
    show(io, MIME"text/plain"(), AD9361ClockPlan())
    out = String(take!(io))
    @test occursin("AD9361ClockPlan", out)
    @test occursin("dividers", out)
end

@testset "recommend_sampling_freqs — factor-string helper" begin
    f = Acquisition._factor_string
    @test f(1) == "1"
    @test f(2) == "2"
    @test f(8) == "2³"
    @test f(744) == "2³·3·31"   # the case the user asked about
    @test f(2048) == "2¹¹"
    @test f(10080) == "2⁵·3²·5·7"
end

@testset "recommend_sampling_freqs — show methods" begin
    rs = recommend_sampling_freqs(GPSL1(); fs_min = 2e6Hz, fs_max = 5e6Hz, num_alternatives = 2)
    # Force a wide IO so pretty_table doesn't truncate columns we want to test.
    io = IOContext(IOBuffer(), :displaysize => (24, 200))
    show(io, MIME"text/plain"(), rs)
    out = String(take!(io.io))
    @test occursin("Sampling freq (MHz)", out)
    @test occursin("Inner FFT size", out)
    # Factorization annotation is included in the inner FFT column
    @test occursin("(", out) && occursin(")", out)

    io2 = IOBuffer()
    show(io2, MIME"text/plain"(), first(rs))
    out2 = String(take!(io2))
    @test occursin("SamplingFreqRecommendation", out2)
    @test occursin("samples_per_code", out2)
    # Single-item show should also include the factorization
    @test occursin("inner_fft=", out2)

    # Empty vector: show should not error
    empty_rs = SamplingFreqRecommendation[]
    io3 = IOBuffer()
    show(io3, MIME"text/plain"(), empty_rs)
    @test occursin("no candidates", String(take!(io3)))
end
