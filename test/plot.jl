# test/plot.jl
# Tests for src/plot.jl: _fmdbzp_column_to_tau, _fmdbzp_sort_by_chip, and the @recipe

@testset "_fmdbzp_column_to_tau — block-permuted column to delay" begin
    # With num_blocks=4, block_size=8 (samples_per_code=32):
    #   column c=0: r=0, fine=0 → mod(4-0,4)*8 + 0 = 0
    #   column c=1: r=0, fine=1 → 0*8 + 1 = 1
    #   column c=8: r=1, fine=0 → mod(4-1,4)*8 + 0 = 3*8 = 24
    #   column c=16: r=2, fine=0 → mod(4-2,4)*8 + 0 = 2*8 = 16
    #   column c=24: r=3, fine=0 → mod(4-3,4)*8 + 0 = 1*8 = 8
    num_blocks = 4
    block_size = 8

    @test Acquisition._fmdbzp_column_to_tau(0,  num_blocks, block_size) == 0
    @test Acquisition._fmdbzp_column_to_tau(1,  num_blocks, block_size) == 1
    @test Acquisition._fmdbzp_column_to_tau(8,  num_blocks, block_size) == 24
    @test Acquisition._fmdbzp_column_to_tau(16, num_blocks, block_size) == 16
    @test Acquisition._fmdbzp_column_to_tau(24, num_blocks, block_size) == 8

    # tau=0 should always appear at c=0 (block r=0 maps to tau 0 within first block)
    @test Acquisition._fmdbzp_column_to_tau(0, 32, 64) == 0

    # Verify the inverse: for all c in 0:samples_per_code-1, tau values are a permutation of 0:samples_per_code-1
    samples_per_code = num_blocks * block_size
    taus = [Acquisition._fmdbzp_column_to_tau(c, num_blocks, block_size) for c in 0:samples_per_code-1]
    @test sort(taus) == collect(0:samples_per_code-1)
end

@testset "_fmdbzp_sort_by_chip — chip-ordered columns" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz
    plan = plan_acquire(system, sampling_freq, [1];
        min_doppler_coverage = 5_000Hz, num_coherently_integrated_code_periods = 1,
        num_noncoherent_accumulations = 1)

    f_code = Float64(ustrip(Hz, get_code_frequency(system)))
    fs     = Float64(ustrip(Hz, sampling_freq))

    # Build a dummy power_bins matrix (all ones — shape correctness test)
    num_doppler_bins = length(plan.doppler_freqs)
    dummy_bins = ones(Float32, num_doppler_bins, plan.samples_per_code)

    chip_axis, ordered = Acquisition._fmdbzp_sort_by_chip(
        dummy_bins, plan.num_blocks, plan.block_size, f_code, fs)

    # Output shape matches input
    @test length(chip_axis) == plan.samples_per_code
    @test size(ordered) == size(dummy_bins)

    # chip_axis is sorted (monotone non-decreasing)
    @test all(diff(chip_axis) .>= 0)

    # chip_axis covers [0, code_length) without duplicates
    code_length = plan.samples_per_code * f_code / fs
    @test minimum(chip_axis) >= 0.0
    @test maximum(chip_axis) < code_length
    @test length(unique(chip_axis)) == plan.samples_per_code
end

@testset "plot recipe — axes and data shape via RecipesBase" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz
    prn = 1

    plan = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 5_000Hz, num_coherently_integrated_code_periods = 1,
        num_noncoherent_accumulations = 1)

    (; signal) = generate_test_signal(system, prn;
        num_samples = plan.samples_per_code,
        doppler = 1000Hz, code_phase = 100.0,
        sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = 60)

    results = acquire!(plan, ComplexF32.(signal), [prn]; interm_freq = 0.0Hz, store_power_bins = true)
    result  = only(results)

    # Invoke the recipe without a plot backend
    recipe_list = RecipesBase.apply_recipe(Dict{Symbol,Any}(), result)
    @test length(recipe_list) == 1
    rd = only(recipe_list)

    doppler_hz, chip_axis, z = rd.args
    @test length(doppler_hz) == length(plan.doppler_freqs)
    @test length(chip_axis)  == plan.samples_per_code
    @test size(z)            == (plan.samples_per_code, length(plan.doppler_freqs))

    # chip_axis is sorted
    @test all(diff(chip_axis) .>= 0)

    # doppler_hz matches plan (in Hz, not Unitful)
    @test doppler_hz ≈ ustrip.(Hz, plan.doppler_freqs)

    # log_scale=true variant — z values should all be finite (no -Inf from log(0))
    recipe_list_log = RecipesBase.apply_recipe(Dict{Symbol,Any}(), result, true)
    _, _, z_log = only(recipe_list_log).args
    @test all(isfinite, z_log)
    @test size(z_log) == size(z)
    @test maximum(z_log) > minimum(z_log)  # non-trivial dynamic range
end
