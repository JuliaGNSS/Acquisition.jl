# test/generic_acquire.jl — public-API tests for the generic circular-correlation PCPS
# fallback engine (src/generic_acquire.jl), reached automatically for sampling
# frequencies whose samples_per_code factors too badly for FM-DBZP.

@testset "plan routing — degenerate rates fall back, good rates stay FM-DBZP" begin
    system = GPSL1CA()
    # 2039 (prime) and 2038 = 2·1019 have no divisor near min_num_blocks → generic.
    @test plan_acquire(system, 2.039e6Hz, [1]; fft_flag = FFTW.ESTIMATE) isa
        Acquisition.GenericAcquisitionPlan
    @test plan_acquire(system, 2.038e6Hz, [1]; fft_flag = FFTW.ESTIMATE) isa
        Acquisition.GenericAcquisitionPlan
    # Well-supported rates keep the fast FM-DBZP path (regression guard).
    @test plan_acquire(system, 2.048e6Hz, [1]; fft_flag = FFTW.ESTIMATE) isa
        Acquisition.AcquisitionPlan
    @test plan_acquire(system, 16.368e6Hz, [1];
        min_doppler_coverage = 5_000Hz, num_coherently_integrated_code_periods = 5,
        fft_flag = FFTW.ESTIMATE) isa Acquisition.AcquisitionPlan
    # min_doppler_coverage beyond what the rate can represent still errors (no silent
    # fallback to an aliased grid).
    @test_throws ArgumentError plan_acquire(system, 2.048e6Hz, [1];
        min_doppler_coverage = 1.1e6Hz)
end

@testset "generic acquire — prime samples_per_code detects at correct code phase/Doppler" begin
    system = GPSL1CA()
    sampling_freq = 2.039e6Hz            # samples_per_code = 2039 (prime)
    prn = 1

    (; signal, code_phase, interm_freq) = generate_test_signal(
        system, prn;
        num_samples = 2039, doppler = 1000Hz, code_phase = 200.0,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 45, seed = 42)

    result = acquire(system, signal, sampling_freq, prn; interm_freq)

    @test result isa AcquisitionResults
    @test result.prn == prn
    @test is_detected(result)
    @test result.code_phase ≈ code_phase atol = 1.0
    @test abs(result.carrier_doppler / 1Hz - 1000) < ustrip(Hz, step(result.dopplers))
    @test result.secondary_code_phase === nothing   # graceful degradation
    @test result.power_bins === nothing              # not stored by default
end

@testset "generic acquire! — multiple PRNs, planted detected & absent rejected" begin
    system = GPSL1CA()
    sampling_freq = 2.038e6Hz            # samples_per_code = 2038 = 2·1019
    plan = plan_acquire(system, sampling_freq, [1, 2]; fft_flag = FFTW.ESTIMATE)
    @test plan isa Acquisition.GenericAcquisitionPlan

    # Doppler on the search grid (step = fs/samples_per_code = 1000 Hz here) to avoid
    # worst-case scalloping loss — the grid resolution equals the FM-DBZP path's.
    (; signal, code_phase) = generate_test_signal(system, 1;
        num_samples = plan.samples_per_code, doppler = -1000Hz, code_phase = 512.0,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 45, seed = 7)

    results = acquire!(plan, ComplexF32.(signal), [1, 2]; interm_freq = 0.0Hz)
    @test length(results) == 2
    r1 = only(filter(r -> r.prn == 1, results))
    r2 = only(filter(r -> r.prn == 2, results))
    @test is_detected(r1)
    @test !is_detected(r2)
    @test r1.code_phase ≈ code_phase atol = 1.0
end

@testset "generic acquire! — single-PRN Integer overload + PRN-not-in-plan error" begin
    system = GPSL1CA()
    sampling_freq = 2.039e6Hz
    plan = plan_acquire(system, sampling_freq, [1]; fft_flag = FFTW.ESTIMATE)
    (; signal) = generate_test_signal(system, 1;
        num_samples = plan.samples_per_code, sampling_freq, interm_freq = 0.0Hz, CN0 = 45)

    result = acquire!(plan, ComplexF32.(signal), 1; interm_freq = 0.0Hz)   # Integer overload
    @test result isa AcquisitionResults
    @test result.prn == 1

    @test_throws ArgumentError acquire!(plan, ComplexF32.(signal), [3])
end

@testset "generic acquire — non-coherent integration detects weaker signal" begin
    system = GPSL1CA()
    sampling_freq = 2.039e6Hz
    prn = 3
    (; signal, code_phase, interm_freq) = generate_test_signal(system, prn;
        num_samples = 4 * 2039, doppler = 1000Hz, code_phase = 100.0,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 40, seed = 99)

    result = acquire(system, signal, sampling_freq, prn;
        interm_freq, num_noncoherent_accumulations = 4)

    @test result.num_noncoherent_integrations == 4
    @test is_detected(result)
    @test result.code_phase ≈ code_phase atol = 1.5
end

@testset "generic acquire — non-zero intermediate frequency" begin
    system = GPSL1CA()
    sampling_freq = 2.039e6Hz
    prn = 5
    interm_freq = 1000Hz
    (; signal, code_phase) = generate_test_signal(system, prn;
        num_samples = 2039, doppler = 1000Hz, code_phase = 150.0,
        sampling_freq, interm_freq, CN0 = 45, seed = 33)

    result = acquire(system, signal, sampling_freq, prn; interm_freq)
    @test is_detected(result)
    @test result.code_phase ≈ code_phase atol = 1.5
end

@testset "generic acquire — store_power_bins surface + plot recipe" begin
    system = GPSL1CA()
    sampling_freq = 2.039e6Hz
    prn = 1
    plan = plan_acquire(system, sampling_freq, [prn]; fft_flag = FFTW.ESTIMATE)
    (; signal) = generate_test_signal(system, prn;
        num_samples = plan.samples_per_code, doppler = 1000Hz, code_phase = 100.0,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 60)

    result = acquire!(plan, ComplexF32.(signal), prn;
        interm_freq = 0.0Hz, store_power_bins = true)
    @test result.power_bins isa Matrix{Float32}
    @test size(result.power_bins) == (length(plan.doppler_freqs), plan.samples_per_code)
    # get_num_cells uses num_blocks(1) * block_size(spc) * num_secondary_rotations(1).
    @test get_num_cells(result) == length(plan.doppler_freqs) * plan.samples_per_code

    # The stored surface plots through the shared recipe (num_blocks = 1 path).
    recipe_list = RecipesBase.apply_recipe(Dict{Symbol,Any}(), result)
    rd = only(recipe_list)
    doppler_hz, chip_axis, z = rd.args
    @test length(chip_axis) == plan.samples_per_code
    @test size(z) == (plan.samples_per_code, length(plan.doppler_freqs))
    @test all(diff(chip_axis) .>= 0)
end

@testset "generic acquire — subsample_interpolation does not worsen code phase" begin
    system = GPSL1CA()
    sampling_freq = 2.039e6Hz
    prn = 2
    true_code_phase = 300.7
    (; signal, interm_freq) = generate_test_signal(system, prn;
        num_samples = 2039, doppler = 0Hz, code_phase = true_code_phase,
        sampling_freq, interm_freq = 0.0Hz, CN0 = 45, seed = 5555)

    r_grid = acquire(system, signal, sampling_freq, prn; interm_freq, subsample_interpolation = false)
    r_interp = acquire(system, signal, sampling_freq, prn; interm_freq, subsample_interpolation = true)

    err_grid = abs(r_grid.code_phase - true_code_phase)
    err_interp = abs(r_interp.code_phase - true_code_phase)
    @test err_grid < 2.0
    @test err_interp <= err_grid + 0.5
end
