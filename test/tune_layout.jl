@testset "Measured layout choice keeps stable improvements" begin
    timings = Dict(:baseline => 100, :close => 97, :fast => 60, :slow => 80)
    comparisons = Tuple{Symbol,Symbol}[]
    function measure(a, b)
        push!(comparisons, (a, b))
        return timings[a], timings[b]
    end
    winner = Acquisition._select_measured_plan(:baseline, [:close, :fast, :slow],
        identity, measure)
    @test winner == :fast
    @test comparisons == [(:baseline, :close), (:baseline, :fast), (:fast, :slow)]
    @test Acquisition._select_measured_plan(:baseline, Symbol[], identity, measure) == :baseline
end

@testset "Measured layouts obey the search constraints" begin
    system = GPSL1CA()
    plan = plan_acquire(system, 2.048e6Hz, [1];
        min_doppler_coverage = 10000Hz, num_coherently_integrated_code_periods = 20)
    layouts = Acquisition._tuning_layouts(plan, 1, 10000Hz, nothing)
    @test !isempty(layouts)
    @test length(layouts) <= 2
    @test all(layout -> layout != Acquisition._plan_layout(plan), layouts)
    for layout in layouts
        grid = Acquisition._doppler_grid_hz(layout, 1000.0, 20)
        @test first(grid) <= -10000.0
        @test last(grid) >= 10000.0
        @test plan.samples_per_code % layout.num_blocks == 0
        @test (layout.bins_per_pass ÷ 2 + 0.5) * get_code_center_frequency_ratio(system) <=
            Acquisition._MAX_CODE_DOPPLER_DRIFT
    end
    capped = plan_acquire(system, 2.048e6Hz, [1];
        min_doppler_coverage = 10000Hz, num_coherently_integrated_code_periods = 20,
        max_doppler_passes = 1)
    @test isempty(Acquisition._tuning_layouts(capped, 1, 10000Hz, 1))
end

@testset "Plan-time tuning returns a reusable acquisition plan" begin
    system = GPSL1CA()
    kwargs = (; min_doppler_coverage = 10000Hz,
        num_coherently_integrated_code_periods = 20)
    # Time a subset while retaining every requested PRN in the returned plan.
    Random.seed!(4321)
    expected_random = rand()
    Random.seed!(4321)
    plan = plan_acquire(system, 2.048e6Hz, [1, 2]; kwargs..., tune = true,
        tuning_prns = [1], tuning_interm_freq = 243Hz)
    @test rand() == expected_random
    @test plan.avail_prns == [1, 2]
    @test !plan.in_use[]
    @test length(plan.scratch_free) == length(plan.thread_scratch)
    @test all(iszero, plan.signal_block_ffts)
    @test all(iszero, plan.sig_buf)
    @test first(plan.doppler_freqs) <= -10000Hz
    @test last(plan.doppler_freqs) >= 10000Hz
    (; signal) = generate_test_signal(system, 2; sampling_freq = 2.048e6Hz,
        num_samples = 20 * plan.samples_per_code, doppler = 3000Hz,
        interm_freq = 243Hz, CN0 = 60)
    result = acquire!(plan, signal, 2; interm_freq = 243Hz)
    @test is_detected(result)
    @test abs(result.carrier_doppler - 3000Hz) <= step(plan.doppler_freqs)
    @test !plan.in_use[]
    @test acquire!(plan, signal, 2; interm_freq = 243Hz).carrier_doppler == result.carrier_doppler
    # Also exercise the full multistep pipeline while tuning, with the
    # caller's cooperative-preemption setting enabled on every candidate.
    multistep = plan_acquire(system, 1.542e6Hz, [1]; tune = true,
        num_noncoherent_accumulations = 2, max_blocking_time = 1u"ms")
    @test multistep.preempt_block_stride[] >= 1
    @test all(iszero, multistep.signal_block_ffts)
    (; signal) = generate_test_signal(system, 1; sampling_freq = 1.542e6Hz,
        num_samples = 2 * multistep.samples_per_code, doppler = 3000Hz,
        interm_freq = 0Hz, CN0 = 70)
    multistep_result = acquire!(multistep, signal, 1)
    @test is_detected(multistep_result)
    @test multistep_result.num_noncoherent_integrations == 2
    # No alternatives: tuning must still honour cooperative preemption.
    single = plan_acquire(system, 2.048e6Hz, [1]; kwargs..., tune = true,
        max_doppler_passes = 1, max_blocking_time = 1u"ns")
    @test single.num_doppler_passes == 1
    @test single.preempt_block_stride[] >= 1
    @test_throws ArgumentError plan_acquire(system, 2.048e6Hz, [1];
        tune = true, tuning_prns = Int[])
    @test_throws ArgumentError plan_acquire(system, 2.048e6Hz, [1];
        tune = true, tuning_prns = [2])
    @test_throws ArgumentError plan_acquire(system, 2.048e6Hz, [1];
        tune = true, tuning_interm_freq = NaN * Hz)
end
