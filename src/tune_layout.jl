# Opt-in plan-time layout measurements. The model shortlists candidates; the
# production acquire! pipeline determines the final choice on this machine.
const _LAYOUT_TUNING_CANDIDATES = 3
const _LAYOUT_TUNING_MIN_GAIN = 0.95

_plan_layout(plan::AcquisitionPlan) =
    DopplerLayout(plan.num_blocks, plan.num_doppler_passes, plan.doppler_bins_per_pass)

function _tuning_layouts(plan::AcquisitionPlan, num_prns::Int,
                         min_doppler_coverage, max_doppler_passes)
    N = plan.num_coherently_integrated_code_periods
    width = ustrip(Hz, plan.sampling_freq) / plan.samples_per_code
    coverage = ustrip(Hz, min_doppler_coverage)
    ratio = get_code_center_frequency_ratio(plan.system)
    candidates = DopplerLayout[]
    _doppler_search_layout(plan.samples_per_code, width, coverage, N,
        max_doppler_passes, num_prns, ratio, candidates)
    baseline = _plan_layout(plan)
    unique!(candidates)
    filter!(layout -> layout != baseline, candidates)
    compensate = _code_doppler_compensation(coverage, ratio, N, width)
    sort!(candidates; by = layout -> _search_cost(plan.samples_per_code,
        layout.num_blocks, N, layout.num_passes, num_prns, compensate))
    resize!(candidates, min(length(candidates), _LAYOUT_TUNING_CANDIDATES - 1))
    return candidates
end

# Keep the incumbent unless the measured improvement clears a noise margin.
# Only one challenger is built at a time; losing plans are not retained.
function _select_measured_plan(best, candidates, build, measure_pair)
    for layout in candidates
        challenger = build(layout)
        incumbent_ns, challenger_ns = measure_pair(best, challenger)
        if challenger_ns < _LAYOUT_TUNING_MIN_GAIN * incumbent_ns
            best = challenger
        end
        challenger = nothing
        GC.gc(false)
    end
    return best
end

function _measure_plan_pair(incumbent::AcquisitionPlan, challenger::AcquisitionPlan,
                            signal, prns, interm_freq)
    # Compile and touch buffers before timing. Alternate the order to reduce
    # bias from changes in host load. Timings include preparation and the full
    # requested PRN batch, including task scheduling and result extraction.
    acquire!(incumbent, signal, prns; interm_freq)
    acquire!(challenger, signal, prns; interm_freq)
    GC.gc(false)
    incumbent_ns = challenger_ns = typemax(UInt64)
    elapsed_ns = UInt64(0)
    for round in 1:5
        plans = isodd(round) ? (incumbent, challenger) : (challenger, incumbent)
        for plan in plans
            start = time_ns()
            acquire!(plan, signal, prns; interm_freq)
            elapsed = time_ns() - start
            elapsed_ns += elapsed
            if plan === incumbent
                incumbent_ns = min(incumbent_ns, elapsed)
            else
                challenger_ns = min(challenger_ns, elapsed)
            end
        end
        # Expensive searches need fewer repetitions. Always take at least two
        # samples per layout; cheap searches get five.
        round >= 2 && elapsed_ns >= 20_000_000 && break
    end
    return incumbent_ns, challenger_ns
end

function _tune_acquisition_plan(plan::AcquisitionPlan, prns, interm_freq, build;
                                min_doppler_coverage, max_doppler_passes)
    candidates = _tuning_layouts(plan, length(prns), min_doppler_coverage,
        max_doppler_passes)
    isempty(candidates) && return plan
    n = plan.num_coherently_integrated_code_periods * plan.samples_per_code *
        plan.num_noncoherent_accumulations
    # A private RNG keeps planning from changing the caller's random stream.
    signal = randn(Xoshiro(0x107), ComplexF32, n)
    measure_pair(a, b) = _measure_plan_pair(a, b, signal, prns, interm_freq)
    best = _select_measured_plan(plan, candidates, build, measure_pair)
    fill!(best.signal_block_ffts, 0f0)
    fill!(best.sig_buf, 0f0)
    return best
end
