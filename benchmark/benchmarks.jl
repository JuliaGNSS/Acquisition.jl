# benchmark/benchmarks.jl
#
# Acquisition Benchmark Suite — compatible with both:
#   - master (AcquisitionPlan / CoarseFineAcquisitionPlan API)
#   - ss/fmdbzp-refactor (plan_acquire / FM-DBZP API)
#
# AirSpeedVelocity.jl runs this file on each branch to compare performance.
#
# Usage with PkgBenchmark:
#   using PkgBenchmark
#   results = benchmarkpkg("Acquisition")
#
# Or manually:
#   include("benchmark/benchmarks.jl")
#   run(SUITE)

using BenchmarkTools
using Unitful: Hz, ustrip
using Acquisition
using GNSSSignals
using FFTW
using Random

const SUITE = BenchmarkGroup()

system = GPSL1()

# ============================================================================
# API detection
#
# FM-DBZP branch exports `plan_acquire`; master does not.
# ============================================================================

const _is_fmdbzp = isdefined(Acquisition, :plan_acquire)

# ============================================================================
# Helpers — unified interface over both APIs
# ============================================================================

"""
    _make_plan(fs, num_coh_ms; prns, num_noncoherent_accumulations)

Create an acquisition plan. On the FM-DBZP branch uses `plan_acquire`; on
master uses `AcquisitionPlan` with `num_coh_ms * samples_per_code` samples.
"""
function _make_plan(fs, num_coh_ms; prns = 1:32, num_noncoherent_accumulations = 1)
    if _is_fmdbzp
        plan_acquire(system, fs, collect(prns);
            min_doppler_coverage = 10_000Hz,
            num_coherently_integrated_code_periods = num_coh_ms,
            num_noncoherent_accumulations = num_noncoherent_accumulations)
    else
        samples_per_code = ceil(Int,
            get_code_length(system) / ustrip(Hz, get_code_frequency(system)) * ustrip(Hz, fs))
        num_samples = num_coh_ms * samples_per_code
        AcquisitionPlan(system, num_samples, fs;
            prns = collect(prns), fft_flag = FFTW.MEASURE)
    end
end

"""
    _signal_length(plan, num_accumulations)

Return the number of samples the signal needs for the given plan and number of
noncoherent accumulation steps.
"""
function _signal_length(plan, num_accumulations)
    if _is_fmdbzp
        num_accumulations * plan.num_coherently_integrated_code_periods * plan.samples_per_code
    else
        # master: DBZP requires 2× the coherent chunk size per accumulation
        2 * plan.num_samples_to_integrate_coherently
    end
end

function _make_signal(plan, num_accumulations)
    ComplexF32.(randn(ComplexF64, _signal_length(plan, num_accumulations)))
end

"""
    _acquire!(plan, signal, prns)

Call `acquire!` with the right keyword arguments for each API.
"""
function _acquire!(plan, signal, prns)
    if _is_fmdbzp
        acquire!(plan, signal, collect(prns); interm_freq = 0.0Hz)
    else
        acquire!(plan, signal, collect(prns))
    end
end

# ============================================================================
# Acquire — varying sampling frequency, integration length, and PRN count
# ============================================================================

SUITE["Acquire"] = BenchmarkGroup()

for (fs, fs_label) in [(2.048e6Hz, "2.048MHz"), (5.0e6Hz, "5MHz")]
    SUITE["Acquire"][fs_label] = BenchmarkGroup()

    for N_ms in [1, 5, 10, 20]
        SUITE["Acquire"][fs_label]["$(N_ms)ms"] = BenchmarkGroup()

        for num_prns in [1, 32]
            prns   = collect(1:num_prns)
            plan   = _make_plan(fs, N_ms; prns = 1:32, num_noncoherent_accumulations = 1)
            signal = _make_signal(plan, 1)

            SUITE["Acquire"][fs_label]["$(N_ms)ms"]["$(num_prns)prns"] =
                @benchmarkable _acquire!($plan, $signal, $prns)
        end
    end
end

# ============================================================================
# Noncoherent accumulation — 2.048 MHz, 1 ms coherent, varying M
#
# Only meaningful on the FM-DBZP branch (master has no noncoherent API here).
# ============================================================================

if _is_fmdbzp
    SUITE["NonCoherent"] = BenchmarkGroup()

    for M in [1, 5, 10]
        plan   = _make_plan(2.048e6Hz, 1; prns = [1], num_noncoherent_accumulations = M)
        signal = _make_signal(plan, M)

        SUITE["NonCoherent"]["M=$(M)"] =
            @benchmarkable _acquire!($plan, $signal, $([1]))
    end
end

# ============================================================================
# Long coherent integration — 2.048 MHz, single PRN
# ============================================================================

SUITE["LongCoherent"] = BenchmarkGroup()

for N_ms in [1, 5, 10, 20, 40]
    plan   = _make_plan(2.048e6Hz, N_ms; prns = [1], num_noncoherent_accumulations = 1)
    signal = _make_signal(plan, 1)

    SUITE["LongCoherent"]["$(N_ms)ms"] =
        @benchmarkable _acquire!($plan, $signal, $([1]))
end
