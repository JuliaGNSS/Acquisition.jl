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
using LinearAlgebra: mul!
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

# ============================================================================
# BatchFFTCrossover — batched 2-D FFT along dim 1 vs per-column FFTs.
#
# Justifies the `BATCH_FFT_THRESHOLD = 320` constant in src/plan.jl: batched wins
# for small num_doppler_bins (amortises FFTW dispatch across columns); individual
# column FFTs win for large num_doppler_bins (better cache behaviour).
#
# Sweep covers both sides of the crossover (ndop from ~8 up to several thousand)
# at three realistic sampling rates. Only meaningful on the FM-DBZP branch,
# which exposes `col_batch_fft_plan`.
# ============================================================================

if _is_fmdbzp
    SUITE["BatchFFTCrossover"] = BenchmarkGroup()

    for (fs, fs_label) in [(2.048e6Hz, "2.048MHz"), (5.0e6Hz, "5MHz"), (10.0e6Hz, "10MHz")]
        for N_ms in [1, 5, 10, 20]
            plan = _make_plan(fs, N_ms; prns = [1], num_noncoherent_accumulations = 1)
            ndop = plan.num_coherently_integrated_code_periods * plan.num_blocks
            cim    = plan.thread_scratch[Threads.threadid()].coherent_integration_matrix
            col_buf = plan.thread_scratch[Threads.threadid()].col_buf
            spc    = plan.samples_per_code
            group_key = "ndop=$(ndop)_fs=$(fs_label)_N=$(N_ms)ms"
            SUITE["BatchFFTCrossover"][group_key] = BenchmarkGroup()

            # Batched path only meaningful below the threshold (plan stores `nothing` above it).
            if plan.col_batch_fft_plan !== nothing
                batch_plan = plan.col_batch_fft_plan
                SUITE["BatchFFTCrossover"][group_key]["batched"] =
                    @benchmarkable mul!($cim, $batch_plan, $cim)
            end

            col_plan = plan.col_fft_plan
            SUITE["BatchFFTCrossover"][group_key]["individual"] = @benchmarkable begin
                for c in 1:$spc
                    copyto!($col_buf, 1, $cim, (c - 1) * $ndop + 1, $ndop)
                    mul!($col_buf, $col_plan, $col_buf)
                end
            end
        end
    end
end

# ============================================================================
# BitEdgeSearch — cost of bit_edge_search_steps for a 20 ms coherent window.
#
# 20 ms is the GPS L1 C/A bit period, so num_data_bits = 1 but a bit transition
# can still land inside the window. bit_edge_search_steps runs the data-bit
# path over N candidate alignments and keeps the strongest. Values must divide
# bit_period_codes (= 20), so 1, 2, 5, 10 all apply.
# FM-DBZP only.
# ============================================================================

if _is_fmdbzp
    SUITE["BitEdgeSearch"] = BenchmarkGroup()

    for N_be in [1, 2, 5, 10]
        plan = plan_acquire(system, 2.048e6Hz, [1];
            min_doppler_coverage = 10_000Hz,
            num_coherently_integrated_code_periods = 20,
            bit_edge_search_steps = N_be,
            num_noncoherent_accumulations = 1)
        signal = _make_signal(plan, 1)

        SUITE["BitEdgeSearch"]["N_be=$(N_be)"] =
            @benchmarkable _acquire!($plan, $signal, $([1]))
    end
end
