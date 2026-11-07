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
# FFTW stays: the PR benchmark job runs this file against the base commit too,
# and before the switch to FFTL `plan_acquire` took `fft_flag = FFTW.MEASURE`.
using FFTW
using LinearAlgebra: mul!
using Random

const SUITE = BenchmarkGroup()

# GNSSSignals v2 renamed GPSL1 -> GPSL1CA. Pick whichever exists so the same
# benchmark script runs on both master (v1) and this branch (v2).
system = isdefined(GNSSSignals, :GPSL1CA) ? GNSSSignals.GPSL1CA() : GNSSSignals.GPSL1()

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

# ============================================================================
# Signal sweep — `plan_acquire` and `acquire!` across the GNSS signals this
# package supports, at 4 PRNs each. Captures both runtime and RAM footprint
# (BenchmarkTools' `memory` field). Together these track Issue #60, which
# targets the `plan_acquire` RAM footprint.
#
# Parameters are sized so the suite fits inside the GitHub Actions free-tier
# runner under `--threads=auto`:
#   - 4 PRNs (vs 32 default) — per-PRN buffers scale linearly, trend preserved
#   - `min_doppler_coverage = 2000Hz` — shrinks `num_doppler_bins`, the
#     dominant size driver at `num_coh=1`. Smaller than the default 7000Hz
#     but the relative before/after RAM trend the issue cares about is the
#     same; raise to default locally if you want headline numbers.
#
# Sampling frequencies match each signal's practical minimum: L1CA/E1B at
# 5/15 MHz (CBOC needs >= 12.276 MHz), L5I at 25 MHz, L1C-P at 16 MHz (the
# heavy case from Issue #60).
#
# `plan_acquire` sizes its per-thread scratch by `Threads.maxthreadid()` at
# the time of the call, so the PlanAcquire memory grows with the launched
# thread count. To compare before/after thread-floor changes (Issue #60
# projects ~5.5 GB → ~0.4 GB at 1 thread, ~10 GB → ~4.9 GB at 16 threads),
# run with both `julia -t 1` and `julia -t 16` and diff the results.
#
# FM-DBZP only — `plan_acquire` does not exist on master.
# ============================================================================

if _is_fmdbzp
    SUITE["PlanAcquire"] = BenchmarkGroup()
    SUITE["AcquireSignals"] = BenchmarkGroup()

    signal_cases = [
        (GNSSSignals.GPSL1CA,     5.0e6Hz,     "L1CA_5MHz"),
        (GNSSSignals.GalileoE1B, 15.0e6Hz,     "E1B_15MHz"),
        (GNSSSignals.GPSL5I,     25.0e6Hz,     "L5I_25MHz"),
        (GNSSSignals.GPSL1C_P,   16.0e6Hz,     "L1CP_16MHz"),
        # Non-smooth rate: 16.368 MHz (= 16 × 1.023 MHz, the canonical GNSS IF)
        # gives samples_per_code 16368 = 2⁴·3·11·31, so the inner double-block
        # FFT lands on a large prime. `plan_acquire` pads it to a 2·3·5·7-smooth
        # size; this case guards that fast path (and the FFT-size regression it
        # fixes) against future changes. The other rates above are all smooth, so
        # padding is a no-op there.
        (GNSSSignals.GPSL1CA,    16.368e6Hz,   "L1CA_16.368MHz"),
    ]
    bench_prns = collect(1:4)

    bench_min_doppler = 2000Hz

    for (sys_ctor, fs, label) in signal_cases
        # Construct the signal object once and interpolate it, so PlanAcquire times
        # `plan_acquire` and not the signal constructor. In GNSSSignals 3 the
        # constructor builds the embedded per-signal LUT (baking any BOC/CBOC
        # subcarrier), which dominates a per-iteration `sys_ctor()` for E1B/L1CP;
        # real callers build the signal once and reuse it across plans.
        sig_obj = sys_ctor()
        SUITE["PlanAcquire"][label] =
            @benchmarkable plan_acquire($sig_obj, $fs, $bench_prns;
                min_doppler_coverage = $bench_min_doppler)

        plan = plan_acquire(sig_obj, fs, bench_prns;
            min_doppler_coverage = bench_min_doppler)
        signal = _make_signal(plan, 1)
        SUITE["AcquireSignals"][label] =
            @benchmarkable acquire!($plan, $signal, $bench_prns; interm_freq = 0.0Hz)
    end

    # Multistep simple path — exercises the fused FFT+|x|²+code-drift+fftshift
    # kernel (Issue #62) over N_nc accumulation steps. L1CA at 5 MHz with
    # N_coh=1, N_nc=8 is the canonical simple-path noncoherent workload; the
    # entry is here so RAM/runtime diffs around the multistep fusion land in a
    # stable AirSpeedVelocity series.
    let
        fs = 5.0e6Hz
        label = "L1CA_5MHz_Nnc8"
        N_nc = 8
        sig_obj = GNSSSignals.GPSL1CA()
        plan = plan_acquire(sig_obj, fs, bench_prns;
            min_doppler_coverage = bench_min_doppler,
            num_noncoherent_accumulations = N_nc)
        signal = _make_signal(plan, N_nc)
        SUITE["AcquireSignals"][label] =
            @benchmarkable acquire!($plan, $signal, $bench_prns; interm_freq = 0.0Hz)
        SUITE["PlanAcquire"][label] =
            @benchmarkable plan_acquire($sig_obj, $fs, $bench_prns;
                min_doppler_coverage = $bench_min_doppler,
                num_noncoherent_accumulations = $N_nc)
    end

    # Secondary-code rotation search path — exercises
    # `_sign_search_step_with_rotations!` and the (samples_per_code × L)-wide
    # noncoherent buffer. L5I at 12 MHz / N_coh=10 is the canonical rotation
    # workload (the case the rotation-vs-LongL5I sensitivity work in
    # `claude_scratch/sensitivity-fix.md` is tuned for). Without this entry,
    # the default L5I bench above runs at N_coh=1 and never touches the sign-
    # search path, so plan-size/runtime savings in that path go unmeasured.
    let
        fs = 12.0e6Hz
        label = "L5I_12MHz_Ncoh10"
        N_coh = 10
        sig_obj = GNSSSignals.GPSL5I()
        plan = plan_acquire(sig_obj, fs, bench_prns;
            min_doppler_coverage = bench_min_doppler,
            num_coherently_integrated_code_periods = N_coh)
        signal = _make_signal(plan, 1)
        SUITE["AcquireSignals"][label] =
            @benchmarkable acquire!($plan, $signal, $bench_prns; interm_freq = 0.0Hz)
        SUITE["PlanAcquire"][label] =
            @benchmarkable plan_acquire($sig_obj, $fs, $bench_prns;
                min_doppler_coverage = $bench_min_doppler,
                num_coherently_integrated_code_periods = $N_coh)
    end
end

# ============================================================================
# SparseDivisorRates — sampling frequencies whose `samples_per_code` has sparse
# divisors (issue #101). At 1.542 MHz (1542 = 2·3·257) the smallest divisor
# reaching ±7 kHz in one FM-DBZP grid is 257; at 3.069 MHz (3069 = 3·11·31) it
# is 31. Doppler passes cover the same range with 6 and 11 blocks. GPS L1 C/A,
# default ±7 kHz, 32 PRNs. 1.500 MHz is the well-behaved neighbour (20 blocks)
# that 1.542 MHz should now land next to. FM-DBZP only.
# ============================================================================

if _is_fmdbzp
    SUITE["SparseDivisorRates"] = BenchmarkGroup()
    for (fs, fs_label) in [(1.5e6Hz, "1.500MHz"), (1.542e6Hz, "1.542MHz"),
                           (3.069e6Hz, "3.069MHz")]
        plan = plan_acquire(system, fs, collect(1:32))
        signal = _make_signal(plan, 1)
        SUITE["SparseDivisorRates"][fs_label] =
            @benchmarkable _acquire!($plan, $signal, $(1:32))
    end
end
