# benchmark/benchmarks.jl
#
# Acquisition Benchmark Suite
#
# AirSpeedVelocity.jl runs this file against the PR head and its base to
# compare performance.
#
# Usage with PkgBenchmark:
#   using PkgBenchmark
#   results = benchmarkpkg("Acquisition")
#
# Or manually:
#   include("benchmark/benchmarks.jl")
#   run(SUITE)

using BenchmarkTools
using Unitful: Hz
using Acquisition
using GNSSSignals
using LinearAlgebra: mul!
using Random

const SUITE = BenchmarkGroup()

const l1ca = GNSSSignals.GPSL1CA()

# ============================================================================
# Feature flags — opt-in groups that are noisy/large to run by default.
# ============================================================================

# `BatchFFTCrossover` exists to justify the `BATCH_FFT_THRESHOLD = 320` constant
# in src/plan.jl; it's a one-off comparison sweep, not something that needs to
# run on every commit. Off by default — set ACQUISITION_BENCH_CROSSOVER=1 to
# re-run the sweep when re-tuning the threshold.
const INCLUDE_BATCH_FFT_CROSSOVER = get(ENV, "ACQUISITION_BENCH_CROSSOVER", "0") == "1"

# ============================================================================
# Helpers
# ============================================================================

"""
    _make_plan(fs, num_coh_ms; prns, num_noncoherent_accumulations)

Create a GPS L1 C/A acquisition plan.
"""
function _make_plan(fs, num_coh_ms; prns = 1:32, num_noncoherent_accumulations = 1)
    plan_acquire(l1ca, fs, collect(prns);
        min_doppler_coverage = 10_000Hz,
        num_coherently_integrated_code_periods = num_coh_ms,
        num_noncoherent_accumulations = num_noncoherent_accumulations)
end

"""
    _signal_length(plan, num_accumulations)

Return the number of samples the signal needs for the given plan and number of
noncoherent accumulation steps.
"""
_signal_length(plan, num_accumulations) =
    num_accumulations * plan.num_coherently_integrated_code_periods * plan.samples_per_code

function _make_samples(plan, num_accumulations)
    ComplexF32.(randn(ComplexF64, _signal_length(plan, num_accumulations)))
end

_acquire!(plan, samples, prns) = acquire!(plan, samples, collect(prns); interm_freq = 0.0Hz)

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
            samples = _make_samples(plan, 1)

            SUITE["Acquire"][fs_label]["$(N_ms)ms"]["$(num_prns)prns"] =
                @benchmarkable _acquire!($plan, $samples, $prns)
        end
    end
end

# ============================================================================
# Noncoherent accumulation — 2.048 MHz, 1 ms coherent, varying M
# ============================================================================

SUITE["NonCoherent"] = BenchmarkGroup()

for M in [1, 5, 10]
    plan   = _make_plan(2.048e6Hz, 1; prns = [1], num_noncoherent_accumulations = M)
    samples = _make_samples(plan, M)

    SUITE["NonCoherent"]["M=$(M)"] =
        @benchmarkable _acquire!($plan, $samples, $([1]))
end

# ============================================================================
# Long coherent integration — 2.048 MHz, single PRN
# ============================================================================

SUITE["LongCoherent"] = BenchmarkGroup()

for N_ms in [1, 5, 10, 20, 40]
    plan   = _make_plan(2.048e6Hz, N_ms; prns = [1], num_noncoherent_accumulations = 1)
    samples = _make_samples(plan, 1)

    SUITE["LongCoherent"]["$(N_ms)ms"] =
        @benchmarkable _acquire!($plan, $samples, $([1]))
end

# ============================================================================
# BatchFFTCrossover — batched 2-D FFT along dim 1 vs per-column FFTs.
#
# Justifies the `BATCH_FFT_THRESHOLD = 320` constant in src/plan.jl: batched wins
# for small num_doppler_bins (amortises FFTW dispatch across columns); individual
# column FFTs win for large num_doppler_bins (better cache behaviour).
#
# Sweep covers both sides of the crossover (ndop from ~8 up to several thousand)
# at three realistic sampling rates.
# ============================================================================

if INCLUDE_BATCH_FFT_CROSSOVER
    SUITE["BatchFFTCrossover"] = BenchmarkGroup()

    for (fs, fs_label) in [(2.048e6Hz, "2.048MHz"), (5.0e6Hz, "5MHz"), (10.0e6Hz, "10MHz")]
        for N_ms in [1, 5, 10, 20]
            plan = _make_plan(fs, N_ms; prns = [1], num_noncoherent_accumulations = 1)
            ndop = plan.num_coherently_integrated_code_periods * plan.num_blocks
            # The pipeline consumes one (ndop × block_size) tile at a time, so
            # the crossover question is now batched-tile vs per-column FFTs
            # over a tile, repeated num_blocks times per PRN.
            tile    = Acquisition._default_scratch(plan).coherent_tile
            col_buf = Acquisition._default_scratch(plan).col_buf
            bsize   = plan.block_size
            group_key = "ndop=$(ndop)_fs=$(fs_label)_N=$(N_ms)ms"
            SUITE["BatchFFTCrossover"][group_key] = BenchmarkGroup()

            # Batched path only meaningful below the threshold (plan stores `nothing` above it).
            if plan.col_batch_fft_plan !== nothing
                batch_plan = plan.col_batch_fft_plan
                SUITE["BatchFFTCrossover"][group_key]["batched"] =
                    @benchmarkable mul!($tile, $batch_plan, $tile)
            end

            col_plan = plan.col_fft_plan
            SUITE["BatchFFTCrossover"][group_key]["individual"] = @benchmarkable begin
                for c in 1:$bsize
                    copyto!($col_buf, 1, $tile, (c - 1) * $ndop + 1, $ndop)
                    mul!($col_buf, $col_plan, $col_buf)
                end
            end
        end
    end
end

# ============================================================================
# BitEdgeSearch — cost of bit_edge_search_steps for a 40 ms coherent window.
#
# 40 ms spans two GPS L1 C/A bits, so num_bits = 2. bit_edge_search_steps runs
# the bit path over N candidate alignments of the bit boundaries and keeps the
# strongest. Values must divide the 20 code periods per bit, so 1, 2, 5, 10 all
# apply.
# ============================================================================

SUITE["BitEdgeSearch"] = BenchmarkGroup()

for N_be in [1, 2, 5, 10]
    plan = plan_acquire(l1ca, 2.048e6Hz, [1];
        min_doppler_coverage = 10_000Hz,
        num_coherently_integrated_code_periods = 40,
        bit_edge_search_steps = N_be,
        num_noncoherent_accumulations = 1)
    samples = _make_samples(plan, 1)

    SUITE["BitEdgeSearch"]["N_be=$(N_be)"] =
        @benchmarkable _acquire!($plan, $samples, $([1]))
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
# ============================================================================

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
    # `plan_acquire` and not the signal constructor. The
    # constructor builds the embedded per-signal LUT (baking any BOC/CBOC
    # subcarrier), which dominates a per-iteration `sys_ctor()` for E1B/L1CP;
    # real callers build the signal once and reuse it across plans.
    sig_obj = sys_ctor()
    SUITE["PlanAcquire"][label] =
        @benchmarkable plan_acquire($sig_obj, $fs, $bench_prns;
            min_doppler_coverage = $bench_min_doppler)

    plan = plan_acquire(sig_obj, fs, bench_prns;
        min_doppler_coverage = bench_min_doppler)
    samples = _make_samples(plan, 1)
    SUITE["AcquireSignals"][label] =
        @benchmarkable acquire!($plan, $samples, $bench_prns; interm_freq = 0.0Hz)
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
    samples = _make_samples(plan, N_nc)
    SUITE["AcquireSignals"][label] =
        @benchmarkable acquire!($plan, $samples, $bench_prns; interm_freq = 0.0Hz)
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
    samples = _make_samples(plan, 1)
    SUITE["AcquireSignals"][label] =
        @benchmarkable acquire!($plan, $samples, $bench_prns; interm_freq = 0.0Hz)
    SUITE["PlanAcquire"][label] =
        @benchmarkable plan_acquire($sig_obj, $fs, $bench_prns;
            min_doppler_coverage = $bench_min_doppler,
            num_coherently_integrated_code_periods = $N_coh)
end
