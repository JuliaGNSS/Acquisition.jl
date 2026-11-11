# Usage Guide

## Basic Acquisition

The simplest way to acquire GNSS signals is with the [`acquire`](@ref) function:

```@example guide
using Acquisition, GNSSSignals
import Unitful: Hz

system = GPSL1CA()

# Generate a synthetic GPS L1 signal for PRN 1 (1 ms, 4 MHz)
(; signal, sampling_freq, interm_freq) = generate_test_signal(system, 1;
    num_samples = 4096, sampling_freq = 4e6Hz)
nothing # hide
```

Acquire multiple PRNs at once:

```@example guide
results = acquire(system, signal, sampling_freq, 1:3; interm_freq)
```

Acquire a single PRN:

```@example guide
result = acquire(system, signal, sampling_freq, 1; interm_freq)
```

Each result contains:
- `carrier_doppler`: Estimated Doppler frequency
- `code_phase`: Code phase in chips
- `CN0`: Carrier-to-noise density ratio (dB-Hz)
- `peak_to_noise_ratio`: Peak correlation power divided by noise power
- `power_bins`: Correlation power matrix (only populated when `store_power_bins=true`)

## Detecting Satellites

Use [`is_detected`](@ref) to decide whether a satellite signal is present:

```@example guide
# Filter to detected satellites (1% false alarm probability)
detected = filter(is_detected, results)
```

```@example guide
# Custom false alarm probability
detected = filter(r -> is_detected(r; pfa = 0.001), results)
```

Under the hood, this compares each result's `peak_to_noise_ratio` against a
CFAR (Constant False Alarm Rate) threshold computed by [`cfar_threshold`](@ref).

### CFAR Threshold — How It Works

The detector uses the statistic

```
peak_to_noise_ratio = peak_power / noise_power
```

Under the null hypothesis (noise only), the peak across
`num_cells = num_doppler_bins × num_code_phases` search cells follows a scaled
chi-squared distribution with `2M` degrees of freedom, where `M` is the number
of non-coherent accumulations stored in the result. [`cfar_threshold`](@ref)
returns the quantile of that distribution at the requested per-grid false alarm
probability, using a Bonferroni-like correction across `num_cells`:

```@example guide
threshold = cfar_threshold(0.01, get_num_cells(result); num_noncoherent_integrations = 1)
```

`is_detected(result; pfa)` takes care of both arguments for you — it reads
`num_cells` and `num_noncoherent_integrations` directly from the result — so
you only need [`cfar_threshold`](@ref) when you want to inspect the threshold
separately (e.g. for plotting or custom logic). Prefer
`cfar_threshold(result; pfa)`, which is what `is_detected` uses:

```@example guide
threshold = cfar_threshold(result; pfa = 0.01)
```

**Sign searches.** With a data-bit polarity search (a window longer than one
data bit) or a bit-edge search (`bit_edge_search_steps > 1`), each cell is the
maximum over several sign hypotheses. Under noise that maximum has a higher
mean — which the noise estimate absorbs — and a much shorter tail relative to
it than one χ² cell, so the χ² threshold would be far too high: no false
alarms at all, and weak signals missed. [`plan_acquire`](@ref) therefore
computes the null distribution of the maximum from the plan's sign patterns,
and `cfar_threshold(result; pfa)` / `is_detected` use it. The raw
`peak_to_noise_ratio` of a sign search is consequently lower than without one,
for the same signal; compare it to its own threshold, not across plans.

Choosing `pfa`:

- `pfa = 0.01` (the default) is a reasonable starting point for a cold acquire.
- Lower values (`1e-4`, `1e-6`) are typical when a false track is expensive —
  the threshold rises modestly because the chi-squared tail is steep.
- Only the *ratio* matters: the threshold is independent of the absolute noise
  power, so it works unchanged at any CN0 and any sampling frequency.

### Noise Power Estimation

The denominator of `peak_to_noise_ratio` — the noise power — is estimated
from the Doppler × code-phase search grid itself. The CFAR formula assumes
the per-cell noise power is independent and identically distributed across
the grid: under that assumption the sample mean over the grid is the
lowest-variance unbiased estimate, and the threshold derived from it
controls the false-alarm rate exactly.

That assumption fails on real-world IF recordings. Hardware artefacts —
DC offset, IF spurs, narrowband interferers, ADC nonlinearity — inflate
*specific Doppler rows* of the grid. A global-mean estimator then folds the
hot rows into the noise floor: the threshold rises with them, and any peak
whose Doppler bin happens to land in a quiet row passes the inflated
threshold. False alarms cluster at predictable Doppler offsets — typically
near 0 Hz (DC offset) or at clock-comb harmonics.

Acquisition.jl ships two estimators, dispatched on a singleton type stored
on the [`AcquisitionPlan`](@ref):

- [`OppositeRowNoiseEstimator`](@ref) **(default)** — averages a single
  Doppler row, the one exactly half the search grid away from the peak.
  The estimate is conditioned on Doppler instead of pooled across it, so a
  hot Doppler row no longer biases the threshold for peaks elsewhere. The
  opposite row is also a safe choice signal-wise: it is far enough from any
  real signal's mainlobe that sidelobes cannot contaminate the estimate.
  This mirrors the behaviour of GNSS-SDR's
  `pcps_acquisition::max_to_input_power_statistic`.

- [`GlobalMeanNoiseEstimator`](@ref) — averages all cells of the grid except
  a `±samples_per_chip` exclusion zone around the peak column. Lowest
  variance when the noise really is independent and identically distributed
  across the grid. Useful for synthetic AWGN test signals; not recommended
  for real signals.

Pick the estimator at plan-construction time:

```julia
# Default — robust to Doppler-conditional noise variation
plan = plan_acquire(system, sampling_freq, prns)

# Explicit form
plan = plan_acquire(system, sampling_freq, prns;
    noise_estimator = OppositeRowNoiseEstimator())

# AWGN test path
plan = plan_acquire(system, sampling_freq, prns;
    noise_estimator = GlobalMeanNoiseEstimator())
```

The choice flows through `acquire!` and `is_detected`; no other code needs
to change. The CFAR threshold formula is unchanged either way — only the
noise-power estimate fed to it differs.

## Using Acquisition Plans

For repeated acquisitions — e.g. tracking many epochs or processing a file — pre-compute
a plan once to avoid repeated FFT planning and memory allocation:

```@example guide
plan = plan_acquire(system, sampling_freq, collect(1:3))

results = acquire!(plan, signal, 1:3; interm_freq)
```

### Avoiding first-call latency

The search is compiled for each plan *shape* — signal system, sampling frequency,
coherent and non-coherent lengths, Doppler grid and search options — and for the
sample type you pass. Shapes covered by Acquisition's own precompile workload are
ready when the package loads. For any other shape, compilation happens on first use:

- [`plan_acquire`](@ref) compiles everything that depends only on the plan — the FFT
  kernels and the per-PRN search kernels. That takes one to two seconds for a new shape,
  and nothing for a shape compiled before in the session.
- The first [`acquire!`](@ref) then compiles what depends on your sample type and PRN
  collection (downconversion, the PRN loop, result extraction): roughly another
  0.4 s per sample type.

A receiver usually cannot afford either once samples are flowing. Put your production
configuration into your own package's [PrecompileTools](https://github.com/JuliaLang/PrecompileTools.jl)
workload: then all of it is compiled when *your* package is precompiled, and the first
plan and acquisition at run time are as fast as every later one.

```julia
module MyReceiver

using Acquisition, GNSSSignals, PrecompileTools
import Unitful: Hz

make_plan() = plan_acquire(GPSL1CA(), 6.5e6Hz, collect(1:32);
    num_coherently_integrated_code_periods = 3)

@setup_workload begin
    @compile_workload begin
        plan = make_plan()
        # Same element type and container as your front end delivers; the values do
        # not matter.
        n = plan.num_coherently_integrated_code_periods * plan.samples_per_code
        signal = Complex{Int16}.(rand(Int16(-100):Int16(100), n), rand(Int16(-100):Int16(100), n))
        acquire!(plan, signal, collect(1:32); interm_freq = 0.0Hz)
    end
end

end
```

With this workload, at a sampling frequency Acquisition's own image does not cover,
`make_plan()` took 1.3 ms and the first `acquire!` 31 ms — its steady-state time —
instead of about 1.5 s and 0.4 s.

Make the workload match production exactly, or the part that differs compiles at run
time after all:

- the same plan keywords (sampling frequency, `num_coherently_integrated_code_periods`,
  `num_noncoherent_accumulations`, `min_doppler_coverage`, `bit_edge_search_steps`, …);
- the same sample container: element type (`ComplexF32`, `Complex{Int16}`, …) *and*
  container (`Vector`, or the `view` of your ring buffer you actually pass);
- the same PRN collection type (`Vector{Int}` or `UnitRange`);
- the entry points you use — call [`acquire_stream!`](@ref) in the workload if that is
  what you use, and pass `store_power_bins` / `subsample_interpolation` if you set them.

The unit of `interm_freq` does not matter: it is converted to `Float64` before the
search, so `243Hz` and `0.243kHz` share one compilation.

In a script or notebook, where there is no package to precompile, run one
`acquire!` on representative data right after `plan_acquire` and before the
time-critical part.

## Multi-threaded Acquisition

When processing many PRNs, start Julia with multiple threads and the PRN loop runs in
parallel automatically — no code changes required:

```bash
julia -t 4
```

The plan allocates per-thread scratch buffers at construction time based on the number
of threads available when `plan_acquire` is called. If you create the plan and later
run with more threads, restart Julia with the desired thread count before calling
`plan_acquire`.

The per-thread scratch is small: the acquisition pipeline is *tiled*, producing and
reducing the correlation surface one `num_doppler_bins × block_size` column block at a
time, so at `num_noncoherent_accumulations = 1` no per-thread buffer scales with
`samples_per_code` at all. The full Doppler × code-phase power surface is only
materialised when you ask for it (`store_power_bins = true`, one cached buffer per
PRN) or when `num_noncoherent_accumulations > 1` — and even then the PRN loop runs
PRN-outer, so only `min(threads, cores, #PRNs)` accumulation surfaces exist, never
one per PRN. The multistep path additionally caches the signal-block FFTs of all
segments (16 bytes per signal sample) so no work is recomputed across PRNs.

### Acquiring next to a real-time loop

Julia's scheduler is cooperative: it never takes a thread away from a running task.
The PRN loop spawns one chunk per scratch slot, so while an `acquire!` is in flight
every thread is inside a chunk that will not yield until it has finished all of its
PRNs. Anything else in the process — a tracking loop servicing a hardware correlator,
say — waits for a whole chunk. At 8 threads and 32 PRNs that is about 1 ms of GPS
L1 C/A at 2.048 MHz, which is the entire epoch budget of a 1 kHz loop.

`max_blocking_time` bounds that wait. The per-PRN kernels then yield every few
FM-DBZP code blocks, so acquisition stays preemptible without holding any thread out
of the search:

```julia
import Unitful: Hz, µs

plan = plan_acquire(GPSL1CA(), 2.048e6Hz, 1:32; max_blocking_time = 100µs)
```

`plan_acquire` times one PRN pass to convert the bound into a block stride, so the
same setting means the same thing at a different sampling rate, Doppler grid or
machine. Measured with a 1 kHz tracking loop (1 ms epoch, ~120 µs of correlator
servicing) as an ordinary task alongside a continuous 32-PRN `acquire!`, 8 threads:

| `max_blocking_time` | p99 wake lateness | acquisitions/s |
|---|---|---|
| `nothing` (default) | 0.753 ms | 845 |
| `270µs` (one PRN)   | 0.218 ms | 878 |
| `70µs`              | 0.078 ms | 895 |
| `35µs`              | 0.040 ms | 881 |
| `8µs` (one block)   | 0.009 ms | 755 |

The observed p99 tracks the requested bound closely. It is a target rather than a
guarantee, though: `plan_acquire` derives the stride from a timing measurement, so
verify against your own loop if the deadline is hard.

Each yield costs on the order of 250 ns, so the throughput cost grows with how many
the bound implies. Down to roughly an eighth of a PRN pass it stays within run-to-run
noise; only the one-block extreme is visibly expensive (a solo `acquire!` goes from
1.016 ms to 1.250 ms here). The default is `nothing`, which yields nowhere and
behaves exactly as before, and results never depend on the setting.

## Streaming Results as They Finish

The PRNs of one `acquire!` are searched in parallel, but the vector it returns is only
complete once the *last* PRN is done — a satellite found in the first tenth of a
millisecond of a 32-PRN search still waits for the other 31. [`acquire_stream!`](@ref)
starts the same search on its own task and returns a `Channel` immediately, publishing
each result the moment its own PRN finishes, so a consumer can start tracking that
satellite while the rest of the search is still running:

```julia
acquire_stream!(plan, signal, 1:32; interm_freq) do results
    for result in results
        is_detected(result) && start_tracking(result)
    end
end
```

That is the whole API: no channel to construct, no task to spawn, nothing to close.
The channel belongs to the call and carries **one result per entry**, so the loop
above sees satellites, not batches.

The `do` block is a scope, not a callback queue — it runs on the calling task, and the
search is shut down when the block is left, *however* it is left: a normal return, a
`break`, or an exception. That is what makes it safe to stop early, which is the whole
reason to stream in the first place:

```@example guide
acquire_stream!(plan, signal, 1:3; interm_freq) do results
    for result in results
        is_detected(result) && return result.prn   # stop at the first hit
    end
    return nothing
end
```

Nothing has to consume the results concurrently, though. `acquire_stream!` without a
block hands back the bare channel, which can be read after the fact like any other —
as long as it is read to the end:

```@example guide
channel = acquire_stream!(plan, signal, 1:3; interm_freq)

[result.prn for result in channel]   # completion order
```

Note the `buffer_size` default of `length(prns)` is what makes that safe: every result
fits, so the search finishes whether or not anyone is reading. With a smaller buffer,
a channel nobody drains will stall the search instead.

The comprehension terminates because the channel does: `acquire_stream!` closes it once
the last PRN has been published. A consumer never has to know how many PRNs were
searched, and a search that *fails* closes the channel with its exception, so the
failure is raised out of the loop instead of leaving the consumer waiting on a channel
nothing will ever fill. (The exception reaches the consumer wrapped in a
`TaskFailedException`, since it was raised on the search task — the original is at
`err.task`.)

Leaving a *bare-channel* stream early is the one case that needs care: a `for` loop
does not close the channel it iterates, so a plain `break` leaves the search running
and still holding the plan. Nothing is corrupted — the next search on that plan raises
`PlanInUseError` — but nothing is stopped either. Use the `do`-block form, or `close`
the channel and drain it yourself.

[`acquire_stream`](@ref) is the same thing one level up — it plans and streams in a
single call, like [`acquire`](@ref) does for the batch API.

### What it buys

A 32-PRN GPS L1 C/A search at 2.048 MHz, 4 threads, medians over 180 runs (times
relative to the start of the call):

|  | first result available | last result | whole call |
|---|---|---|---|
| `acquire!` | 0.72 ms | 0.72 ms | 0.72 ms |
| `acquire_stream!` | **0.09 ms** | 1.00 ms | 1.00 ms |

The first result is readable **8× earlier**, and the rest stream out as their PRNs
finish — result 8 at 0.25 ms, result 16 at 0.43 ms. The gain scales with how many PRNs
share a chunk, so it grows with the PRN count and shrinks as threads are added: at 8
PRNs on 8 threads every chunk holds a single PRN, they all finish together, and
streaming buys only 1.5×. Throughput is unchanged — the same search through either API
measures within run-to-run noise at 1, 8 and 32 PRNs.

### What to keep in mind

  - **Results arrive in completion order**, not in the order of `prns` — which is the
    point, but it does mean a consumer must read the PRN off the result rather than
    infer it from position. `acquire!` still returns the same results as a vector
    ordered by `prns`, so the batch and streaming views can be mixed freely.
  - **The plan is in use until the search finishes.** The search writes into the plan's
    buffers from another task, so the same `plan` cannot serve a second `acquire!` or
    `acquire_stream!` until the stream is done — trying raises `PlanInUseError` rather
    than letting two searches share one set of buffers. Iterating the channel to
    completion ends the search; the `do`-block form also ends it on `break`, `return`
    and exceptions. The `signal` must stay unmodified for the same span.
  - **`buffer_size` sets the backpressure.** It defaults to `length(prns)`, so every
    result fits and the search never waits for the consumer. A smaller channel makes
    the chunk whose `put!` blocks wait — holding no scratch buffer while it does, so
    the other chunks keep searching — and a consumer that stops reading altogether will
    eventually stall the search.
  - **Abandoning a stream**: the `do`-block form handles this — it cancels the search,
    discards whatever was in flight and waits for the plan to be free before the block
    returns. With the bare channel you have to do it yourself: `close` it and the
    search unwinds at its next publish, joining its chunks first so the plan is left
    whole, but it may still be unwinding when `close` returns and there is no handle
    here to wait on. Draining the channel is the tidy way to stop.
  - **`store_power_bins = true` aliases plan-owned memory.** Each result's
    `power_bins` is a buffer the next `acquire!` for that PRN overwrites, so a
    consumer that keeps the surface must copy it before the next call. This is true of
    the batch API too, but a streaming consumer is far more likely to still be holding
    a result when the next search starts.

If the consumer has a deadline of its own, combine this with `max_blocking_time` (see
[Acquiring next to a real-time loop](@ref)): the PRN chunks occupy every thread while
they run, and that setting bounds how long they can keep another task in the process
waiting.

## Non-coherent Integration

At low CN0, accumulate power across multiple successive signal segments:

```@example guide
plan_ni = plan_acquire(system, sampling_freq, [1];
    num_coherently_integrated_code_periods = 10,
    num_noncoherent_accumulations = 8)

long_signal = generate_test_signal(system, 1;
    num_samples = 8 * 10 * 4096,
    sampling_freq = sampling_freq, CN0 = 30).signal

result_ni = acquire!(plan_ni, long_signal, [1])
nothing # hide
```

The signal must contain at least
`num_noncoherent_accumulations × num_coherently_integrated_code_periods × samples_per_code`
samples.

## Plotting Results

Acquisition results can be plotted directly with Plots.jl.
Pass `store_power_bins = true` to retain the correlation power surface.

The three examples below use the same PRN 1 signal at default CN0 (45 dB-Hz) and
show how coherent and non-coherent integration affect the correlation surface.
A single 2 ms signal is generated once; the 1 ms baseline uses only the first half.

### 1 ms coherent integration (baseline)

1 ms gives ~1000 Hz Doppler bin spacing — coarse but fast:

```@example guide
using Plots
plotlyjs()

fs = 4e6Hz
prn = 1
doppler = 1500Hz

# Generate one 2 ms signal (8000 samples at 4 MHz) — reused across all three plots
signal_2ms = generate_test_signal(system, prn;
    num_samples = 8000, sampling_freq = fs, doppler = doppler).signal

plan_1ms = plan_acquire(system, fs, [prn])
result_1ms = acquire!(plan_1ms, signal_2ms[1:4000], [prn]; store_power_bins = true)
plot(result_1ms[1])
```

### 2 ms coherent integration

2× longer integration → 2× finer Doppler bins (~500 Hz spacing).
The correlation spike narrows visibly in the Doppler dimension:

```@example guide
plan_2ms = plan_acquire(system, fs, [prn];
    num_coherently_integrated_code_periods = 2)
result_2ms = acquire!(plan_2ms, signal_2ms, [prn]; store_power_bins = true)
plot(result_2ms[1])
```

### 1 ms coherent + 2 non-coherent accumulations

Non-coherent integration adds power from 2 successive 1 ms segments,
improving sensitivity without requiring longer phase-coherent integration.
The Doppler resolution stays at ~1000 Hz but the peak-to-noise ratio improves:

```@example guide
plan_ni = plan_acquire(system, fs, [prn]; num_noncoherent_accumulations = 2)
result_ni = acquire!(plan_ni, signal_2ms, [prn]; store_power_bins = true)
plot(result_ni[1])
```

---

## Algorithm Constraints and Trade-offs

The FM-DBZP algorithm (Heckler & Garrison 2009) has different constraints from a
classical serial Doppler search. Understanding them is important for choosing
acquisition parameters.

### Doppler Resolution and Coverage

The coherent integration time `T_coh` determines **both** the Doppler resolution
and the Doppler bin spacing:

```
T_coh = num_coherently_integrated_code_periods × samples_per_code / sampling_freq

Doppler bin spacing = 1 / T_coh
```

For GPS L1 C/A at 4 MHz (`samples_per_code = 4092`):

| Integration length | `T_coh` | Doppler bin spacing |
|--------------------|---------|---------------------|
| 1 ms (1 code period) | 1 ms | ~1000 Hz |
| 2 ms | 2 ms | ~500 Hz |
| 5 ms | 5 ms | ~200 Hz |
| 10 ms | 10 ms | ~100 Hz |
| 20 ms | 20 ms | ~50 Hz |

Unlike a classical search where you can choose any Doppler step independently of
integration length, **FM-DBZP fixes the Doppler step at `1 / T_coh`**. To get
finer Doppler resolution you must integrate longer.

The total Doppler coverage is:

```
Doppler coverage = num_blocks / T_coh = num_blocks × Doppler bin spacing
```

where `num_blocks` is chosen automatically to cover at least `min_doppler_coverage`
on each side (default ±7000 Hz). You can widen the search with:

```julia
plan = plan_acquire(system, sampling_freq, prns; min_doppler_coverage = 10_000Hz)
```

#### What `min_doppler_coverage` actually guarantees

The Doppler grid stored in `plan.doppler_freqs` is

```
range(-coverage/2, step = bin_spacing, length = num_doppler_bins)
```

where `coverage = num_blocks × (sampling_freq / samples_per_code)` and
`num_doppler_bins = num_coherently_integrated_code_periods × num_blocks`.
This is a half-open interval — the grid spans `[-coverage/2, +coverage/2)`,
so the **highest searched bin is `+coverage/2 - bin_spacing`**, not `+coverage/2`.

`min_doppler_coverage` is the *minimum guaranteed reach on both ends*:
`plan_acquire` chooses `num_blocks` such that

```
last(plan.doppler_freqs)  ≥ +min_doppler_coverage
first(plan.doppler_freqs) ≤ -min_doppler_coverage
```

Concretely, with the default `min_doppler_coverage = 7000Hz`:

| `sampling_freq` | `T_coh` | `num_blocks` | `bin_spacing` | `plan.doppler_freqs` |
|---|---|---|---|---|
| 2.048 MHz | 1 ms | 16 | 1000 Hz | `-8000 : 1000 : +7000 Hz` |
| 4 MHz | 1 ms | 16 | 1000 Hz | `-8000 : 1000 : +7000 Hz` |
| 4 MHz | 10 ms | 16 | 100 Hz | `-8000 : 100 : +7900 Hz` |
| 5 MHz | 1 ms | 20 | 1000 Hz | `-10000 : 1000 : +9000 Hz` |
| 36 MHz | 1 ms | 16 | 1000 Hz | `-8000 : 1000 : +7000 Hz` |

The asymmetry is a consequence of the FFT bin layout: a length-N DFT covers
exactly N bins worth of bandwidth, and centering those N bins on 0 leaves the
upper edge open. It is *not* a bug — the highest *searched* Doppler is the
last bin, and that bin is guaranteed to be ≥ `+min_doppler_coverage`.

### The `num_blocks` Divisibility Constraint

`num_blocks` must divide `samples_per_code` exactly so that each block has an
integer number of samples (`block_size = samples_per_code ÷ num_blocks`).
`plan_acquire` finds the smallest valid divisor automatically, but this means
**not all sampling frequencies support all Doppler coverages**.

For example, at 2.048 MHz (`samples_per_code = 2048`) the valid divisors are
1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 — so `num_blocks` will always
be a power of two and Doppler coverage is always a multiple of the bin spacing.
At 5 MHz (`samples_per_code = 5000`) the divisors include 5, 10, 20, 25, … —
giving more choices but potentially a larger jump to the next valid `num_blocks`.

When the divisors of `samples_per_code` are far apart, the smallest one that
reaches the coverage can be far larger than needed. `plan_acquire` then searches
the Doppler range in several narrow passes with a smaller divisor instead (see
[Doppler Passes](#Doppler-Passes)). It throws an `ArgumentError` only when the
requested coverage is wider than the sampling frequency itself.

#### Why the *smallest* valid divisor?

Once `min_doppler_coverage` fixes a lower bound, several divisors of
`samples_per_code` are usually admissible (e.g. at 5 MHz with ±9 kHz requested,
both `num_blocks = 20` and `num_blocks = 25` are valid). It is tempting to pick
a larger divisor: that shrinks `block_size` and thus the inner double-block
FFT of size `2 × block_size`. In practice the opposite is the right default —
`plan_acquire` picks the *smallest* valid divisor, which corresponds to the
narrowest Doppler coverage that still meets your request.

The reason is that acquisition has two FFT stages with opposite scaling in
`num_blocks`:

- **Inner double-block stage.** `num_blocks` forward + inverse FFTs of size
  `2 × block_size = 2 × samples_per_code / num_blocks`. Total work scales as
  `2 × samples_per_code × log₂(2 × samples_per_code / num_blocks)` — it
  shrinks *logarithmically* as `num_blocks` grows.
- **Column FFT stage.** `samples_per_code` FFTs of size
  `num_doppler_bins = num_coherently_integrated_code_periods × num_blocks`.
  Total work scales as
  `samples_per_code × num_doppler_bins × log₂(num_doppler_bins)` — it grows
  roughly *linearly* (× log) in `num_blocks`. The power reduction sweeps
  `num_doppler_bins × samples_per_code` cells (streamed tile by tile), so its
  work scales with `num_blocks` as well.

A concrete comparison at 4 MHz, `num_coherently_integrated_code_periods = 1`:

| `num_blocks` | inner FFT cost | column FFT cost | total |
|---|---|---|---|
| 16 (≈±8 kHz) | 8000 · log₂(500) ≈ 71.7 k | 4000 · 16 · log₂(16) = 256 k | **≈ 328 k** |
| 20 (±10 kHz) | 8000 · log₂(400) ≈ 69.1 k | 4000 · 20 · log₂(20) ≈ 346 k | **≈ 415 k** |

The inner stage barely changes; the column stage grows ~35 %, and overall cost
rises by ~26 %. The gap widens with longer coherent integration, because
`num_doppler_bins` then equals `num_coherently_integrated_code_periods × num_blocks`,
so every extra divisor unit costs `num_coherently_integrated_code_periods` extra
column-FFT bins.

The inner FFT size no longer factors into this choice. It is zero-padded to a
fast length regardless of `block_size` (see
[Sampling Frequency and FFT Performance](#Sampling-Frequency-and-FFT-Performance)),
so a `block_size` with a large prime factor no longer penalises a smaller
divisor — the smallest valid divisor is unconditionally the right default. The
only prime-factorization concern left is the column FFT length `num_doppler_bins`,
which the [`recommend_sampling_freqs`](@ref) helper accounts for.

This is the rule for a *single* Doppler grid. When that divisor is far above
the need (sparse divisors), `plan_acquire` also weighs every smaller divisor
split into narrow passes, and picks that instead when it is clearly cheaper (see
[Doppler Passes](#Doppler-Passes)).

### Sampling Frequency and FFT Performance

Acquisition's cost is dominated by FFTs (computed with
[FFTL.jl](https://github.com/JuliaMath/FFTL.jl)), which run fastest when the
transform lengths factor into small primes — best of all powers of two — and
slower when they carry a large prime (11, 13, 31, 257, …). Both lengths are ultimately set by the
sampling frequency, since `samples_per_code = ceil(code_length × fs / code_freq)`.
There are two FFT stages, and they behave very differently:

- **Inner double-block FFT** (size `2 × block_size`) — handled automatically.
  `plan_acquire` zero-pads it up to the next length of the form `2^a` times at
  most two factors from `{3, 5, 7}` (e.g. `250 → 256`, `1250 → 1280`,
  `2046 → 2048`), so neither a `block_size` with a large prime factor nor one
  with a long radix-5 chain selects a slow kernel. The padded transform returns
  the same correlation values: any length `≥ 2 × block_size` preserves the kept
  correlation lags, and the inverse FFT is normalised by the actual length. This used to be the biggest trap — e.g.
  16.368 MHz (= 16 × 1.023 MHz) gives `block_size 1023 → 2046 = 2·3·11·31`, whose
  radix-31 FFT was ~5× slower than the neighbouring 16.384 MHz rate; padding to
  `2048 = 2¹¹` closes that gap. You no longer need to avoid a rate on the inner
  FFT's account.

- **Column FFT** (size `num_doppler_bins = num_coherently_integrated_code_periods ×
  num_blocks`) — **not** zero-paddable, because it is a true DFT over the Doppler
  axis and padding would move its bins. With a single Doppler grid, an unlucky `fs`
  inflates `num_doppler_bins` *and* lands it on a large prime: `num_blocks` must
  divide `samples_per_code` (see
  [The `num_blocks` Divisibility Constraint](#The-num_blocks-Divisibility-Constraint)),
  and when its factorization is sparse the smallest divisor that reaches the
  coverage can be far too large. The pathological case is 1.542 MHz
  (`1542 = 2·3·257`): its only divisors are `1, 2, 3, 6, 257, 514, 771, 1542`, so
  one grid reaching ±7 kHz needs 257 blocks — 257 Doppler bins of 6-sample blocks,
  ~13× more cells than the 20 bins of the 1.500 MHz neighbour. `plan_acquire`
  avoids this automatically by splitting the search into Doppler passes, below.

#### Doppler Passes

Instead of one wide grid, `plan_acquire` can cover the Doppler range with `P`
narrow passes that use a small divisor `B`. Pass `k` downconverts the signal by
an extra offset `f_k`, which moves the band around `f_k` to baseband, and runs the
unchanged FM-DBZP pipeline with `B` blocks. Each pass keeps its central bins, and
the passes tile one uniform Doppler grid, so `power_bins`, the noise estimators
and the CFAR cell count work exactly as with a single grid. The samples are never
resampled. At 1.542 MHz, ±7 kHz becomes 3 passes of 6 blocks (each keeping 5 of
its 6 bins), a 15-bin grid from −7 to +7 kHz.

FM-DBZP correlates block by block, so a Doppler `f` away from the grid centre
loses `sinc²(f · T_block)` (`T_block` = block duration). A pass with few blocks
has long blocks, so its edges lose more. The passes therefore keep only bins
whose loss stays within what a single grid loses at its coverage edge on an
*ideal* rate — one where the smallest divisor reaching the coverage is exactly
the minimum needed (−2.9 dB for ±7 kHz at 1 ms, e.g. 16 blocks at 2.048 MHz). No
Doppler in the searched range is worse off than on a well-chosen rate. Near
0 Hz, though, a grid with more blocks than it needs is nearly loss-free, and
passes give up some of that.

For searches with negligible code drift, passes are only considered on
sparse-divisor rates, where the single
grid needs at least 1.9× the blocks the coverage asks for (1.542 MHz: 257 for
16; 1.023 and 3.069 MHz: 31 for 16). Rates whose next divisor is closer — 1.5 MHz
(20 for 16), 2.048 MHz at ±10 kHz (32 for 22), 4 MHz at ±25 kHz (80 for 52) —
keep their single grid. On a sparse-divisor rate `plan_acquire` chooses between
the single grid and every pass layout with a cost model fitted to measured
`acquire!` timings (see [`recommend_sampling_freqs`](@ref)), and only leaves the
single grid when passes are expected to be at least 10% cheaper. Pass
`max_doppler_passes = 1` to always use one grid.

Measured on GPS L1 C/A, 32 PRNs, `min_doppler_coverage = 7000Hz`, one thread
(illustrative — absolute times are machine-dependent):

| Sampling freq | single grid | `acquire!(1:32)` | with Doppler passes | `acquire!(1:32)` |
|:--------------|:-----------:|:----------------:|:-------------------:|:----------------:|
| 1.500 MHz | 20 blocks | 2.10 ms | (not sparse: single grid) | 2.10 ms |
| **1.542 MHz** | **257 blocks** | **111.4 ms** | **3 × 6 blocks** | **1.83 ms** |
| **3.069 MHz** | **31 blocks** | **12.6 ms** | **2 × 11 blocks** | **4.53 ms** |

So a 0.1 % change in `fs` (1.500 → 1.542 MHz) no longer costs ~50×. The same
applies at longer coherent integration: 1.542 MHz with
`num_coherently_integrated_code_periods = 10` goes from 1.29 s to 18 ms.
For these low-drift L1 searches, rates with suitable divisors keep their single grid.

For signals whose code Doppler would drift by more than 0.1 chip across the
requested coherent window and Doppler coverage, the planner also uses narrow
passes to compensate that drift. Each pass delays the signal by its centre's
code advance once per code period, using phase ramps on the signal-block FFTs
shared by all PRNs. Phase-ramp coefficients are computed when the plan is built;
only complex multiplications are repeated during acquisition. The cache stores
one frequency half per code period and pass, reusing it across blocks and
noncoherent steps. The cost model counts both the extra passes and their signal
preparation. Low-drift searches (GPS L1 C/A at ±7 kHz through 20 ms, for example)
keep the original layout and FFTs exactly.

Pass widths target at most 0.1 chip of residual drift across the coherent
window, in addition to the uncompensated drift within one code period. This is
an approximation, not a power-loss guarantee: fractional delays of sampled
codes and block boundaries also affect the peak. By default,
`max_doppler_passes = nothing` lets the cost model choose the pass count
without a fixed cap, including for wide searches such as L5 at ±15 kHz over
20 ms. Each pass performs signal preparation and a PRN search, so the phase
ramp's small cost does not make extra passes free. Fewer blocks per pass can
offset that work, and the cost model compares the complete layouts. Shared
signal preparation runs serially, so its cost is amortized over the PRNs per
active worker, accounting for the Julia thread count and available cores.
The plan is optimized for the PRN batch passed to `plan_acquire`; searching a
subset can change the preparation-to-correlation balance.

An explicit integer `max_doppler_passes` limits time and cache memory by
restricting the available layouts. If none meets the tolerance, the planner
falls back to the original layout. `max_doppler_passes = 1` keeps the
uncompensated single grid. Code-drift correction between noncoherent steps continues to use each
bin's true Doppler.

### Picking a Good Sampling Frequency

When you have flexibility in the front-end clock,
[`recommend_sampling_freqs`](@ref) sweeps a frequency range and returns the
candidates with the smoothest FFT factorizations and lowest estimated cost.
It accounts for the [`num_blocks` divisibility constraint](#The-num_blocks-Divisibility-Constraint),
both FFT stages of the algorithm (inner double-block + column FFT), and the
[Doppler passes](#Doppler-Passes) `plan_acquire` would use, so it reports exactly
the layout the plan will run.

```@example guide
using Acquisition, GNSSSignals
import Unitful: Hz

recommend_sampling_freqs(GPSL1CA();
    fs_min = 2e6Hz,
    fs_max = 5e6Hz,
    min_doppler_coverage = 7000Hz,
    num_coherently_integrated_code_periods = 1,
    num_alternatives = 5,
)
```

For a custom code geometry (e.g. a non-standard system) pass `code_length` and
`code_freq` directly:

```@example guide
recommend_sampling_freqs(10_000, 36e6Hz;
    fs_min = 36e6Hz,
    fs_max = 40e6Hz,
    min_doppler_coverage = 10_000Hz,
    num_coherently_integrated_code_periods = 36,
    num_alternatives = 5,
)
```

By default candidates are ranked by estimated FFT cost; pass
`sort_by = :smoothness` to rank by largest prime factor of `num_doppler_bins`
(the column FFT) instead. Use `max_prime` to tighten or relax the smoothness
budget (default `7`). Both apply to the column FFT only —
the inner FFT is zero-padded to a fast size regardless, so a rate like
16.368 MHz (non-smooth `samples_per_code`, but smooth `num_doppler_bins`) is now
recommended rather than rejected. Likewise a sparse-divisor rate such as
1.542 MHz is recommended with its Doppler-pass layout; pass
`max_doppler_passes = 1` to restrict the search to single-grid rates.

#### Filtering against an SDR's hardware constraints

Most SDRs can only reach a discrete subset of sampling frequencies — the
output of a master clock, divider tree, and PLL. Pass an
[`AbstractSDRClockPlan`](@ref) via `sdr_clock_plan` to skip rates the
hardware can't produce:

```@example guide
recommend_sampling_freqs(GPSL1CA();
    fs_min = 2e6Hz,
    fs_max = 8e6Hz,
    min_doppler_coverage = 7000Hz,
    num_alternatives = 5,
    sdr_clock_plan = AD9361ClockPlan(),
)
```

[`AD9361ClockPlan`](@ref) reproduces the validation done by the AD9361 driver
shipped with `litex_m2sdr` (550 kHz – 61.44 MSPS, divider search through
`{12, 8, 6, 4, 3, 2, 1}` against an ADC clock window of 25–640 MHz, and a
reachable BBPLL). Other SDRs can be supported by subtyping
[`AbstractSDRClockPlan`](@ref) and implementing
[`is_valid_sample_rate`](@ref) (and optionally [`sample_rate_range`](@ref)).

### Coherent Integration with Data Bits (GPS L1 C/A)

GPS L1 C/A data bits flip every 20 ms. Integrating across a bit transition cancels
signal energy. The library handles this automatically:

**Windows of at most one bit** (`num_coherently_integrated_code_periods ≤ 20`):
At most one bit transition can fall in the window. Use `bit_edge_search_steps > 1`
to search for it: the window is split into `bit_edge_search_steps` equal parts,
and a sign flip is tried at each of the `bit_edge_search_steps − 1` inner part
boundaries, next to the no-transition hypothesis. Each hypothesis costs one
Doppler FFT, so the search runs about `bit_edge_search_steps` times as long as
without it. `bit_edge_search_steps` must divide the window length in code
periods:

```julia
plan = plan_acquire(system, sampling_freq, prns;
    num_coherently_integrated_code_periods = 10,
    bit_edge_search_steps = 10)
```

**Multi-bit integration** (`num_coherently_integrated_code_periods ≥ 20`):
The window spans whole data bit periods. Two constraints apply:
1. `num_coherently_integrated_code_periods` must be divisible by `bit_period_codes` (= 20 for GPS L1)
2. `bit_period_codes` must be divisible by `bit_edge_search_steps`

The algorithm searches all `2^(num_data_bits - 1)` sign-flip combinations across
the data bits in the window. With `bit_edge_search_steps > 1` the first bit edge
is also placed at `bit_edge_search_steps` evenly spaced positions within a bit
period; away from the window start the window then holds a partial bit at each
end, and all `2^num_data_bits` sign combinations of the partial and whole bits
are searched.

**Pilot channels** (e.g. GPS L5-Q, Galileo E1-C): no data bit constraint; set
`num_coherently_integrated_code_periods` freely.

### Sub-sample Interpolation

By default, code phase and Doppler estimates are quantised to the search grid:

- Code-phase step = `1 / sampling_freq` in seconds (0.25 µs at 4 MHz ≈ 0.25 chips for GPS L1 C/A).
- Doppler step = `1 / T_coh` (see the [Doppler Resolution and Coverage](#Doppler-Resolution-and-Coverage) table).

Pass `subsample_interpolation = true` to refine both below the grid spacing
using a parabolic fit across the peak bin and its two neighbours:

```@example guide
result_interp = acquire(system, signal, sampling_freq, 1;
    interm_freq, subsample_interpolation = true)
```

The fit is only applied when the neighbouring bins exceed `√noise_power`
(so it doesn't chase noise on non-detections). With `store_power_bins = true`
the neighbour cells are read back from the stored surface; without it they are
recomputed on demand for the up-to-three columns involved — about
`1/block_size` of one PRN's work, once per PRN. Either way the cost is
negligible compared to the acquisition itself.

### Storing the Power Surface

Pass `store_power_bins = true` to keep the full Doppler × code-phase
correlation matrix in the returned result. It's required for plotting
(see [Plotting Results](#Plotting-Results)) and useful for post-hoc analysis.
Without it the `power_bins` field is `nothing` — and at
`num_noncoherent_accumulations = 1` the surface is never even computed: the
peak and noise statistics are reduced on the fly while the pipeline streams
through the search grid one column block at a time. Opting in allocates one
cached buffer per PRN on first use, which subsequent calls reuse in place, so
repeated stored acquisitions stay allocation-free too.

```julia
result = acquire!(plan, signal, [1]; store_power_bins = true)
```
