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
separately (e.g. for plotting or custom logic).

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

If `plan_acquire` cannot find a valid `num_blocks` for your sampling frequency and
`min_doppler_coverage`, it throws an `ArgumentError`. Try a slightly different
sampling frequency or reduce `min_doppler_coverage`.

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

The inner FFT size no longer factors into this choice. It is zero-padded to an
FFTW-fast length regardless of `block_size` (see
[Sampling Frequency and FFT Performance](#Sampling-Frequency-and-FFT-Performance)),
so a `block_size` with a large prime factor no longer penalises a smaller
divisor — the smallest valid divisor is unconditionally the right default. The
only prime-factorization concern left is the column FFT length `num_doppler_bins`,
which the [`recommend_sampling_freqs`](@ref) helper accounts for.

### Sampling Frequency and FFT Performance

Acquisition's cost is dominated by FFTs, and FFTW runs fastest when the
transform lengths factor into small primes (2, 3, 5, 7) and slower when they
carry a large prime (11, 13, 31, 257, …). Both lengths are ultimately set by the
sampling frequency, since `samples_per_code = ceil(code_length × fs / code_freq)`.
There are two FFT stages, and they behave very differently:

- **Inner double-block FFT** (size `2 × block_size`) — handled automatically.
  `plan_acquire` zero-pads it up to the next `2·3·5·7`-smooth length
  (`fft_size = nextprod((2, 3, 5, 7), 2 × block_size)`), so a `block_size` with a
  large prime factor no longer selects a slow FFTW kernel. The padded transform
  runs in the fast regime and returns **bit-identical results**: any length
  `≥ 2 × block_size` preserves the kept correlation lags, and the inverse FFT is
  normalised by the actual length. This used to be the biggest trap — e.g.
  16.368 MHz (= 16 × 1.023 MHz) gives `block_size 1023 → 2046 = 2·3·11·31`, whose
  radix-31 FFT was ~5× slower than the neighbouring 16.384 MHz rate; padding to
  `2048 = 2¹¹` closes that gap. You no longer need to avoid a rate on the inner
  FFT's account.

- **Column FFT** (size `num_doppler_bins = num_coherently_integrated_code_periods ×
  num_blocks`) — **not** zero-paddable, because it is a true DFT over the Doppler
  axis and padding would move its bins. This is the remaining sampling-frequency
  sensitivity. `num_blocks` is the smallest divisor of `samples_per_code` that
  meets `min_doppler_coverage` (see
  [The `num_blocks` Divisibility Constraint](#The-num_blocks-Divisibility-Constraint)),
  so if every admissible divisor carries a large prime the column FFT is stuck
  with it. The pathological case is 1.542 MHz (`1542 = 2·3·257`): the only divisor
  `≥ min_num_blocks` is 257, forcing a radix-257 column FFT.

How costly is that? Measured on GPS L1 C/A, 32 PRNs, `min_doppler_coverage =
7000Hz` (illustrative — absolute times are machine- and thread-dependent, the
adjacent-rate ratios much less so):

| Sampling freq | `num_doppler_bins` | `acquire!(1:32)` | vs. smooth neighbour |
|:--------------|:------------------:|:----------------:|:--------------------:|
| 1.500 MHz | 20 (`2²·5`) | 0.6 ms | — |
| **1.542 MHz** | **257** (prime) | **21 ms** | **~35× slower** |
| 3.000 MHz | 20 (`2²·5`) | 0.9 ms | — |
| **3.069 MHz** | **31** (prime) | **3.0 ms** | **~3× slower** |

The inner FFT is padded in every row, so the whole gap is the column FFT: a
0.1 % change in `fs` (1.500 → 1.542 MHz) can cost ~35×. (Rates that *used* to be
slow only because of the inner FFT — e.g. 6.138 MHz, or 16.368 MHz above — are
no longer affected: padding handles them and `num_blocks` lands on a smooth
value.)

!!! note "Most of that ~35× is the bin-count jump, not the prime FFT kernel"
    It is tempting to read the penalty as "the length-257 FFT is a slow FFTW
    kernel," but that is the smaller effect. The dominant term is that
    `num_doppler_bins` itself jumps **20 → 257**: you genuinely resolve 257
    Doppler hypotheses instead of 20. From the cost model above, the power
    reduction sweeps `num_doppler_bins × samples_per_code` cells — `(257×1542) /
    (20×1500) ≈ 13×` more, with *zero* dependence on FFT smoothness — and the
    column FFT stage grows by `(257·log₂257) / (20·log₂20) ≈ 24×` from size
    alone, i.e. even if 257 were perfectly smooth. The non-smooth (prime) length
    adds only a further ~5× on top of the FFT stage itself. So a hypothetical
    smooth 256-bin rate would still be ~10× slower than the 20-bin neighbour:
    the fix is a rate with a *small smooth* admissible divisor (fewer bins),
    not merely a smooth `num_doppler_bins` of the same size.

    This is also why padding the column FFT to a smooth length would not help,
    beyond moving its bins: the exact "pad-to-smooth-then-correct" transform for
    a prime length is the chirp-z / Bluestein (or Rader) algorithm, which FFTW
    already applies internally for prime sizes — and it does not beat the native
    257-point transform here, let alone recover the bin-count cost.

As a rule of thumb for GPS L1 C/A, prefer a sampling frequency whose
`samples_per_code` (≈ `fs / 1000` in Hz, rounded up) is smooth. Powers of two
(2.048, 4.096, 8.192, 16.384 MHz) and `2^a · 5^b` rates (2, 2.5, 5, 10.24 MHz)
are always safe: every divisor is smooth, so `num_blocks` and the column FFT are
smooth too. The inner FFT is already taken care of by padding, so the only rates
worth avoiding now are those that force a large prime into `num_blocks`.

### Picking a Good Sampling Frequency

When you have flexibility in the front-end clock,
[`recommend_sampling_freqs`](@ref) sweeps a frequency range and returns the
candidates with the smoothest FFT factorizations and lowest estimated cost.
It accounts for the [`num_blocks` divisibility constraint](#The-num_blocks-Divisibility-Constraint)
and both FFT stages of the algorithm (inner double-block + column FFT).

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
budget (default `7`, FFTW's "fast" regime). Both apply to the column FFT only —
the inner FFT is zero-padded to a fast size regardless, so a rate like
16.368 MHz (non-smooth `samples_per_code`, but smooth `num_doppler_bins`) is now
recommended rather than rejected.

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

**Sub-bit integration** (`num_coherently_integrated_code_periods < 20`):
At most one bit transition can fall in the window. Use `bit_edge_search_steps > 1`
to search over candidate alignment positions:

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
the data bits in the window.

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
