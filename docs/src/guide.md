# Usage Guide

## Basic Acquisition

The simplest way to acquire GNSS signals is with the [`acquire`](@ref) function:

```@example guide
using Acquisition, GNSSSignals
import Unitful: Hz

system = GPSL1()

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

### Sampling Frequency and FFT Performance

Acquisition runs many in-place FFTs at size `2 × block_size`, where
`block_size = samples_per_code ÷ num_blocks`. FFTW is very fast when those sizes
factor into small primes (2, 3, 5, 7) and noticeably slower when they contain
a large prime factor such as 11, 13, 31, or 257. Since `samples_per_code`
itself is `ceil(code_length × fs / code_freq)`, its prime factorization is
decided entirely by your sampling frequency — a 0.1 % change in `fs` can make
the inner FFT 3-5× faster.

As a rule of thumb for GPS L1 C/A:

- Prefer sampling frequencies where `samples_per_code` (≈ `fs / 1000` in Hz,
  rounded up) has only small prime factors.
- Powers of two (2.048, 4.096, 8.192, 16.384 MHz) and
  `2^a · 5^b` rates (2, 2.5, 5, 10.24 MHz) are always fast.
- Avoid rates whose `samples_per_code` contains a prime ≥ 11. The commonly
  recommended 16.368 MHz (= 16 × 1.023 MHz) is a notable offender:
  `16368 = 2^4 · 3 · 11 · 31`, and the radix-31 inner FFT is ~5× slower than
  nearby smooth sizes.

Measured per-PRN `acquire!` times on a Ryzen 7 PRO 5850U (16 threads), GPS
L1 C/A, `min_doppler_coverage = 10 000 Hz`:

| Sampling freq | `samples_per_code` (factors) | 1 ms coherent | 20 ms coherent |
|---|---|---|---|
| **Fast** (smooth factorization): |||
| 1.500 MHz | `1500 = 2²·3·5³` | 0.19 ms | 4.0 ms |
| 2.000 MHz | `2000 = 2⁴·5³` | 0.21 ms | 4.5 ms |
| 2.048 MHz | `2048 = 2¹¹` | 0.28 ms | 10.1 ms |
| 2.500 MHz | `2500 = 2²·5⁴` | 0.34 ms | 7.5 ms |
| 4.000 MHz | `4000 = 2⁵·5³` | 0.39 ms | 10.7 ms |
| 4.096 MHz | `4096 = 2¹²` | 0.63 ms | 22.9 ms |
| 5.000 MHz | `5000 = 2³·5⁴` | 0.60 ms | 17.0 ms |
| 8.192 MHz | `8192 = 2¹³` | 1.3 ms | 50.3 ms |
| 10.240 MHz | `10240 = 2¹¹·5` | 1.1 ms | 32.4 ms |
| 16.384 MHz | `16384 = 2¹⁴` | 2.4 ms | 112 ms |
| **Slow** (large prime factor): |||
| 1.542 MHz | `1542 = 2·3·257` | **6.6 ms** | **167 ms** |
| 3.069 MHz | `3069 = 3²·11·31` | 1.3 ms | 33 ms |
| 6.138 MHz | `6138 = 2·3²·11·31` | 2.7 ms | 63 ms |
| 8.184 MHz | `8184 = 2³·3·11·31` | 3.4 ms | 82 ms |
| 16.368 MHz | `16368 = 2⁴·3·11·31` | 7.2 ms | 167 ms |

Notice the near-identical-rate pairs:
**8.192 MHz** runs at 50 ms vs. **8.184 MHz** at 82 ms (1.6× faster);
**16.384 MHz** runs at 112 ms vs. **16.368 MHz** at 167 ms (1.5× faster).
If you control the RF front-end clock, picking a smooth rate is usually the
single biggest performance lever available.

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
(so it doesn't chase noise on non-detections), and it costs four extra
array reads per PRN — negligible compared to the acquisition itself.

### Storing the Power Surface

Pass `store_power_bins = true` to keep the full Doppler × code-phase
correlation matrix in the returned result. It's required for plotting
(see [Plotting Results](#Plotting-Results)) and useful for post-hoc analysis;
without it the `power_bins` field is `nothing` and the plan's internal buffer
is reused across calls to keep acquisitions allocation-free.

```julia
result = acquire!(plan, signal, [1]; store_power_bins = true)
```
