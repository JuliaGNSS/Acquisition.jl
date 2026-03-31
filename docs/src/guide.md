# Usage Guide

## Basic Acquisition

The simplest way to acquire GNSS signals is with the [`acquire`](@ref) function:

```@example guide
using Acquisition, GNSSSignals
import Unitful: Hz

system = GPSL1()

# Generate a synthetic GPS L1 signal for PRN 1
(; signal, sampling_freq, interm_freq) = generate_test_signal(system, 1)
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
- `power_bins`: Correlation power matrix

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

## Coarse-Fine Acquisition

For better Doppler resolution without the computational cost of a high-resolution search, use [`coarse_fine_acquire`](@ref):

```@example guide
results = coarse_fine_acquire(system, signal, sampling_freq, 1:3;
    interm_freq,
    coarse_step = 250Hz,
    fine_step = 25Hz
)
```

This performs an initial coarse search, then refines the Doppler estimate around detected peaks.

## Custom Doppler Range

Specify custom Doppler search bounds or a custom range:

```@example guide
# Custom bounds
results = acquire(system, signal, sampling_freq, 1:3;
    interm_freq,
    min_doppler = -5000Hz,
    max_doppler = 5000Hz
)
```

```@example guide
# Custom range with specific step
results = acquire(system, signal, sampling_freq, 1:3;
    interm_freq,
    dopplers = -7000Hz:100Hz:7000Hz
)
```

## Using Acquisition Plans

For repeated acquisitions with the same parameters, pre-compute an acquisition plan to avoid redundant FFT planning and memory allocation:

```@example guide
(; signal, num_samples, sampling_freq, interm_freq) = generate_test_signal(system, 1)

# Create plan once
plan = AcquisitionPlan(system, num_samples, sampling_freq; prns=1:3)

# Reuse for multiple signals
results = acquire!(plan, signal, 1:3; interm_freq)
```

Similarly, for coarse-fine acquisition:

```@example guide
plan = CoarseFineAcquisitionPlan(system, num_samples, sampling_freq; prns=1:3)
results = acquire!(plan, signal, 1:3; interm_freq)
```

## Plotting Results

Acquisition results can be plotted directly with Plots.jl:

```@example guide
using Plots
plotlyjs()

# Use a smaller signal for plotting (fewer samples = smaller power_bins matrix)
plot_data = generate_test_signal(system, 1; num_samples = 5000, sampling_freq = 2e6Hz)
plot_result = acquire(system, plot_data.signal, plot_data.sampling_freq, 1; interm_freq = plot_data.interm_freq, dopplers = 0Hz:100Hz:2000Hz)

# 3D surface plot of correlation power
plot(plot_result)
```

```@example guide
# Log scale (dB)
plot(plot_result, true)
```
