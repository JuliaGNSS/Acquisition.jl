# Usage Guide

## Basic Acquisition

The simplest way to acquire GNSS signals is with the [`acquire`](@ref) function:

```julia
using Acquisition, GNSSSignals

# Acquire all GPS L1 C/A PRNs
results = acquire(GPSL1(), signal, 5e6Hz, 1:32)

# Acquire a single PRN
result = acquire(GPSL1(), signal, 5e6Hz, 1)
```

Each result contains:
- `carrier_doppler`: Estimated Doppler frequency
- `code_phase`: Code phase in chips
- `CN0`: Carrier-to-noise density ratio (dB-Hz)
- `power_bins`: Correlation power matrix

## Coarse-Fine Acquisition

For better Doppler resolution without the computational cost of a high-resolution search, use [`coarse_fine_acquire`](@ref):

```julia
results = coarse_fine_acquire(GPSL1(), signal, 5e6Hz, 1:32;
    coarse_step = 250Hz,  # Initial search step
    fine_step = 25Hz      # Refined search step
)
```

This performs an initial coarse search, then refines the Doppler estimate around detected peaks.

## Custom Doppler Range

Specify custom Doppler search bounds or a custom range:

```julia
# Custom bounds
results = acquire(GPSL1(), signal, 5e6Hz, 1:32;
    min_doppler = -5000Hz,
    max_doppler = 5000Hz
)

# Custom range with specific step
results = acquire(GPSL1(), signal, 5e6Hz, 1:32;
    dopplers = -7000Hz:100Hz:7000Hz
)
```

## Using Acquisition Plans

For repeated acquisitions with the same parameters, pre-compute an acquisition plan to avoid redundant FFT planning and memory allocation:

```julia
# Create plan once
plan = AcquisitionPlan(GPSL1(), 10000, 5e6Hz; prns=1:32)

# Reuse for multiple signals
for signal in signals
    results = acquire!(plan, signal, 1:32)
end
```

Similarly for coarse-fine acquisition:

```julia
plan = CoarseFineAcquisitionPlan(GPSL1(), 10000, 5e6Hz; prns=1:32)
results = acquire!(plan, signal, 1:32)
```

## Plotting Results

Acquisition results can be plotted directly with Plots.jl:

```julia
using Plots

# 3D surface plot of correlation power
plot(results[1])

# Log scale (dB)
plot(results[1], true)
```

## Interpreting Results

A successful acquisition typically shows:
- **CN0 > 35-40 dB-Hz**: Signal is likely present
- Clear correlation peak in the power bins

Print results in a table format:

```julia
julia> results
┌─────┬─────────────┬─────────────────────┬────────────────────┐
│ PRN │ CN0 (dBHz)  │ Carrier Doppler (Hz)│ Code phase (chips) │
├─────┼─────────────┼─────────────────────┼────────────────────┤
│   1 │      45.234 │             1250.0  │           234.567  │
│   2 │      32.100 │             -500.0  │           891.234  │
...
```

PRNs with CN0 > 42 dB-Hz are highlighted in green, others in red.
