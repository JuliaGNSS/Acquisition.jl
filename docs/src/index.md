# Acquisition.jl

GNSS signal acquisition using parallel code phase search.

Acquisition.jl is part of the [JuliaGNSS](https://github.com/JuliaGNSS) ecosystem and works with [GNSSSignals.jl](https://github.com/JuliaGNSS/GNSSSignals.jl) for signal definitions.

## Installation

```julia
using Pkg
Pkg.add("Acquisition")
```

## Quick Start

```julia
using Acquisition, GNSSSignals

# Load your signal data
signal = # ... your complex baseband samples

# Acquire GPS L1 C/A signals
results = acquire(GPSL1(), signal, 5e6Hz, 1:32)

# Or use coarse-fine acquisition for better Doppler resolution
results = coarse_fine_acquire(GPSL1(), signal, 5e6Hz, 1:32)
```

## Features

- **Parallel code phase search**: FFT-based correlation for fast acquisition
- **Coarse-fine acquisition**: Two-stage search for improved Doppler resolution
- **Pre-computed plans**: Reuse FFT plans and buffers for batch processing
- **Plotting support**: Visualize acquisition results with Plots.jl

## Background

For an introduction to GNSS signal acquisition, see:
- Kaplan & Hegarty, *Understanding GPS/GNSS: Principles and Applications*, Chapter 8
- [GPS signal acquisition (navipedia.net)](https://gssc.esa.int/navipedia/index.php/Acquisition)

## Contents

```@contents
Pages = ["guide.md", "api.md"]
Depth = 2
```
