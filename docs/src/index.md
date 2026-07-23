# Acquisition.jl

GNSS signal acquisition using the FM-DBZP (Frequency-domain Modified Double Block Zero Padding) algorithm.

Acquisition.jl is part of the [JuliaGNSS](https://github.com/JuliaGNSS) ecosystem and works with [GNSSSignals.jl](https://github.com/JuliaGNSS/GNSSSignals.jl) for signal definitions.

## Installation

```julia
using Pkg
Pkg.add("Acquisition")
```

## Quick Start

```julia
using Acquisition, GNSSSignals
import Unitful: Hz

system = GPSL1CA()
(; signal, sampling_freq, interm_freq) = generate_test_signal(system, 1)

# Acquire GPS L1 C/A signals for PRNs 1–32
results = acquire(system, signal, sampling_freq, 1:32; interm_freq)

# Reuse a pre-computed plan for repeated acquisitions
plan = plan_acquire(system, sampling_freq, 1:32)
results = acquire!(plan, signal, 1:32; interm_freq)
```

!!! tip "Pick a fast sampling frequency"
    Acquisition performance depends on the **prime factorization** of
    `samples_per_code`. The inner FFT is zero-padded to a fast size
    automatically, but a rate that forces `num_blocks` onto a large prime
    (e.g. 1.542 MHz → 257) still gives a slow column FFT. See
    [Sampling Frequency and FFT Performance](@ref) for the rule of thumb, and
    [`recommend_sampling_freqs`](@ref) to pick a good rate.

## Algorithm Overview

This library implements the **FM-DBZP** algorithm from Heckler & Garrison (2009), which performs simultaneous correlation across all code phases and Doppler bins using a two-dimensional FFT structure.

The signal segment of length `N = num_coherently_integrated_code_periods × samples_per_code`
is divided into `num_blocks` equal sub-blocks. A row-wise inverse FFT performs circular
correlation against each PRN sub-block, forming a 2D coherent integration surface
`(num_doppler_bins × samples_per_code)`. A column-wise FFT then resolves the Doppler
dimension, producing a power surface that is peak-searched for acquisition.

The implementation never materialises these surfaces: it processes one
`num_doppler_bins × block_size` column-block tile at a time and reduces the power
cells on the fly, so memory stays tile-sized (cache-resident) no matter how large
the search grid is. The full power surface only exists if you request it with
`store_power_bins = true`.

## Features

- **FM-DBZP algorithm**: Joint code-phase and Doppler search via 2-D FFT structure
- **Pre-computed plans**: Reuse FFT plans and buffers across acquisitions with [`plan_acquire`](@ref)
- **Streaming, tile-based reduction**: The search surface is processed one cache-resident column block at a time — plan memory stays in the MiB range even for wide searches, and the full power surface is only materialised on request (`store_power_bins = true`)
- **Multi-threaded**: PRNs processed in parallel — start Julia with `-t N` to use N threads; extra threads cost only tile-sized scratch
- **Non-coherent integration**: Accumulate power across multiple signal segments to improve sensitivity at low CN0
- **Data bit handling**: Bit-edge search and sign-combination search for coherent integration spanning multiple data bits
- **Sub-sample interpolation**: Parabolic interpolation for Doppler and code-phase refinement below grid resolution
- **Plotting support**: Visualise acquisition results with Plots.jl

## Contents

```@contents
Pages = ["guide.md", "api.md"]
Depth = 2
```
