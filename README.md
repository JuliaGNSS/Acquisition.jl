[![Tests](https://github.com/JuliaGNSS/Acquisition.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaGNSS/Acquisition.jl/actions)
[![codecov](https://codecov.io/gh/JuliaGNSS/Acquisition.jl/branch/master/graph/badge.svg?token=GFRAHP6R3S)](https://codecov.io/gh/JuliaGNSS/Acquisition.jl)

# Acquisition.jl
Acquire GNSS signals

## Getting started

Install:
```julia
julia> ]
pkg> add Acquisition
```

## Usage

```julia
using Acquisition, Plots
import Acquisition: GPSL1, Hz
stream1 = open("signal.dat")
signal = Vector{Complex{Int16}}(undef, 10000)
read!(stream1, signal)
gpsl1 = GPSL1()
acq_res = [acquire(gpsl1, signal, 5e6Hz, prn) for prn = 1:32]
plot(acq_res[1])
```

![Acquisition plot](media/acquisition_plot.png)

The acquisition results include: `CN0`, `carrier_doppler`, `code_phase`, etc.

## License

MIT License
