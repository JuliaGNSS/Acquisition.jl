# Acquisition.jl Benchmark Suite
#
# This file defines the main benchmark suite including CPU and GPU benchmarks.
#
# Usage:
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

const SUITE = BenchmarkGroup()

# ============================================================================
# Include CPU benchmarks
# ============================================================================

include(joinpath(@__DIR__, "cpu_benchmarks.jl"))

for (key, group) in CPU_SUITE
    SUITE["CPU $key"] = group
end

# ============================================================================
# Include KernelAbstractions GPU benchmarks (only if KAAcquisitionPlan is defined)
# ============================================================================

if isdefined(Acquisition, :KAAcquisitionPlan)
    include(joinpath(@__DIR__, "ka_benchmarks.jl"))

    for (key, group) in KA_SUITE
        SUITE["GPU $key"] = group
    end
end
