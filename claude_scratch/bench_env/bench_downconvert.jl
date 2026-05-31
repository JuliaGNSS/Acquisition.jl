# Time _downconvert! in isolation to scope the FastSinCos.jl payoff.
using Pkg
Pkg.activate(@__DIR__)
using Acquisition, BenchmarkTools, Printf

segment_length    = 120_000
sampling_freq_hz  = 12e6
interm_freq_hz    = 1.5e6           # non-trivial IF so sincos actually runs
signal_complexf64 = randn(ComplexF64, segment_length + 1000)
sig_buf           = zeros(ComplexF32, segment_length)

# Warmup
Acquisition._downconvert!(sig_buf, signal_complexf64, 1, segment_length, interm_freq_hz, sampling_freq_hz)

println("=== _downconvert!  segment_length=$segment_length, IF=$(interm_freq_hz/1e6) MHz, fs=12 MHz ===")
b = @benchmark Acquisition._downconvert!($sig_buf, $signal_complexf64, 1, $segment_length, $interm_freq_hz, $sampling_freq_hz) samples=50 seconds=10
@printf("median = %.3f ms  (min %.3f ms, mean %.3f ms)\n",
        median(b.times)/1e6, minimum(b.times)/1e6, mean(b.times)/1e6)

# For comparison: with IF=0 (the no-sincos fast path)
b0 = @benchmark Acquisition._downconvert!($sig_buf, $signal_complexf64, 1, $segment_length, 0.0, $sampling_freq_hz) samples=50 seconds=10
@printf("IF=0 fast path: median = %.3f ms\n", median(b0.times)/1e6)
