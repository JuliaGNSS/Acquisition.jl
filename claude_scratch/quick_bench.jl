# Tight runtime check: time a single acquire! call with rotation search on a
# realistic L5I 10 ms test signal. Compares against the doc baseline of ~118 ms
# per PRN BEFORE the phase-ramp fix.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Acquisition, GNSSSignals, Unitful, BenchmarkTools, Printf
import Unitful: Hz

system        = GPSL5I()
sampling_freq = 12e6Hz
prn           = 1

(; signal) = generate_test_signal(system, prn;
    num_samples = 10 * 12000,
    doppler = 1234Hz, code_phase = 1234.5,
    sampling_freq, interm_freq = 0.0Hz, CN0 = 45)

plan = plan_acquire(system, sampling_freq, [prn];
    num_coherently_integrated_code_periods = 10, num_noncoherent_accumulations = 1)

# Warm up FFTW / compile
_ = acquire!(plan, signal, prn; interm_freq = 0.0Hz)

# 20 samples, 15 s budget — matches the doc's bench_l5i.jl
b = @benchmark acquire!($plan, $signal, $prn; interm_freq = 0.0Hz) samples=20 seconds=15
println("Acquire! rotation search (GPSL5I, N_coh=10, fs=12 MHz):")
@printf("  median:  %.2f ms\n", median(b).time / 1e6)
@printf("  minimum: %.2f ms\n", minimum(b).time / 1e6)
@printf("  mean:    %.2f ms\n", mean(b).time / 1e6)
@printf("  allocs:  %d allocs, %d bytes\n", b.allocs, b.memory)
