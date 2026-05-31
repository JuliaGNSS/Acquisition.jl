# Bench acquire! over multiple PRNs to see if the inner @batch cooperates
# (no regression) when the outer per=core PRN loop already fills cores.
using Pkg
Pkg.activate(@__DIR__)
using Acquisition, GNSSSignals, BenchmarkTools, Random, Unitful, Printf
import Unitful: Hz

system = GPSL5I()
sampling_freq = 12e6Hz
N_coh = 10
prns = 1:8       # 8 PRNs

plan = plan_acquire(system, sampling_freq, collect(prns);
    num_coherently_integrated_code_periods = N_coh,
    num_noncoherent_accumulations = 1)

Random.seed!(42)
(; signal) = generate_test_signal(system, 1;
    num_samples = N_coh * plan.samples_per_code,
    doppler = 1000Hz, code_phase = 5115.0,
    sampling_freq, interm_freq = 0.0Hz, CN0 = 45, seed = 42)

# Warmup
acquire!(plan, signal, collect(prns); interm_freq = 0.0Hz)

@printf("nthreads = %d   PRNs = %s\n", Threads.nthreads(), collect(prns))
b = @benchmark acquire!($plan, $signal, $(collect(prns)); interm_freq = 0.0Hz) samples=10 seconds=20
@printf("median: %.2f ms   min: %.2f ms\n", median(b.times)/1e6, minimum(b.times)/1e6)
