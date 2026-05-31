using Pkg
Pkg.activate(@__DIR__)
using Acquisition, GNSSSignals, Random, Unitful, Printf, BenchmarkTools
import Unitful: Hz
include(joinpath(@__DIR__, "long_code.jl"))

function bench()
    fs = 12e6Hz
    prn = 1
    Random.seed!(42)
    (; signal) = generate_test_signal(GPSL5I(), prn;
        num_samples = 120000,
        doppler = 1000Hz, code_phase = 5115.0,
        sampling_freq = fs, interm_freq = 0.0Hz, CN0 = 45, seed = 42)

    # Match default min_doppler_coverage for both plans for fairness.
    plan_fix = plan_acquire(GPSL5I(), fs, [prn];
        num_coherently_integrated_code_periods = 10,
        num_noncoherent_accumulations = 1)
    plan_long = plan_acquire(LongL5I(), fs, [prn];
        num_coherently_integrated_code_periods = 1,
        num_noncoherent_accumulations = 1)

    println("plan_fix  : samples_per_code=$(plan_fix.samples_per_code)  num_blocks=$(plan_fix.num_blocks)  num_doppler_bins=$(length(plan_fix.doppler_freqs))")
    println("plan_long : samples_per_code=$(plan_long.samples_per_code)  num_blocks=$(plan_long.num_blocks)  num_doppler_bins=$(length(plan_long.doppler_freqs))")

    # Warmup
    acquire!(plan_fix, signal, prn; interm_freq = 0.0Hz)
    acquire!(plan_long, signal, prn; interm_freq = 0.0Hz)

    b_fix  = @benchmark acquire!($plan_fix,  $signal, $prn; interm_freq = 0.0Hz) samples=20 seconds=15
    b_long = @benchmark acquire!($plan_long, $signal, $prn; interm_freq = 0.0Hz) samples=20 seconds=20

    @printf("\nGPSL5I N_coh=10 (with fix):  median=%.1f ms  min=%.1f ms\n",
            median(b_fix.times)/1e6,  minimum(b_fix.times)/1e6)
    @printf("LongL5I N_coh=1 (102300 chips): median=%.1f ms  min=%.1f ms\n",
            median(b_long.times)/1e6, minimum(b_long.times)/1e6)
    @printf("\nSlowdown of LongL5I vs GPSL5I-fix: %.2fx (median),  %.2fx (min)\n",
            median(b_long.times)/median(b_fix.times),
            minimum(b_long.times)/minimum(b_fix.times))
end

bench()
