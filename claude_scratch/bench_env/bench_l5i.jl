using Pkg
Pkg.activate(@__DIR__)

using Acquisition, GNSSSignals, Random, Unitful, Printf, BenchmarkTools
import Unitful: Hz

function bench()
    system = GPSL5I()
    prn = 1
    sampling_freq = 12e6Hz
    N_coh = 10

    plan = plan_acquire(system, sampling_freq, [prn];
        num_coherently_integrated_code_periods = N_coh,
        num_noncoherent_accumulations = 1)

    Random.seed!(42)
    (; signal) = generate_test_signal(system, prn;
        num_samples = N_coh * plan.samples_per_code,
        doppler = 1000Hz, code_phase = 5115.0,
        sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = 45,
        seed = 42)

    # Warmup
    acquire!(plan, signal, prn; interm_freq = 0.0Hz)

    println("=== Single PRN, L5I 10 ms coherent, rotation search active ===")
    println("samples_per_code=$(plan.samples_per_code), num_blocks=$(plan.num_blocks), num_doppler_bins=$(length(plan.doppler_freqs))")
    println("num_secondary_rotations=$(plan.num_secondary_rotations)")
    println("pattern matrix: $(size(plan.sign_patterns_by_prn[prn]))")

    b = @benchmark acquire!($plan, $signal, $prn; interm_freq = 0.0Hz) samples=20 seconds=15
    show(stdout, MIME"text/plain"(), b)
    println()
    @printf("\nmedian: %.2f ms   minimum: %.2f ms   mean: %.2f ms\n",
        median(b.times) / 1e6, minimum(b.times) / 1e6, mean(b.times) / 1e6)
end

bench()
