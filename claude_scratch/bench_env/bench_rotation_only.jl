# Bench just _sign_search_step_with_rotations! to isolate parallelisation effect.
using Pkg
Pkg.activate(@__DIR__)
using Acquisition, GNSSSignals, BenchmarkTools, Printf, Random
import Unitful: Hz

system = GPSL5I(); prn = 1; sampling_freq = 12e6Hz; N_coh = 10
plan = plan_acquire(system, sampling_freq, [prn];
    num_coherently_integrated_code_periods = N_coh,
    num_noncoherent_accumulations = 1)

scratch  = plan.thread_scratch[1]
samples_per_code = plan.samples_per_code
num_doppler_bins = N_coh * plan.num_blocks
num_blocks       = plan.num_blocks
block_size       = plan.block_size

# Fill the CIM with random data
Random.seed!(42)
cim = ComplexF32.(randn(ComplexF64, num_doppler_bins, samples_per_code))
noncoh_buf = zeros(Float32, num_doppler_bins, plan.samples_per_code_eff)
tile_re = plan.tiled_phase_patterns_re_by_prn[prn]
tile_im = plan.tiled_phase_patterns_im_by_prn[prn]

# Warmup
Acquisition._sign_search_step_with_rotations!(
    noncoh_buf, cim, plan.thread_scratch, plan.col_fft_plan,
    samples_per_code, num_doppler_bins, N_coh, num_blocks, block_size,
    0, tile_re, tile_im)

println("=== _sign_search_step_with_rotations! direct bench ===")
println("nthreads = $(Threads.nthreads())")
println("samples_per_code = $samples_per_code, num_doppler_bins = $num_doppler_bins")
println("num_coh_periods = $N_coh, num_blocks = $num_blocks")

b = @benchmark Acquisition._sign_search_step_with_rotations!(
    $noncoh_buf, $cim, $(plan.thread_scratch), $(plan.col_fft_plan),
    $samples_per_code, $num_doppler_bins, $N_coh, $num_blocks, $block_size,
    0, $tile_re, $tile_im) samples=30 seconds=15

@printf("median: %.2f ms   min: %.2f ms   mean: %.2f ms\n",
        median(b.times)/1e6, minimum(b.times)/1e6, mean(b.times)/1e6)
