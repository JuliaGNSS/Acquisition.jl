# Measure plan memory footprint for GPS L5I rotation acquire.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Acquisition, GNSSSignals, Unitful, Printf
import Unitful: Hz

system = GPSL5I()
sampling_freq = 12e6Hz
N_coh = 10

function fmt(bytes)
    bytes < 1024 ? "$bytes B" :
    bytes < 1024^2 ? @sprintf("%.1f KiB", bytes/1024) :
    bytes < 1024^3 ? @sprintf("%.2f MiB", bytes/1024^2) :
                     @sprintf("%.2f GiB", bytes/1024^3)
end

for nprn in (1, 32)
    plan = plan_acquire(system, sampling_freq, collect(1:nprn);
        num_coherently_integrated_code_periods = N_coh,
        num_noncoherent_accumulations = 1)
    total = Base.summarysize(plan)
    println("="^72)
    @printf("GPSL5I N_coh=%d  fs=12 MHz  %d PRN(s)  nthreads=%d\n", N_coh, nprn, Threads.nthreads())
    @printf("  plan.samples_per_code     = %d\n",      plan.samples_per_code)
    @printf("  plan.samples_per_code_eff = %d  (×%d rotations)\n",
            plan.samples_per_code_eff,
            plan.samples_per_code_eff ÷ plan.samples_per_code)
    @printf("  num_doppler_bins          = %d\n", length(plan.doppler_freqs))
    @printf("  TOTAL plan size           = %s\n", fmt(total))
    println("Per-field breakdown:")
    for f in (
        :prn_conj_ffts, :signal_block_ffts, :noncoherent_integration_matrices,
        :thread_scratch, :sign_patterns_by_prn,
        :tiled_phase_patterns_re_by_prn, :tiled_phase_patterns_im_by_prn,
        :doppler_freqs, :fftshift_perm, :result_buffers, :acq_results_buf,
    )
        sz = Base.summarysize(getfield(plan, f))
        @printf("  %-36s %s\n", string(f), fmt(sz))
    end
    # Drill into one thread's scratch
    s = plan.thread_scratch[1]
    println("\n  per-thread scratch breakdown (thread 1):")
    for f in (
        :coherent_integration_matrix, :sign_search_max_buf,
        :noncoherent_integration_buf, :noncoherent_integration_accumulator,
        :sub_block_ffts, :col_buf, :combine_buf_re, :combine_buf_im,
        :row_buf, :row_shift_buf, :col_sums_buf, :sig_buf,
        :double_block_buf, :corr_buf,
    )
        sz = Base.summarysize(getfield(s, f))
        @printf("    %-36s %s\n", string(f), fmt(sz))
    end
    @printf("  one scratch total = %s\n", fmt(Base.summarysize(s)))
    @printf("  × %d threads      = %s\n", length(plan.thread_scratch),
            fmt(Base.summarysize(plan.thread_scratch)))
    println()
end
