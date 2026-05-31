using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Acquisition, GNSSSignals, Random, Unitful, Printf, Statistics
import Unitful: Hz
include(joinpath(@__DIR__, "long_code.jl"))

# Wider Doppler range — both on-grid and off-grid, low → high
dopplers_hz = [0.0, 50.0, 100.0, 500.0, 1000.0, 1050.0, 2500.0, 3333.0,
               5000.0, 5050.0, 7000.0, 7500.0, 10000.0, 12500.0, 15000.0,
               20000.0, 30000.0]

function per_doppler_stats(cn0, sys, label, doppler_step)
    prn = 1
    sampling_freq = 12e6Hz
    plan = if sys isa LongL5I
        plan_acquire(sys, sampling_freq, [prn];
            num_coherently_integrated_code_periods = 1,
            num_noncoherent_accumulations = 1,
            min_doppler_coverage = 30_000Hz)
    else
        plan_acquire(sys, sampling_freq, [prn];
            num_coherently_integrated_code_periods = 10,
            num_noncoherent_accumulations = 1,
            min_doppler_coverage = 30_000Hz)
    end
    num_cells = length(plan.doppler_freqs) * plan.samples_per_code
    thresh = cfar_threshold(1e-2, num_cells; num_noncoherent_integrations = 1)

    code_length_l5i = 10230
    cps_chips = collect(range(0.0, stop = 10 * code_length_l5i, length = 60))[1:end-1]

    println("\n=== $label  CN0=$(cn0) dB-Hz  doppler_step=$(round(doppler_step, digits=1)) Hz ===")
    @printf("  %-10s %10s %10s %10s %10s %10s %8s\n",
        "dop (Hz)", "n", "mean err", "median", "p90 err", "max err", "% > 1 bin")

    Random.seed!(0xDEADBEEF)
    n = 0
    for dop_hz in dopplers_hz
        errs = Float64[]
        for cp in cps_chips
            n += 1
            cp_within = mod(cp, code_length_l5i)
            (; signal) = generate_test_signal(GPSL5I(), prn;
                num_samples = 120000,
                doppler = dop_hz * Hz, code_phase = cp_within,
                sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = cn0, seed = n)
            r = acquire!(plan, signal, prn; interm_freq = 0.0Hz)
            push!(errs, abs(ustrip(Hz, r.carrier_doppler) - dop_hz))
        end
        @printf("  %-10.1f %10d %10.1f %10.1f %10.1f %10.1f %7.1f%%\n",
            dop_hz, length(errs), mean(errs), median(errs),
            quantile(errs, 0.90), maximum(errs),
            100 * count(>(doppler_step), errs) / length(errs))
    end
end

# Get the doppler step from a representative plan
sampling_freq = 12e6Hz
plan_fix = plan_acquire(GPSL5I(), sampling_freq, [1];
    num_coherently_integrated_code_periods = 10,
    min_doppler_coverage = 30_000Hz)
plan_long = plan_acquire(LongL5I(), sampling_freq, [1];
    num_coherently_integrated_code_periods = 1,
    min_doppler_coverage = 30_000Hz)
dop_step_fix  = ustrip(Hz, step(plan_fix.doppler_freqs))
dop_step_long = ustrip(Hz, step(plan_long.doppler_freqs))

per_doppler_stats(45, GPSL5I(), "GPSL5I (N_coh=10, with per-column shift fix)", dop_step_fix)
per_doppler_stats(45, LongL5I(), "LongL5I (N_coh=1 over 102_300-chip code)",     dop_step_long)
