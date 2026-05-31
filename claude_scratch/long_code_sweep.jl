using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Acquisition, GNSSSignals, Random, Unitful, Printf, Statistics
import Unitful: Hz
include(joinpath(@__DIR__, "long_code.jl"))

function sweep_at_cn0(cn0, sys, label)
    prn = 1
    sampling_freq = 12e6Hz
    # Acquisition plan dimensions depend on the system. We want a "10 ms coherent
    # integration" in both cases:
    #   - GPSL5I + N_coh=10: 10 × 1ms primary periods, rotation search active.
    #   - LongL5I + N_coh=1: 1 × 10ms long period, simple pilot path.
    plan = if sys isa LongL5I
        plan_acquire(sys, sampling_freq, [prn];
            num_coherently_integrated_code_periods = 1,
            num_noncoherent_accumulations = 1)
    else
        plan_acquire(sys, sampling_freq, [prn];
            num_coherently_integrated_code_periods = 10,
            num_noncoherent_accumulations = 1)
    end

    # Generate test signals with GPSL5I (the only system that knows how to bake
    # NH10 into samples), then acquire with either GPSL5I (N_coh=10 + fix) or
    # LongL5I (N_coh=1 long).
    code_length_l5i = 10230
    cps_chips = collect(range(0.0, stop = 10 * code_length_l5i, length = 60))[1:end-1]
    dopplers_hz = [0.0, 50.0, 100.0, 1000.0, 1050.0, 3333.0, 5000.0, 5050.0]
    num_cells = length(plan.doppler_freqs) * plan.samples_per_code
    thresh = cfar_threshold(1e-2, num_cells; num_noncoherent_integrations = 1)

    Random.seed!(0xDEADBEEF)
    n = 0; nd = 0; pns = Float64[]; dop_errs = Float64[]
    for cp in cps_chips, dop_hz in dopplers_hz
        n += 1
        cp_within = mod(cp, code_length_l5i)
        (; signal) = generate_test_signal(GPSL5I(), prn;
            num_samples = 120000,   # always 10 ms of signal
            doppler = dop_hz * Hz, code_phase = cp_within,
            sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = cn0, seed = n)
        r = acquire!(plan, signal, prn; interm_freq = 0.0Hz)
        push!(pns, r.peak_to_noise_ratio)
        push!(dop_errs, abs(ustrip(Hz, r.carrier_doppler) - dop_hz))
        is_detected(r) || (nd += 1)
    end
    @printf("%s @ CN0=%.0f dB-Hz: %3d/%d not det (%4.1f%%)  median p/N=%6.1f  min p/N=%5.1f  worst dop err=%5.1f Hz  threshold=%.1f\n",
        label, cn0, nd, n, 100*nd/n, median(pns), minimum(pns),
        maximum(dop_errs), thresh)
end

println("Comparison: 10ms coherent — GPSL5I (N_coh=10, fix) vs LongL5I (N_coh=1)")
println()
for cn0 in (40, 42, 45, 47, 50, 55)
    sweep_at_cn0(cn0, GPSL5I(), "GPSL5I-fix ")
    sweep_at_cn0(cn0, LongL5I(),  "LongL5I-N=1")
    println()
end
