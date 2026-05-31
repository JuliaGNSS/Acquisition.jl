using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Acquisition, GNSSSignals, Random, Unitful, Printf, Statistics
import Unitful: Hz

function sweep_at_cn0(cn0)
    system = GPSL5I()
    prn = 1
    sampling_freq = 12e6Hz
    N_coh = 10
    plan = plan_acquire(system, sampling_freq, [prn];
        num_coherently_integrated_code_periods = N_coh,
        num_noncoherent_accumulations = 1)
    code_length = get_code_length(system)
    cps_chips = collect(range(0.0, stop = 10 * code_length, length = 60))[1:end-1]
    dopplers_hz = [0.0, 50.0, 100.0, 1000.0, 1050.0, 3333.0, 5000.0, 5050.0]
    num_cells = length(plan.doppler_freqs) * plan.samples_per_code
    thresh = cfar_threshold(1e-2, num_cells; num_noncoherent_integrations = 1)

    Random.seed!(0xDEADBEEF)
    n = 0; nd = 0; pns = Float64[]
    for cp in cps_chips, dop_hz in dopplers_hz
        n += 1
        (; signal) = generate_test_signal(system, prn;
            num_samples = N_coh * plan.samples_per_code,
            doppler = dop_hz * Hz, code_phase = mod(cp, code_length),
            sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = cn0, seed = n)
        r = acquire!(plan, signal, prn; interm_freq = 0.0Hz)
        push!(pns, r.peak_to_noise_ratio)
        is_detected(r) || (nd += 1)
    end
    @printf("CN0=%.0f dB-Hz: %3d/%d not detected (%4.1f%%)  median peak/N=%.1f  min=%.1f  p10=%.1f  threshold=%.1f\n",
        cn0, nd, n, 100*nd/n, median(pns), minimum(pns), quantile(pns, 0.10), thresh)
end

for cn0 in (40, 42, 45, 47, 50, 55)
    sweep_at_cn0(cn0)
end
