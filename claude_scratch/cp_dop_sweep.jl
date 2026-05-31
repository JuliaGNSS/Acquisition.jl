using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Acquisition, GNSSSignals, Random, Unitful, Printf
import Unitful: Hz

function run_sweep()
    system = GPSL5I()
    prn = 1
    sampling_freq = 12e6Hz
    N_coh = 10

    plan = plan_acquire(system, sampling_freq, [prn];
        num_coherently_integrated_code_periods = N_coh,
        num_noncoherent_accumulations = 1)
    println("plan: samples_per_code=$(plan.samples_per_code), num_blocks=$(plan.num_blocks), block_size=$(plan.block_size)")
    println("plan: num_secondary_rotations=$(plan.num_secondary_rotations), num_doppler_bins=$(length(plan.doppler_freqs))")
    doppler_step_hz = ustrip(Hz, step(plan.doppler_freqs))
    println("plan: doppler bin spacing = $doppler_step_hz Hz")

    code_length = get_code_length(system)
    NH10_periods = 10
    cps_chips = collect(range(0.0, stop = NH10_periods * code_length, length = 60))[1:end-1]
    dopplers_hz = [0.0, 50.0, 100.0, 1000.0, 1050.0, 3333.0, 5000.0, 5050.0]

    worst_cp_err = 0.0
    worst_dop_err = 0.0
    worst_pn = Inf
    best_pn = -Inf
    sum_dop_err = 0.0
    n_trials = 0
    cp_err_offenders = Tuple{Float64,Float64,Float64}[]
    dop_err_offenders = Tuple{Float64,Float64,Float64}[]

    Random.seed!(0xDEADBEEF)

    for cp in cps_chips
        cp_within_primary = mod(cp, code_length)
        for dop_hz in dopplers_hz
            n_trials += 1
            (; signal) = generate_test_signal(system, prn;
                num_samples = N_coh * plan.samples_per_code,
                doppler = dop_hz * Hz, code_phase = cp_within_primary,
                sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = 45,
                seed = n_trials)

            result = acquire!(plan, signal, prn; interm_freq = 0.0Hz, subsample_interpolation = true)

            cp_err = abs(result.code_phase - cp_within_primary)
            cp_err = min(cp_err, code_length - cp_err)
            dop_err = abs(ustrip(Hz, result.carrier_doppler) - dop_hz)
            pn = result.peak_to_noise_ratio

            worst_cp_err = max(worst_cp_err, cp_err)
            worst_dop_err = max(worst_dop_err, dop_err)
            worst_pn = min(worst_pn, pn)
            best_pn = max(best_pn, pn)
            sum_dop_err += dop_err

            if cp_err > 1.0
                push!(cp_err_offenders, (cp_within_primary, dop_hz, cp_err))
            end
            if dop_err > doppler_step_hz
                push!(dop_err_offenders, (cp_within_primary, dop_hz, dop_err))
            end
        end
    end

    @printf("Trials: %d  (cps = %d × dops = %d)\n", n_trials, length(cps_chips), length(dopplers_hz))
    @printf("Worst code-phase error: %.2f chips\n", worst_cp_err)
    @printf("Worst Doppler error:    %.2f Hz  (bin spacing %.1f Hz)\n", worst_dop_err, doppler_step_hz)
    @printf("Mean Doppler error:     %.2f Hz\n", sum_dop_err / n_trials)
    @printf("Peak/N range across all trials: [%.1f, %.1f]\n", worst_pn, best_pn)
    @printf("Code-phase offenders (>1 chip): %d / %d\n", length(cp_err_offenders), n_trials)
    @printf("Doppler offenders (> %.1f Hz):  %d / %d\n", doppler_step_hz, length(dop_err_offenders), n_trials)

    if !isempty(cp_err_offenders)
        println("First 5 CP offenders (cp, dop, err):")
        for off in first(cp_err_offenders, 5); println("  $off"); end
    end
    if !isempty(dop_err_offenders)
        println("First 10 Doppler offenders (cp, dop, err):")
        for off in first(dop_err_offenders, 10); println("  $off"); end
    end
end

run_sweep()
