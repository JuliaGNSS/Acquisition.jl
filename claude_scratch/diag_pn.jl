# Diagnose where the rotation-kernel peak/N gap vs LongL5I comes from at
# CN0=45 dB-Hz: is it signal peak power or noise-floor estimate?
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Acquisition, GNSSSignals, Random, Unitful, Printf, Statistics
import Unitful: Hz
include(joinpath(@__DIR__, "long_code.jl"))

system        = GPSL5I()
sampling_freq = 12e6Hz
prn           = 1
true_doppler  = 1000Hz
true_cp       = 5115.0

(; signal) = generate_test_signal(system, prn;
    num_samples = 10 * 12000,
    doppler = true_doppler, code_phase = true_cp,
    sampling_freq, interm_freq = 0.0Hz, CN0 = 45)

plan_rot = plan_acquire(system, sampling_freq, [prn];
    num_coherently_integrated_code_periods = 10, num_noncoherent_accumulations = 1)
plan_long = plan_acquire(LongL5I(), sampling_freq, [prn];
    num_coherently_integrated_code_periods = 1, num_noncoherent_accumulations = 1)

r_rot  = acquire!(plan_rot,  signal, prn; interm_freq = 0.0Hz, store_power_bins = true)
r_long = acquire!(plan_long, signal, prn; interm_freq = 0.0Hz, store_power_bins = true)

p_rot  = r_rot.power_bins
p_long = r_long.power_bins

# Peak powers
peak_rot  = maximum(p_rot)
peak_long = maximum(p_long)

# Noise floor estimates (manually compute mean of the matrix excluding the peak's row)
peak_row_rot,  _ = Tuple(argmax(p_rot))
peak_row_long, _ = Tuple(argmax(p_long))
# Mean of all cells excluding peak's row
mean_rot  = (sum(p_rot)  - sum(view(p_rot,  peak_row_rot,  :))) / (length(p_rot)  - size(p_rot,  2))
mean_long = (sum(p_long) - sum(view(p_long, peak_row_long, :))) / (length(p_long) - size(p_long, 2))

# Median across all cells
med_rot  = median(p_rot)
med_long = median(p_long)

# Also: median of the doppler bin OPPOSITE to peak (typical CFAR noise estimator)
opp_row_rot  = mod1(peak_row_rot  + size(p_rot,  1) ÷ 2, size(p_rot,  1))
opp_row_long = mod1(peak_row_long + size(p_long, 1) ÷ 2, size(p_long, 1))
opp_med_rot  = median(view(p_rot,  opp_row_rot,  :))
opp_med_long = median(view(p_long, opp_row_long, :))

println("Rotation kernel (GPSL5I, N_coh=10):")
@printf("  shape           = %s\n", size(p_rot))
@printf("  peak power      = %.3e\n", peak_rot)
@printf("  mean (ex-row)   = %.3e\n", mean_rot)
@printf("  median all      = %.3e\n", med_rot)
@printf("  median opp row  = %.3e\n", opp_med_rot)
@printf("  peak/median_all = %.2f\n", peak_rot / med_rot)
@printf("  peak/N from acquire! = %.2f\n", r_rot.peak_to_noise_ratio)
println()
println("LongL5I (N_coh=1):")
@printf("  shape           = %s\n", size(p_long))
@printf("  peak power      = %.3e\n", peak_long)
@printf("  mean (ex-row)   = %.3e\n", mean_long)
@printf("  median all      = %.3e\n", med_long)
@printf("  median opp row  = %.3e\n", opp_med_long)
@printf("  peak/median_all = %.2f\n", peak_long / med_long)
@printf("  peak/N from acquire! = %.2f\n", r_long.peak_to_noise_ratio)
println()
@printf("Peak ratio rot/long   = %.3f  (%.2f dB)\n",
        peak_rot/peak_long, 10*log10(peak_rot/peak_long))
@printf("Median ratio rot/long = %.3f  (%.2f dB inflation)\n",
        med_rot/med_long, 10*log10(med_rot/med_long))
