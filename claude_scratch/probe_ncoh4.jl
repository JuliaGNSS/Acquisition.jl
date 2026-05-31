# Sanity-check the fix with N_coh = 4 (< L = 10): noiseless peak-power
# equivalence vs a 4 ms LongL5I-equivalent reference.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Acquisition, GNSSSignals, Unitful, Printf
import Unitful: Hz

const _Frequency = Union{Unitful.Quantity{T, Unitful.𝐓^-1, U},
                         Unitful.Level{L, S, Unitful.Quantity{T, Unitful.𝐓^-1, U}} where {L, S}} where {T, U}
# 4 ms reference: bake 4 primary periods × NH10 modulation into a 40920-chip
# "primary code" with no secondary, equivalent to N_coh=1 on a 4 ms code.
struct LongL5I4 <: GNSSSignals.AbstractGNSSSignal{Matrix{Int16}}
    inner::GPSL5I{Matrix{Int16}}
end
LongL5I4() = LongL5I4(GPSL5I())
GNSSSignals.get_code_length(::LongL5I4)             = 4 * 10_230
GNSSSignals.get_code_frequency(::LongL5I4)          = GNSSSignals.get_code_frequency(GPSL5I())
GNSSSignals.get_center_frequency(::LongL5I4)        = GNSSSignals.get_center_frequency(GPSL5I())
GNSSSignals.get_data_frequency(::LongL5I4)          = 0.0Hz
GNSSSignals.get_secondary_code(::LongL5I4)          = GNSSSignals.NoSecondaryCode()
GNSSSignals.get_secondary_code_length(::LongL5I4)   = 1
GNSSSignals.get_code_type(::LongL5I4)               = Int16
function GNSSSignals.gen_code!(buffer::AbstractVector, ::LongL5I4, prn::Integer,
                               sampling_frequency::_Frequency, code_frequency::_Frequency,
                               start_phase, start_index::Integer)
    GNSSSignals.gen_code!(buffer, GPSL5I(), prn, sampling_frequency,
                          code_frequency, start_phase, start_index)
end

system        = GPSL5I()
sampling_freq = 12e6Hz
prn           = 1
N_coh         = 4
spc           = 12000
N_total       = N_coh * spc

fc        = get_code_frequency(system)
code_full = ComplexF32.(gen_code(N_total, system, prn, sampling_freq, fc, 0.0))

plan_rot = plan_acquire(system, sampling_freq, [prn];
    num_coherently_integrated_code_periods = N_coh, num_noncoherent_accumulations = 1)
plan_ref = plan_acquire(LongL5I4(), sampling_freq, [prn];
    num_coherently_integrated_code_periods = 1, num_noncoherent_accumulations = 1)

@printf("N_coh = %d, num_secondary_rotations = %d\n", N_coh, plan_rot.num_secondary_rotations)
@printf("plan_rot doppler bins = %d, doppler spacing = %g Hz\n",
        length(plan_rot.doppler_freqs), ustrip(Hz, step(plan_rot.doppler_freqs)))
@printf("plan_ref doppler bins = %d, doppler spacing = %g Hz\n",
        length(plan_ref.doppler_freqs), ustrip(Hz, step(plan_ref.doppler_freqs)))

println()
println("Doppler  peak_rot         peak_ref         ratio   dB_loss")
for dop in (0.0, 100.0, 250.0, 500.0, 750.0, 1000.0)
    ω = 2π * dop / ustrip(Hz, sampling_freq)
    carrier = cis.(ω .* (0:N_total-1))
    signal  = ComplexF32.(carrier .* code_full)
    r_rot = acquire!(plan_rot, signal, prn; interm_freq = 0.0Hz, store_power_bins = true)
    r_ref = acquire!(plan_ref, signal, prn; interm_freq = 0.0Hz, store_power_bins = true)
    p_rot = maximum(r_rot.power_bins)
    p_ref = maximum(r_ref.power_bins)
    @printf("%7.1f  %14.4g  %14.4g  %6.3f  %5.2f dB\n",
        dop, p_rot, p_ref, p_rot/p_ref, 10*log10(p_rot/p_ref))
end
