# Probe the off-grid Doppler gap between the current ±1 rotation kernel and
# LongL5I's full 10 ms FFT. Confirms the ~5 dB structural gap that the
# phase-ramp fix will close.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Acquisition, GNSSSignals, Unitful, Printf
import Unitful: Hz

const _Frequency = Union{Unitful.Quantity{T, Unitful.𝐓^-1, U},
                         Unitful.Level{L, S, Unitful.Quantity{T, Unitful.𝐓^-1, U}} where {L, S}} where {T, U}
struct LongL5I <: GNSSSignals.AbstractGNSSSignal{Matrix{Int16}}
    inner::GPSL5I{Matrix{Int16}}
end
LongL5I() = LongL5I(GPSL5I())
GNSSSignals.get_code_length(::LongL5I)             = 102_300
GNSSSignals.get_code_frequency(::LongL5I)          = GNSSSignals.get_code_frequency(GPSL5I())
GNSSSignals.get_center_frequency(::LongL5I)        = GNSSSignals.get_center_frequency(GPSL5I())
GNSSSignals.get_data_frequency(::LongL5I)          = 0.0Hz
GNSSSignals.get_secondary_code(::LongL5I)          = GNSSSignals.NoSecondaryCode()
GNSSSignals.get_secondary_code_length(::LongL5I)   = 1
GNSSSignals.get_code_type(::LongL5I)               = Int16
function GNSSSignals.gen_code!(buffer::AbstractVector, ::LongL5I, prn::Integer,
                               sampling_frequency::_Frequency, code_frequency::_Frequency,
                               start_phase, start_index::Integer)
    GNSSSignals.gen_code!(buffer, GPSL5I(), prn, sampling_frequency,
                          code_frequency, start_phase, start_index)
end

system        = GPSL5I()
long_system   = LongL5I()
sampling_freq = 12e6Hz
prn           = 1
true_cp       = 0.0
N_coh         = 10
spc           = 12000
N_total       = N_coh * spc

fc        = get_code_frequency(system)
code_full = ComplexF32.(gen_code(N_total, system, prn, sampling_freq, fc, true_cp))

plan_rot  = plan_acquire(system,      sampling_freq, [prn];
    num_coherently_integrated_code_periods = N_coh, num_noncoherent_accumulations = 1)
plan_long = plan_acquire(long_system, sampling_freq, [prn];
    num_coherently_integrated_code_periods = 1,    num_noncoherent_accumulations = 1)

println("Doppler  peak_rot         peak_long        ratio   dB_loss")
for dop in (0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1050.0, 1100.0, 3333.0)
    ω = 2π * dop / ustrip(Hz, sampling_freq)
    carrier = cis.(ω .* (0:N_total-1))
    signal  = ComplexF32.(carrier .* code_full)
    r_rot  = acquire!(plan_rot,  signal, prn; interm_freq = 0.0Hz, store_power_bins = true)
    r_long = acquire!(plan_long, signal, prn; interm_freq = 0.0Hz, store_power_bins = true)
    p_rot  = maximum(r_rot.power_bins)
    p_long = maximum(r_long.power_bins)
    @printf("%7.1f  %14.4g  %14.4g  %6.3f  %5.2f dB\n",
        dop, p_rot, p_long, p_rot/p_long, 10*log10(p_rot/p_long))
end
