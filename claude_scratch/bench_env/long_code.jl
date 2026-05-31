using GNSSSignals, Unitful
import Unitful: Hz

const Frequency = Union{Unitful.Quantity{T, Unitful.𝐓^-1, U}, Unitful.Level{L, S, Unitful.Quantity{T, Unitful.𝐓^-1, U}} where {L, S}} where {T, U}

struct LongL5I <: GNSSSignals.AbstractGNSSSignal{Matrix{Int16}}
    inner::GPSL5I{Matrix{Int16}}
end
LongL5I() = LongL5I(GPSL5I())

GNSSSignals.get_code_length(::LongL5I)         = 102_300
GNSSSignals.get_code_frequency(::LongL5I)      = GNSSSignals.get_code_frequency(GPSL5I())
GNSSSignals.get_center_frequency(::LongL5I)    = GNSSSignals.get_center_frequency(GPSL5I())
GNSSSignals.get_data_frequency(::LongL5I)      = 0.0Hz
GNSSSignals.get_secondary_code(::LongL5I)      = GNSSSignals.NoSecondaryCode()
GNSSSignals.get_secondary_code_length(::LongL5I) = 1
GNSSSignals.get_code_type(::LongL5I)           = Int16

function GNSSSignals.gen_code!(
    buffer::AbstractVector,
    ::LongL5I,
    prn::Integer,
    sampling_frequency::Frequency,
    code_frequency::Frequency,
    start_phase,
    start_index::Integer,
)
    GNSSSignals.gen_code!(buffer, GPSL5I(), prn, sampling_frequency,
                          code_frequency, start_phase, start_index)
end
