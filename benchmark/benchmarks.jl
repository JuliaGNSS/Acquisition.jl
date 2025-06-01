using BenchmarkTools
using Unitful: Hz
using Acquisition
using GNSSSignals

sampling_freq = 15e6Hz
num_samples = 20000

const SUITE = BenchmarkGroup()

for type in [Float64, Float32, Int16, Int32]
    for system_type in [GPSL1, GalileoE1B]#[GPSL1, GPSL5, GalileoE1B]
        system = system_type()
        signal = randn(ComplexF64, num_samples)
        if type <: Integer
            signal_typed =
                complex.(floor.(type, real.(signal)), floor.(type, imag.(signal)))
        else
            signal_typed = Complex{type}.(signal)
        end
        SUITE["Acquire"][type][system_type] =
            @benchmarkable acquire($system, $signal_typed, $sampling_freq, $1)
        SUITE["Coarse Fine Acquire"][type][system_type] =
            @benchmarkable coarse_fine_acquire($system, $signal_typed, $sampling_freq, $1)
    end
end