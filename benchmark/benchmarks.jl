using BenchmarkTools
using Unitful: Hz
using Acquisition
using GNSSSignals
using FFTW

sampling_freq = 15e6Hz
num_samples = 20000

const SUITE = BenchmarkGroup()

# Check if eltype keyword is supported (added in later versions)
const _supports_eltype = hasmethod(AcquisitionPlan,
    Tuple{typeof(GPSL1()), Int, typeof(1.0Hz)},
    (:prns, :fft_flag, :eltype))

function _make_acq_plan(system, num_samples, sampling_freq, buffer_eltype)
    if _supports_eltype
        AcquisitionPlan(system, num_samples, sampling_freq;
            prns = 1:1, fft_flag = FFTW.ESTIMATE, eltype = buffer_eltype)
    else
        AcquisitionPlan(system, num_samples, sampling_freq;
            prns = 1:1, fft_flag = FFTW.ESTIMATE)
    end
end

function _make_coarse_fine_plan(system, num_samples, sampling_freq, buffer_eltype)
    if _supports_eltype
        CoarseFineAcquisitionPlan(system, num_samples, sampling_freq;
            prns = 1:1, fft_flag = FFTW.ESTIMATE, eltype = buffer_eltype)
    else
        CoarseFineAcquisitionPlan(system, num_samples, sampling_freq;
            prns = 1:1, fft_flag = FFTW.ESTIMATE)
    end
end

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
        # Use matching eltype for floating point signals to avoid allocations
        buffer_eltype = type <: AbstractFloat ? type : Float32
        acq_plan = _make_acq_plan(system, num_samples, sampling_freq, buffer_eltype)
        SUITE["Acquire"][type][system_type] =
            @benchmarkable acquire!($acq_plan, $signal_typed, $(1:1))
        coarse_fine_acq_plan = _make_coarse_fine_plan(system, num_samples, sampling_freq, buffer_eltype)
        SUITE["Coarse Fine Acquire"][type][system_type] =
            @benchmarkable acquire!($coarse_fine_acq_plan, $signal_typed, $(1:1))
    end
end