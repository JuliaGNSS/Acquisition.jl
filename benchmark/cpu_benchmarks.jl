# CPU Benchmark Suite for Acquisition.jl
#
# Benchmarks for CPU-based acquisition functions.

using FFTW

const CPU_SUITE = BenchmarkGroup()

const CPU_SAMPLE_SIZES = [16368, 32736, 60000]
const CPU_PRN_COUNTS = [1, 8, 32]
const CPU_SAMPLING_FREQ = 16.368e6Hz
const CPU_SIGNAL_TYPES = [Float64, Float32, Int16, Int32]
const CPU_SYSTEM_TYPES = [GPSL1, GalileoE1B]

# Check if eltype keyword is supported (added in later versions)
const _supports_eltype = hasmethod(AcquisitionPlan,
    Tuple{typeof(GPSL1()), Int, typeof(1.0Hz)},
    (:prns, :fft_flag, :eltype))

function _make_acq_plan(system, num_samples, sampling_freq, buffer_eltype; prns=1:32)
    if _supports_eltype
        AcquisitionPlan(system, num_samples, sampling_freq;
            prns=prns, fft_flag=FFTW.ESTIMATE, eltype=buffer_eltype)
    else
        AcquisitionPlan(system, num_samples, sampling_freq;
            prns=prns, fft_flag=FFTW.ESTIMATE)
    end
end

function _make_coarse_fine_plan(system, num_samples, sampling_freq, buffer_eltype; prns=1:32)
    if _supports_eltype
        CoarseFineAcquisitionPlan(system, num_samples, sampling_freq;
            prns=prns, fft_flag=FFTW.ESTIMATE, eltype=buffer_eltype)
    else
        CoarseFineAcquisitionPlan(system, num_samples, sampling_freq;
            prns=prns, fft_flag=FFTW.ESTIMATE)
    end
end

function _make_signal(num_samples, ::Type{T}) where T
    signal = randn(ComplexF64, num_samples)
    if T <: Integer
        complex.(floor.(T, real.(signal)), floor.(T, imag.(signal)))
    else
        Complex{T}.(signal)
    end
end

# ============================================================================
# AcquisitionPlan benchmarks (varying samples and PRNs)
# ============================================================================

CPU_SUITE["Acquire"] = BenchmarkGroup()

for num_samples in CPU_SAMPLE_SIZES
    CPU_SUITE["Acquire"]["$(num_samples)_samples"] = BenchmarkGroup()

    for num_prns in CPU_PRN_COUNTS
        prns = 1:num_prns
        system = GPSL1()
        signal = _make_signal(num_samples, Float32)
        plan = _make_acq_plan(system, num_samples, CPU_SAMPLING_FREQ, Float32; prns=1:32)

        CPU_SUITE["Acquire"]["$(num_samples)_samples"]["$(num_prns)_prns"] =
            @benchmarkable acquire!($plan, $signal, $prns)
    end
end

# ============================================================================
# CoarseFineAcquisitionPlan benchmarks (varying samples and PRNs)
# ============================================================================

CPU_SUITE["CoarseFine"] = BenchmarkGroup()

for num_samples in CPU_SAMPLE_SIZES
    CPU_SUITE["CoarseFine"]["$(num_samples)_samples"] = BenchmarkGroup()

    for num_prns in CPU_PRN_COUNTS
        prns = 1:num_prns
        system = GPSL1()
        signal = _make_signal(num_samples, Float32)
        plan = _make_coarse_fine_plan(system, num_samples, CPU_SAMPLING_FREQ, Float32; prns=1:32)

        CPU_SUITE["CoarseFine"]["$(num_samples)_samples"]["$(num_prns)_prns"] =
            @benchmarkable acquire!($plan, $signal, $prns)
    end
end

# ============================================================================
# Type and system benchmarks (varying signal types and GNSS systems)
# ============================================================================

const TYPE_BENCHMARK_SAMPLES = 20000

CPU_SUITE["Types"] = BenchmarkGroup()

for signal_type in CPU_SIGNAL_TYPES
    CPU_SUITE["Types"][string(signal_type)] = BenchmarkGroup()

    for system_type in CPU_SYSTEM_TYPES
        system = system_type()
        signal = _make_signal(TYPE_BENCHMARK_SAMPLES, signal_type)
        buffer_eltype = signal_type <: AbstractFloat ? signal_type : Float32
        plan = _make_acq_plan(system, TYPE_BENCHMARK_SAMPLES, CPU_SAMPLING_FREQ, buffer_eltype; prns=1:1)

        CPU_SUITE["Types"][string(signal_type)][string(system_type)] =
            @benchmarkable acquire!($plan, $signal, $(1:1))
    end
end

CPU_SUITE["CoarseFine Types"] = BenchmarkGroup()

for signal_type in CPU_SIGNAL_TYPES
    CPU_SUITE["CoarseFine Types"][string(signal_type)] = BenchmarkGroup()

    for system_type in CPU_SYSTEM_TYPES
        system = system_type()
        signal = _make_signal(TYPE_BENCHMARK_SAMPLES, signal_type)
        buffer_eltype = signal_type <: AbstractFloat ? signal_type : Float32
        plan = _make_coarse_fine_plan(system, TYPE_BENCHMARK_SAMPLES, CPU_SAMPLING_FREQ, buffer_eltype; prns=1:1)

        CPU_SUITE["CoarseFine Types"][string(signal_type)][string(system_type)] =
            @benchmarkable acquire!($plan, $signal, $(1:1))
    end
end
