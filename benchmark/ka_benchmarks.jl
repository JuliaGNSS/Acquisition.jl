# KernelAbstractions GPU Benchmark Suite for Acquisition.jl
#
# Benchmarks for GPU-based acquisition using KernelAbstractions.

using LinearAlgebra

# GPU backend detection
const HAS_ROCM = try
    using AMDGPU
    AMDGPU.functional()
catch
    false
end

const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

const GPUArrayType, KA_BACKEND_LABEL = if HAS_ROCM
    @info "GPU backend: AMDGPU (ROCm)"
    AMDGPU.ROCArray, "ROCm"
elseif HAS_CUDA
    @info "GPU backend: CUDA"
    CUDA.CuArray, "CUDA"
else
    @info "GPU backend: None (using CPU Array fallback)"
    Array, "KA-CPU"
end

const KA_SUITE = BenchmarkGroup()

const KA_SAMPLE_SIZES = [16368, 32736, 60000]
const KA_PRN_COUNTS = [1, 32]
const KA_SAMPLING_FREQ = 16.368e6Hz

function _ka_make_signal(num_samples)
    Complex{Float32}.(randn(ComplexF64, num_samples))
end

for num_samples in KA_SAMPLE_SIZES
    KA_SUITE["$(num_samples)_samples"] = BenchmarkGroup()

    for num_prns in KA_PRN_COUNTS
        prns = 1:num_prns
        system = GPSL1()
        signal_cpu = _ka_make_signal(2 * num_samples)

        plan = KAAcquisitionPlan(
            system,
            num_samples,
            KA_SAMPLING_FREQ,
            GPUArrayType;
            prns = 1:32,
        )
        signal = GPUArrayType(signal_cpu)

        KA_SUITE["$(num_samples)_samples"]["$(num_prns)_prns"] =
            @benchmarkable acquire!($plan, $signal, $prns)
    end
end
