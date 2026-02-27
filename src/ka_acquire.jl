using KernelAbstractions
using KernelAbstractions: @kernel, @index, get_backend, synchronize
import AbstractFFTs: plan_fft, plan_bfft

"""
    KAAcquisitionPlan

GPU-accelerated acquisition plan using KernelAbstractions.jl for portable GPU execution.

Processes Doppler frequencies in batches to limit GPU memory usage. Downconversion
phases are computed at runtime via a GPU kernel (no precomputed phase matrix).

Works with any GPU backend supported by KernelAbstractions.jl (CUDA, ROCm,
Metal, oneAPI) via the AbstractFFTs interface.

# See also

[`acquire!`](@ref), [`AcquisitionPlan`](@ref)
"""
struct KAAcquisitionPlan{
    T,
    S<:AbstractGNSS,
    A1<:AbstractVector,
    A2<:AbstractMatrix,
    A3<:AbstractMatrix,
    A4<:AbstractArray,
    AI<:AbstractVector{Int},
    P,
    BP,
}
    system::S
    num_samples_to_integrate_coherently::Int
    linear_fft_size::Int
    bfft_size::Int
    doppler_batch_size::Int
    dopplers::StepRangeLen{
        Float64,
        Base.TwicePrecision{Float64},
        Base.TwicePrecision{Float64},
    }
    sampling_frequency::typeof(1.0Hz)
    codes_freq_domain::A4                   # (samples × prns × code_dopplers)
    code_doppler_indices::Vector{Int}       # maps doppler_idx → code_doppler_idx
    code_doppler_step::Float64              # code Doppler step in Hz
    code_doppler_offset_idx::Int            # index of zero code Doppler
    num_code_dopplers::Int                  # total number of code Doppler bins
    dopplers_gpu::A1                        # Doppler frequencies on GPU (num_dopplers,)
    signal_baseband::A2                     # (linear_fft_size × batch_size)
    signal_baseband_freq_domain::A2         # (linear_fft_size × batch_size)
    code_baseband_freq_domain::A2           # (bfft_size × batch_size)
    code_baseband::A2                       # (bfft_size × batch_size)
    power_buffer::A3                        # (code_samples × batch_size)
    power_accumulator::A3                   # (code_samples × batch_size)
    findmax_vals_gpu::A1                    # Thread-local max values for reduction (GPU)
    findmax_vals_cpu::Vector{T}             # Thread-local max values (CPU for final reduction)
    findmax_idxs_gpu::AI                    # Thread-local max indices (GPU)
    findmax_idxs_cpu::Vector{Int}           # Thread-local max indices (CPU for final reduction)
    sum_buffer_gpu::A1                      # Thread-local partial sums for reduction (GPU)
    sum_buffer_cpu::Vector{T}               # Thread-local partial sums (CPU for final reduction)
    fft_plan::P
    bfft_plan::BP
    avail_prn_channels::Vector{Int}
end

"""
    KAAcquisitionPlan(system, num_samples_to_integrate_coherently, sampling_freq, ArrayType; kwargs...)

Create a GPU-accelerated acquisition plan with Doppler batching.

# Arguments

  - `system`: GNSS system (e.g., `GPSL1()`)
  - `num_samples_to_integrate_coherently`: Number of samples per coherent integration chunk
  - `sampling_freq`: Sampling frequency of the signal
  - `ArrayType`: GPU array constructor (e.g., `CuArray`, `ROCArray`, `Array` for CPU)

# Keyword Arguments

  - `eltype`: Element type for internal buffers (default: `Float32`)
  - `max_doppler`: Maximum Doppler frequency to search (default: `7000Hz`)
  - `min_doppler`: Minimum Doppler frequency to search (default: `-max_doppler`)
  - `dopplers`: Custom Doppler range (default: computed from `doppler_step`)
  - `doppler_step`: Doppler frequency step size
  - `doppler_step_factor`: Factor for computing Doppler step from integration time (default: `1//3`)
  - `prns`: PRN channels to prepare (default: `1:34`)
  - `max_code_doppler_loss`: Maximum acceptable correlation loss in dB from code Doppler
    mismatch (default: `0.5`).
  - `fft_flag`: FFTW planning flag for CPU Arrays (default: `FFTW.MEASURE`).
  - `max_gpu_memory`: Maximum GPU memory budget in bytes for work buffers.
    The batch size is computed automatically to fit within this budget.
    Default: `4 * 1024^3` (4 GiB). Set to `nothing` to disable batching
    (allocate all Dopplers at once).

# See also

[`acquire!`](@ref), [`AcquisitionPlan`](@ref)
"""
function KAAcquisitionPlan(
    system::S,
    num_samples_to_integrate_coherently::Integer,
    sampling_freq,
    ArrayType::Type{<:AbstractArray};
    eltype::Type{T} = Float32,
    max_doppler = 7000Hz,
    min_doppler = -max_doppler,
    doppler_step_factor = 1//3,
    doppler_step = doppler_step_factor * sampling_freq /
                   num_samples_to_integrate_coherently,
    dopplers = min_doppler:doppler_step:max_doppler,
    max_code_doppler_loss = 0.5dB,
    prns = 1:34,
    zero_pad_power::Int = 0,
    fft_flag = FFTW.MEASURE,
    max_gpu_memory = 4 * 1024^3,
) where {T<:AbstractFloat,S<:AbstractGNSS}
    # Normalize dopplers to Float64 StepRangeLen for consistency
    dopplers_hz =
        Float64(ustrip(first(dopplers))):Float64(ustrip(step(dopplers))):Float64(
            ustrip(last(dopplers)),
        )
    num_dopplers = length(dopplers_hz)
    num_prns = length(prns)

    # Double Block Zero Padding (DBZP): pad to >= 2N for linear correlation
    linear_fft_size = fftw_friendly_size(2 * num_samples_to_integrate_coherently)
    bfft_size = fftw_friendly_size(linear_fft_size << zero_pad_power)

    # Convert dB loss to max phase error (cycles), then to code Doppler step (Hz)
    T_coh = num_samples_to_integrate_coherently / sampling_freq
    max_phase_err = max_phase_error_from_loss(max_code_doppler_loss)
    code_doppler_step_val = max_phase_err > 0 ? max_phase_err / ustrip(T_coh) : Inf
    code_dopplers_grid, n_neg =
        compute_code_doppler_grid(system, min_doppler, max_doppler, code_doppler_step_val)
    code_doppler_offset_idx = n_neg + 1
    num_code_dopplers_val = length(code_dopplers_grid)
    code_doppler_indices = compute_code_doppler_indices(
        dopplers_hz,
        system,
        code_doppler_step_val,
        code_doppler_offset_idx,
        num_code_dopplers_val,
    )

    # Compute num_code_samples (needed for batch size calculation)
    Δt = num_samples_to_integrate_coherently / sampling_freq
    code_period = get_code_length(system) / get_code_frequency(system)
    effective_sampling_freq = sampling_freq * bfft_size / linear_fft_size
    num_code_samples = ceil(Int, effective_sampling_freq * min(Δt, code_period))

    # Compute batch size from memory budget
    bytes_per_doppler = (
        sizeof(Complex{T}) * linear_fft_size * 2 +   # signal_baseband + freq_domain
        sizeof(Complex{T}) * bfft_size * 2 +          # code_baseband_freq_domain + code_baseband
        sizeof(T) * num_code_samples * 2              # power_buffer + power_accumulator
    )
    doppler_batch_size = if isnothing(max_gpu_memory)
        num_dopplers
    else
        clamp(div(Int(max_gpu_memory), bytes_per_doppler), 1, num_dopplers)
    end

    # Generate 3D codes on CPU: (linear_fft_size × num_prns × num_code_dopplers)
    # Zero-pad codes from N to linear_fft_size for DBZP linear correlation
    codes_cpu = zeros(T, linear_fft_size, num_prns, num_code_dopplers_val)
    for (cd_idx, cd) in enumerate(code_dopplers_grid)
        for (p_idx, prn) in enumerate(prns)
            codes_cpu[1:num_samples_to_integrate_coherently, p_idx, cd_idx] .= gen_code(
                num_samples_to_integrate_coherently,
                system,
                prn,
                sampling_freq,
                get_code_frequency(system) + cd * 1.0Hz,
            )
        end
    end

    # Store all Doppler frequencies on GPU (small vector, used by downconvert kernel)
    dopplers_gpu = ArrayType{T}(collect(T, dopplers_hz))

    # Allocate batch-sized buffers on GPU — reused per Doppler batch
    # Signal buffers are linear_fft_size (DBZP: 2N) for forward FFT
    signal_baseband = ArrayType{Complex{T}}(undef, linear_fft_size, doppler_batch_size)
    signal_baseband_freq_domain = similar(signal_baseband)
    # BFFT buffers are sized at bfft_size for additional zero-padding
    code_baseband_freq_domain = ArrayType{Complex{T}}(undef, bfft_size, doppler_batch_size)
    code_baseband = similar(code_baseband_freq_domain)
    # Preallocate power buffer using effective sampling freq for proper row count
    power_buffer = ArrayType{T}(undef, num_code_samples, doppler_batch_size)
    power_accumulator = ArrayType{T}(undef, num_code_samples, doppler_batch_size)

    # Preallocate reduction buffers (1024 threads is typical)
    num_reduction_threads = 1024
    findmax_vals_gpu = ArrayType{T}(undef, num_reduction_threads)
    findmax_vals_cpu = Vector{T}(undef, num_reduction_threads)
    findmax_idxs_gpu = ArrayType{Int}(undef, num_reduction_threads)
    findmax_idxs_cpu = Vector{Int}(undef, num_reduction_threads)
    sum_buffer_gpu = ArrayType{T}(undef, num_reduction_threads)
    sum_buffer_cpu = Vector{T}(undef, num_reduction_threads)

    # Create FFT plans. For CPU Arrays, use FFTW with configurable planning flags
    # (MEASURE/PATIENT find faster algorithms). For GPU arrays, use AbstractFFTs
    # which delegates to cuFFT/rocFFT (no planning flags needed).
    fft_plan, bfft_plan = if ArrayType <: Array && fft_flag != FFTW.ESTIMATE
        with_fftw_wisdom() do
            plan_fft(signal_baseband, 1; flags = fft_flag),
            plan_bfft(code_baseband_freq_domain, 1; flags = fft_flag)
        end
    elseif ArrayType <: Array
        plan_fft(signal_baseband, 1; flags = fft_flag),
        plan_bfft(code_baseband_freq_domain, 1; flags = fft_flag)
    else
        plan_fft(signal_baseband, 1), plan_bfft(code_baseband_freq_domain, 1)
    end

    # Transform codes to frequency domain
    # Transfer 3D array to GPU and FFT each (sample) column
    codes_gpu = ArrayType{Complex{T}}(codes_cpu)
    codes_freq_domain = similar(codes_gpu)
    for cd_idx = 1:num_code_dopplers_val
        for p_idx = 1:num_prns
            codes_freq_domain[:, p_idx, cd_idx] .= fft(view(codes_gpu, :, p_idx, cd_idx))
        end
    end

    KAAcquisitionPlan{
        T,
        S,
        typeof(dopplers_gpu),
        typeof(signal_baseband),
        typeof(power_buffer),
        typeof(codes_freq_domain),
        typeof(findmax_idxs_gpu),
        typeof(fft_plan),
        typeof(bfft_plan),
    }(
        system,
        num_samples_to_integrate_coherently,
        linear_fft_size,
        bfft_size,
        doppler_batch_size,
        dopplers_hz,
        sampling_freq,
        codes_freq_domain,
        code_doppler_indices,
        code_doppler_step_val,
        code_doppler_offset_idx,
        num_code_dopplers_val,
        dopplers_gpu,
        signal_baseband,
        signal_baseband_freq_domain,
        code_baseband_freq_domain,
        code_baseband,
        power_buffer,
        power_accumulator,
        findmax_vals_gpu,
        findmax_vals_cpu,
        findmax_idxs_gpu,
        findmax_idxs_cpu,
        sum_buffer_gpu,
        sum_buffer_cpu,
        fft_plan,
        bfft_plan,
        collect(prns),
    )
end

# Convenience constructor that defaults num_samples_to_integrate_coherently to one bit period
function KAAcquisitionPlan(
    system::S,
    sampling_freq,
    ArrayType::Type{<:AbstractArray};
    kwargs...,
) where {S<:AbstractGNSS}
    # Default to one bit period worth of samples for optimal non-coherent integration
    data_frequency = get_data_frequency(system)
    num_samples_to_integrate_coherently = ceil(Int, sampling_freq / data_frequency)
    KAAcquisitionPlan(
        system,
        num_samples_to_integrate_coherently,
        sampling_freq,
        ArrayType;
        kwargs...,
    )
end

# ============================================================================
# KernelAbstractions Kernels
# ============================================================================

"""
Fused downconvert + zero-padding kernel. Downconverts signal samples via runtime sincos
and writes zeros to the DBZP padding region, all in a single kernel launch.
Each thread handles one (sample, doppler) pair over the full linear_fft_size.
"""
@kernel function ka_downconvert_and_pad_kernel!(
    signal_baseband,
    @Const(signal),
    @Const(dopplers),
    freq_offset,
    inv_sampling_frequency,
    num_signal_samples,
)
    i, d = @index(Global, NTuple)

    if i <= num_signal_samples
        @fastmath begin
            doppler = dopplers[d] + freq_offset
            phase =
                Float32(-6.283185307179586) * doppler * Float32(i - 1) *
                inv_sampling_frequency
            s, c = sincos(phase)
            signal_baseband[i, d] = signal[i] * Complex{Float32}(c, s)
        end
    else
        # DBZP zero-padding region
        signal_baseband[i, d] = zero(eltype(signal_baseband))
    end
end

"""
Fused frequency-domain correlation kernel (no DBZP padding case, N == N_pad).
Multiplies code replica (1D) × conj(signal) (2D) in a single kernel launch,
replacing broadcasting which generates multiple GPU kernel launches and temporaries.
"""
@kernel function ka_correlate_kernel!(
    code_baseband_freq_domain,
    @Const(code_freq_domain),
    @Const(signal_baseband_freq_domain),
)
    i, d = @index(Global, NTuple)
    @inbounds code_baseband_freq_domain[i, d] =
        code_freq_domain[i] * conj(signal_baseband_freq_domain[i, d])
end

"""
Fused frequency-domain correlation kernel with DBZP zero-padding (N_pad > N).
Maps positive frequencies (1:pos_end) and negative frequencies (N_pad-neg_count+1:N_pad)
from the N-length code/signal to the N_pad-length output, zeroing the gap.
"""
@kernel function ka_correlate_padded_kernel!(
    code_baseband_freq_domain,
    @Const(code_freq_domain),
    @Const(signal_baseband_freq_domain),
    pos_end,
    neg_start_out,  # N_pad - neg_count + 1
    neg_start_in,   # pos_end + 1
    N_pad,
)
    i, d = @index(Global, NTuple)
    @inbounds if i <= pos_end
        # Positive frequencies
        code_baseband_freq_domain[i, d] =
            code_freq_domain[i] * conj(signal_baseband_freq_domain[i, d])
    elseif i >= neg_start_out
        # Negative frequencies: map output index to input index
        in_idx = neg_start_in + (i - neg_start_out)
        code_baseband_freq_domain[i, d] =
            code_freq_domain[in_idx] * conj(signal_baseband_freq_domain[in_idx, d])
    else
        # Zero gap between positive and negative frequencies
        code_baseband_freq_domain[i, d] = zero(eltype(code_baseband_freq_domain))
    end
end

"""
Fused abs2 + accumulate kernel. Computes abs2 of complex BFFT output and either
assigns or accumulates into the power buffer, eliminating a separate broadcasting pass.
"""
@kernel function ka_abs2_accumulate_kernel!(power_buffer, @Const(code_baseband), accumulate)
    i, d = @index(Global, NTuple)
    @inbounds begin
        val = abs2(code_baseband[i, d])
        if accumulate
            power_buffer[i, d] += val
        else
            power_buffer[i, d] = val
        end
    end
end

"""
Fused findmax + sum kernel. Computes maximum value/index AND total sum in a single pass.
"""
@kernel function ka_findmax_and_sum_kernel!(
    thread_vals,
    thread_idxs,
    thread_sums,
    @Const(data),
    n,
)
    tid = @index(Global)
    num_threads = @uniform @ndrange()[1]

    # Grid-stride loop to find local maximum AND compute sum
    best_val = typemin(eltype(data))
    best_idx = 0
    partial_sum = zero(eltype(data))

    idx = tid
    @inbounds while idx <= n
        val = data[idx]
        partial_sum += val
        if val > best_val
            best_val = val
            best_idx = idx
        end
        idx += num_threads
    end

    thread_vals[tid] = best_val
    thread_idxs[tid] = best_idx
    thread_sums[tid] = partial_sum
end

"""
    gpu_findmax_and_sum!(vals_gpu, vals_cpu, idxs_gpu, idxs_cpu, sums_gpu, sums_cpu, data, backend)

Fused findmax + sum in single GPU pass. Returns (max_value, linear_index, total_sum).
"""
function gpu_findmax_and_sum!(
    vals_gpu,
    vals_cpu,
    idxs_gpu,
    idxs_cpu,
    sums_gpu,
    sums_cpu,
    data,
    backend,
)
    n = length(data)
    num_threads = length(vals_gpu)

    # Single kernel computes both findmax and sum
    ka_findmax_and_sum_kernel!(backend)(
        vals_gpu,
        idxs_gpu,
        sums_gpu,
        data,
        n;
        ndrange = num_threads,
    )

    # Single GPU→CPU transfer for all three buffers
    copyto!(vals_cpu, vals_gpu)
    copyto!(idxs_cpu, idxs_gpu)
    copyto!(sums_cpu, sums_gpu)

    # CPU reduction for findmax
    best_val = typemin(eltype(data))
    best_idx = 0
    @inbounds for i = 1:num_threads
        if vals_cpu[i] > best_val
            best_val = vals_cpu[i]
            best_idx = idxs_cpu[i]
        end
    end

    # CPU reduction for sum
    total_sum = sum(sums_cpu)
    return best_val, best_idx, total_sum
end

# ============================================================================
# Acquisition Functions
# ============================================================================

"""
    acquire!(plan::KAAcquisitionPlan, signal, prns; kwargs...) -> Vector{AcquisitionResults}

Perform GPU-accelerated acquisition using a pre-computed [`KAAcquisitionPlan`](@ref).

Dopplers are processed in batches to limit GPU memory usage. Within each batch,
downconversion, FFT, correlation and power computation run on the GPU. The per-batch
findmax/sum results are accumulated on the CPU to produce the final result.

# Arguments

  - `plan`: Pre-computed [`KAAcquisitionPlan`](@ref)
  - `signal`: Complex baseband signal samples (GPU array)
  - `prns`: PRN numbers to search (must be subset of `plan.avail_prn_channels`)

# Keyword Arguments

  - `interm_freq`: Intermediate frequency (default: `0.0Hz`)
  - `doppler_offset`: Offset added to Doppler search range (default: `0.0Hz`)

# Returns

Vector of [`AcquisitionResults`](@ref), one per PRN.
"""
function acquire!(
    plan::KAAcquisitionPlan{T,S},
    signal::AbstractArray,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
) where {T,S}
    all(prn -> prn in plan.avail_prn_channels, prns) ||
        throw(ArgumentError("All requested PRNs must be in plan.avail_prn_channels"))

    isempty(prns) && return AcquisitionResults{S,Float32}[]

    # Precompute constants
    chunk_samples = plan.num_samples_to_integrate_coherently
    num_signal_samples = length(signal)
    num_chunks = cld(num_signal_samples, chunk_samples)
    code_period = get_code_length(plan.system) / get_code_frequency(plan.system)
    Δt = chunk_samples / plan.sampling_frequency
    effective_sampling_freq = plan.sampling_frequency * plan.bfft_size / plan.linear_fft_size
    num_code_samples = ceil(Int, effective_sampling_freq * min(Δt, code_period))
    samples_per_chip = floor(Int, effective_sampling_freq / get_code_frequency(plan.system))
    backend = get_backend(plan.signal_baseband)

    # Precompute doppler range (avoid allocation per result)
    result_dopplers =
        iszero(ustrip(doppler_offset)) ? plan.dopplers :
        (plan.dopplers .+ Float64(ustrip(doppler_offset)))

    # Get PRN indices upfront
    prn_indices = [findfirst(==(prn), plan.avail_prn_channels) for prn in prns]
    num_prns = length(prns)
    num_dopplers = length(plan.dopplers)
    batch_size = plan.doppler_batch_size

    # Compute active code Doppler indices (handle doppler_offset)
    active_cd_indices = if !iszero(ustrip(doppler_offset))
        ratio = get_code_center_frequency_ratio(plan.system)
        [
            code_doppler_index(
                plan.dopplers[d] + Float64(ustrip(doppler_offset)),
                ratio,
                plan.code_doppler_step,
                plan.code_doppler_offset_idx,
                plan.num_code_dopplers,
            ) for d = 1:num_dopplers
        ]
    else
        plan.code_doppler_indices
    end

    N = plan.linear_fft_size
    N_pad = plan.bfft_size
    needs_padding = N_pad > N
    freq_offset = T(ustrip(doppler_offset + interm_freq))
    inv_sampling_freq = T(1.0 / ustrip(plan.sampling_frequency))

    # Per-PRN accumulators across Doppler batches (CPU-side, tiny)
    best_power = fill(typemin(T), num_prns)
    best_code_idx = zeros(Int, num_prns)
    best_doppler_idx = zeros(Int, num_prns)
    total_power = zeros(T, num_prns)
    total_samples = 0

    for batch_start = 1:batch_size:num_dopplers
        batch_end = min(batch_start + batch_size - 1, num_dopplers)
        actual_batch = batch_end - batch_start + 1

        # Doppler slice for this batch
        batch_dopplers = view(plan.dopplers_gpu, batch_start:batch_end)
        batch_cd_indices = view(active_cd_indices, batch_start:batch_end)

        for chunk_idx = 1:num_chunks
            start_idx = (chunk_idx - 1) * chunk_samples + 1
            end_idx = min(chunk_idx * chunk_samples, num_signal_samples)
            actual_chunk_size = end_idx - start_idx + 1
            signal_chunk = view(signal, start_idx:end_idx)

            # 1. Fused downconvert + zero-padding in a single kernel launch.
            #    Writes signal × cis(-2πft) for samples 1:actual_chunk_size,
            #    and zeros for the DBZP padding region (actual_chunk_size+1:N).
            ka_downconvert_and_pad_kernel!(backend)(
                view(plan.signal_baseband, 1:N, 1:actual_batch),
                signal_chunk,
                batch_dopplers,
                freq_offset,
                inv_sampling_freq,
                Int32(actual_chunk_size);
                ndrange = (N, actual_batch),
            )

            # 2. Batched FFT
            mul!(plan.signal_baseband_freq_domain, plan.fft_plan, plan.signal_baseband)

            # 3-5. Process each PRN: correlate → power → findmax
            for (p_idx, prn_idx) in enumerate(prn_indices)
                _ka_correlate_prn!(
                    plan,
                    prn_idx,
                    batch_cd_indices,
                    actual_batch,
                    N,
                    N_pad,
                    needs_padding,
                    backend,
                )

                # Fused abs2 + accumulate in a single kernel launch
                ka_abs2_accumulate_kernel!(backend)(
                    view(plan.power_buffer, :, 1:actual_batch),
                    view(plan.code_baseband, 1:num_code_samples, 1:actual_batch),
                    chunk_idx > 1;
                    ndrange = (num_code_samples, actual_batch),
                )

                # After last chunk: GPU findmax+sum on this batch, merge into CPU accumulators
                if chunk_idx == num_chunks
                    batch_max, batch_lin_idx, batch_sum = gpu_findmax_and_sum!(
                        plan.findmax_vals_gpu,
                        plan.findmax_vals_cpu,
                        plan.findmax_idxs_gpu,
                        plan.findmax_idxs_cpu,
                        plan.sum_buffer_gpu,
                        plan.sum_buffer_cpu,
                        view(plan.power_buffer, :, 1:actual_batch),
                        backend,
                    )

                    # Map batch-local linear index to global Doppler index
                    batch_dop_idx = div(batch_lin_idx - 1, num_code_samples) + 1
                    batch_code_idx = mod(batch_lin_idx - 1, num_code_samples) + 1

                    total_power[p_idx] += batch_sum
                    if batch_max > best_power[p_idx]
                        best_power[p_idx] = batch_max
                        best_code_idx[p_idx] = batch_code_idx
                        best_doppler_idx[p_idx] = batch_start + batch_dop_idx - 1
                    end
                end
            end
        end

        total_samples += num_code_samples * actual_batch
    end

    # 6. Build results from accumulated values across all Doppler batches
    map(enumerate(prns)) do (p_idx, prn)
        code_index = best_code_idx[p_idx]
        doppler_index = best_doppler_idx[p_idx]
        signal_noise_power = best_power[p_idx]

        noise_power = total_samples > 0 ? (total_power[p_idx] / total_samples) : zero(T)
        signal_power = signal_noise_power - noise_power

        snr = max(signal_power / noise_power, T(1e-10))
        CN0 = 10 * log10(snr / code_period / 1.0Hz)
        doppler =
            (doppler_index - 1) * step(plan.dopplers) +
            first(plan.dopplers) +
            Float64(ustrip(doppler_offset))
        code_doppler = doppler * get_code_center_frequency_ratio(plan.system)
        code_phase =
            (code_index - 1) / (
                effective_sampling_freq /
                (get_code_frequency(plan.system) + code_doppler * 1.0Hz)
            )

        AcquisitionResults(
            plan.system,
            prn,
            effective_sampling_freq,
            doppler * 1.0Hz,
            code_phase,
            CN0,
            Float32(noise_power),
            Matrix{Float32}(undef, 0, 0),
            result_dopplers,
        )
    end
end

"""
Correlate a single PRN: code × signal multiplication in frequency domain + backward FFT.

Uses fused KA kernels instead of broadcasting to minimize GPU kernel launches.
Groups adjacent Doppler bins that share the same code Doppler index and launches
a single 2D kernel per group. `batch_cd_indices` maps batch-local Doppler index
to code Doppler index. `actual_batch` is the number of active Dopplers in this batch.
"""
function _ka_correlate_prn!(plan, prn_idx, batch_cd_indices, actual_batch, N, N_pad, needs_padding, backend)
    if needs_padding
        pos_end = N ÷ 2 + 1
        neg_count = N - pos_end
        neg_start_out = N_pad - neg_count + 1
        neg_start_in = pos_end + 1
        group_start = 1
        for d_idx = 1:actual_batch
            cd_idx = batch_cd_indices[d_idx]
            if d_idx == actual_batch || batch_cd_indices[d_idx + 1] != cd_idx
                # Launch fused padded correlation kernel for this group of Dopplers
                code_col = view(plan.codes_freq_domain, :, prn_idx, cd_idx)
                num_cols = d_idx - group_start + 1
                ka_correlate_padded_kernel!(backend)(
                    view(plan.code_baseband_freq_domain, :, group_start:d_idx),
                    code_col,
                    view(plan.signal_baseband_freq_domain, :, group_start:d_idx),
                    Int32(pos_end),
                    Int32(neg_start_out),
                    Int32(neg_start_in),
                    Int32(N_pad);
                    ndrange = (N_pad, num_cols),
                )
                group_start = d_idx + 1
            end
        end
    else
        group_start = 1
        for d_idx = 1:actual_batch
            cd_idx = batch_cd_indices[d_idx]
            if d_idx == actual_batch || batch_cd_indices[d_idx + 1] != cd_idx
                # Launch fused correlation kernel for this group of Dopplers
                code_col = view(plan.codes_freq_domain, :, prn_idx, cd_idx)
                num_cols = d_idx - group_start + 1
                ka_correlate_kernel!(backend)(
                    view(plan.code_baseband_freq_domain, :, group_start:d_idx),
                    code_col,
                    view(plan.signal_baseband_freq_domain, :, group_start:d_idx);
                    ndrange = (N, num_cols),
                )
                group_start = d_idx + 1
            end
        end
    end

    # Backward FFT: frequency domain → time domain correlation
    mul!(plan.code_baseband, plan.bfft_plan, plan.code_baseband_freq_domain)
    return nothing
end

# Single PRN convenience method
function acquire!(
    plan::KAAcquisitionPlan{T,S},
    signal::AbstractArray,
    prn::Integer;
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
)::AcquisitionResults{S,Float32} where {T,S}
    only(acquire!(plan, signal, [prn]; interm_freq, doppler_offset))
end
