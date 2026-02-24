using KernelAbstractions
using KernelAbstractions: @kernel, @index, get_backend, synchronize
import AbstractFFTs: plan_fft, plan_bfft

"""
    KAAcquisitionPlan

GPU-accelerated acquisition plan using KernelAbstractions.jl for portable GPU execution.

Performs fully-batched FFT-based acquisition across all Doppler frequencies and PRNs
simultaneously. Works with any GPU backend supported by KernelAbstractions.jl (CUDA, ROCm,
Metal, oneAPI) via the AbstractFFTs interface.

# Type Parameters

  - `T`: Element type (e.g., `Float32`, `Float64`)
  - `S`: GNSS system type
  - `A`: Array type (e.g., `CuArray`, `ROCArray`)
  - `P`: FFT plan type
  - `IP`: Inverse FFT plan type

# Fields

  - `system`: GNSS system (e.g., `GPSL1()`)
  - `dopplers`: Range of Doppler frequencies to search
  - `sampling_frequency`: Sampling frequency of the signal
  - `codes_freq_domain`: Pre-computed frequency-domain replica codes (samples × prns)
  - `signal_baseband`: Batched downconverted signal buffer (samples × dopplers)
  - `signal_baseband_freq_domain`: Batched FFT of downconverted signal (samples × dopplers)
  - `code_baseband_freq_domain`: Batched correlation in frequency domain (samples × dopplers × prns)
  - `code_baseband`: Batched correlation in time domain (samples × dopplers × prns)
  - `fft_plan`: Forward FFT plan
  - `bfft_plan`: Backward FFT plan (unnormalized inverse)
  - `avail_prn_channels`: PRN channels available in this plan

# Example

```julia
using Acquisition, GNSSSignals, AMDGPU  # or CUDA

# Create GPU plan
plan = KAAcquisitionPlan(GPSL1(), 16368, 16.368e6Hz, ROCArray; prns = 1:32)

# Transfer signal to GPU and acquire
signal_gpu = ROCArray(signal_cpu)
results = acquire!(plan, signal_gpu, 1:32)
```

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
    downconvert_phases::A2                  # Precomputed phase factors (samples × dopplers)
    signal_baseband::A2                     # (samples × dopplers)
    signal_baseband_freq_domain::A2         # (samples × dopplers)
    code_baseband_freq_domain::A2           # (samples × dopplers) - reused per PRN
    code_baseband::A2                       # (samples × dopplers) - reused per PRN
    power_buffer::A3                        # (code_samples × dopplers) - reused for abs2
    power_accumulator::A3                   # (code_samples × dopplers) - accumulated powers across chunks
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

Create a GPU-accelerated acquisition plan.

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
  - `doppler_step`: Doppler frequency step size (default: `doppler_step_factor / T` where
    `T = num_samples_to_integrate_coherently / sampling_freq`)
  - `doppler_step_factor`: Factor for computing Doppler step from integration time (default: `1//3`)
  - `prns`: PRN channels to prepare (default: `1:34`)
  - `max_code_doppler_loss`: Maximum acceptable correlation loss in dB from code Doppler
    mismatch (default: `0.5`). Controls how many code replicas are pre-computed at different
    code Doppler offsets. Works uniformly across all GNSS systems regardless of chip rate.

# Example

```julia
using Acquisition, GNSSSignals, AMDGPU

plan = KAAcquisitionPlan(GPSL1(), 16.368e6Hz, ROCArray; prns = 1:32)
```

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
) where {T<:AbstractFloat,S<:AbstractGNSS}
    # Normalize dopplers to Float64 StepRangeLen for consistency
    dopplers_hz =
        Float64(ustrip(first(dopplers))):Float64(ustrip(step(dopplers))):Float64(
            ustrip(last(dopplers)),
        )
    num_dopplers = length(dopplers_hz)
    num_prns = length(prns)

    # Double Block Zero Padding (DBZP): pad to >= 2N for linear correlation
    # (same as CPU path in AcquisitionPlan)
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

    # Store Doppler frequencies on GPU (small vector for kernel when offset needed)
    dopplers_gpu = ArrayType{T}(collect(T, dopplers_hz))

    # Precompute downconversion phase factors on CPU, then transfer to GPU
    # phase[i,d] = cis(-2π * doppler[d] * (i-1) / sampling_freq)
    # Only N phases needed; indices N+1:linear_fft_size are zero-padded (DBZP)
    sampling_freq_val = T(ustrip(sampling_freq))
    downconvert_phases_cpu = [
        i <= num_samples_to_integrate_coherently ?
        cis(T(-2) * T(π) * T(dopplers_hz[d]) * T(i - 1) / sampling_freq_val) :
        zero(Complex{T}) for i = 1:linear_fft_size, d = 1:num_dopplers
    ]
    downconvert_phases = ArrayType{Complex{T}}(downconvert_phases_cpu)

    # Allocate buffers on GPU - use 2D arrays to reduce memory (reused per PRN)
    # Signal buffers are linear_fft_size (DBZP: 2N) for forward FFT
    signal_baseband = ArrayType{Complex{T}}(undef, linear_fft_size, num_dopplers)
    signal_baseband_freq_domain = similar(signal_baseband)
    # BFFT buffers are sized at bfft_size for additional zero-padding
    code_baseband_freq_domain = ArrayType{Complex{T}}(undef, bfft_size, num_dopplers)
    code_baseband = similar(code_baseband_freq_domain)

    # Preallocate power buffer using effective sampling freq for proper row count
    Δt = num_samples_to_integrate_coherently / sampling_freq
    code_period = get_code_length(system) / get_code_frequency(system)
    effective_sampling_freq = sampling_freq * bfft_size / linear_fft_size
    num_code_samples = ceil(Int, effective_sampling_freq * min(Δt, code_period))
    power_buffer = ArrayType{T}(undef, num_code_samples, num_dopplers)
    power_accumulator = ArrayType{T}(undef, num_code_samples, num_dopplers)

    # Preallocate reduction buffers (1024 threads is typical)
    num_reduction_threads = 1024
    findmax_vals_gpu = ArrayType{T}(undef, num_reduction_threads)
    findmax_vals_cpu = Vector{T}(undef, num_reduction_threads)       # CPU side for final reduction
    findmax_idxs_gpu = ArrayType{Int}(undef, num_reduction_threads)  # GPU side
    findmax_idxs_cpu = Vector{Int}(undef, num_reduction_threads)     # CPU side for final reduction
    sum_buffer_gpu = ArrayType{T}(undef, num_reduction_threads)
    sum_buffer_cpu = Vector{T}(undef, num_reduction_threads)         # CPU side for final reduction

    # Create FFT plans using AbstractFFTs (will use cuFFT/rocFFT etc.)
    # Plan along dimension 1 (samples) — FFT on original-size signal
    fft_plan = plan_fft(signal_baseband, 1)
    # BFFT plan on the (possibly padded) buffer
    bfft_plan = plan_bfft(code_baseband_freq_domain, 1)

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
        dopplers_hz,
        sampling_freq,
        codes_freq_domain,
        code_doppler_indices,
        code_doppler_step_val,
        code_doppler_offset_idx,
        num_code_dopplers_val,
        dopplers_gpu,
        downconvert_phases,
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
Batched downconversion kernel with @fastmath for performance.

Downconverts the input signal to baseband for all Doppler frequencies simultaneously.
Each thread handles one (sample, doppler) pair. Uses @fastmath for ~10% speedup.
"""
@kernel function ka_downconvert_kernel!(
    signal_baseband,
    @Const(signal),
    @Const(dopplers),
    freq_offset,              # Additional frequency offset (interm_freq + doppler_offset)
    inv_sampling_frequency,   # 1/fs for faster multiply
)
    i, d = @index(Global, NTuple)

    @fastmath begin
        doppler = dopplers[d] + freq_offset
        phase =
            Float32(-6.283185307179586) * doppler * Float32(i - 1) * inv_sampling_frequency
        s, c = sincos(phase)
        signal_baseband[i, d] = signal[i] * Complex{Float32}(c, s)
    end
end

"""
Fused findmax + sum kernel.

Computes both the maximum value/index AND total sum in a single pass over the data.
Reduces GPU→CPU synchronizations from 3 to 1 per PRN.
"""
@kernel function ka_findmax_and_sum_kernel!(
    thread_vals,      # Output: max value per thread
    thread_idxs,      # Output: linear index of max per thread
    thread_sums,      # Output: partial sum per thread
    @Const(data),     # Input data
    n,                # Total elements
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

This reduces GPU→CPU synchronizations from 3 to 1 per PRN, giving ~15-20% speedup
by computing the peak and total power sum in a single kernel launch.
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

# Arguments

  - `plan`: Pre-computed [`KAAcquisitionPlan`](@ref)
  - `signal`: Complex baseband signal samples (GPU array)
  - `prns`: PRN numbers to search (must be subset of `plan.avail_prn_channels`)

# Keyword Arguments

  - `interm_freq`: Intermediate frequency (default: `0.0Hz`)
  - `doppler_offset`: Offset added to Doppler search range (default: `0.0Hz`)
  - `store_powers`: If `true`, store full power matrix in results for plotting (default: `false`)

# Returns

Vector of [`AcquisitionResults`](@ref), one per PRN.

# Example

```julia
using Acquisition, GNSSSignals, AMDGPU

plan = KAAcquisitionPlan(GPSL1(), 16368, 16.368e6Hz, ROCArray; prns = 1:32)
signal_gpu = ROCArray(signal_cpu)
results = acquire!(plan, signal_gpu, 1:32)
```

# See also

[`KAAcquisitionPlan`](@ref), [`acquire!`](@ref)
"""
function acquire!(
    plan::KAAcquisitionPlan{T,S},
    signal::AbstractArray,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
    store_powers = false,
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
    dopplers_hz = plan.dopplers

    # Compute active code Doppler indices (handle doppler_offset)
    active_cd_indices = if !iszero(ustrip(doppler_offset))
        ratio = get_code_center_frequency_ratio(plan.system)
        [
            code_doppler_index(
                dopplers_hz[d] + Float64(ustrip(doppler_offset)),
                ratio,
                plan.code_doppler_step,
                plan.code_doppler_offset_idx,
                plan.num_code_dopplers,
            ) for d = 1:num_dopplers
        ]
    else
        plan.code_doppler_indices
    end

    # Allocate per-PRN power accumulators (reuse power_buffer for first PRN, allocate rest)
    # We need to track accumulated powers per PRN across chunks
    power_accumulators = [similar(plan.power_accumulator) for _ = 1:num_prns]

    # Process signal in chunks - downconvert/FFT once per chunk for all PRNs
    for chunk_idx = 1:num_chunks
        start_idx = (chunk_idx - 1) * chunk_samples + 1
        end_idx = min(chunk_idx * chunk_samples, num_signal_samples)
        actual_chunk_size = end_idx - start_idx + 1
        signal_chunk = view(signal, start_idx:end_idx)

        # 1. Batched downconversion for this chunk (ONCE per chunk, not per PRN)
        # Zero the full buffer (DBZP: signal_baseband is linear_fft_size ≈ 2N,
        # only first actual_chunk_size samples get signal data)
        plan.signal_baseband .= zero(eltype(plan.signal_baseband))

        if iszero(ustrip(interm_freq)) && iszero(ustrip(doppler_offset))
            # Fast path: use precomputed phases (phases beyond N are already zero)
            signal_view = view(plan.signal_baseband, 1:actual_chunk_size, :)
            phases_view = view(plan.downconvert_phases, 1:actual_chunk_size, :)
            signal_view .= signal_chunk .* phases_view
        else
            # Slow path: use kernel for runtime frequency offsets
            freq_offset = T(ustrip(doppler_offset + interm_freq))
            inv_sampling_freq = T(1.0 / ustrip(plan.sampling_frequency))
            ka_downconvert_kernel!(backend)(
                plan.signal_baseband,
                signal_chunk,
                plan.dopplers_gpu,
                freq_offset,
                inv_sampling_freq;
                ndrange = (actual_chunk_size, size(plan.signal_baseband, 2)),
            )
        end

        # 2. Batched FFT along sample dimension (ONCE per chunk, not per PRN)
        mul!(plan.signal_baseband_freq_domain, plan.fft_plan, plan.signal_baseband)

        # 3-5. Process each PRN with the already-transformed signal
        N = plan.linear_fft_size
        N_pad = plan.bfft_size
        needs_padding = N_pad > N
        for (p_idx, prn_idx) in enumerate(prn_indices)
            # 3. Code multiplication for this PRN, grouped by code Doppler index
            # When zero-padding, clear the padded buffer first, then write
            # positive freqs (1..N÷2+1) and negative freqs (N_pad-neg_count+1..N_pad)
            if needs_padding
                plan.code_baseband_freq_domain .=
                    zero(eltype(plan.code_baseband_freq_domain))
                pos_end = N ÷ 2 + 1
                neg_count = N - pos_end
                group_start = 1
                for d_idx = 1:num_dopplers
                    cd_idx = active_cd_indices[d_idx]
                    if d_idx == num_dopplers || active_cd_indices[d_idx+1] != cd_idx
                        code_col = view(plan.codes_freq_domain, :, prn_idx, cd_idx)
                        doppler_range = group_start:d_idx
                        sig_view = view(plan.signal_baseband_freq_domain, :, doppler_range)
                        # Positive frequencies
                        view(plan.code_baseband_freq_domain, 1:pos_end, doppler_range) .=
                            view(code_col, 1:pos_end) .* conj.(view(sig_view, 1:pos_end, :))
                        # Negative frequencies
                        view(
                            plan.code_baseband_freq_domain,
                            (N_pad-neg_count+1):N_pad,
                            doppler_range,
                        ) .=
                            view(code_col, (pos_end+1):N) .*
                            conj.(view(sig_view, (pos_end+1):N, :))
                        group_start = d_idx + 1
                    end
                end
            else
                group_start = 1
                for d_idx = 1:num_dopplers
                    cd_idx = active_cd_indices[d_idx]
                    if d_idx == num_dopplers || active_cd_indices[d_idx+1] != cd_idx
                        code_col = view(plan.codes_freq_domain, :, prn_idx, cd_idx)
                        doppler_range = group_start:d_idx
                        view(plan.code_baseband_freq_domain, :, doppler_range) .=
                            code_col .*
                            conj.(view(plan.signal_baseband_freq_domain, :, doppler_range))
                        group_start = d_idx + 1
                    end
                end
            end

            # 4. Backward FFT
            mul!(plan.code_baseband, plan.bfft_plan, plan.code_baseband_freq_domain)

            # 5. Compute power and accumulate
            if chunk_idx == 1
                power_accumulators[p_idx] .=
                    abs2.(view(plan.code_baseband, 1:num_code_samples, :))
            else
                power_accumulators[p_idx] .+=
                    abs2.(view(plan.code_baseband, 1:num_code_samples, :))
            end
        end
    end

    # 6. Process results for each PRN
    map(enumerate(prns)) do (p_idx, prn)
        power_acc = power_accumulators[p_idx]

        # Fused findmax + sum on accumulated powers
        signal_noise_power, linear_idx, total_power = gpu_findmax_and_sum!(
            plan.findmax_vals_gpu,
            plan.findmax_vals_cpu,
            plan.findmax_idxs_gpu,
            plan.findmax_idxs_cpu,
            plan.sum_buffer_gpu,
            plan.sum_buffer_cpu,
            power_acc,
            backend,
        )

        # Convert linear index to (code_index, doppler_index)
        doppler_index = div(linear_idx - 1, num_code_samples) + 1
        code_index = mod(linear_idx - 1, num_code_samples) + 1

        # Compute noise power from total minus peak region
        num_dopplers = size(power_acc, 2)
        lower_end = max(1, code_index - samples_per_chip)
        upper_start = min(num_code_samples, code_index + samples_per_chip) + 1
        peak_rows = (upper_start - 1) - (lower_end - 1)
        peak_region_samples = peak_rows * num_dopplers
        total_samples = length(power_acc)
        noise_samples = total_samples - peak_region_samples

        noise_power = noise_samples > 0 ? (total_power / total_samples) : zero(T)
        signal_power = signal_noise_power - noise_power

        # CN0 includes coherent integration gain (normalized by code_period).
        # Non-coherent integration improves detection probability but doesn't increase
        # the measured SNR ratio, so no explicit gain is added here.
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

        # Only keep full power matrix if requested
        result_powers = if store_powers
            Matrix{Float32}(Array(power_acc))
        else
            Matrix{Float32}(undef, 0, 0)
        end

        AcquisitionResults(
            plan.system,
            prn,
            effective_sampling_freq,
            doppler * 1.0Hz,
            code_phase,
            CN0,
            Float32(noise_power),
            result_powers,
            result_dopplers,
        )
    end
end

# Single PRN convenience method
function acquire!(
    plan::KAAcquisitionPlan{T,S},
    signal::AbstractArray,
    prn::Integer;
    interm_freq = 0.0Hz,
    doppler_offset = 0.0Hz,
    store_powers = false,
)::AcquisitionResults{S,Float32} where {T,S}
    only(acquire!(plan, signal, [prn]; interm_freq, doppler_offset, store_powers))
end
