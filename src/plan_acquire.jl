"""
    AcquisitionPlan

Pre-computed acquisition plan for efficient repeated acquisition over the same signal parameters.

Contains pre-allocated buffers and FFT plans to avoid repeated memory allocation when acquiring
multiple signals with the same length and sampling frequency.

# Fields

  - `system`: GNSS system (e.g., `GPSL1()`)
  - `num_samples_to_integrate_coherently`: Number of samples per coherent integration chunk
  - `sampling_freq`: Sampling frequency
  - `dopplers`: Range of Doppler frequencies to search
  - `avail_prn_channels`: PRN channels available in this plan

# See also

[`acquire!`](@ref), [`CoarseFineAcquisitionPlan`](@ref)
"""
struct AcquisitionPlan{T<:AbstractFloat,S,DS,CS,P,IP,PS}
    system::S
    num_samples_to_integrate_coherently::Int
    sampling_freq::typeof(1.0Hz)
    dopplers::DS
    codes_freq_domain::CS
    signal_baseband::Vector{Complex{T}}
    signal_baseband_freq_domain::Vector{Complex{T}}
    code_freq_baseband_freq_domain::Vector{ComplexF32}
    code_baseband::Vector{ComplexF32}
    signal_powers::Vector{Matrix{Float32}}
    fft_plan::P
    ifft_plan::IP
    avail_prn_channels::PS
    results::Vector{AcquisitionResults{S,Float32,DS}}
    prn_indices::Vector{Int}
    output_results::Vector{AcquisitionResults{S,Float32,DS}}
end

"""
    AcquisitionPlan(system, sampling_freq; kwargs...)
    AcquisitionPlan(system, num_samples_to_integrate_coherently, sampling_freq; kwargs...)

Create an acquisition plan for efficient repeated acquisition.

# Arguments

  - `system`: GNSS system (e.g., `GPSL1()`)
  - `sampling_freq`: Sampling frequency of the signal
  - `num_samples_to_integrate_coherently`: Number of samples per coherent integration chunk (optional).
    Defaults to one bit period worth of samples, which enables non-coherent integration
    across bit transitions for longer signals.

# Keyword Arguments

  - `eltype`: Element type for internal buffers (default: `Float32`). Use `Float64` for
    `ComplexF64` signals to avoid allocations.
  - `max_doppler`: Maximum Doppler frequency to search (default: `7000Hz`)
  - `min_doppler`: Minimum Doppler frequency to search (default: `-max_doppler`)
  - `dopplers`: Custom Doppler range (default: `min_doppler:250Hz:max_doppler`)
  - `prns`: PRN channels to prepare (default: `1:34`)
  - `fft_flag`: FFTW planning flag (default: `FFTW.MEASURE`)

# Example

```julia
using Acquisition, GNSSSignals

# Create plan with default chunk size (one bit period)
plan = AcquisitionPlan(GPSL1(), 5e6Hz; prns = 1:32)
results = acquire!(plan, signal, 1:32)

# For ComplexF64 signals, use Float64 buffers for best performance:
plan64 = AcquisitionPlan(GPSL1(), 5e6Hz; prns = 1:32, eltype = Float64)
```

# See also

[`acquire!`](@ref), [`CoarseFineAcquisitionPlan`](@ref)
"""
function AcquisitionPlan(
    system,
    num_samples_to_integrate_coherently,
    sampling_freq;
    eltype::Type{T} = Float32,
    max_doppler = 7000Hz,
    min_doppler = -max_doppler,
    dopplers = min_doppler:250Hz:max_doppler,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
) where {T<:AbstractFloat}
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan,
    ifft_plan = common_buffers(T, system, num_samples_to_integrate_coherently, sampling_freq, prns, fft_flag)
    chunk_duration = num_samples_to_integrate_coherently / sampling_freq
    code_interval = get_code_length(system) / get_code_frequency(system)
    signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, sampling_freq * min(chunk_duration, code_interval)),
            length(dopplers),
        ) for _ in prns
    ]
    results = [
        AcquisitionResults(
            system,
            prn,
            sampling_freq,
            0.0Hz,
            0.0,
            0.0,
            Float32(0),
            signal_powers[i],
            dopplers,
        )
        for (i, prn) in enumerate(prns)
    ]
    prn_indices = Vector{Int}(undef, length(prns))
    output_results = Vector{Base.eltype(results)}(undef, length(prns))
    AcquisitionPlan(
        system,
        num_samples_to_integrate_coherently,
        sampling_freq,
        dopplers,
        codes_freq_domain,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        signal_powers,
        fft_plan,
        ifft_plan,
        prns,
        results,
        prn_indices,
        output_results,
    )
end

# Convenience constructor that defaults num_samples_to_integrate_coherently to one bit period
function AcquisitionPlan(
    system,
    sampling_freq;
    kwargs...
)
    # Default to one bit period worth of samples for optimal non-coherent integration
    data_frequency = get_data_frequency(system)
    num_samples_to_integrate_coherently = ceil(Int, sampling_freq / data_frequency)
    AcquisitionPlan(system, num_samples_to_integrate_coherently, sampling_freq; kwargs...)
end

"""
    CoarseFineAcquisitionPlan

Pre-computed acquisition plan for two-stage coarse-fine acquisition.

Performs an initial coarse search to find approximate Doppler, then refines with a
fine search around that estimate. This approach is more efficient than a single
high-resolution search.

# Fields

  - `coarse_plan`: [`AcquisitionPlan`](@ref) for the initial coarse search
  - `fine_plan`: [`AcquisitionPlan`](@ref) for the refined fine search

# See also

[`coarse_fine_acquire`](@ref), [`AcquisitionPlan`](@ref)
"""
struct CoarseFineAcquisitionPlan{C<:AcquisitionPlan,F<:AcquisitionPlan}
    coarse_plan::C
    fine_plan::F
end

"""
    CoarseFineAcquisitionPlan(system, sampling_freq; kwargs...)
    CoarseFineAcquisitionPlan(system, num_samples_to_integrate_coherently, sampling_freq; kwargs...)

Create a two-stage coarse-fine acquisition plan.

# Arguments

  - `system`: GNSS system (e.g., `GPSL1()`)
  - `sampling_freq`: Sampling frequency of the signal
  - `num_samples_to_integrate_coherently`: Number of samples per coherent integration chunk (optional).
    Defaults to one bit period worth of samples.

# Keyword Arguments

  - `eltype`: Element type for internal buffers (default: `Float32`). Use `Float64` for
    `ComplexF64` signals to avoid allocations.
  - `max_doppler`: Maximum Doppler frequency to search (default: `7000Hz`)
  - `min_doppler`: Minimum Doppler frequency to search (default: `-max_doppler`)
  - `coarse_step`: Doppler step size for coarse search (default: `250Hz`)
  - `fine_step`: Doppler step size for fine search (default: `25Hz`)
  - `prns`: PRN channels to prepare (default: `1:34`)
  - `fft_flag`: FFTW planning flag (default: `FFTW.MEASURE`)

# Example

```julia
using Acquisition, GNSSSignals

# Create plan with default chunk size (one bit period)
plan = CoarseFineAcquisitionPlan(GPSL1(), 5e6Hz; prns = 1:32)
results = acquire!(plan, signal, 1:32)

# For ComplexF64 signals, use Float64 buffers for best performance:
plan64 = CoarseFineAcquisitionPlan(GPSL1(), 5e6Hz; prns = 1:32, eltype = Float64)
```

# See also

[`coarse_fine_acquire`](@ref), [`AcquisitionPlan`](@ref)
"""
function CoarseFineAcquisitionPlan(
    system,
    num_samples_to_integrate_coherently,
    sampling_freq;
    eltype::Type{T} = Float32,
    max_doppler = 7000Hz,
    min_doppler = -max_doppler,
    coarse_step = 250Hz,
    fine_step = 25Hz,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
) where {T<:AbstractFloat}
    coarse_dopplers = min_doppler:coarse_step:max_doppler
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan,
    ifft_plan = common_buffers(T, system, num_samples_to_integrate_coherently, sampling_freq, prns, fft_flag)
    Δt = num_samples_to_integrate_coherently / sampling_freq
    code_interval = get_code_length(system) / get_code_frequency(system)
    coarse_signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(coarse_dopplers),
        ) for _ in prns
    ]
    fine_doppler_range = -2*coarse_step:fine_step:2*coarse_step
    fine_signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(fine_doppler_range),
        ) for _ in prns
    ]
    coarse_results = [
        AcquisitionResults(
            system,
            prn,
            sampling_freq,
            0.0Hz,
            0.0,
            0.0,
            Float32(0),
            coarse_signal_powers[i],
            coarse_dopplers,
        )
        for (i, prn) in enumerate(prns)
    ]
    coarse_prn_indices = Vector{Int}(undef, length(prns))
    coarse_output_results = Vector{Base.eltype(coarse_results)}(undef, length(prns))
    coarse_plan = AcquisitionPlan(
        system,
        num_samples_to_integrate_coherently,
        sampling_freq,
        coarse_dopplers,
        codes_freq_domain,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        coarse_signal_powers,
        fft_plan,
        ifft_plan,
        prns,
        coarse_results,
        coarse_prn_indices,
        coarse_output_results,
    )
    fine_results = [
        AcquisitionResults(
            system,
            prn,
            sampling_freq,
            0.0Hz,
            0.0,
            0.0,
            Float32(0),
            fine_signal_powers[i],
            fine_doppler_range,
        )
        for (i, prn) in enumerate(prns)
    ]
    fine_prn_indices = Vector{Int}(undef, length(prns))
    fine_output_results = Vector{Base.eltype(fine_results)}(undef, length(prns))
    fine_plan = AcquisitionPlan(
        system,
        num_samples_to_integrate_coherently,
        sampling_freq,
        fine_doppler_range,
        codes_freq_domain,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        fine_signal_powers,
        fft_plan,
        ifft_plan,
        prns,
        fine_results,
        fine_prn_indices,
        fine_output_results,
    )
    CoarseFineAcquisitionPlan(coarse_plan, fine_plan)
end

# Convenience constructor that defaults num_samples_to_integrate_coherently to one bit period
function CoarseFineAcquisitionPlan(
    system,
    sampling_freq;
    kwargs...
)
    # Default to one bit period worth of samples for optimal non-coherent integration
    data_frequency = get_data_frequency(system)
    num_samples_to_integrate_coherently = ceil(Int, sampling_freq / data_frequency)
    CoarseFineAcquisitionPlan(system, num_samples_to_integrate_coherently, sampling_freq; kwargs...)
end

function common_buffers(
    ::Type{T},
    system,
    num_samples_to_integrate_coherently,
    sampling_freq,
    prns,
    fft_flag,
) where {T}
    codes = [gen_code(num_samples_to_integrate_coherently, system, sat_prn, sampling_freq) for sat_prn in prns]
    signal_baseband = Vector{Complex{T}}(undef, num_samples_to_integrate_coherently)
    signal_baseband_freq_domain = similar(signal_baseband)
    code_freq_baseband_freq_domain = Vector{ComplexF32}(undef, num_samples_to_integrate_coherently)
    code_baseband = similar(code_freq_baseband_freq_domain)
    fft_plan = plan_fft(signal_baseband; flags = fft_flag)
    ifft_plan = plan_ifft(code_freq_baseband_freq_domain; flags = fft_flag)
    codes_freq_domain = map(code -> fft_plan * code, codes)
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan,
    ifft_plan
end