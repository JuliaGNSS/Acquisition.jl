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
    code_doppler_indices::Vector{Int}
    code_doppler_step::Float64
    code_doppler_offset_idx::Int
    num_code_dopplers::Int
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
  - `dopplers`: Custom Doppler range (default: computed from `doppler_step`)
  - `doppler_step`: Doppler frequency step size (default: `doppler_step_factor / T` where
    `T = num_samples_to_integrate_coherently / sampling_freq`)
  - `doppler_step_factor`: Factor for computing Doppler step from integration time (default: `1//3`)
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
    doppler_step_factor = 1//3,
    doppler_step = doppler_step_factor * sampling_freq / num_samples_to_integrate_coherently,
    dopplers = min_doppler:doppler_step:max_doppler,
    code_doppler_tolerance = 0.01,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
) where {T<:AbstractFloat}
    # Code Doppler step: tolerance (cycles of phase error) / integration time (s) = Hz
    T_coh = num_samples_to_integrate_coherently / sampling_freq
    code_doppler_step = code_doppler_tolerance / ustrip(T_coh)
    code_dopplers, n_neg = compute_code_doppler_grid(system, min_doppler, max_doppler, code_doppler_step)
    code_doppler_offset_idx = n_neg + 1
    num_code_dopplers = length(code_dopplers)
    code_doppler_indices = compute_code_doppler_indices(dopplers, system, code_doppler_step, code_doppler_offset_idx, num_code_dopplers)

    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan,
    ifft_plan = common_buffers(T, system, num_samples_to_integrate_coherently, sampling_freq, prns, fft_flag, code_dopplers)
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
        code_doppler_indices,
        code_doppler_step,
        code_doppler_offset_idx,
        num_code_dopplers,
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
  - `coarse_step`: Doppler step size for coarse search (default: computed from `doppler_step_factor`)
  - `fine_step`: Doppler step size for fine search (default: `coarse_step / 10`)
  - `doppler_step_factor`: Factor for computing coarse step from integration time (default: `1//3`)
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
    doppler_step_factor = 1//3,
    coarse_step = doppler_step_factor * sampling_freq / num_samples_to_integrate_coherently,
    fine_step = coarse_step / 10,
    code_doppler_tolerance = 0.01,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
) where {T<:AbstractFloat}
    # Code Doppler step: tolerance (cycles of phase error) / integration time (s) = Hz
    T_coh = num_samples_to_integrate_coherently / sampling_freq
    code_doppler_step = code_doppler_tolerance / ustrip(T_coh)
    code_dopplers, n_neg = compute_code_doppler_grid(system, min_doppler, max_doppler, code_doppler_step)
    code_doppler_offset_idx = n_neg + 1
    num_code_dopplers = length(code_dopplers)

    coarse_dopplers = min_doppler:coarse_step:max_doppler
    coarse_code_doppler_indices = compute_code_doppler_indices(coarse_dopplers, system, code_doppler_step, code_doppler_offset_idx, num_code_dopplers)

    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan,
    ifft_plan = common_buffers(T, system, num_samples_to_integrate_coherently, sampling_freq, prns, fft_flag, code_dopplers)
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
    # Fine plan indices are precomputed assuming offset=0, but the fine search always
    # uses a non-zero doppler_offset (the coarse estimate), so cd_idx is recomputed
    # inline in acquire!. These indices satisfy the struct field requirement.
    fine_code_doppler_indices = compute_code_doppler_indices(fine_doppler_range, system, code_doppler_step, code_doppler_offset_idx, num_code_dopplers)

    fine_signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(fine_doppler_range),
        ) for _ in prns
    ]
    coarse_results = [
        AcquisitionResults(system, prn, sampling_freq, 0.0Hz, 0.0, 0.0, Float32(0), coarse_signal_powers[i], coarse_dopplers)
        for (i, prn) in enumerate(prns)
    ]
    coarse_prn_indices = Vector{Int}(undef, length(prns))
    coarse_output_results = Vector{Base.eltype(coarse_results)}(undef, length(prns))
    coarse_plan = AcquisitionPlan(
        system, num_samples_to_integrate_coherently, sampling_freq, coarse_dopplers,
        codes_freq_domain, coarse_code_doppler_indices, code_doppler_step, code_doppler_offset_idx, num_code_dopplers,
        signal_baseband, signal_baseband_freq_domain, code_freq_baseband_freq_domain, code_baseband,
        coarse_signal_powers, fft_plan, ifft_plan, prns, coarse_results, coarse_prn_indices, coarse_output_results,
    )
    fine_results = [
        AcquisitionResults(system, prn, sampling_freq, 0.0Hz, 0.0, 0.0, Float32(0), fine_signal_powers[i], fine_doppler_range)
        for (i, prn) in enumerate(prns)
    ]
    fine_prn_indices = Vector{Int}(undef, length(prns))
    fine_output_results = Vector{Base.eltype(fine_results)}(undef, length(prns))
    fine_plan = AcquisitionPlan(
        system, num_samples_to_integrate_coherently, sampling_freq, fine_doppler_range,
        codes_freq_domain, fine_code_doppler_indices, code_doppler_step, code_doppler_offset_idx, num_code_dopplers,
        signal_baseband, signal_baseband_freq_domain, code_freq_baseband_freq_domain, code_baseband,
        fine_signal_powers, fft_plan, ifft_plan, prns, fine_results, fine_prn_indices, fine_output_results,
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

"""
    compute_code_doppler_grid(system, min_doppler, max_doppler, code_doppler_step)

Compute a grid of code Doppler offsets centered at 0 Hz that covers the code Doppler
range implied by the carrier Doppler search range.

Returns `(code_dopplers, n_neg)` where `code_dopplers` is a range of code Doppler values
in Hz (without Unitful units) and `n_neg` is the number of negative steps.
"""
function compute_code_doppler_grid(system, min_doppler, max_doppler, code_doppler_step)
    ratio = get_code_center_frequency_ratio(system)
    min_code_doppler = ustrip(min_doppler) * ratio
    max_code_doppler = ustrip(max_doppler) * ratio
    n_neg = ceil(Int, -min_code_doppler / code_doppler_step)
    n_pos = ceil(Int, max_code_doppler / code_doppler_step)
    code_dopplers = (-n_neg:n_pos) .* code_doppler_step
    return code_dopplers, n_neg
end

"""
    compute_code_doppler_indices(dopplers, system, code_doppler_step, code_doppler_offset_idx, num_code_dopplers)

Map each carrier Doppler bin to the index of the nearest pre-computed code Doppler replica.
"""
function compute_code_doppler_indices(dopplers, system, code_doppler_step, code_doppler_offset_idx, num_code_dopplers)
    ratio = get_code_center_frequency_ratio(system)
    [clamp(round(Int, ustrip(doppler) * ratio / code_doppler_step) + code_doppler_offset_idx, 1, num_code_dopplers) for doppler in dopplers]
end

const FFTW_WISDOM_DIR = @get_scratch!("fftw_wisdom")

function with_fftw_wisdom(f)
    wisdom_path = joinpath(FFTW_WISDOM_DIR, "wisdom")
    isfile(wisdom_path) && FFTW.import_wisdom(wisdom_path)
    result = f()
    FFTW.export_wisdom(wisdom_path)
    result
end

function common_buffers(
    ::Type{T},
    system,
    num_samples_to_integrate_coherently,
    sampling_freq,
    prns,
    fft_flag,
    code_dopplers,
) where {T}
    codes = [
        [gen_code(num_samples_to_integrate_coherently, system, sat_prn, sampling_freq,
                  get_code_frequency(system) + code_doppler * 1.0Hz)
         for code_doppler in code_dopplers]
        for sat_prn in prns
    ]
    signal_baseband = Vector{Complex{T}}(undef, num_samples_to_integrate_coherently)
    signal_baseband_freq_domain = similar(signal_baseband)
    code_freq_baseband_freq_domain = Vector{ComplexF32}(undef, num_samples_to_integrate_coherently)
    code_baseband = similar(code_freq_baseband_freq_domain)
    fft_plan, ifft_plan = if fft_flag != FFTW.ESTIMATE
        with_fftw_wisdom() do
            plan_fft(signal_baseband; flags = fft_flag),
            plan_ifft(code_freq_baseband_freq_domain; flags = fft_flag)
        end
    else
        plan_fft(signal_baseband; flags = fft_flag),
        plan_ifft(code_freq_baseband_freq_domain; flags = fft_flag)
    end
    codes_freq_domain = [
        [fft_plan * code for code in prn_codes]
        for prn_codes in codes
    ]
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan,
    ifft_plan
end