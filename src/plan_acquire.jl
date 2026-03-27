"""
    AcquisitionPlan

Pre-computed acquisition plan for efficient repeated acquisition over the same signal parameters.

Contains pre-allocated buffers and FFT plans to avoid repeated memory allocation when acquiring
multiple signals with the same length and sampling frequency.

# Fields

  - `system`: GNSS system (e.g., `GPSL1()`)
  - `num_samples_to_integrate_coherently`: Number of samples per coherent integration chunk
  - `linear_fft_size`: FFT size for linear correlation (2 × num_samples_to_integrate_coherently)
  - `sampling_freq`: Sampling frequency
  - `dopplers`: Range of Doppler frequencies to search
  - `avail_prn_channels`: PRN channels available in this plan

# See also

[`acquire!`](@ref), [`CoarseFineAcquisitionPlan`](@ref)
"""
struct AcquisitionPlan{T<:AbstractFloat,S,DS,CS,P,BP,PS}
    system::S
    num_samples_to_integrate_coherently::Int
    linear_fft_size::Int
    bfft_size::Int
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
    bfft_plan::BP
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
  - `max_code_doppler_loss`: Maximum acceptable correlation loss in dB from code Doppler
    mismatch (default: `0.5`). Controls how many code replicas are pre-computed at different
    code Doppler offsets. This parameter works uniformly across all GNSS systems regardless
    of chip rate. Typical values: `0.5` (negligible loss), `1.0` (mild loss), `3.0` (aggressive).

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
    doppler_step = doppler_step_factor * sampling_freq /
                   num_samples_to_integrate_coherently,
    dopplers = min_doppler:doppler_step:max_doppler,
    max_code_doppler_loss = 0.5dB,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
    zero_pad_power::Int = 0,
) where {T<:AbstractFloat}
    # Convert dB loss to max phase error (cycles), then to code Doppler step (Hz)
    T_coh = num_samples_to_integrate_coherently / sampling_freq
    max_phase_err = max_phase_error_from_loss(max_code_doppler_loss)
    code_doppler_step = max_phase_err > 0 ? max_phase_err / ustrip(T_coh) : Inf
    code_dopplers, n_neg =
        compute_code_doppler_grid(system, min_doppler, max_doppler, code_doppler_step)
    code_doppler_offset_idx = n_neg + 1
    num_code_dopplers = length(code_dopplers)
    code_doppler_indices = compute_code_doppler_indices(
        dopplers,
        system,
        code_doppler_step,
        code_doppler_offset_idx,
        num_code_dopplers,
    )

    # Double Block Zero Padding (DBZP): zero-pad signal and code to >= 2N before FFT to
    # convert circular correlation into linear correlation, eliminating boundary artifacts
    # when code_length * sampling_freq / code_freq is non-integer.
    # See: D.J.R. van Nee and A.J.R.M. Coenen, "New Fast GPS Code-Acquisition Technique
    # Using FFT," Electronics Letters, vol. 27, no. 2, pp. 158-160, Jan. 1991.
    linear_fft_size = fftw_friendly_size(2 * num_samples_to_integrate_coherently)
    bfft_size = fftw_friendly_size(linear_fft_size << zero_pad_power)
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan,
    bfft_plan = common_buffers(
        T,
        system,
        num_samples_to_integrate_coherently,
        linear_fft_size,
        bfft_size,
        sampling_freq,
        prns,
        fft_flag,
        code_dopplers,
    )
    chunk_duration = num_samples_to_integrate_coherently / sampling_freq
    code_interval = get_code_length(system) / get_code_frequency(system)
    effective_sampling_freq = sampling_freq * bfft_size / linear_fft_size
    signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, effective_sampling_freq * min(chunk_duration, code_interval)),
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
            Float32(0),
            signal_powers[i],
            dopplers,
        ) for (i, prn) in enumerate(prns)
    ]
    prn_indices = Vector{Int}(undef, length(prns))
    output_results = Vector{Base.eltype(results)}(undef, length(prns))
    AcquisitionPlan(
        system,
        num_samples_to_integrate_coherently,
        linear_fft_size,
        bfft_size,
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
        bfft_plan,
        prns,
        results,
        prn_indices,
        output_results,
    )
end

# Convenience constructor that defaults num_samples_to_integrate_coherently to one bit period
function AcquisitionPlan(system, sampling_freq; kwargs...)
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
  - `max_code_doppler_loss`: Maximum acceptable correlation loss in dB from code Doppler
    mismatch (default: `0.5`). Controls how many code replicas are pre-computed at different
    code Doppler offsets. This parameter works uniformly across all GNSS systems regardless
    of chip rate. Typical values: `0.5` (negligible loss), `1.0` (mild loss), `3.0` (aggressive).

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
    max_code_doppler_loss = 0.5dB,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
    zero_pad_power::Int = 0,
) where {T<:AbstractFloat}
    # Convert dB loss to max phase error (cycles), then to code Doppler step (Hz)
    T_coh = num_samples_to_integrate_coherently / sampling_freq
    max_phase_err = max_phase_error_from_loss(max_code_doppler_loss)
    code_doppler_step = max_phase_err > 0 ? max_phase_err / ustrip(T_coh) : Inf
    code_dopplers, n_neg =
        compute_code_doppler_grid(system, min_doppler, max_doppler, code_doppler_step)
    code_doppler_offset_idx = n_neg + 1
    num_code_dopplers = length(code_dopplers)

    coarse_dopplers = min_doppler:coarse_step:max_doppler
    coarse_code_doppler_indices = compute_code_doppler_indices(
        coarse_dopplers,
        system,
        code_doppler_step,
        code_doppler_offset_idx,
        num_code_dopplers,
    )

    # DBZP: see comment in AcquisitionPlan constructor
    linear_fft_size = fftw_friendly_size(2 * num_samples_to_integrate_coherently)
    bfft_size = fftw_friendly_size(linear_fft_size << zero_pad_power)
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan,
    bfft_plan = common_buffers(
        T,
        system,
        num_samples_to_integrate_coherently,
        linear_fft_size,
        bfft_size,
        sampling_freq,
        prns,
        fft_flag,
        code_dopplers,
    )
    Δt = num_samples_to_integrate_coherently / sampling_freq
    code_interval = get_code_length(system) / get_code_frequency(system)
    effective_sampling_freq = sampling_freq * bfft_size / linear_fft_size
    coarse_signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, effective_sampling_freq * min(Δt, code_interval)),
            length(coarse_dopplers),
        ) for _ in prns
    ]

    fine_doppler_range = (-2*coarse_step):fine_step:(2*coarse_step)
    # Fine plan indices are precomputed assuming offset=0, but the fine search always
    # uses a non-zero doppler_offset (the coarse estimate), so cd_idx is recomputed
    # inline in acquire!. These indices satisfy the struct field requirement.
    fine_code_doppler_indices = compute_code_doppler_indices(
        fine_doppler_range,
        system,
        code_doppler_step,
        code_doppler_offset_idx,
        num_code_dopplers,
    )

    fine_signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, effective_sampling_freq * min(Δt, code_interval)),
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
            Float32(0),
            coarse_signal_powers[i],
            coarse_dopplers,
        ) for (i, prn) in enumerate(prns)
    ]
    coarse_prn_indices = Vector{Int}(undef, length(prns))
    coarse_output_results = Vector{Base.eltype(coarse_results)}(undef, length(prns))
    coarse_plan = AcquisitionPlan(
        system,
        num_samples_to_integrate_coherently,
        linear_fft_size,
        bfft_size,
        sampling_freq,
        coarse_dopplers,
        codes_freq_domain,
        coarse_code_doppler_indices,
        code_doppler_step,
        code_doppler_offset_idx,
        num_code_dopplers,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        coarse_signal_powers,
        fft_plan,
        bfft_plan,
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
            Float32(0),
            fine_signal_powers[i],
            fine_doppler_range,
        ) for (i, prn) in enumerate(prns)
    ]
    fine_prn_indices = Vector{Int}(undef, length(prns))
    fine_output_results = Vector{Base.eltype(fine_results)}(undef, length(prns))
    fine_plan = AcquisitionPlan(
        system,
        num_samples_to_integrate_coherently,
        linear_fft_size,
        bfft_size,
        sampling_freq,
        fine_doppler_range,
        codes_freq_domain,
        fine_code_doppler_indices,
        code_doppler_step,
        code_doppler_offset_idx,
        num_code_dopplers,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        fine_signal_powers,
        fft_plan,
        bfft_plan,
        prns,
        fine_results,
        fine_prn_indices,
        fine_output_results,
    )
    CoarseFineAcquisitionPlan(coarse_plan, fine_plan)
end

# Convenience constructor that defaults num_samples_to_integrate_coherently to one bit period
function CoarseFineAcquisitionPlan(system, sampling_freq; kwargs...)
    # Default to one bit period worth of samples for optimal non-coherent integration
    data_frequency = get_data_frequency(system)
    num_samples_to_integrate_coherently = ceil(Int, sampling_freq / data_frequency)
    CoarseFineAcquisitionPlan(
        system,
        num_samples_to_integrate_coherently,
        sampling_freq;
        kwargs...,
    )
end

"""
    max_phase_error_from_loss(loss_dB)

Compute the maximum code phase error (in cycles) that produces at most `loss_dB` of
correlation loss. The correlation loss from a linear code frequency mismatch over the
integration interval is `sinc²(ε)` where `ε` is the accumulated phase error in cycles.

Accepts a Unitful `Gain` value (e.g. `0.5u"dB"`).

Uses Newton's method to invert `sinc(ε) = 10^(-loss_dB/20)` for `ε ∈ [0, 0.5]`.
"""
function max_phase_error_from_loss(loss::Gain)
    loss_dB = loss.val
    loss_dB <= 0 && return 0.0
    target = 10^(-loss_dB / 20)  # sinc(ε) threshold
    target >= 1 && return 0.0
    # Initial guess from small-angle approximation: sinc(ε) ≈ 1 - π²ε²/6
    ε = sqrt(6 * (1 - target)) / π
    # Newton iterations on f(ε) = sinc(ε) - target = sin(πε)/(πε) - target
    for _ = 1:10
        πε = π * ε
        s, c = sincos(πε)
        sinc_val = s / πε
        # d/dε sinc(ε) = (cos(πε) - sinc(ε)) / ε
        sinc_deriv = (c - sinc_val) / ε
        ε -= (sinc_val - target) / sinc_deriv
        ε = clamp(ε, 0.0, 0.5)
    end
    return ε
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
    code_dopplers = ((-n_neg):n_pos) .* code_doppler_step
    return code_dopplers, n_neg
end

"""
    code_doppler_index(doppler_hz, ratio, code_doppler_step, code_doppler_offset_idx, num_code_dopplers)

Map a single carrier Doppler value (in Hz, unitless) to the index of the nearest
pre-computed code Doppler replica.
"""
@inline function code_doppler_index(
    doppler_hz,
    ratio,
    code_doppler_step,
    code_doppler_offset_idx,
    num_code_dopplers,
)
    clamp(
        round(Int, doppler_hz * ratio / code_doppler_step) + code_doppler_offset_idx,
        1,
        num_code_dopplers,
    )
end

"""
    compute_code_doppler_indices(dopplers, system, code_doppler_step, code_doppler_offset_idx, num_code_dopplers)

Map each carrier Doppler bin to the index of the nearest pre-computed code Doppler replica.
"""
function compute_code_doppler_indices(
    dopplers,
    system,
    code_doppler_step,
    code_doppler_offset_idx,
    num_code_dopplers,
)
    ratio = get_code_center_frequency_ratio(system)
    [
        code_doppler_index(
            ustrip(doppler),
            ratio,
            code_doppler_step,
            code_doppler_offset_idx,
            num_code_dopplers,
        ) for doppler in dopplers
    ]
end

"""
    fftw_friendly_size(n)

Return the smallest integer `≥ n` whose prime factors are all in {2, 3, 5, 7}.
FFTW is efficient for these "smooth" sizes; padding to the next power of 2 can be
much larger than necessary (e.g. 20000 → 32768) and hurt performance.
"""
fftw_friendly_size(n::Integer) = nextprod((2, 3, 5, 7), n)

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
    linear_fft_size,
    bfft_size,
    sampling_freq,
    prns,
    fft_flag,
    code_dopplers,
) where {T}
    codes = [
        [
            begin
                code = gen_code(
                    num_samples_to_integrate_coherently,
                    system,
                    sat_prn,
                    sampling_freq,
                    get_code_frequency(system) + code_doppler * 1.0Hz,
                )
                padded = zeros(eltype(code), linear_fft_size)
                padded[1:num_samples_to_integrate_coherently] .= code
                padded
            end for code_doppler in code_dopplers
        ] for sat_prn in prns
    ]
    signal_baseband = Vector{Complex{T}}(undef, linear_fft_size)
    signal_baseband_freq_domain = similar(signal_baseband)
    code_freq_baseband_freq_domain = Vector{ComplexF32}(undef, bfft_size)
    code_baseband = similar(code_freq_baseband_freq_domain)
    fft_plan, bfft_plan = if fft_flag != FFTW.ESTIMATE
        with_fftw_wisdom() do
            plan_fft(signal_baseband; flags = fft_flag),
            plan_bfft(code_freq_baseband_freq_domain; flags = fft_flag)
        end
    else
        plan_fft(signal_baseband; flags = fft_flag),
        plan_bfft(code_freq_baseband_freq_domain; flags = fft_flag)
    end
    codes_freq_domain = [[fft_plan * code for code in prn_codes] for prn_codes in codes]
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan,
    bfft_plan
end
