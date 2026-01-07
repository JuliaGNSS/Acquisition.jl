"""
    AcquisitionPlan

Pre-computed acquisition plan for efficient repeated acquisition over the same signal parameters.

Contains pre-allocated buffers and FFT plans to avoid repeated memory allocation when acquiring
multiple signals with the same length and sampling frequency.

# Fields
- `system`: GNSS system (e.g., `GPSL1()`)
- `signal_length`: Number of samples in the signal
- `sampling_freq`: Sampling frequency
- `dopplers`: Range of Doppler frequencies to search
- `avail_prn_channels`: PRN channels available in this plan

# See also
[`acquire!`](@ref), [`CoarseFineAcquisitionPlan`](@ref)
"""
struct AcquisitionPlan{S,DS,CS,P,PS}
    system::S
    signal_length::Int
    sampling_freq::typeof(1.0Hz)
    dopplers::DS
    codes_freq_domain::CS
    signal_baseband::Vector{ComplexF32}
    signal_baseband_freq_domain::Vector{ComplexF32}
    code_freq_baseband_freq_domain::Vector{ComplexF32}
    code_baseband::Vector{ComplexF32}
    signal_powers::Vector{Matrix{Float32}}
    fft_plan::P
    avail_prn_channels::PS
end

"""
    AcquisitionPlan(system, signal_length, sampling_freq; kwargs...)

Create an acquisition plan for efficient repeated acquisition.

# Arguments
- `system`: GNSS system (e.g., `GPSL1()`)
- `signal_length`: Number of samples in the signal
- `sampling_freq`: Sampling frequency of the signal

# Keyword Arguments
- `max_doppler`: Maximum Doppler frequency to search (default: `7000Hz`)
- `min_doppler`: Minimum Doppler frequency to search (default: `-max_doppler`)
- `dopplers`: Custom Doppler range (default: `min_doppler:250Hz:max_doppler`)
- `prns`: PRN channels to prepare (default: `1:34`)
- `fft_flag`: FFTW planning flag (default: `FFTW.MEASURE`)

# Example
```julia
using Acquisition, GNSSSignals
plan = AcquisitionPlan(GPSL1(), 10000, 5e6Hz; prns=1:32)
results = acquire!(plan, signal, 1:32)
```

# See also
[`acquire!`](@ref), [`CoarseFineAcquisitionPlan`](@ref)
"""
function AcquisitionPlan(
    system,
    signal_length,
    sampling_freq;
    max_doppler = 7000Hz,
    min_doppler = -max_doppler,
    dopplers = min_doppler:250Hz:max_doppler,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
)
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan = common_buffers(system, signal_length, sampling_freq, prns, fft_flag)
    Δt = signal_length / sampling_freq
    code_interval = get_code_length(system) / get_code_frequency(system)
    signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(dopplers),
        ) for _ in prns
    ]
    AcquisitionPlan(
        system,
        signal_length,
        sampling_freq,
        dopplers,
        codes_freq_domain,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        signal_powers,
        fft_plan,
        prns,
    )
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
    CoarseFineAcquisitionPlan(system, signal_length, sampling_freq; kwargs...)

Create a two-stage coarse-fine acquisition plan.

# Arguments
- `system`: GNSS system (e.g., `GPSL1()`)
- `signal_length`: Number of samples in the signal
- `sampling_freq`: Sampling frequency of the signal

# Keyword Arguments
- `max_doppler`: Maximum Doppler frequency to search (default: `7000Hz`)
- `min_doppler`: Minimum Doppler frequency to search (default: `-max_doppler`)
- `coarse_step`: Doppler step size for coarse search (default: `250Hz`)
- `fine_step`: Doppler step size for fine search (default: `25Hz`)
- `prns`: PRN channels to prepare (default: `1:34`)
- `fft_flag`: FFTW planning flag (default: `FFTW.MEASURE`)

# Example
```julia
using Acquisition, GNSSSignals
plan = CoarseFineAcquisitionPlan(GPSL1(), 10000, 5e6Hz; prns=1:32)
results = acquire!(plan, signal, 1:32)
```

# See also
[`coarse_fine_acquire`](@ref), [`AcquisitionPlan`](@ref)
"""
function CoarseFineAcquisitionPlan(
    system,
    signal_length,
    sampling_freq;
    max_doppler = 7000Hz,
    min_doppler = -max_doppler,
    coarse_step = 250Hz,
    fine_step = 25Hz,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
)
    coarse_dopplers = min_doppler:coarse_step:max_doppler
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan = common_buffers(system, signal_length, sampling_freq, prns, fft_flag)
    Δt = signal_length / sampling_freq
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
    coarse_plan = AcquisitionPlan(
        system,
        signal_length,
        sampling_freq,
        coarse_dopplers,
        codes_freq_domain,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        coarse_signal_powers,
        fft_plan,
        prns,
    )
    fine_plan = AcquisitionPlan(
        system,
        signal_length,
        sampling_freq,
        fine_doppler_range,
        codes_freq_domain,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        fine_signal_powers,
        fft_plan,
        prns,
    )
    CoarseFineAcquisitionPlan(coarse_plan, fine_plan)
end

function common_buffers(system, signal_length, sampling_freq, prns, fft_flag)
    codes = [gen_code(signal_length, system, sat_prn, sampling_freq) for sat_prn in prns]
    signal_baseband = Vector{ComplexF32}(undef, signal_length)
    signal_baseband_freq_domain = similar(signal_baseband)
    code_freq_baseband_freq_domain = similar(signal_baseband)
    code_baseband = similar(signal_baseband)
    fft_plan = plan_fft(signal_baseband; flags = fft_flag)
    codes_freq_domain = map(code -> fft_plan * code, codes)
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan
end