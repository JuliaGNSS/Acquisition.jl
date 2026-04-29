# src/recommend_sampling_freq.jl
# Search for sampling frequencies that give fast FM-DBZP acquisition.

"""
    SamplingFreqRecommendation

A single sampling-frequency candidate from [`recommend_sampling_freqs`](@ref).

# Fields

  - `sampling_freq::typeof(1.0Hz)`: Sampling frequency
  - `samples_per_code::Int`: Samples per PRN code period at this `sampling_freq`
  - `num_blocks::Int`: Smallest valid `num_blocks` that meets `min_doppler_coverage`
    (= the value `plan_acquire` would pick)
  - `block_size::Int`: `samples_per_code ÷ num_blocks`
  - `inner_fft_size::Int`: Inner double-block FFT length (`2 × block_size`)
  - `num_doppler_bins::Int`: Column-FFT length
    (`num_coherently_integrated_code_periods × num_blocks`)
  - `doppler_coverage::typeof(1.0Hz)`: Total Doppler coverage at this configuration
  - `inner_max_prime::Int`: Largest prime factor of `inner_fft_size` — smaller is
    better for FFTW
  - `num_doppler_bins_max_prime::Int`: Largest prime factor of `num_doppler_bins`
  - `cost::Float64`: Estimated combined FFT cost
    (`inner_fft_cost + column_fft_cost`, see [`recommend_sampling_freqs`](@ref))
"""
struct SamplingFreqRecommendation
    sampling_freq::typeof(1.0Hz)
    samples_per_code::Int
    num_blocks::Int
    block_size::Int
    inner_fft_size::Int
    num_doppler_bins::Int
    doppler_coverage::typeof(1.0Hz)
    inner_max_prime::Int
    num_doppler_bins_max_prime::Int
    cost::Float64
end

function _prime_factors(n::Integer)
    n <= 1 && return Int[]
    facs = Int[]
    p = 2
    while p * p <= n
        while n % p == 0
            push!(facs, Int(p))
            n ÷= p
        end
        p += 1
    end
    n > 1 && push!(facs, Int(n))
    facs
end

const _SUPERSCRIPT_DIGITS = ('⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹')

_to_superscript(k::Integer) = String([_SUPERSCRIPT_DIGITS[d - '0' + 1] for d in string(k)])

# Pretty-print a positive integer as e.g. "2³·3·31".
function _factor_string(n::Integer)
    n <= 1 && return string(n)
    facs = _prime_factors(n)
    parts = String[]
    i = 1
    while i <= length(facs)
        p = facs[i]
        j = i
        while j <= length(facs) && facs[j] == p
            j += 1
        end
        k = j - i
        push!(parts, k == 1 ? string(p) : string(p, _to_superscript(k)))
        i = j
    end
    join(parts, "·")
end

function _largest_prime_factor(n::Integer)
    n <= 1 && return 1
    largest = 1
    p = 2
    while p * p <= n
        while n % p == 0
            largest = p
            n ÷= p
        end
        p += 1
    end
    n > 1 ? Int(n) : largest
end

# Smallest divisor of `samples_per_code` that is >= `min_num_blocks`.
# Returns `nothing` if no such divisor exists.
function _smallest_valid_num_blocks(samples_per_code::Integer, min_num_blocks::Integer)
    for d in min_num_blocks:samples_per_code
        samples_per_code % d == 0 && return Int(d)
    end
    nothing
end

# Pretty-print the Doppler grid as "-fmin Hz : step Hz : +fmax Hz".
# Mirrors the grid `plan_acquire` builds: range(-cov/2, step=spacing, length=N),
# so the last bin is +cov/2 - spacing (asymmetric, one fewer positive bin).
function _doppler_grid_string(coverage_hz::Real, num_doppler_bins::Integer)
    spacing = coverage_hz / num_doppler_bins
    fmin = -coverage_hz / 2
    fmax = fmin + (num_doppler_bins - 1) * spacing
    fmt(x) = let r = round(x, digits = 1)
        isinteger(r) ? string(Int(r)) : string(r)
    end
    string(fmt(fmin), " : ", fmt(spacing), " : ", fmt(fmax), " Hz")
end

# Cost model matching the docstring of `recommend_sampling_freqs`.
function _fft_cost(samples_per_code, num_blocks, num_doppler_bins, inner_fft_size)
    # 2 (forward + inverse) × num_blocks FFTs of size inner_fft_size, each ~ N log N
    inner = 2.0 * num_blocks * inner_fft_size * log2(inner_fft_size)
    # samples_per_code column FFTs of size num_doppler_bins, each ~ N log N
    column = float(samples_per_code) * num_doppler_bins * log2(num_doppler_bins)
    inner + column
end

"""
    recommend_sampling_freqs(code_length, code_freq;
        fs_min = code_freq,
        fs_max = 1.5 * fs_min,
        min_doppler_coverage = 7000Hz,
        num_coherently_integrated_code_periods = 1,
        num_alternatives = 5,
        max_prime = 7,
        sort_by = :cost,
        fs_step = 1000Hz,
        sdr_clock_plan = nothing,
    ) -> Vector{SamplingFreqRecommendation}

Search the range `[fs_min, fs_max]` for sampling frequencies with smooth-prime
factorizations and return the `num_alternatives` best candidates.

The smallest valid `num_blocks` (the one [`plan_acquire`](@ref) would pick) is
computed for every candidate, and candidates are ranked by either FFT cost or
factorization smoothness. Candidates with the same `samples_per_code` are
deduplicated — the first matching `sampling_freq` is reported.

# Arguments

  - `code_length`: Number of chips per PRN code period
  - `code_freq`: Code chipping rate (a frequency, e.g. `1.023e6Hz`)

# Keyword Arguments

  - `fs_min`, `fs_max`: Sweep bounds for the sampling frequency
  - `min_doppler_coverage`: Minimum guaranteed one-sided Doppler reach. The
    recommender mirrors [`plan_acquire`](@ref) — both ends of the searched
    grid will be at least `±min_doppler_coverage`.
  - `num_coherently_integrated_code_periods`: Number of code periods per coherent
    integration block. Affects `num_doppler_bins` and therefore the column-FFT cost.
  - `num_alternatives`: Maximum number of candidates to return (default `5`)
  - `max_prime`: Reject candidates whose `samples_per_code`, `inner_fft_size`,
    or `num_doppler_bins` contain a prime factor larger than this value
    (default `7` — FFTW's "fast" regime)
  - `sort_by`: `:cost` (default) ranks by estimated FFT FLOPs; `:smoothness`
    ranks by largest prime factor of `inner_fft_size` (ties broken by cost)
  - `fs_step`: Sampling-frequency sweep step (default `1000Hz`)
  - `sdr_clock_plan`: Optional [`AbstractSDRClockPlan`](@ref) describing
    hardware sample-rate constraints. When provided, the sweep range is
    intersected with [`sample_rate_range`](@ref)`(plan)` and every
    candidate must satisfy [`is_valid_sample_rate`](@ref)`(plan, fs)`.
    Default `nothing` — no hardware filtering.

# Cost Model

Both FFT stages of FM-DBZP are accounted for:

```
inner_fft_cost  = 2 × num_blocks × inner_fft_size × log₂(inner_fft_size)
column_fft_cost = samples_per_code × num_doppler_bins × log₂(num_doppler_bins)
cost            = inner_fft_cost + column_fft_cost
```

This mirrors the per-PRN per coherent-step work, ignoring constant factors.

# See also

[`recommend_sampling_freqs(::AbstractGNSS)`](@ref), [`plan_acquire`](@ref)
"""
function recommend_sampling_freqs(
    code_length::Integer,
    code_freq;
    fs_min = code_freq,
    fs_max = 1.5 * fs_min,
    min_doppler_coverage = 7000Hz,
    num_coherently_integrated_code_periods::Int = 1,
    num_alternatives::Int = 5,
    max_prime::Int = 7,
    sort_by::Symbol = :cost,
    fs_step = 1000Hz,
    sdr_clock_plan::Union{AbstractSDRClockPlan,Nothing} = nothing,
)
    sort_by in (:cost, :smoothness) || throw(ArgumentError(
        "sort_by must be :cost or :smoothness, got :$sort_by"))
    num_alternatives >= 1 || throw(ArgumentError("num_alternatives must be >= 1"))
    max_prime >= 2 || throw(ArgumentError("max_prime must be >= 2"))
    num_coherently_integrated_code_periods >= 1 || throw(ArgumentError(
        "num_coherently_integrated_code_periods must be >= 1"))

    fs_min_hz = ustrip(Hz, fs_min)
    fs_max_hz = ustrip(Hz, fs_max)
    fs_step_hz = ustrip(Hz, fs_step)
    code_freq_hz = ustrip(Hz, code_freq)
    min_doppler_coverage_hz = ustrip(Hz, min_doppler_coverage)

    fs_min_hz <= fs_max_hz || throw(ArgumentError("fs_min must be <= fs_max"))
    fs_step_hz > 0 || throw(ArgumentError("fs_step must be > 0"))

    if sdr_clock_plan !== nothing
        plan_min, plan_max = sample_rate_range(sdr_clock_plan)
        fs_min_hz = max(fs_min_hz, plan_min)
        fs_max_hz = min(fs_max_hz, plan_max)
    end

    T_code = code_length / code_freq_hz
    seen_spc = Set{Int}()
    candidates = SamplingFreqRecommendation[]

    fs_hz = fs_min_hz
    while fs_hz <= fs_max_hz
        if sdr_clock_plan !== nothing && !is_valid_sample_rate(sdr_clock_plan, fs_hz)
            fs_hz += fs_step_hz
            continue
        end
        samples_per_code = ceil(Int, T_code * fs_hz)
        if !(samples_per_code in seen_spc)
            push!(seen_spc, samples_per_code)
            spc_max_prime = _largest_prime_factor(samples_per_code)
            if spc_max_prime <= max_prime
                bin_width = fs_hz / samples_per_code
                # Mirror plan_acquire's geometry: the highest searched bin must reach
                # +min_doppler_coverage. See `plan_acquire` for the derivation.
                min_num_blocks = ceil(Int,
                    2 * min_doppler_coverage_hz / bin_width
                    + 2 / num_coherently_integrated_code_periods)
                num_blocks = _smallest_valid_num_blocks(samples_per_code, min_num_blocks)
                if num_blocks !== nothing
                    block_size = samples_per_code ÷ num_blocks
                    inner_fft_size = 2 * block_size
                    num_doppler_bins = num_coherently_integrated_code_periods * num_blocks
                    inner_max_prime = _largest_prime_factor(inner_fft_size)
                    ndb_max_prime = _largest_prime_factor(num_doppler_bins)
                    if inner_max_prime <= max_prime && ndb_max_prime <= max_prime
                        push!(candidates, SamplingFreqRecommendation(
                            fs_hz * Hz,
                            samples_per_code,
                            num_blocks,
                            block_size,
                            inner_fft_size,
                            num_doppler_bins,
                            (num_blocks * bin_width) * Hz,
                            inner_max_prime,
                            ndb_max_prime,
                            _fft_cost(samples_per_code, num_blocks, num_doppler_bins, inner_fft_size),
                        ))
                    end
                end
            end
        end
        fs_hz += fs_step_hz
    end

    if sort_by === :cost
        sort!(candidates, by = c -> (c.cost, c.inner_max_prime))
    else
        sort!(candidates, by = c -> (c.inner_max_prime, c.cost))
    end

    return first(candidates, num_alternatives)
end

"""
    recommend_sampling_freqs(system::AbstractGNSS; kwargs...) -> Vector{SamplingFreqRecommendation}

Convenience method that derives `code_length` and `code_freq` from `system` via
`get_code_length(system)` and `get_code_frequency(system)`.

All keyword arguments are forwarded to
[`recommend_sampling_freqs(::Integer, ::Any)`](@ref).
"""
function recommend_sampling_freqs(system::AbstractGNSS; kwargs...)
    recommend_sampling_freqs(
        get_code_length(system),
        get_code_frequency(system);
        kwargs...,
    )
end

function Base.show(io::IO, ::MIME"text/plain", r::SamplingFreqRecommendation)
    print(io, "SamplingFreqRecommendation(",
        "fs=", r.sampling_freq,
        ", samples_per_code=", r.samples_per_code, " (", _factor_string(r.samples_per_code), ")",
        ", num_blocks=", r.num_blocks,
        ", block_size=", r.block_size,
        ", inner_fft=", r.inner_fft_size, " (", _factor_string(r.inner_fft_size), ")",
        ", num_doppler_bins=", r.num_doppler_bins, " (", _factor_string(r.num_doppler_bins), ")",
        ", doppler_grid=", _doppler_grid_string(ustrip(Hz, r.doppler_coverage), r.num_doppler_bins),
        ", cost≈", round(r.cost, sigdigits = 4),
        ")")
end

function Base.show(
    io::IO,
    ::MIME"text/plain",
    rs::Vector{SamplingFreqRecommendation},
)
    isempty(rs) && return print(io, "0-element Vector{SamplingFreqRecommendation} (no candidates)")
    column_labels = [
        "Sampling freq (MHz)",
        "Samples per code",
        "Num blocks",
        "Block size",
        "Inner FFT size (factors)",
        "Num Doppler bins (factors)",
        "Doppler grid",
        "Estimated cost",
    ]
    data = reduce(
        vcat,
        map(
            r -> permutedims([
                round(ustrip(Hz, r.sampling_freq) / 1e6, digits = 6),
                r.samples_per_code,
                r.num_blocks,
                r.block_size,
                string(r.inner_fft_size, " (", _factor_string(r.inner_fft_size), ")"),
                string(r.num_doppler_bins, " (", _factor_string(r.num_doppler_bins), ")"),
                _doppler_grid_string(ustrip(Hz, r.doppler_coverage), r.num_doppler_bins),
                round(r.cost, sigdigits = 4),
            ]),
            rs,
        ),
    )
    pretty_table(io, data; column_labels = column_labels)
end
