# src/sdr_clock_plans.jl
# Hardware sample-rate constraints for SDR front ends.
#
# Most SDRs derive their sample rate from a master clock through integer
# dividers and multipliers, so only a discrete (but dense) set of rates is
# actually settable. `recommend_sampling_freqs` consults a clock plan via
# [`is_valid_sample_rate`](@ref) to skip unreachable candidates.

"""
    AbstractSDRClockPlan

Hardware constraints that decide which sampling frequencies an SDR can produce.

Concrete subtypes implement [`is_valid_sample_rate`](@ref). They may also
override [`sample_rate_range`](@ref) to narrow the sweep range.

See [`AD9361ClockPlan`](@ref) for an example implementation.
"""
abstract type AbstractSDRClockPlan end

"""
    is_valid_sample_rate(plan::AbstractSDRClockPlan, fs_hz::Real) -> Bool

Return `true` if `fs_hz` is reachable by the SDR described by `plan`.
"""
function is_valid_sample_rate end

"""
    sample_rate_range(plan::AbstractSDRClockPlan) -> Tuple{Float64,Float64}

Return `(fs_min_hz, fs_max_hz)`, the inclusive sample-rate envelope of the
SDR. Used by [`recommend_sampling_freqs`](@ref) to clamp the sweep range.
The default fallback returns `(0.0, Inf)` so concrete plans only need to
override it when there's a meaningful envelope.
"""
sample_rate_range(::AbstractSDRClockPlan) = (0.0, Inf)

"""
    AD9361ClockPlan(; ref_clk = 38.4e6, adc_clk_min = 25e6, adc_clk_max = 640e6,
                      fs_min = 0.55e6, fs_max = 61.44e6,
                      bbpll_min = 715e6, bbpll_max = 1430e6,
                      bbpll_div_min = 2, bbpll_div_max = 64,
                      dividers = [12, 8, 6, 4, 3, 2, 1])

Sample-rate validation for the Analog Devices **AD9361** RFIC, as used by the
LiteX-M2SDR and many other SDR boards.

A sample rate `fs` is reachable when *some* divider `d ∈ dividers` makes
`adc_clk = fs × d` land in `[adc_clk_min, adc_clk_max]`, and that `adc_clk`
can be reached from the BBPLL: `adc_clk × bbpll_div ∈ [bbpll_min, bbpll_max]`
for some power-of-two `bbpll_div ∈ [bbpll_div_min, bbpll_div_max]`.

The defaults reproduce the limits in the AD9361 driver shipped with
`litex_m2sdr` (see `software/user/ad9361/ad9361.h`).
"""
struct AD9361ClockPlan <: AbstractSDRClockPlan
    ref_clk::Float64
    adc_clk_min::Float64
    adc_clk_max::Float64
    fs_min::Float64
    fs_max::Float64
    bbpll_min::Float64
    bbpll_max::Float64
    bbpll_div_min::Int
    bbpll_div_max::Int
    dividers::Vector{Int}
end

function AD9361ClockPlan(;
    ref_clk = 38.4e6,
    adc_clk_min = 25e6,
    adc_clk_max = 640e6,
    fs_min = 0.55e6,
    fs_max = 61.44e6,
    bbpll_min = 715e6,
    bbpll_max = 1430e6,
    bbpll_div_min = 2,
    bbpll_div_max = 64,
    dividers = [12, 8, 6, 4, 3, 2, 1],
)
    AD9361ClockPlan(
        Float64(ref_clk), Float64(adc_clk_min), Float64(adc_clk_max),
        Float64(fs_min), Float64(fs_max),
        Float64(bbpll_min), Float64(bbpll_max),
        Int(bbpll_div_min), Int(bbpll_div_max),
        Vector{Int}(dividers),
    )
end

sample_rate_range(plan::AD9361ClockPlan) = (plan.fs_min, plan.fs_max)

function _bbpll_reachable(plan::AD9361ClockPlan, adc_clk::Real)
    div = plan.bbpll_div_min
    while div <= plan.bbpll_div_max
        rate = adc_clk * div
        plan.bbpll_min <= rate <= plan.bbpll_max && return true
        div *= 2
    end
    return false
end

function is_valid_sample_rate(plan::AD9361ClockPlan, fs_hz::Real)
    (plan.fs_min <= fs_hz <= plan.fs_max) || return false
    for d in plan.dividers
        adc_clk = fs_hz * d
        if plan.adc_clk_min <= adc_clk <= plan.adc_clk_max && _bbpll_reachable(plan, adc_clk)
            return true
        end
    end
    return false
end

function Base.show(io::IO, ::MIME"text/plain", plan::AD9361ClockPlan)
    print(io, "AD9361ClockPlan(",
        "fs ∈ [", plan.fs_min/1e6, ", ", plan.fs_max/1e6, "] MHz, ",
        "ADC ∈ [", plan.adc_clk_min/1e6, ", ", plan.adc_clk_max/1e6, "] MHz, ",
        "dividers=", plan.dividers,
        ")")
end
