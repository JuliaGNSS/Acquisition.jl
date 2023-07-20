module Acquisition

using DocStringExtensions,
    GNSSSignals, RecipesBase, FFTW, Statistics, LinearAlgebra, LoopVectorization, Unitful

import Unitful: s, Hz

using PrecompileTools

using ThreadsX

using PrettyTables

export acquire,
    plot_acquisition_results,
    coarse_fine_acquire,
    coarse_fine_acquire!,
    acquire!,
    AcquisitionPlan,
    CoarseFineAcquisitionPlan,
    noncoherent_integrate,
    noncoherent_integrate_manual_timeshift,
    noncoherent_integrate_manual_timeshift_dopplers

struct AcquisitionResults{S<:AbstractGNSS,T}
    system::S
    prn::Int
    sampling_frequency::typeof(1.0Hz)
    carrier_doppler::typeof(1.0Hz)
    code_phase::Float64
    CN0::Float64
    noise_power::T
    power_bins::Matrix{T}
    complex_signal::Matrix{Complex{T}}
    dopplers::StepRangeLen{
        Float64,
        Base.TwicePrecision{Float64},
        Base.TwicePrecision{Float64},
    }
end

include("plan_acquire.jl")
include("downconvert.jl")
include("plot.jl")
include("calc_powers.jl")
include("est_signal_noise_power.jl")
include("acquire.jl")

function _precompile_()
    Base.precompile(Tuple{typeof(Core.kwcall),Core.NamedTuple{(:max_doppler, :coarse_step, :fine_step, :prns, :fft_flag), Tuple{Unitful.Quantity{Core.Float64, 𝐓^-1, Unitful.FreeUnits{(Hz,), 𝐓^-1, nothing}}, Unitful.Quantity{Core.Float64, 𝐓^-1, Unitful.FreeUnits{(Hz,), 𝐓^-1, nothing}}, Unitful.Quantity{Core.Float64, 𝐓^-1, Unitful.FreeUnits{(Hz,), 𝐓^-1, nothing}}, Base.UnitRange{Core.Int64}, Core.UInt32}},Type{CoarseFineAcquisitionPlan},GPSL1{Matrix{Int16}},Int64,Unitful.Quantity{Core.Float64, 𝐓^-1, Unitful.FreeUnits{(Hz,), 𝐓^-1, nothing}}})   # time: 0.34097806
    isdefined(Acquisition, Symbol("#27#28")) && Base.precompile(Tuple{getfield(Acquisition, Symbol("#27#28")),Vector{ComplexF32},Matrix{Float32},Matrix{ComplexF32}})   # time: 0.16505001
    isdefined(Acquisition, Symbol("#35#37")) && Base.precompile(Tuple{getfield(Acquisition, Symbol("#35#37")),Matrix{Float32},Matrix{ComplexF32},Int64})   # time: 0.10682159
    isdefined(Acquisition, Symbol("#35#37")) && Base.precompile(Tuple{getfield(Acquisition, Symbol("#35#37")),Any,Any,Any})   # time: 0.040406585
    Base.precompile(Tuple{typeof(est_signal_noise_power),Matrix{Float32},Unitful.Quantity{Core.Float64, 𝐓^-1, Unitful.FreeUnits{(Hz,), 𝐓^-1, nothing}},Unitful.Quantity{Core.Int64, 𝐓^-1, Unitful.FreeUnits{(Hz,), 𝐓^-1, nothing}},Nothing})   # time: 0.016112743
    Base.precompile(Tuple{typeof(est_signal_noise_power),Matrix{Float32},Unitful.Quantity{Core.Float64, 𝐓^-1, Unitful.FreeUnits{(Hz,), 𝐓^-1, nothing}},Unitful.Quantity{Core.Int64, 𝐓^-1, Unitful.FreeUnits{(Hz,), 𝐓^-1, nothing}},Float32})   # time: 0.002283945
end

 
end
