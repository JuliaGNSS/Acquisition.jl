# src/generic_acquire.jl — generic circular-correlation PCPS fallback engine.
#
# The FM-DBZP engine (plan.jl / acquire.jl) is fast but requires `num_blocks` to be a
# divisor of `samples_per_code`. For sampling frequencies whose `samples_per_code`
# factors badly (prime, or large prime factors) the only valid block count is
# degenerate or produces FFTW-hostile FFT sizes, so those rates acquire poorly. This
# engine is a plain parallel code-phase search (PCPS) that works at ANY sampling
# frequency: it has no divisibility constraint at all. `plan_acquire` routes here
# automatically when the FM-DBZP configuration it would pick is unsuitable (see the
# fallback trigger in plan.jl).
#
# It produces the exact same public `AcquisitionResults` type as the FM-DBZP path and
# reuses the shared noise-power/CFAR machinery (`est_signal_noise_power`,
# `cfar_threshold`). It degrades gracefully relative to FM-DBZP: no secondary-code
# rotation search (`secondary_code_phase` is always `nothing`) and no bit-edge search.

"""
    GenericAcquisitionPlan

Pre-computed plan for the generic circular-correlation PCPS engine — the fallback used
by [`plan_acquire`](@ref) for sampling frequencies the FM-DBZP engine
([`AcquisitionPlan`](@ref)) cannot serve efficiently.

Interchangeable with [`AcquisitionPlan`](@ref) at every public call site: it carries the
same public fields (`system`, `sampling_freq`, `samples_per_code`, `doppler_freqs`,
`num_blocks`, `block_size`, `num_secondary_rotations`, …) so [`acquire!`](@ref),
[`get_num_cells`](@ref), [`is_detected`](@ref) and plotting all work unchanged. The
FM-DBZP-shaped fields take their degenerate values here: `num_blocks == 1`,
`block_size == samples_per_code`, `num_secondary_rotations == 1`.
"""
struct GenericAcquisitionPlan{S<:AbstractGNSSSignal,DS,P,BP,R,E<:AbstractNoiseEstimator} <:
       AbstractAcquisitionPlan
    system::S
    sampling_freq::typeof(1.0Hz)
    samples_per_code::Int
    samples_per_code_eff::Int              # == samples_per_code (no rotation expansion)
    num_blocks::Int                        # == 1
    block_size::Int                        # == samples_per_code
    num_coherently_integrated_code_periods::Int
    num_data_bits::Int                     # == 1 (bit-edge search not supported)
    bit_edge_search_steps::Int             # == 1
    num_noncoherent_accumulations::Int
    num_secondary_rotations::Int           # == 1
    # Doppler grid: sorted StepRangeLen of Hz values, same geometry as the FM-DBZP grid
    # but built from the ideal (unconstrained) block count so coverage/resolution match.
    doppler_freqs::DS
    coherent_window::Int                   # M = num_coherently_integrated_code_periods * samples_per_code
    # conj(FFT(length-M code replica)) per PRN — the periodic PRN tiled N_coh times.
    prn_conj_code_ffts::Dict{Int,Vector{ComplexF32}}
    fft_plan::P                            # in-place forward FFT, length M
    bfft_plan::BP                          # in-place backward FFT (unnormalised), length M
    # Working buffers (single-threaded engine — one set suffices). The downconverted
    # window is FFT'd in place, so `signal_baseband` holds the signal spectrum after the
    # forward transform and is reused across PRNs within a (segment, Doppler) cell.
    signal_baseband::Vector{ComplexF32}    # length M, downconverted signal window / its spectrum
    corr_buf::Vector{ComplexF32}           # length M, per-PRN correlation
    col_sums_buf::Vector{Float32}          # length samples_per_code, noise-estimator scratch
    # Per-avail-PRN power accumulator surface (num_doppler_bins × samples_per_code).
    signal_powers::Vector{Matrix{Float32}}
    # Per-avail-PRN copy returned when the caller passes store_power_bins=true.
    result_buffers::Vector{Union{Nothing,Matrix{Float32}}}
    avail_prns::Vector{Int}
    noise_estimator::E
    acq_results_buf::Vector{R}
end

# Constructor. Called from `plan_acquire` with the already-computed `samples_per_code`
# once the FM-DBZP configuration has been judged unsuitable. Mirrors the FM-DBZP Doppler
# grid geometry (see plan.jl) but with the *ideal* block count `min_num_blocks`, which
# need not divide `samples_per_code`.
function _plan_generic_acquire(
    system::AbstractGNSSSignal,
    sampling_freq,
    prns::AbstractVector{<:Integer},
    samples_per_code::Int;
    min_doppler_coverage = 7000Hz,
    num_coherently_integrated_code_periods::Int = 1,
    num_noncoherent_accumulations::Int = 1,
    noise_estimator::AbstractNoiseEstimator = OppositeRowNoiseEstimator(),
    fft_flag = FFTW.MEASURE,
)
    sampling_freq_hz = ustrip(Hz, sampling_freq)
    N_coh = num_coherently_integrated_code_periods

    # Doppler grid — identical geometry to the FM-DBZP grid, but using the ideal block
    # count `min_num_blocks` directly (the FM-DBZP path is forced to round this up to a
    # divisor of samples_per_code; here there is no such constraint).
    bin_width = sampling_freq_hz / samples_per_code
    min_doppler_coverage_hz = ustrip(Hz, min_doppler_coverage)
    min_num_blocks = ceil(Int, 2 * min_doppler_coverage_hz / bin_width + 2 / N_coh)
    num_doppler_bins = N_coh * min_num_blocks
    doppler_coverage_hz = min_num_blocks * bin_width
    doppler_bin_spacing_hz = doppler_coverage_hz / num_doppler_bins   # == bin_width / N_coh
    doppler_freqs =
        range(-doppler_coverage_hz / 2, step = doppler_bin_spacing_hz, length = num_doppler_bins) .* Hz

    coherent_window = N_coh * samples_per_code

    # In-place length-M FFT plans on a ComplexF32 prototype.
    proto = zeros(ComplexF32, coherent_window)
    fft_plan = plan_fft!(proto; flags = fft_flag)
    bfft_plan = plan_bfft!(proto; flags = fft_flag)

    # Conjugated FFT of each PRN's length-M code replica. `gen_code` over M samples tiles
    # the periodic PRN N_coh times, so a length-M circular correlation integrates
    # coherently over all N_coh code periods.
    code_freq = get_code_frequency(system)
    prn_conj_code_ffts = Dict{Int,Vector{ComplexF32}}()
    fft_scratch = zeros(ComplexF32, coherent_window)
    for prn in prns
        code = gen_code(coherent_window, system, prn, sampling_freq, code_freq, 0.0)
        fft_scratch .= ComplexF32.(code)
        mul!(fft_scratch, fft_plan, fft_scratch)
        prn_conj_code_ffts[prn] = conj.(fft_scratch)
    end

    avail_prns_vec = collect(Int, prns)
    signal_powers =
        [Matrix{Float32}(undef, num_doppler_bins, samples_per_code) for _ in avail_prns_vec]
    result_buffers = Union{Nothing,Matrix{Float32}}[nothing for _ in avail_prns_vec]

    dummy_result = AcquisitionResults(
        system, 0, convert(typeof(1.0Hz), sampling_freq), 0.0Hz, 0.0, nothing, 0.0,
        0f0, 0f0, 0, nothing, doppler_freqs, 1, samples_per_code, 1)
    acq_results_buf = Vector{typeof(dummy_result)}(undef, length(avail_prns_vec))

    GenericAcquisitionPlan(
        system,
        convert(typeof(1.0Hz), sampling_freq),
        samples_per_code,
        samples_per_code,          # samples_per_code_eff
        1,                         # num_blocks
        samples_per_code,          # block_size
        num_coherently_integrated_code_periods,
        1,                         # num_data_bits
        1,                         # bit_edge_search_steps
        num_noncoherent_accumulations,
        1,                         # num_secondary_rotations
        doppler_freqs,
        coherent_window,
        prn_conj_code_ffts,
        fft_plan,
        bfft_plan,
        zeros(ComplexF32, coherent_window),
        zeros(ComplexF32, coherent_window),
        zeros(Float32, samples_per_code),
        signal_powers,
        result_buffers,
        avail_prns_vec,
        noise_estimator,
        acq_results_buf,
    )
end

# Fill `signal_baseband` with the length-M window starting at `seg_start`, wiping off the
# carrier at (interm + doppler). One `sincos` per sample; ComplexF32 throughout.
function _generic_downconvert!(plan::GenericAcquisitionPlan, signal, seg_start::Int, freq_hz::Float64)
    sig = plan.signal_baseband
    fs = ustrip(Hz, plan.sampling_freq)
    phase_step = Float32(-2π * freq_hz / fs)
    @inbounds for n in 1:plan.coherent_window
        s, c = sincos(phase_step * (n - 1))
        sig[n] = ComplexF32(signal[seg_start + n - 1]) * Complex(c, s)
    end
    return sig
end

"""
    acquire!(plan::GenericAcquisitionPlan, signal, prns; interm_freq=0.0Hz, subsample_interpolation=false, store_power_bins=false) -> Vector{AcquisitionResults}

Generic PCPS acquisition using a pre-computed [`GenericAcquisitionPlan`](@ref). Same
signature and return type as the FM-DBZP [`acquire!`](@ref). See
[`GenericAcquisitionPlan`](@ref) for the graceful-degradation caveats.
"""
function acquire!(
    plan::GenericAcquisitionPlan,
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    subsample_interpolation::Bool = false,
    store_power_bins::Bool = false,
)
    all(prn -> prn in plan.avail_prns, prns) ||
        throw(ArgumentError("All requested PRNs must be in plan.avail_prns. Got: $prns, available: $(plan.avail_prns)"))

    M = plan.coherent_window
    num_segments = length(signal) ÷ M
    num_segments >= plan.num_noncoherent_accumulations ||
        throw(ArgumentError(
            "Signal has $(length(signal)) samples → $num_segments full segments of $M, " *
            "but plan.num_noncoherent_accumulations=$(plan.num_noncoherent_accumulations). " *
            "Provide a longer signal."))

    interm_freq_hz = ustrip(Hz, interm_freq)
    num_doppler_bins = length(plan.doppler_freqs)

    prn_indices = [findfirst(==(prn), plan.avail_prns)::Int for prn in prns]

    # Build the full power surface for each requested PRN: outer loop over segments and
    # Doppler bins (signal downconvert+FFT is shared across PRNs), inner loop over PRNs.
    for step_idx in 1:plan.num_noncoherent_accumulations
        seg_start = (step_idx - 1) * M + 1
        accumulate = step_idx > 1
        for (doppler_idx, doppler) in enumerate(plan.doppler_freqs)
            doppler_hz = ustrip(Hz, doppler)
            _generic_downconvert!(plan, signal, seg_start, interm_freq_hz + doppler_hz)
            # Forward FFT in place: signal_baseband now holds the signal spectrum.
            mul!(plan.signal_baseband, plan.fft_plan, plan.signal_baseband)
            for prn_idx in prn_indices
                prn = plan.avail_prns[prn_idx]
                conj_code = plan.prn_conj_code_ffts[prn]
                corr = plan.corr_buf
                @inbounds @. corr = plan.signal_baseband * conj_code
                mul!(corr, plan.bfft_plan, corr)
                power_row = view(plan.signal_powers[prn_idx], doppler_idx, :)
                spc = plan.samples_per_code
                if accumulate
                    @inbounds @simd for c in 1:spc
                        power_row[c] += abs2(corr[c])
                    end
                else
                    @inbounds @simd for c in 1:spc
                        power_row[c] = abs2(corr[c])
                    end
                end
            end
        end
    end

    code_freq_hz = ustrip(Hz, get_code_frequency(plan.system))
    code_length = get_code_length(plan.system)
    code_period = code_length / get_code_frequency(plan.system)
    sampling_freq_hz = ustrip(Hz, plan.sampling_freq)
    doppler_step = step(plan.doppler_freqs)

    results = resize!(plan.acq_results_buf, length(prns))
    for (i, prn_idx) in enumerate(prn_indices)
        prn = plan.avail_prns[prn_idx]
        power_bins = plan.signal_powers[prn_idx]
        results[i] = _extract_generic_result!(
            plan, prn, prn_idx, power_bins, code_freq_hz, code_length, code_period,
            sampling_freq_hz, num_doppler_bins, doppler_step,
            subsample_interpolation, store_power_bins)
    end
    return results
end

# Assemble an AcquisitionResults from a fully-accumulated generic power surface.
# power_bins layout: (num_doppler_bins, samples_per_code); rows = sorted Doppler bins,
# column `c` = code-phase lag with delay `c-1` samples. Code phase uses the same
# convention as the FM-DBZP path and the plot recipe:
# `mod(-(c-1) * code_freq / fs, code_length)`.
function _extract_generic_result!(
    plan::GenericAcquisitionPlan, prn::Int, prn_idx::Int, power_bins::Matrix{Float32},
    code_freq_hz, code_length, code_period, sampling_freq_hz,
    num_doppler_bins, doppler_step, subsample_interpolation, store_power_bins,
)
    signal_power, noise_power, code_bin_idx, doppler_bin_idx = est_signal_noise_power(
        power_bins, sampling_freq_hz, code_freq_hz, plan.col_sums_buf, plan.noise_estimator)

    peak_to_noise = (signal_power + noise_power) / noise_power
    CN0 = 10 * log10(signal_power / noise_power / code_period / 1.0Hz)

    delay_samples = code_bin_idx - 1
    code_phase = mod(-delay_samples * code_freq_hz / sampling_freq_hz, code_length)

    if subsample_interpolation
        num_code_bins = size(power_bins, 2)
        col_left = mod(code_bin_idx - 2, num_code_bins) + 1
        col_right = mod(code_bin_idx, num_code_bins) + 1
        power_left = power_bins[doppler_bin_idx, col_left]
        power_peak = power_bins[doppler_bin_idx, code_bin_idx]
        power_right = power_bins[doppler_bin_idx, col_right]
        if max(power_left, power_right) > sqrt(noise_power)
            frac = _parabolic_interp(power_left, power_peak, power_right)
            code_phase = mod(-(delay_samples + frac) * code_freq_hz / sampling_freq_hz, code_length)
        end
    end

    doppler = plan.doppler_freqs[doppler_bin_idx]
    if subsample_interpolation
        dop_left = power_bins[doppler_bin_idx == 1 ? num_doppler_bins : doppler_bin_idx - 1, code_bin_idx]
        dop_peak = power_bins[doppler_bin_idx, code_bin_idx]
        dop_right = power_bins[doppler_bin_idx == num_doppler_bins ? 1 : doppler_bin_idx + 1, code_bin_idx]
        if max(dop_left, dop_right) > sqrt(noise_power)
            frac = _parabolic_interp(dop_left, dop_peak, dop_right)
            doppler = doppler + frac * doppler_step
        end
    end

    result_buf = if store_power_bins
        cached = plan.result_buffers[prn_idx]
        buf = cached === nothing ? similar(power_bins) : cached
        plan.result_buffers[prn_idx] = buf
        copyto!(buf, power_bins)
    else
        nothing
    end

    return AcquisitionResults(
        plan.system,
        prn,
        plan.sampling_freq,
        doppler,
        code_phase,
        nothing,                 # secondary_code_phase — not searched on the generic path
        CN0,
        Float32(noise_power),
        Float32(peak_to_noise),
        plan.num_noncoherent_accumulations,
        result_buf,
        plan.doppler_freqs,
        plan.num_blocks,         # 1
        plan.block_size,         # samples_per_code
        plan.num_secondary_rotations,  # 1
    )
end
