# src/plan.jl

# Crossover above which individual column FFTs beat a batched 2-D FFT along dim 1.
# Batched wins for small num_doppler_bins (amortises FFTW dispatch across columns);
# per-column wins for large num_doppler_bins (better cache behaviour).
# Reproduce/re-tune via `SUITE["BatchFFTCrossover"]` in benchmark/benchmarks.jl.
const BATCH_FFT_THRESHOLD = 320

# Per-thread mutable scratch buffers used during acquisition.
# Allocated once per thread in `plan_acquire` to enable multi-threaded PRN processing.
# Internal — not part of the public API.
struct AcquisitionScratch
    double_block_buf::Vector{ComplexF32}            # length double_block_size
    corr_buf::Vector{ComplexF32}                    # length double_block_size
    sig_buf::Vector{ComplexF32}                     # length segment_length — downconverted signal segment (thread 1 only)
    col_buf::Vector{ComplexF32}                     # length num_doppler_bins
    col_fftshift_buf::Vector{ComplexF32}            # length num_doppler_bins
    row_buf::Vector{Float32}                        # length samples_per_code
    row_shift_buf::Vector{Float32}                  # length samples_per_code
    coherent_integration_matrix::Matrix{ComplexF32}         # (num_doppler_bins, samples_per_code)
    noncoherent_integration_max_buf::Matrix{Float32}         # (num_doppler_bins, samples_per_code) — 0x0 when simple path is taken (see plan_acquire)
    noncoherent_integration_buf::Matrix{Float32}             # (num_doppler_bins, samples_per_code)
    # At num_noncoherent_accumulations == 1 this matrix replaces the per-PRN
    # `noncoherent_integration_matrices` vector: each PRN runs one build+accumulate
    # into this per-thread buffer, then extracts its result before the next PRN
    # overwrites it. 0x0 when N_nc > 1 (the per-PRN layout is required there).
    noncoherent_integration_accumulator::Matrix{Float32}
    sub_block_ffts::Matrix{ComplexF32}              # (num_doppler_bins, num_data_bits) — 0x0 when simple path is taken (see plan_acquire)
    col_sums_buf::Vector{Float32}                   # length samples_per_code — scratch for est_signal_noise_power
end

"""
    AcquisitionPlan

Pre-computed acquisition plan for FM-DBZP (Heckler & Garrison 2009).

See [`plan_acquire`](@ref) and [`acquire`](@ref).
"""
struct AcquisitionPlan{S<:AbstractGNSSSignal,DS,P1,P2,P3,P4,R,E<:AbstractNoiseEstimator}
    system::S
    sampling_freq::typeof(1.0Hz)
    samples_per_code::Int       # paper N_τ  — samples per code period
    num_blocks::Int             # paper N_step — blocks per code period (power of 2)
    block_size::Int             # paper B_size — samples per block = samples_per_code ÷ num_blocks
    num_coherently_integrated_code_periods::Int       # paper N     — total code periods per coherent integration
    num_data_bits::Int          # paper N_db  — data bit periods per coherent block (1 for pilot/sub-bit)
    bit_edge_search_steps::Int          # paper N_be  — bit edge search positions (1 = disabled)
    num_noncoherent_accumulations::Int  # incoherent integration steps
    # Conjugated PRN FFTs: prn_conj_ffts[prn] is (double_block_size, num_blocks);
    # column k+1 = conj(FFT(zero-padded PRN sub-block k)), precomputed to avoid
    # conjugation in the inner loop of _build_coherent_integration_matrix!
    prn_conj_ffts::Dict{Int,Matrix{ComplexF32}}
    double_block_fft_plan::P1   # in-place forward FFT, size double_block_size
    double_block_bfft_plan::P2  # in-place backward FFT (unnormalised), size double_block_size
    col_fft_plan::P3            # in-place forward FFT, size num_doppler_bins = num_coherently_integrated_code_periods*num_blocks
    col_batch_fft_plan::P4      # in-place forward FFT along dim 1 of (num_doppler_bins, samples_per_code) matrix; `nothing` when num_doppler_bins > BATCH_FFT_THRESHOLD
    # doppler_freqs: StepRangeLen of num_doppler_bins Hz values, sorted from -doppler_coverage_hz/2 to doppler_coverage_hz/2-doppler_bin_spacing_hz
    doppler_freqs::DS
    signal_block_ffts::Matrix{ComplexF32} # (double_block_size, num_coh*num_blocks) — precomputed once per acquire!, shared across PRNs
    noncoherent_integration_matrices::Vector{Matrix{Float32}}  # per PRN, (num_doppler_bins, samples_per_code); index i corresponds to avail_prns[i]
    fftshift_perm::Vector{Int}               # length num_doppler_bins — pre-computed fftshift row permutation
    # Per PRN: holds a copy of power_bins when the caller of `acquire!` passes
    # store_power_bins=true. Initialised to `nothing` and allocated on the first
    # opt-in call per PRN; subsequent calls reuse the same matrix in-place.
    result_buffers::Vector{Union{Nothing,Matrix{Float32}}}
    avail_prns::Vector{Int}
    # Per-thread scratch, indexed by Threads.threadid() (length = nthreads at plan_acquire time).
    # Thread 1's entry doubles as the ambient single-threaded scratch — see `_default_scratch`.
    thread_scratch::Vector{AcquisitionScratch}
    # Pre-allocated results buffer, concrete-typed to avoid boxing allocations
    acq_results_buf::Vector{R}
    # Singleton selecting the noise-power estimator used by `est_signal_noise_power`.
    noise_estimator::E
end

# Thread 1's scratch is the ambient single-threaded scratch reused by `acquire!`
# (downconversion + signal-block FFT precompute happen before the per-PRN parallel
# loop) and by test/REPL call sites that drive the kernels directly. Naming this
# convention here keeps it from being re-implemented as `plan.thread_scratch[1]`
# across the codebase.
_default_scratch(plan::AcquisitionPlan) = plan.thread_scratch[1]

"""
    _precompute_prn_ffts!(prn_ffts, system, prn, sampling_freq, samples_per_code, num_blocks, block_size, double_block_size, fft_plan)

Populate `prn_ffts[prn]` with a `(double_block_size, num_blocks)` matrix.
Column `k+1` (1-indexed) holds `FFT([code[k*block_size+1:(k+1)*block_size] ; zeros(block_size)])`,
i.e. the FFT of PRN sub-block `k` zero-padded to `double_block_size`.

## Algorithm

The coherent integration matrix cell `[row_k, col_block_r]` correlates signal double-block `k` with PRN
sub-block `(k + r) mod num_blocks` (Prasad 2009, Steps 3 & 5). Each row `k` of the coherent integration matrix
uses a different PRN sub-block depending on which column block `r` is being filled, with the
sub-block index wrapping circularly. Together, all num_blocks sub-blocks tile the full samples_per_code-chip
PRN, so every chip contributes to the correlation — giving full samples_per_code-chip correlation gain rather
than the block_size gain that would result from using a single sub-block for all rows.
"""
function _precompute_prn_ffts!(
    prn_conj_ffts::Dict{Int,Matrix{ComplexF32}},
    system,
    prn::Int,
    sampling_freq,
    samples_per_code::Int,
    num_blocks::Int,
    block_size::Int,
    double_block_size::Int,
    fft_plan,
)
    prn_code = gen_code(samples_per_code, system, prn, sampling_freq, get_code_frequency(system), 0.0)
    conj_fft_matrix = zeros(ComplexF32, double_block_size, num_blocks)
    fft_buf = zeros(ComplexF32, double_block_size)
    for block_idx in 0:num_blocks-1
        fft_buf .= 0
        fft_buf[1:block_size] .= ComplexF32.(prn_code[block_idx*block_size+1:(block_idx+1)*block_size])
        mul!(fft_buf, fft_plan, fft_buf)
        @. conj_fft_matrix[:, block_idx+1] = conj(fft_buf)   # store conjugate — avoids conj in inner loop
    end
    prn_conj_ffts[prn] = conj_fft_matrix
end

"""
    plan_acquire(system, sampling_freq, prns; kwargs...) -> AcquisitionPlan

Pre-compute an acquisition plan for FM-DBZP acquisition.

Allocates all buffers, computes FFT plans, and pre-computes conjugated PRN FFTs.
The returned plan can be passed to [`acquire!`](@ref) repeatedly without re-allocating.
Per-thread scratch buffers are sized for `Threads.maxthreadid()` at the time of the call;
start Julia with `-t N` before calling `plan_acquire` to enable multi-threaded acquisition.

# Arguments

- `system`: GNSS system (e.g. `GPSL1CA()`)
- `sampling_freq`: Sampling frequency
- `prns`: PRN numbers to pre-compute (e.g. `1:32`)

# Keyword Arguments

- `min_doppler_coverage`: Minimum guaranteed one-sided Doppler reach
  (default: `7000Hz`). Both ends of the searched grid will be at least
  `±min_doppler_coverage`. Controls `num_blocks` — see
  [Algorithm Constraints and Trade-offs](@ref) for details.
- `num_coherently_integrated_code_periods`: Number of code periods per coherent
  integration block (default: `1`). Determines Doppler resolution:
  `bin_spacing = 1 / (num_coherently_integrated_code_periods × T_code)`.
- `bit_edge_search_steps`: Number of bit-edge alignment candidates to search
  (default: `1`, disabled). Must divide `bit_period_codes`. Only valid when
  `num_coherently_integrated_code_periods > 1`.
- `num_noncoherent_accumulations`: Number of successive incoherent integration
  steps (default: `1`). Signal passed to `acquire!` must contain at least
  `num_noncoherent_accumulations` full segments.
- `noise_estimator`: how `acquire!` estimates noise power for the CFAR
  test statistic. Defaults to [`OppositeRowNoiseEstimator`](@ref) (averages
  one Doppler row, robust to Doppler-conditional banding from DC offset/IF
  spurs/narrowband interferers — recommended for real-world IF signals).
  Pass [`GlobalMeanNoiseEstimator`](@ref) for synthetic-AWGN tests where
  the noise floor is independent and identically distributed across the
  grid and the lower variance of the global mean is preferred. See the
  [Detecting Satellites](@ref) section of the guide for the rationale.
- `fft_flag`: FFTW planning flag (default: `FFTW.MEASURE`).

# See also

[`acquire!`](@ref), [`acquire`](@ref)
"""
function plan_acquire(
    system::AbstractGNSSSignal,
    sampling_freq,
    prns::AbstractVector{<:Integer};
    min_doppler_coverage = 7000Hz,
    num_coherently_integrated_code_periods::Int = 1,
    bit_edge_search_steps::Int = 1,
    num_noncoherent_accumulations::Int = 1,
    noise_estimator::AbstractNoiseEstimator = OppositeRowNoiseEstimator(),
    fft_flag = FFTW.MEASURE,
)
    num_noncoherent_accumulations >= 1 || throw(ArgumentError("num_noncoherent_accumulations must be >= 1, got $num_noncoherent_accumulations"))
    num_coherently_integrated_code_periods >= 1 || throw(ArgumentError("num_coherently_integrated_code_periods must be >= 1"))
    bit_edge_search_steps >= 1 || throw(ArgumentError("bit_edge_search_steps must be >= 1, got $bit_edge_search_steps"))
    if num_coherently_integrated_code_periods == 1 && bit_edge_search_steps > 1
        throw(ArgumentError("bit_edge_search_steps must be 1 when num_coherently_integrated_code_periods == 1 (sub-bit mode; no bit boundary search applies), got $bit_edge_search_steps"))
    end

    # Samples per code period
    samples_per_code = ceil(Int, get_code_length(system) / get_code_frequency(system) * sampling_freq)

    # Determine data bit period in code periods (bit_period_codes)
    data_freq = get_data_frequency(system)
    data_freq_hz = ustrip(Hz, data_freq)
    has_data = isfinite(data_freq_hz) && data_freq_hz > 0
    bit_period_codes = if has_data
        bit_samples = ceil(Int, sampling_freq / data_freq)
        bit_samples ÷ samples_per_code
    else
        num_coherently_integrated_code_periods  # no constraint for pilot channels
    end

    # Blocks per code period: smallest divisor of samples_per_code such that the
    # final Doppler grid `range(-cov/2, step=spacing, length=num_doppler_bins)`
    # covers `[-min_doppler_coverage, +min_doppler_coverage]` at *both* ends.
    #
    # Geometry:
    #   coverage = num_blocks * bin_width                         (bin_width = fs / samples_per_code)
    #   num_doppler_bins = num_coherently_integrated_code_periods * num_blocks
    #   spacing = coverage / num_doppler_bins = bin_width / num_coherently_integrated_code_periods
    #   highest searched value = +coverage/2 - spacing            (last bin of the half-open range)
    #
    # Requiring `+coverage/2 - spacing >= min_doppler_coverage` gives
    #   num_blocks >= 2*(min_doppler_coverage + spacing) / bin_width
    #            >= (2*min_doppler_coverage / bin_width) + 2/num_coherently_integrated_code_periods
    # so we add `2/N_coh` (rounded up) to the naive bin count to guarantee the upper edge.
    sampling_freq_hz = ustrip(Hz, sampling_freq)
    bin_width = sampling_freq_hz / samples_per_code
    min_doppler_coverage_hz = ustrip(Hz, min_doppler_coverage)
    min_num_blocks = ceil(Int, 2 * min_doppler_coverage_hz / bin_width
                              + 2 / num_coherently_integrated_code_periods)
    num_blocks = let found = nothing
        for candidate in min_num_blocks:samples_per_code
            if samples_per_code % candidate == 0
                found = candidate
                break
            end
        end
        found
    end
    if isnothing(num_blocks)
        throw(ArgumentError(
            "No valid num_blocks found: samples_per_code=$samples_per_code has no divisor " *
            ">= min_num_blocks=$min_num_blocks. " *
            "This should not happen for standard GNSS sampling frequencies — please file a bug."))
    end

    # Validate data-channel coherent integration length.
    # Sub-bit integration (num_coherently_integrated_code_periods < bit_period_codes) is always valid — a bit
    # boundary may still fall within the window (bit edge search handles this), but there
    # can be at most one transition so num_data_bits=1 and no bit combination search is needed.
    # Multi-bit integration (num_coherently_integrated_code_periods >= bit_period_codes) requires the window to span
    # a whole number of bit periods so the FM-DBZP sub-block structure is consistent.
    if has_data && num_coherently_integrated_code_periods >= bit_period_codes
        num_coherently_integrated_code_periods % bit_period_codes == 0 || throw(ArgumentError(
            "num_coherently_integrated_code_periods=$num_coherently_integrated_code_periods must be divisible by " *
            "bit_period_codes=$bit_period_codes (code periods per data bit)."))
        bit_period_codes % bit_edge_search_steps == 0 || throw(ArgumentError(
            "bit_edge_search_steps=$bit_edge_search_steps must divide bit_period_codes=$bit_period_codes."))
    end

    block_size = samples_per_code ÷ num_blocks
    double_block_size = 2 * block_size
    # num_data_bits (paper: N_db): number of full data bit periods within the coherent window.
    # 1 for pilot channels and sub-bit integration (< one bit period); bit combination search
    # is only needed when num_data_bits > 1 (multiple bits → unknown polarity transitions).
    num_data_bits = (has_data && num_coherently_integrated_code_periods >= bit_period_codes) ? num_coherently_integrated_code_periods ÷ bit_period_codes : 1
    num_doppler_bins = num_coherently_integrated_code_periods * num_blocks  # paper C_fd

    # FFT plans (in-place, size double_block_size and num_doppler_bins)
    double_block_proto = zeros(ComplexF32, double_block_size)
    double_block_fft_plan = plan_fft!(double_block_proto; flags = fft_flag)
    double_block_bfft_plan = plan_bfft!(double_block_proto; flags = fft_flag)
    col_proto = zeros(ComplexF32, num_doppler_bins)
    col_fft_plan = plan_fft!(col_proto; flags = fft_flag)
    # Batched column FFT plan: only allocate when num_doppler_bins is small enough
    # that the batched approach outperforms individual column FFTs (see BATCH_FFT_THRESHOLD).
    # Above the threshold, store `nothing` — the non-batched path is used instead.
    col_batch_fft_plan = if num_doppler_bins <= BATCH_FFT_THRESHOLD
        col_batch_proto = zeros(ComplexF32, num_doppler_bins, samples_per_code)
        plan_fft!(col_batch_proto, 1; flags = fft_flag)
    else
        nothing
    end

    # Precompute conjugated PRN FFTs
    prn_conj_ffts = Dict{Int,Matrix{ComplexF32}}()
    for prn in prns
        _precompute_prn_ffts!(prn_conj_ffts, system, prn, sampling_freq,
            samples_per_code, num_blocks, block_size, double_block_size, double_block_fft_plan)
    end

    # Doppler grid (fftshift order: sorted from -doppler_coverage_hz/2)
    doppler_coverage_hz = num_blocks * sampling_freq_hz / samples_per_code
    doppler_bin_spacing_hz = doppler_coverage_hz / num_doppler_bins
    doppler_freqs = range(-doppler_coverage_hz / 2, step = doppler_bin_spacing_hz, length = num_doppler_bins) .* Hz

    segment_length = num_coherently_integrated_code_periods * samples_per_code
    # Pre-allocated signal-block FFT cache: holds FFT(signal double-block k) for
    # k ∈ 0..num_coh*num_blocks-1. Filled once per acquire! call before the PRN
    # loop and reused across all PRNs.
    signal_block_ffts = zeros(ComplexF32, double_block_size, num_coherently_integrated_code_periods * num_blocks)
    # At N_nc == 1 each PRN's noncoherent matrix is written once and consumed
    # once for result extraction; we collapse the per-PRN vector into a
    # per-thread accumulator (see `noncoherent_integration_accumulator` in
    # AcquisitionScratch). The multistep path (N_nc > 1) still needs the
    # per-PRN vector because it accumulates across steps.
    sequential_prn_mode = num_noncoherent_accumulations == 1
    noncoherent_integration_matrices = sequential_prn_mode ?
        Matrix{Float32}[] :
        [zeros(Float32, num_doppler_bins, samples_per_code) for _ in prns]
    accumulator_rows = sequential_prn_mode ? num_doppler_bins : 0
    accumulator_cols = sequential_prn_mode ? samples_per_code : 0
    fftshift_perm = [mod(r - 1 + num_doppler_bins ÷ 2, num_doppler_bins) + 1 for r in 1:num_doppler_bins]
    result_buffers = Union{Nothing,Matrix{Float32}}[nothing for _ in prns]

    # The sign-search path in `_accumulate_noncoherent_integration_step!` runs only
    # when num_data_bits > 1 OR bit_edge_search_steps > 1. Otherwise the simple
    # (pilot) path is taken, which never reads `noncoherent_integration_max_buf` or
    # `sub_block_ffts`. Decide here at plan time so each thread can skip those
    # buffers when they will provably never be read.
    sign_search_path_active = num_data_bits > 1 || bit_edge_search_steps > 1
    sign_search_max_rows = sign_search_path_active ? num_doppler_bins  : 0
    sign_search_max_cols = sign_search_path_active ? samples_per_code  : 0
    sub_block_cols       = sign_search_path_active ? num_data_bits     : 0

    # Per-thread scratch: one entry per thread, indexed by Threads.threadid().
    # Use maxthreadid() to cover Julia's internal task-switching threads (always >= nthreads()).
    # `sig_buf` lives here so that thread 1's scratch — the ambient single-threaded
    # scratch — owns the downconverted-signal buffer used by `acquire!` before the
    # per-PRN parallel loop. Other threads' sig_buf entries are unused (a few MB
    # × nthreads, dwarfed by the matrices already sized per-thread).
    nthreads = Threads.maxthreadid()
    thread_scratch = [
        AcquisitionScratch(
            zeros(ComplexF32, double_block_size),
            zeros(ComplexF32, double_block_size),
            zeros(ComplexF32, segment_length),
            zeros(ComplexF32, num_doppler_bins),
            zeros(ComplexF32, num_doppler_bins),
            zeros(Float32, samples_per_code),
            zeros(Float32, samples_per_code),
            zeros(ComplexF32, num_doppler_bins, samples_per_code),
            zeros(Float32, sign_search_max_rows, sign_search_max_cols),
            zeros(Float32, num_doppler_bins, samples_per_code),
            zeros(Float32, accumulator_rows, accumulator_cols),
            zeros(ComplexF32, sign_search_max_rows, sub_block_cols),
            zeros(Float32, samples_per_code),
        )
        for _ in 1:nthreads
    ]

    avail_prns_vec = collect(Int, prns)

    # Pre-allocate concrete-typed results buffer to avoid boxing allocations in acquire!
    dummy_result = AcquisitionResults(
        system, 0, convert(typeof(1.0Hz), sampling_freq), 0.0Hz, 0.0, 0.0,
        0f0, 0f0, 0, nothing, doppler_freqs, num_blocks, block_size)
    acq_results_buf = Vector{typeof(dummy_result)}(undef, length(prns))

    return AcquisitionPlan(
        system, convert(typeof(1.0Hz), sampling_freq),
        samples_per_code, num_blocks, block_size, num_coherently_integrated_code_periods, num_data_bits, bit_edge_search_steps, num_noncoherent_accumulations,
        prn_conj_ffts,
        double_block_fft_plan, double_block_bfft_plan, col_fft_plan, col_batch_fft_plan,
        doppler_freqs,
        signal_block_ffts,
        noncoherent_integration_matrices, fftshift_perm,
        result_buffers,
        avail_prns_vec,
        thread_scratch,
        acq_results_buf,
        noise_estimator,
    )
end
