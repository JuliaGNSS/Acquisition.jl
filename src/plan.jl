# src/plan.jl

# Crossover above which individual column FFTs beat a batched 2-D FFT along dim 1.
# Batched wins for small num_doppler_bins (amortises FFTW dispatch across columns);
# per-column wins for large num_doppler_bins (better cache behaviour).
# Reproduce/re-tune via `SUITE["BatchFFTCrossover"]` in benchmark/benchmarks.jl.
const BATCH_FFT_THRESHOLD = 320

# How far the FM-DBZP block count may be inflated above the ideal `min_num_blocks`
# before the configuration is judged degenerate and `plan_acquire` falls back to the
# generic engine. FM-DBZP must pick `num_blocks` as a divisor of `samples_per_code`;
# when `samples_per_code` has no divisor near `min_num_blocks` (prime or near-prime, e.g.
# 2039 → only divisor ≥ min is 2039 itself, `block_size == 1`) the smallest valid divisor
# is forced far above the requirement, collapsing the block structure and blowing up the
# Doppler-bin count. A factor of 2 keeps every well-supported GNSS rate on the fast path
# (16.368 MHz, 5/10/25 MHz all land within ~1.8× of `min_num_blocks`) while routing the
# genuinely degenerate rates to the generic circular-correlation engine.
const FMDBZP_MAX_BLOCK_INFLATION = 2

# Common supertype of the FM-DBZP `AcquisitionPlan` and the generic
# `GenericAcquisitionPlan` (see generic_acquire.jl). Lets the single-PRN `acquire!`
# convenience overloads and one-shot `acquire` dispatch over either engine.
abstract type AbstractAcquisitionPlan end

# Per-thread mutable scratch buffers used during acquisition.
# Allocated once per pool slot in `plan_acquire` to enable multi-threaded PRN processing.
# Internal — not part of the public API.
#
# The FM-DBZP pipeline is tiled: the coherent correlation surface is produced one
# `(num_doppler_bins, block_size)` column-block tile at a time
# (`_build_coherent_tile!`) and reduced immediately. The full
# `(num_doppler_bins, samples_per_code_eff)` power surface is materialised only
# where it is genuinely needed: one `noncoherent_accumulator` per scratch slot
# at `num_noncoherent_accumulations > 1` (accumulation across signal segments
# requires the surface; the PRN-outer loop bounds the count by the concurrency,
# not the PRN count) and the per-PRN `result_buffers` when the caller opts in
# via `store_power_bins = true`.
struct AcquisitionScratch
    double_block_buf::Vector{ComplexF32}            # length double_block_size
    corr_buf::Vector{ComplexF32}                    # length double_block_size
    # One FM-DBZP column-block tile of the coherent-integration surface:
    # (num_doppler_bins, block_size). Every consumer after the build stage is
    # column-oriented, so the full (num_doppler_bins, samples_per_code) matrix
    # never needs to exist — build tile → reduce tile → next tile.
    coherent_tile::Matrix{ComplexF32}
    col_buf::Vector{ComplexF32}                     # length num_doppler_bins
    # Per-column staging for the sign-search kernels:
    # (num_doppler_bins, num_secondary_rotations). Holds ONE code-phase column's
    # power per rotation hypothesis, max-reduced across bit-edge alignments and
    # data-bit polarities, before it is consumed (streamed at N_nc == 1,
    # scattered into the accumulator at N_nc > 1). Replaces the full-surface
    # `noncoherent_integration_buf`/`sign_search_max_buf` pair.
    # 0×0 when the simple/pilot path is taken.
    stage_buf::Matrix{Float32}
    # Multistep (N_nc > 1) power accumulator: (num_doppler_bins,
    # samples_per_code_eff). The multistep PRN loop is PRN-outer — one claimed
    # scratch slot carries one PRN through ALL accumulation steps and extracts
    # its result before moving on — so `min(nthreads, num_cores, num_prns)`
    # accumulators replace the former one-per-PRN matrices (a 32-PRN search on
    # 8 threads holds 8 surfaces instead of 32). 0×0 at N_nc == 1, where the
    # streamed reduction needs no surface at all.
    noncoherent_accumulator::Matrix{Float32}
    # Streaming-reduction state at num_noncoherent_accumulations == 1: running
    # per-row (sorted-Doppler order) power sums feeding the OppositeRow noise
    # estimate. length num_doppler_bins; 0 on the multistep path, which reduces
    # from the materialised per-PRN accumulator instead.
    row_sums_buf::Vector{Float32}
    # Single-column power scratch (length num_doppler_bins) for the on-demand
    # column recompute used by subsample interpolation on the streamed path.
    col_power_buf::Vector{Float32}
    sub_block_ffts::Matrix{ComplexF32}              # (num_doppler_bins, num_sub_blocks) — 0x0 when simple path is taken (see plan_acquire). Scratch for the sign-search kernels.
    # Real/imag-split accumulator for the rotation-kernel's complex-MAC combine loop.
    # Each is length num_doppler_bins, sized 0 when the rotation kernel is not active.
    # Split storage drives Float32 SIMD on the tiled phasor inner loop —
    # ComplexF32 storage is ~3× slower because the compiler can't pack the MAC.
    combine_buf_re::Vector{Float32}
    combine_buf_im::Vector{Float32}
    # length samples_per_code_eff when per-column sums are consumed — multistep
    # extraction via `est_signal_noise_power`, or the GlobalMean noise estimator
    # on the streamed path. length 0 at N_nc == 1 with the (default) OppositeRow
    # estimator, where the streaming reduction needs no column sums at all.
    col_sums_buf::Vector{Float32}
    # Row-wise code-drift shifts pre-filled per accumulation step on the
    # multistep path (drift is folded into the per-column destination scatter).
    # length num_doppler_bins when N_nc > 1; 0 on the sequential path (step 0
    # never drifts).
    code_drift_shifts::Vector{Int}
end

"""
    AcquisitionPlan

Pre-computed acquisition plan for FM-DBZP (Heckler & Garrison 2009).

See [`plan_acquire`](@ref) and [`acquire`](@ref).
"""
struct AcquisitionPlan{S<:AbstractGNSSSignal,DS,P1,P2,P3,P4,R,E<:AbstractNoiseEstimator} <:
       AbstractAcquisitionPlan
    system::S
    sampling_freq::typeof(1.0Hz)
    samples_per_code::Int       # paper N_τ  — samples per code period
    # Effective columns in the noncoherent buffer / result matrix. Equal to
    # `samples_per_code * num_secondary_rotations` when the rotation search is
    # active so each rotation hypothesis occupies its own (cp_within, doppler)
    # cell instead of being collapsed by a cell-wise max. Equals `samples_per_code`
    # otherwise. Result extraction decodes peak col as
    # `(cp_within, rotation_idx) = ((col-1) % samples_per_code + 1, (col-1) ÷ samples_per_code)`.
    samples_per_code_eff::Int
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
    col_batch_fft_plan::P4      # in-place forward FFT along dim 1 of a (num_doppler_bins, block_size) tile; `nothing` when num_doppler_bins > BATCH_FFT_THRESHOLD
    # doppler_freqs: StepRangeLen of num_doppler_bins Hz values, sorted from -doppler_coverage_hz/2 to doppler_coverage_hz/2-doppler_bin_spacing_hz
    doppler_freqs::DS
    # Signal-block FFT cache: (double_block_size, num_coh*num_blocks*N_nc).
    # Holds FFT(signal double-block k) for EVERY accumulation segment,
    # precomputed serially once per acquire! call (the same total FFT work the
    # per-segment cache used to do) and then read-only shared across the PRN
    # loop. Caching all segments — 16 bytes per signal sample, i.e. 2× the
    # ComplexF32 signal itself — is what lets the multistep PRN loop run
    # PRN-outer with `min(nthreads, num_cores, num_prns)` accumulators instead
    # of one surface per PRN, with zero recompute.
    signal_block_ffts::Matrix{ComplexF32}
    # Downconverted-segment staging (length segment_length). Filled once per
    # accumulation step BEFORE the per-PRN parallel loop, so a single instance
    # suffices — keeping it out of the per-thread scratch pool saves
    # (nthreads-1) × segment_length × 8 bytes.
    sig_buf::Vector{ComplexF32}
    fftshift_perm::Vector{Int}               # length num_doppler_bins — pre-computed fftshift row permutation
    # Per PRN: holds a copy of power_bins when the caller of `acquire!` passes
    # store_power_bins=true. Initialised to `nothing` and allocated on the first
    # opt-in call per PRN; subsequent calls reuse the same matrix in-place.
    result_buffers::Vector{Union{Nothing,Matrix{Float32}}}
    avail_prns::Vector{Int}
    # Hard-capped per-thread scratch pool. The per-PRN work runs under
    # `@batch per=core`, which executes at most `min(nthreads, num_cores)` chunks
    # concurrently, so `thread_scratch` holds exactly that many scratches and a
    # chunk claims a free slot for the duration of each PRN (via `scratch_free` /
    # `scratch_lock`) rather than indexing by `Threads.threadid()`. That bounds
    # the footprint at a constant `min(nthreads, num_cores)` scratches no matter
    # how `acquire!` is scheduled (Issue #60); a `threadid()` index would instead
    # need one slot per thread `acquire!` ever runs on — which drifts upward over
    # a long session when driven from `Threads.@spawn`, a risk for a 24/7 receiver.
    # All slots are built up front (the size is known here), so the plan's
    # construction footprint is its steady-state footprint — no first-`acquire!`
    # spike. Since the pipeline is tiled, each scratch is small — its largest
    # member is the (num_doppler_bins × block_size) tile, `num_blocks`× smaller
    # than the coherent matrix it replaced — so the per-thread footprint no
    # longer scales with `samples_per_code`. Slot 1 doubles as the ambient
    # single-threaded scratch — see `_default_scratch`.
    thread_scratch::Vector{AcquisitionScratch}
    scratch_free::Vector{Int}        # indices of currently-free slots in `thread_scratch`
    scratch_lock::Threads.SpinLock   # guards `scratch_free`
    # Pre-allocated results buffer, concrete-typed to avoid boxing allocations
    acq_results_buf::Vector{R}
    # Singleton selecting the noise-power estimator used by `est_signal_noise_power`.
    noise_estimator::E
    # Secondary-code rotation search configuration.
    use_secondary_code::Bool
    max_secondary_code_rotations::Int
    # Derived: number of secondary-code rotation phases to enumerate. Equals
    # get_secondary_code_length(system) when the search is active, else 1.
    num_secondary_rotations::Int
    # Pre-computed ±1 sign-pattern matrix per PRN, consumed by `_sign_search_step!`
    # / `_sign_search_step_with_rotations!`. Built once in `plan_acquire`. Empty
    # when the sign-search path is inactive (i.e. simple/pilot path with no
    # rotation search). For PRN-independent patterns (no rotation, or shared
    # secondary code) the same matrix value is stored under every PRN key — keeps
    # the lookup uniform without per-call allocation.
    sign_patterns_by_prn::Dict{Int,Matrix{Float32}}
    # Pre-computed phase-ramp-multiplied patterns per PRN, pre-tiled along the
    # Doppler-bin axis AND split into separate real/imag Float32 arrays. The
    # tiled form makes the inner ω loop contiguous; the real/imag split lets
    # the compiler pack the complex MAC into Float32 SIMD lanes (~3× faster
    # than the ComplexF32 form, even with the same per-iteration FLOP count).
    # Both arrays have shape `(num_doppler_bins, num_coh_periods, num_patterns)`,
    # indexed `[ω, p, q]`. Empty when the rotation kernel is not active. See
    # [`combined_phase_patterns`](@ref) and [`tile_phase_patterns`](@ref).
    tiled_phase_patterns_re_by_prn::Dict{Int,Array{Float32,3}}
    tiled_phase_patterns_im_by_prn::Dict{Int,Array{Float32,3}}
end

# Slot 1 of the pool is the ambient single-threaded scratch reused by `acquire!`
# (downconversion + signal-block FFT precompute happen before the per-PRN parallel
# loop) and by test/REPL call sites that drive the kernels directly. Naming this
# convention here keeps it from being re-implemented as `plan.thread_scratch[1]`
# across the codebase.
_default_scratch(plan::AcquisitionPlan) = plan.thread_scratch[1]

# Claim an exclusive scratch slot from the pool. Returns `(scratch, slot)`; the
# caller MUST return the slot via `_release_scratch!` (use try/finally).
#
# Under the supported contract — one `acquire!` per plan at a time — the
# `@batch per=core` loop runs at most as many chunks as there are slots, so the
# free list is never empty and this is just a pop. The spin is a non-allocating
# safety net for accidental concurrent use of one plan: it waits for a slot to
# free rather than allocating beyond the pool, preserving the hard cap.
@inline function _claim_scratch!(plan::AcquisitionPlan)
    idx = 0
    while true
        lock(plan.scratch_lock)
        if !isempty(plan.scratch_free)
            idx = pop!(plan.scratch_free)
            unlock(plan.scratch_lock)
            break
        end
        unlock(plan.scratch_lock)
        GC.safepoint()  # only reached under unsupported concurrent use of one plan
    end
    return (@inbounds plan.thread_scratch[idx]), idx
end

@inline function _release_scratch!(plan::AcquisitionPlan, idx::Int)
    lock(plan.scratch_lock)
    push!(plan.scratch_free, idx)
    unlock(plan.scratch_lock)
    return nothing
end

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
The per-thread scratch pool is sized to `min(Threads.nthreads(), num_cores)` at the
time of the call; start Julia with `-t N` before calling `plan_acquire` to enable
multi-threaded acquisition. Scratch buffers are tile-sized (nothing scales with
`samples_per_code`), so extra threads cost little memory.

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
- `use_secondary_code`: enable the secondary-code rotation search (default: `true`).
  When the signal has a secondary code of length `L > 1` and
  `num_coherently_integrated_code_periods > 1`, the search jointly estimates the
  unknown secondary-code start phase, recovering the full coherent gain.
  **`num_coherently_integrated_code_periods` must then be a whole multiple of `L`**
  (e.g. `10` for GPS L5I's NH10). A partial secondary period introduces a ±Doppler
  sign ambiguity (a near-equal mirror peak at `-f`; issue #68) and is rejected with
  an `ArgumentError`. Set `false` to opt out of the search (no constraint, coarser
  effective gain when a secondary code is present).
- `max_secondary_code_rotations`: cap on the secondary-code rotation-search size
  (default: `32`). `plan_acquire` errors if `L` exceeds this, before allocating
  buffers, so large-`L` signals (e.g. GPS L1C-P, `L = 1800`) fail fast.
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
    use_secondary_code::Bool = true,
    max_secondary_code_rotations::Int = 32,
    fft_flag = FFTW.MEASURE,
)
    num_noncoherent_accumulations >= 1 || throw(ArgumentError("num_noncoherent_accumulations must be >= 1, got $num_noncoherent_accumulations"))
    num_coherently_integrated_code_periods >= 1 || throw(ArgumentError("num_coherently_integrated_code_periods must be >= 1"))
    bit_edge_search_steps >= 1 || throw(ArgumentError("bit_edge_search_steps must be >= 1, got $bit_edge_search_steps"))
    if num_coherently_integrated_code_periods == 1 && bit_edge_search_steps > 1
        throw(ArgumentError("bit_edge_search_steps must be 1 when num_coherently_integrated_code_periods == 1 (sub-bit mode; no bit boundary search applies), got $bit_edge_search_steps"))
    end

    # Secondary-code rotation search cap. Fires before any heavy allocation so a
    # user who hits the cap on a large-L signal (e.g. GPS L1C-P, L=1800) is not
    # paying the cost of the plan's buffers just to learn that the search is too big.
    let L = get_secondary_code_length(system)
        if use_secondary_code && L > 1 && num_coherently_integrated_code_periods > 1 && L > max_secondary_code_rotations
            throw(ArgumentError(
                "Secondary-code rotation search size exceeds the per-plan cap for " *
                "$(typeof(system).name.name): " *
                "get_secondary_code_length(system)=$L > max_secondary_code_rotations=$max_secondary_code_rotations " *
                "at num_coherently_integrated_code_periods=$num_coherently_integrated_code_periods. " *
                "Remedies (in order): " *
                "(1) reduce num_coherently_integrated_code_periods; " *
                "(2) raise max_secondary_code_rotations to at least $L; or " *
                "(3) set use_secondary_code = false to opt out of the rotation search."))
        end
        # Rotation-length divisibility: when the rotation search is active, the coherent
        # window must span a WHOLE number of secondary-code periods (N a multiple of L).
        # Two distinct failures motivate this — both resolved by the same constraint:
        #
        #  - N > L, non-multiple: the cyclic-rotation semantics are ill-defined; the
        #    trailing partial period doesn't map to a clean secondary-chip sequence.
        #  - N not a multiple of L (notably the sub-bit regime N < L, e.g. 5 ms on
        #    GPS L5I whose NH10 has L = 10): a ±Doppler SIGN AMBIGUITY (issue #68). Over
        #    a partial secondary period the cross-correlation between a WRONG rotation
        #    hypothesis and the true sequence can fit a frequency-mirrored carrier almost
        #    as well as the true rotation fits the true carrier. The rotation search then
        #    surfaces a near-equal mirror peak at -f, and the acquired Doppler keeps a
        #    stable magnitude but flips sign at random under noise — tracking gets seeded
        #    ~2|f| off and never locks. The mirror cancels only when the window covers a
        #    full secondary-code period (N % L == 0), where the periodic autocorrelation
        #    of the secondary code is bounded.
        #
        # For data signals this mostly coincides with the bit_period_codes check below
        # (bit_period_codes is typically a multiple of L), but it ALSO catches the sub-bit
        # regime N < bit_period_codes, which that check deliberately allows.
        if use_secondary_code && L > 1 &&
           num_coherently_integrated_code_periods > 1 &&
           num_coherently_integrated_code_periods % L != 0
            throw(ArgumentError(
                "Secondary-code rotation search requires num_coherently_integrated_code_periods " *
                "to be a whole multiple of the secondary-code length L=$L, got " *
                "num_coherently_integrated_code_periods=$num_coherently_integrated_code_periods. " *
                "A partial secondary-code period makes the rotation search produce a ±Doppler " *
                "sign ambiguity — a near-equal mirror peak at -f whose sign flips at random under " *
                "noise (issue #68). Remedies (in order): " *
                "(1) set num_coherently_integrated_code_periods to a multiple of $L " *
                "(e.g. $L for one full secondary-code period); " *
                "(2) set num_coherently_integrated_code_periods = 1 and raise " *
                "num_noncoherent_accumulations for a robust sign at coarser Doppler resolution; or " *
                "(3) set use_secondary_code = false to opt out of the rotation search."))
        end
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
            "This usually means min_doppler_coverage exceeds what the sampling frequency " *
            "can represent (min_num_blocks > samples_per_code). Reduce min_doppler_coverage."))
    end

    # Fallback trigger: route to the generic circular-correlation PCPS engine
    # (`GenericAcquisitionPlan`, generic_acquire.jl) when the FM-DBZP block count this
    # planner would pick is degenerate for the requested sampling frequency — i.e.
    # `samples_per_code` has no divisor within `FMDBZP_MAX_BLOCK_INFLATION`× of the ideal
    # `min_num_blocks`, forcing a collapsed block structure (prime / near-prime rates).
    # The decision is read off the divisor FM-DBZP would actually pick, so every
    # well-supported rate stays on the fast path byte-for-byte, while acquisition now
    # works at any sampling frequency.
    if num_blocks > FMDBZP_MAX_BLOCK_INFLATION * min_num_blocks
        return _plan_generic_acquire(
            system, sampling_freq, prns, samples_per_code;
            min_doppler_coverage,
            num_coherently_integrated_code_periods,
            num_noncoherent_accumulations,
            noise_estimator,
            fft_flag,
        )
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
    # Sized to one (num_doppler_bins, block_size) tile — the unit the pipeline
    # processes — not the full samples_per_code width.
    col_batch_fft_plan = if num_doppler_bins <= BATCH_FFT_THRESHOLD
        col_batch_proto = zeros(ComplexF32, num_doppler_bins, block_size)
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
    # every accumulation segment (16 bytes per signal sample). Filled once per
    # acquire! call before the PRN loop and read-only shared across all PRNs —
    # see the field docstring for why all N_nc segments are cached.
    signal_block_ffts = zeros(ComplexF32, double_block_size,
        num_coherently_integrated_code_periods * num_blocks * num_noncoherent_accumulations)
    # At N_nc == 1 each cell of the power surface is produced exactly once, so
    # the result statistics (peak, noise row/column sums) are reduced on the fly
    # per tile and NO noncoherent power matrix exists at all — see
    # `_acquire_prn_streamed!`. The multistep path (N_nc > 1) accumulates across
    # signal segments into one per-claimed-scratch surface (PRN-outer loop).
    # Derive the secondary-code rotation search size. No-op for signals without a
    # secondary code (L = 1) or when the user opts out.
    secondary_code_length = get_secondary_code_length(system)
    rotation_search_active = use_secondary_code &&
                             secondary_code_length > 1 &&
                             num_coherently_integrated_code_periods > 1
    num_secondary_rotations = rotation_search_active ? secondary_code_length : 1

    # When the rotation search is active, expand the cp axis of the noncoherent
    # buffers by `num_secondary_rotations` so each rotation hypothesis writes to
    # its own column slice instead of being collapsed by a cell-wise max. The
    # expanded buffer has the same total cell count as the LongL5I baseline
    # (num_doppler_bins × samples_per_code × num_rotations), recovering LongL5I-
    # equivalent CFAR statistics. Result extraction decodes peak col back into
    # `(cp_within, rotation_idx)`.
    samples_per_code_eff = rotation_search_active ?
        samples_per_code * num_secondary_rotations : samples_per_code

    sequential_prn_mode = num_noncoherent_accumulations == 1
    fftshift_perm = [mod(r - 1 + num_doppler_bins ÷ 2, num_doppler_bins) + 1 for r in 1:num_doppler_bins]
    result_buffers = Union{Nothing,Matrix{Float32}}[nothing for _ in prns]

    # The sign-search path runs when num_data_bits > 1, bit_edge_search_steps > 1,
    # or the secondary-code rotation search is active. Otherwise the simple/pilot
    # path is taken, which never reads `stage_buf` or `sub_block_ffts`.
    sign_search_path_active = num_data_bits > 1 || bit_edge_search_steps > 1 ||
                              rotation_search_active
    # Per-column staging for the sign-search kernels: ONE column of power per
    # rotation hypothesis. Bit-edge alignments and data-bit polarities are
    # max-reduced into it column-locally, so neither search needs a buffer
    # proportional to `samples_per_code` anymore.
    stage_rows = sign_search_path_active ? num_doppler_bins        : 0
    stage_cols = sign_search_path_active ? num_secondary_rotations : 0
    # The data-bit kernel needs one sub-block FFT column per data bit; the
    # rotation kernel needs one per coherent period. Size to the max so the same
    # scratch matrix serves both code paths.
    sub_block_rows = sign_search_path_active ? num_doppler_bins : 0
    sub_block_cols = if sign_search_path_active
        rotation_search_active ? max(num_data_bits, num_coherently_integrated_code_periods) :
                                 num_data_bits
    else
        0
    end

    # Streaming-reduction state exists only on the sequential (N_nc == 1) path;
    # the multistep path reduces from the per-scratch accumulator surface.
    row_sums_len = sequential_prn_mode ? num_doppler_bins : 0
    # Multistep accumulator: one power surface per scratch slot — the PRN-outer
    # multistep loop carries one PRN through all accumulation steps per slot, so
    # `min(nthreads, num_cores, num_prns)` surfaces bound the footprint instead
    # of one per PRN. 0×0 at N_nc == 1 (streamed, no surface).
    accumulator_rows = sequential_prn_mode ? 0 : num_doppler_bins
    accumulator_cols = sequential_prn_mode ? 0 : samples_per_code_eff
    # Column sums are consumed by multistep extraction (`est_signal_noise_power`
    # always fills them) and by the GlobalMean estimator on the streamed path.
    # At N_nc == 1 with the default OppositeRow estimator nothing reads them,
    # so the (samples_per_code_eff-long) buffer is dropped entirely.
    col_sums_len = (!sequential_prn_mode || noise_estimator isa GlobalMeanNoiseEstimator) ?
        samples_per_code_eff : 0

    # The multistep path folds code-drift correction into the per-column
    # destination scatter; shifts are pre-filled once per accumulation step.
    # The sequential path never drifts (step 0 has zero drift by definition).
    code_drift_shifts_len = sequential_prn_mode ? 0 : num_doppler_bins

    # Per-thread scratch pool. Sized to the most chunks `@batch per=core` ever
    # runs at once — `min(nthreads, num_cores, num_prns)` (the PRN loop never
    # has more concurrent chunks than PRNs) — rather than `maxthreadid()`: a
    # chunk claims a free slot per PRN (see `_claim_scratch!`) instead of indexing
    # by `threadid()`, so the footprint is hard-capped at this many scratches no
    # matter how `acquire!` is scheduled (Issue #60). The pipeline is tiled, so
    # at N_nc == 1 each scratch's largest member is the (num_doppler_bins ×
    # block_size) tile — nothing in the pool scales with `samples_per_code`
    # (the L1C-P/16 MHz scratch that used to weigh ~296 MiB is now ~1.5 MiB).
    # At N_nc > 1 each slot additionally carries ONE accumulation surface (see
    # `noncoherent_accumulator`), bounding the multistep footprint by the
    # concurrency instead of the PRN count. All slots are built up front (the
    # size is known here), so the plan's construction footprint is its
    # steady-state footprint — no first-`acquire!` spike.
    nthreads = min(Threads.nthreads(), max(1, Int(num_cores())), max(1, length(prns)))
    thread_scratch = [
        AcquisitionScratch(
            zeros(ComplexF32, double_block_size),
            zeros(ComplexF32, double_block_size),
            zeros(ComplexF32, num_doppler_bins, block_size),
            zeros(ComplexF32, num_doppler_bins),
            zeros(Float32, stage_rows, stage_cols),
            zeros(Float32, accumulator_rows, accumulator_cols),
            zeros(Float32, row_sums_len),
            zeros(Float32, sequential_prn_mode ? num_doppler_bins : 0),
            zeros(ComplexF32, sub_block_rows, sub_block_cols),
            zeros(Float32, rotation_search_active ? num_doppler_bins : 0),
            zeros(Float32, rotation_search_active ? num_doppler_bins : 0),
            zeros(Float32, col_sums_len),
            zeros(Int, code_drift_shifts_len),
        )
        for _ in 1:nthreads
    ]
    scratch_free = collect(1:nthreads)
    scratch_lock = Threads.SpinLock()

    # Single downconversion staging buffer — filled once per accumulation step
    # before the per-PRN parallel loop, so one instance serves every thread.
    sig_buf = zeros(ComplexF32, segment_length)

    avail_prns_vec = collect(Int, prns)

    # Pre-allocate concrete-typed results buffer to avoid boxing allocations in acquire!
    dummy_result = AcquisitionResults(
        system, 0, convert(typeof(1.0Hz), sampling_freq), 0.0Hz, 0.0, nothing, 0.0,
        0f0, 0f0, 0, nothing, doppler_freqs, num_blocks, block_size, num_secondary_rotations)
    acq_results_buf = Vector{typeof(dummy_result)}(undef, length(prns))

    # Pre-compute the ±1 sign-pattern matrix once per PRN. The dispatcher in
    # `_accumulate_noncoherent_integration_step!` reads from this dict instead
    # of recomputing per accumulation step. Patterns that don't depend on PRN
    # (no rotation, or shared secondary code) still get a per-PRN entry — the
    # underlying matrix object can be shared to avoid duplicate storage.
    sign_patterns_by_prn = Dict{Int,Matrix{Float32}}()
    # The rotation kernel consumes the phase-ramp-multiplied patterns, pre-
    # tiled along the Doppler-bin axis so the inner combine loop is contiguous
    # SIMD. The non-rotation sign-search kernel (`_sign_search_step!`) keeps
    # using the plain ±1 `sign_patterns_by_prn`. Both dicts are populated for
    # sign_search_path PRNs so the dispatcher can look up the right form
    # without a branch.
    tiled_phase_patterns_re_by_prn = Dict{Int,Array{Float32,3}}()
    tiled_phase_patterns_im_by_prn = Dict{Int,Array{Float32,3}}()
    empty_tile = Array{Float32,3}(undef, 0, 0, 0)
    function _build_tiled(p)
        rotation_search_active || return (empty_tile, empty_tile)
        compact = combined_phase_patterns(p, num_coherently_integrated_code_periods)
        tiled = tile_phase_patterns(compact, num_doppler_bins)
        # Split into Float32 real/imag arrays so the kernel inner loop runs as
        # Float32 SIMD (≈3× faster than ComplexF32 storage with the same MACs).
        Float32.(real.(tiled)), Float32.(imag.(tiled))
    end
    if sign_search_path_active
        secondary_code_obj = use_secondary_code ? get_secondary_code(system) : nothing
        # When patterns are PRN-independent, build once and alias under every key.
        prn_independent = !rotation_search_active ||
                          !(secondary_code_obj isa GNSSSignals.PerPRNSecondaryCode)
        if prn_independent
            shared = sign_patterns(secondary_code_obj, first(prns), num_data_bits,
                                   num_secondary_rotations,
                                   num_coherently_integrated_code_periods,
                                   use_secondary_code)
            shared_re, shared_im = _build_tiled(shared)
            for prn in prns
                sign_patterns_by_prn[prn] = shared
                tiled_phase_patterns_re_by_prn[prn] = shared_re
                tiled_phase_patterns_im_by_prn[prn] = shared_im
            end
        else
            for prn in prns
                p = sign_patterns(secondary_code_obj, prn,
                    num_data_bits, num_secondary_rotations,
                    num_coherently_integrated_code_periods, use_secondary_code)
                sign_patterns_by_prn[prn] = p
                re, im = _build_tiled(p)
                tiled_phase_patterns_re_by_prn[prn] = re
                tiled_phase_patterns_im_by_prn[prn] = im
            end
        end
    end

    return AcquisitionPlan(
        system, convert(typeof(1.0Hz), sampling_freq),
        samples_per_code, samples_per_code_eff, num_blocks, block_size, num_coherently_integrated_code_periods, num_data_bits, bit_edge_search_steps, num_noncoherent_accumulations,
        prn_conj_ffts,
        double_block_fft_plan, double_block_bfft_plan, col_fft_plan, col_batch_fft_plan,
        doppler_freqs,
        signal_block_ffts,
        sig_buf,
        fftshift_perm,
        result_buffers,
        avail_prns_vec,
        thread_scratch,
        scratch_free,
        scratch_lock,
        acq_results_buf,
        noise_estimator,
        use_secondary_code,
        max_secondary_code_rotations,
        num_secondary_rotations,
        sign_patterns_by_prn,
        tiled_phase_patterns_re_by_prn,
        tiled_phase_patterns_im_by_prn,
    )
end
