# src/plan.jl

# Per-thread mutable scratch buffers used during acquisition.
# Allocated once per pool slot in `plan_acquire` to enable multi-threaded PRN processing.
# Internal — not part of the public API.
#
# The FM-DBZP pipeline is tiled: the coherent correlation surface is produced one
# `(pass_fft_bins, block_size)` column-block tile at a time (per Doppler pass)
# (`_build_coherent_tile!`) and reduced immediately. The full
# `(num_doppler_bins, samples_per_code_eff)` power surface is materialised only
# where it is genuinely needed: one `noncoherent_accumulator` per scratch slot
# at `num_noncoherent_accumulations > 1` (accumulation across signal segments
# requires the surface; the PRN-outer loop bounds the count by the concurrency,
# not the PRN count) and the per-PRN `result_buffers` when the caller opts in
# via `store_power_bins = true`.
struct AcquisitionScratch{P2,P3,P4}
    # This slot's own FFT plans. FFTL plans own their work buffers, so a plan must
    # not be applied by two tasks at once — every kernel in the per-PRN parallel
    # loop therefore takes its plans from the scratch slot it claimed, never from
    # the shared `AcquisitionPlan`. (FFTL shares twiddles/kernels between plans
    # of one size, so the per-slot copies cost a few small work buffers.)
    double_block_bfft_plan::P2                      # in-place backward FFT (unnormalised), size fft_size
    col_fft_plan::P3                                # per-column Doppler FFT (sign-search kernels)
    doppler_fft_plan::P4                            # Doppler FFT along dim 2 of `coherent_tile_t`
    double_block_buf::Vector{ComplexF32}            # length fft_size (padded 2*block_size)
    corr_buf::Vector{ComplexF32}                    # length fft_size (padded 2*block_size)
    # One FM-DBZP column-block tile of the coherent-integration surface:
    # (pass_fft_bins, block_size). Every consumer after the build stage is
    # column-oriented, so the full (pass_fft_bins, samples_per_code) matrix
    # never needs to exist — build tile → reduce tile → next tile.
    coherent_tile::Matrix{ComplexF32}
    # The same memory as `coherent_tile`, reshaped to the transposed
    # (block_size, pass_fft_bins) layout every production kernel uses — see
    # `_build_coherent_tile_t!`. Only the full-matrix test references build
    # `coherent_tile`, so sharing costs nothing.
    coherent_tile_t::Matrix{ComplexF32}
    # Per-code-phase running max / argmax (sorted Doppler row) over one
    # transposed tile, for the streamed peak search: (block_size,
    # num_secondary_rotations), one column per rotation hypothesis so a tile's
    # rotations are folded into the peak code-phase-major (`_fold_tile_peaks`).
    col_max_buf::Matrix{Float32}
    col_max_row_buf::Matrix{Int}
    col_buf::Vector{ComplexF32}                     # length pass_fft_bins
    # Multistep (N_nc > 1) power accumulator: (num_doppler_bins,
    # samples_per_code_eff) over the whole stitched Doppler grid. The multistep PRN loop is PRN-outer — one claimed
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
    # Single-column power scratch (length pass_fft_bins, one pass's sorted
    # rows) for the on-demand column recompute used by subsample interpolation
    # on the streamed path.
    col_power_buf::Vector{Float32}
    # Sign search (`_sign_search_tile_power!`: secondary-code rotations, data-bit
    # polarities, bit-edge alignments), all 0-sized on the simple path:
    # one sign-modulated, Doppler-transformed copy of `coherent_tile_t`,
    # (block_size, pass_fft_bins) — the tile itself is reused by every hypothesis;
    modulated_tile::Matrix{ComplexF32}
    # the power of a rotation's hypotheses max-reduced, (block_size,
    # pass_fft_bins) — only when a rotation has more than one hypothesis (data-bit
    # polarities or bit-edge alignments), otherwise `modulated_tile` is reduced
    # directly;
    hypothesis_power::Matrix{Float32}
    # the per-row hypothesis signs for the tile's two code-phase ranges,
    # (pass_fft_bins, 2);
    row_signs::Matrix{Float32}
    # one Doppler-transformed tile per data-bit segment, (block_size,
    # num_doppler_bins) each, for alignments whose sign combinations are
    # combined by Gray code (`_direct_hypotheses`) — empty when none are; and the
    # running segment signs of that enumeration.
    segment_tiles::Vector{Matrix{ComplexF32}}
    segment_signs::Vector{Float32}
    # length samples_per_code_eff when per-column sums are consumed — multistep
    # extraction via `est_signal_noise_power`, or the GlobalMean noise estimator
    # on the streamed path. length 0 at N_nc == 1 with the (default) OppositeRow
    # estimator, where the streaming reduction needs no column sums at all.
    col_sums_buf::Vector{Float32}
    # Row-wise code-drift shifts pre-filled per accumulation step and Doppler
    # pass on the multistep path (drift is folded into the per-column
    # destination scatter), one per bin the pass keeps (`doppler_bins_per_pass`).
    # 0 on the sequential path (step 0 never drifts).
    code_drift_shifts::Vector{Int}
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
    # Conjugated PRN FFTs: prn_conj_ffts[prn] is (fft_size, num_blocks);
    # column k+1 = conj(FFT(PRN sub-block k zero-padded to fft_size)), precomputed
    # to avoid conjugation in the inner loop of _build_coherent_integration_matrix!
    prn_conj_ffts::Dict{Int,Matrix{ComplexF32}}
    # In-place forward FFT, size fft_size (padded 2*block_size). Only used to fill
    # `signal_block_ffts` before the per-PRN loop, on the calling task. The
    # backward and Doppler-axis plans (the latter of size `pass_fft_bins`)
    # the per-PRN kernels apply live in each scratch slot instead (types P2–P4,
    # see `AcquisitionScratch`).
    double_block_fft_plan::P1
    # The searched Doppler grid: `num_doppler_passes * doppler_bins_per_pass`
    # Hz values at multiples of the bin spacing, from `-(n÷2)·spacing` (the
    # fftshifted FFT-bin order). Every result row indexes into this grid.
    doppler_freqs::DS
    # Doppler passes (issue #101, see src/doppler_passes.jl). The range is
    # searched by `num_doppler_passes` FM-DBZP passes of `pass_fft_bins`
    # (= num_coh * num_blocks) bins each. Pass `k` runs on the signal
    # downconverted by an extra `doppler_pass_offsets_hz[k]` and contributes its
    # `doppler_bins_per_pass` central bins — sorted rows
    # `pass_row_offset+1 : pass_row_offset+doppler_bins_per_pass` of its own
    # grid — to rows `(k-1)*doppler_bins_per_pass+1 : k*doppler_bins_per_pass`
    # of `doppler_freqs`. One pass with `doppler_bins_per_pass ==
    # pass_fft_bins` is the classic single-grid search.
    num_doppler_passes::Int
    doppler_bins_per_pass::Int
    pass_fft_bins::Int
    pass_row_offset::Int
    doppler_pass_offsets_hz::Vector{Float64}
    # Delay per code period in samples, shared across PRNs within each pass.
    # All zeros when the requested code drift is below the tolerance.
    code_doppler_shifts_per_period::Vector{Float64}
    # Positive-frequency phase ramps per pass, excluding period zero. Nothing
    # for inactive passes; conjugate symmetry halves the coefficient storage.
    code_doppler_phasors::Vector{Union{Nothing,Matrix{ComplexF32}}}
    # Signal-block FFT cache: (fft_size, num_coh*num_blocks*N_nc*num_doppler_passes).
    # Holds FFT(signal double-block k) for EVERY accumulation segment and pass
    # (pass-major: see `_signal_block_ffts_for_step`),
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
    fftshift_perm::Vector{Int}               # length pass_fft_bins — pre-computed fftshift row permutation
    # Per PRN: holds a copy of power_bins when the caller of `acquire!` passes
    # store_power_bins=true. Initialised to `nothing` and allocated on the first
    # opt-in call per PRN; subsequent calls reuse the same matrix in-place.
    result_buffers::Vector{Union{Nothing,Matrix{Float32}}}
    avail_prns::Vector{Int}
    # Hard-capped per-thread scratch pool. The per-PRN work runs as
    # `min(nthreads, num_cores)` spawned chunk tasks, so `thread_scratch` holds
    # exactly that many scratches and a chunk claims a free slot for the duration
    # of each PRN (via `scratch_free` / `scratch_lock`) rather than indexing by
    # `Threads.threadid()`. That bounds the footprint at a constant
    # `min(nthreads, num_cores)` scratches no matter how `acquire!` is scheduled
    # (Issue #60); a `threadid()` index would instead need one slot per thread
    # `acquire!` ever runs on — which drifts upward over a long session as the
    # chunk tasks land on different threads, a risk for a 24/7 receiver.
    # All slots are built up front (the size is known here), so the plan's
    # construction footprint is its steady-state footprint — no first-`acquire!`
    # spike. Since the pipeline is tiled, each scratch is small — its largest
    # member is the (num_doppler_bins × block_size) tile, `num_blocks`× smaller
    # than the coherent matrix it replaced — so the per-thread footprint no
    # longer scales with `samples_per_code`. Slot 1 doubles as the ambient
    # single-threaded scratch — see `_default_scratch`.
    thread_scratch::Vector{AcquisitionScratch{P2,P3,P4}}
    scratch_free::Vector{Int}        # indices of currently-free slots in `thread_scratch`
    scratch_lock::Threads.SpinLock   # guards `scratch_free`
    # One search per plan at a time — the contract `_claim_scratch!` describes,
    # here as an enforced invariant rather than a comment. `acquire!` holds this
    # for its own duration; `acquire_stream!` holds it from the call until the
    # search task finishes, which may be long after the call returned. A second
    # search on a plan that is still in use throws `PlanInUseError` instead of
    # quietly writing the same buffers from two tasks.
    in_use::Threads.Atomic{Bool}
    # Cooperative cancellation for `acquire_stream!`: set when a consumer stops
    # early, read by the PRN loop between PRNs (`_run_prn_chunk`). Lives on the
    # plan because the plan hosts exactly one search at a time, so there is
    # nowhere else it would have to be threaded through to. Reset by
    # `_claim_plan!`, so every search starts uncancelled.
    cancelled::Threads.Atomic{Bool}
    # Cooperative-preemption stride: the per-PRN kernels `yield()` after every
    # `preempt_block_stride` FM-DBZP code blocks, bounding how long a chunk task
    # can hold a thread against a co-resident soft-real-time task (Issue #89).
    # `0` disables it — the default, and exactly today's behaviour. Derived from
    # the `max_blocking_time` keyword by `_calibrate_preempt_stride!`, which
    # needs the finished plan to time a PRN pass, hence the `Ref`: it is written
    # once at the end of `plan_acquire` and read-only from then on.
    preempt_block_stride::Base.RefValue{Int}
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
    # Pre-computed ±1 secondary-code pattern per rotation, (num_coh_periods,
    # num_secondary_rotations) per PRN — a single all-+1 column without a
    # rotation search — consumed by `_sign_search_tile_power!`, which adds the
    # data-bit signs itself. Built once in `plan_acquire`. Empty on the simple
    # path. For PRN-independent patterns (no rotation, or shared
    # secondary code) the same matrix value is stored under every PRN key — keeps
    # the lookup uniform without per-call allocation.
    sign_patterns_by_prn::Dict{Int,Matrix{Float32}}
    # CFAR null distribution of a cell that is the max over several sign
    # hypotheses (data-bit polarities × bit-edge alignments, per rotation);
    # `nothing` when each cell is one hypothesis. See `SignSearchNull`.
    sign_search_null::Union{SignSearchNull,Nothing}
end

"""
    PlanInUseError <: Exception

Thrown when a search is started on an [`AcquisitionPlan`](@ref) that is still serving
an earlier one. A plan owns the buffers the search writes into, so exactly one search
may use it at a time.

The usual cause is a stream that was abandoned without being shut down — `break`ing out
of `for result in acquire_stream!(...)` leaves the search running, because a `for` loop
does not close the channel it iterates. Use the `do`-block form of
[`acquire_stream!`](@ref), which shuts the search down however the block is left.
"""
struct PlanInUseError <: Exception end

function Base.showerror(io::IO, ::PlanInUseError)
    print(io, """
        PlanInUseError: this AcquisitionPlan is still in use by an earlier search.

        A plan owns the buffers a search writes into, so only one search may use it at
        a time. If you stopped consuming an `acquire_stream!` early — a `break` out of
        `for result in acquire_stream!(...)`, say — the search is still running: a
        `for` loop does not close the channel it iterates. The `do`-block form shuts
        the search down however the block is left:

            acquire_stream!(plan, signal, prns) do results
                for result in results
                    is_detected(result) && break
                end
            end
        """)
end

# Take exclusive ownership of the plan for one search. `acquire!` holds it for the
# duration of the call; `acquire_stream!` from the call until its search task is done.
# Also clears the cancellation flag, so a plan reused after a cancelled stream starts
# from a clean slate.
function _claim_plan!(plan::AcquisitionPlan)
    Threads.atomic_cas!(plan.in_use, false, true) === false || throw(PlanInUseError())
    plan.cancelled[] = false
    return nothing
end

_release_plan!(plan::AcquisitionPlan) = (plan.in_use[] = false; nothing)

# Number of odd prime factors (with multiplicity) of `n` when `n` is of the form
# 2^a·3^b·5^c·7^d, else `typemax(Int)`.
function _num_odd_small_factors(n::Int)
    k = 0
    for p in (3, 5, 7)
        while n % p == 0
            n ÷= p
            k += 1
        end
    end
    n == 1 << trailing_zeros(n) || return typemax(Int)
    return k
end

"""
    _inner_fft_size(double_block_size) -> Int

Length the inner double-block correlation FFT is zero-padded to: the smallest
`n >= double_block_size` of the form `2^a` times at most two factors from
`{3, 5, 7}` (e.g. 250 → 256, 1250 → 1280, 2046 → 2048, 6250 → 6272).

`samples_per_code` — hence `block_size` — is fixed by the sampling frequency and
can carry large or awkward prime factors (16.368 MHz → 2046 = 2·3·11·31; 5 MHz →
1250 = 2·5⁴). FFTL is fastest on power-of-two-heavy lengths: over 2·block_size
in 128–13000, this rule is within ~0.5% (geometric mean, worst 1.18×) of the
fastest length found by exhaustively timing every 7-smooth candidate up to
+35%, whereas padding to the next 7-smooth size is 24% slower on average and up
to 2.1× slower (radix-5 chains such as 1250 = 2·5⁴ and 6250 = 2·5⁵).
"""
function _inner_fft_size(double_block_size::Int)
    n = double_block_size
    while _num_odd_small_factors(n) > 2
        n += 1
    end
    return n
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
# One search per plan at a time is enforced by `_claim_plan!`, and the PRN loop
# spawns at most as many chunks as there are slots, so the free list is never
# empty and this is just a pop. The retry loop is therefore unreachable; it is
# kept as a belt-and-braces guard that waits for a slot rather than allocating
# beyond the pool, preserving the hard cap if a future caller ever drives the
# kernels outside `acquire!`. It yields rather than spins, so a waiting task
# hands its thread back instead of denying it to the chunk that is about to
# release the slot.
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
        yield()  # only reached under unsupported concurrent use of one plan
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
    _precompute_prn_ffts!(prn_ffts, system, prn, sampling_freq, samples_per_code, num_blocks, block_size, fft_size, fft_plan)

Populate `prn_ffts[prn]` with a `(fft_size, num_blocks)` matrix.
Column `k+1` (1-indexed) holds `FFT([code[k*block_size+1:(k+1)*block_size] ; zeros(fft_size-block_size)])`,
i.e. the FFT of PRN sub-block `k` zero-padded to `fft_size` (the fast inner
FFT length chosen in `plan_acquire`, `>= 2*block_size`).

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
    fft_size::Int,
    fft_plan,
)
    prn_code = gen_code(samples_per_code, system, prn, sampling_freq, get_code_frequency(system), 0.0)
    conj_fft_matrix = zeros(ComplexF32, fft_size, num_blocks)
    fft_buf = zeros(ComplexF32, fft_size)
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

Allocates all buffers, computes FFT plans, pre-computes conjugated PRN FFTs, and
compiles the plan-dependent part of the search (the FFT kernels and the per-PRN
kernels). Compilation is per plan *shape* — sampling frequency, coherent and
non-coherent lengths, Doppler grid, search path — so the first plan of a shape
the package image does not cover takes one to two seconds longer; later plans of that
shape do not. The first [`acquire!`](@ref) then still compiles the part that
depends on the sample type (~0.4 s); see [Avoiding first-call latency](@ref) to
move all of it into your own package's precompilation.
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
  (default: `1`, disabled). Only valid for signals with data bits and
  `num_coherently_integrated_code_periods > 1`. When the window holds at most
  one data bit (`num_coherently_integrated_code_periods ≤ bit_period_codes`),
  it searches for one bit transition at `bit_edge_search_steps − 1` evenly
  spaced code-period boundaries, plus the no-transition hypothesis, and must
  divide `min(num_coherently_integrated_code_periods, bit_period_codes)`. For
  longer windows it places the first bit edge at `bit_edge_search_steps` evenly
  spaced positions within a bit period, must divide `bit_period_codes`, and
  searches every sign of the partial bits at both ends and the whole bits in
  between.
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
- `fft_flag`: FFT planning flag, accepted for compatibility and ignored — FFTL
  plans do not measure (default: `FFTL.MEASURE`).
- `max_doppler_passes`: largest number of narrow Doppler passes the search may
  be split into (default: `nothing`, automatic selection with no fixed cap).
  When `samples_per_code` has sparse divisors,
  the smallest `num_blocks` that reaches `min_doppler_coverage` in one FM-DBZP
  grid can be far too large (1.542 MHz needs 257 for ±7 kHz). `plan_acquire`
  then covers the range in several passes with a small `num_blocks` instead,
  each downconverted to its own centre frequency, and stitches them into one
  uniform Doppler grid. It picks this only when a cost model expects it to be
  clearly cheaper, and keeps every searched Doppler within the edge loss a
  single grid has on an ideal rate — see
  [Doppler Passes](@ref) in the guide. Passes are also used when code Doppler
  would drift by more than 0.1 chip within a coherent window. Each pass
  compensates its centre's drift once per code period; narrower passes keep
  the residual across the window within 0.1 chip when the pass cap permits.
  `1` always uses the original, uncompensated single grid.
- `max_blocking_time`: upper bound on how long the per-PRN work may hold a
  thread without giving the Julia scheduler a chance to run something else
  (default: `nothing`, i.e. unbounded — today's behaviour). Pass a time
  quantity, e.g. `max_blocking_time = 100u"µs"`.

  Julia's scheduler is cooperative: a chunk task grinding through its PRNs never
  yields, so a co-resident soft-real-time task (a tracking loop feeding a
  hardware correlator) waits for a whole chunk of PRNs to finish before it can
  run. With this set, the per-PRN kernels `yield()` every few FM-DBZP code
  blocks instead, which bounds that wait without holding any thread out of the
  search. `plan_acquire` measures one PRN pass to convert the requested time
  into a block stride, so the setting means the same thing across sampling
  rates, Doppler grids and machines.

  Measured on a 1 kHz tracking loop (1 ms epoch) running alongside a continuous
  32-PRN GPS L1 C/A `acquire!` at 2.048 MHz, 8 threads, 6 × 2000 epochs — p99
  wake lateness of the tracking loop and acquisitions per second:

  | `max_blocking_time` | p99 lateness | throughput |
  |---|---|---|
  | `nothing` (default) | 0.753 ms | 845/s |
  | `270µs` (one PRN)   | 0.218 ms | 878/s |
  | `70µs`              | 0.078 ms | 895/s |
  | `35µs`              | 0.040 ms | 881/s |
  | `8µs` (one block)   | 0.009 ms | 755/s |

  The observed p99 tracks the requested bound closely, but it is an estimate
  rather than a guarantee — see `_calibrate_preempt_stride!`. Each yield costs
  on the order of 250 ns, so the throughput cost grows with how many the bound
  implies: down to roughly an eighth of a PRN pass it stays within run-to-run
  noise, and only the one-block extreme is visibly expensive (a solo `acquire!`
  goes from 1.016 ms to 1.250 ms). Results never depend on the setting — a plan
  with this set returns bit-identical results.

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
    fft_flag = FFTL.MEASURE,
    max_blocking_time = nothing,
    max_doppler_passes::Union{Int,Nothing} = nothing,
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

    # Blocks per code period and Doppler passes (src/doppler_passes.jl).
    #
    # Single grid: the smallest divisor of samples_per_code such that the grid
    # `range(-(n÷2)·spacing, step=spacing, length=n)` covers
    # `[-min_doppler_coverage, +min_doppler_coverage]` at *both* ends:
    #
    #   coverage = num_blocks * bin_width                         (bin_width = fs / samples_per_code)
    #   num_doppler_bins = num_coherently_integrated_code_periods * num_blocks
    #   spacing = coverage / num_doppler_bins = bin_width / num_coherently_integrated_code_periods
    #   highest searched value >= +coverage/2 - spacing
    #
    # Requiring `+coverage/2 - spacing >= min_doppler_coverage` gives
    #   num_blocks >= 2*(min_doppler_coverage + spacing) / bin_width
    #            >= (2*min_doppler_coverage / bin_width) + 2/num_coherently_integrated_code_periods
    # so we add `2/N_coh` (rounded up) to the naive bin count to guarantee the upper edge.
    #
    # When samples_per_code's divisors are sparse, that divisor can be far
    # larger than the coverage needs; a smaller one covering the range in several
    # narrow passes is then cheaper (`_doppler_search_layout`).
    sampling_freq_hz = ustrip(Hz, sampling_freq)
    bin_width = sampling_freq_hz / samples_per_code
    min_doppler_coverage_hz = ustrip(Hz, min_doppler_coverage)
    (max_doppler_passes === nothing || max_doppler_passes >= 1) || throw(ArgumentError("max_doppler_passes must be nothing or >= 1, got $max_doppler_passes"))
    layout = _doppler_search_layout(samples_per_code, bin_width, min_doppler_coverage_hz,
        num_coherently_integrated_code_periods, max_doppler_passes, length(prns),
        get_code_center_frequency_ratio(system))
    if isnothing(layout)
        throw(ArgumentError(
            "No valid num_blocks found: samples_per_code=$samples_per_code has no divisor " *
            ">= min_num_blocks=$(_min_num_blocks(min_doppler_coverage_hz, bin_width, num_coherently_integrated_code_periods)). " *
            "This should not happen for standard GNSS sampling frequencies — please file a bug."))
    end
    num_blocks = layout.num_blocks
    num_doppler_passes = layout.num_passes
    doppler_bins_per_pass = layout.bins_per_pass

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
    # Sub-bit windows search a single transition at `bit_edge_search_steps`
    # evenly spaced code-period boundaries (see `_edge_alignment`), so the step
    # count must divide the window.
    if has_data && num_coherently_integrated_code_periods < bit_period_codes
        num_coherently_integrated_code_periods % bit_edge_search_steps == 0 || throw(ArgumentError(
            "bit_edge_search_steps=$bit_edge_search_steps must divide " *
            "num_coherently_integrated_code_periods=$num_coherently_integrated_code_periods " *
            "when the window is shorter than a data bit (bit_period_codes=$bit_period_codes)."))
    end
    # A signal without data bits has no bit edge to search.
    if !has_data && bit_edge_search_steps > 1
        throw(ArgumentError(
            "bit_edge_search_steps must be 1 for $(typeof(system).name.name), which carries no data bits, " *
            "got $bit_edge_search_steps."))
    end

    block_size = samples_per_code ÷ num_blocks
    double_block_size = 2 * block_size
    # Inner correlation FFT length: `double_block_size` zero-padded to a size FFTL
    # transforms fast (see `_inner_fft_size`). Any length >= 2·block_size
    # preserves the first `block_size` linear-correlation lags, and the IFFT is
    # normalised by the actual length (see `_build_coherent_tile!`), so the
    # padding changes neither num_blocks, the Doppler grid, nor the result.
    fft_size = _inner_fft_size(double_block_size)
    # num_data_bits (paper: N_db): number of full data bit periods within the coherent window.
    # 1 for pilot channels and sub-bit integration (< one bit period); bit combination search
    # is only needed when num_data_bits > 1 (multiple bits → unknown polarity transitions).
    num_data_bits = (has_data && num_coherently_integrated_code_periods >= bit_period_codes) ? num_coherently_integrated_code_periods ÷ bit_period_codes : 1
    # One pass's FM-DBZP Doppler FFT length (paper C_fd), and the stitched grid.
    pass_fft_bins = num_coherently_integrated_code_periods * num_blocks
    num_doppler_bins = num_doppler_passes * doppler_bins_per_pass
    pass_row_offset = pass_fft_bins ÷ 2 - doppler_bins_per_pass ÷ 2

    # FFT plans (in-place, size fft_size and pass_fft_bins)
    double_block_proto = zeros(ComplexF32, fft_size)
    # `threads = 1`: the transforms are small and `acquire!` already runs one
    # PRN per task — FFTL would otherwise default to `Threads.nthreads()`.
    double_block_fft_plan = plan_fft!(double_block_proto; flags = fft_flag, threads = 1)
    col_proto = zeros(ComplexF32, pass_fft_bins)

    # Precompute conjugated PRN FFTs
    prn_conj_ffts = Dict{Int,Matrix{ComplexF32}}()
    for prn in prns
        _precompute_prn_ffts!(prn_conj_ffts, system, prn, sampling_freq,
            samples_per_code, num_blocks, block_size, fft_size, double_block_fft_plan)
    end

    # Doppler grid (fftshift order: bin multiples of the spacing, from
    # -(n÷2)·spacing) and the centre frequency each pass is downconverted by.
    doppler_freqs = _doppler_grid_hz(layout, bin_width, num_coherently_integrated_code_periods) .* Hz
    doppler_pass_offsets_hz = _doppler_pass_offsets_hz(layout, bin_width,
        num_coherently_integrated_code_periods)

    compensate = _code_doppler_compensation(min_doppler_coverage_hz,
        get_code_center_frequency_ratio(system), num_coherently_integrated_code_periods,
        bin_width)
    code_doppler_shifts_per_period = compensate ?
        doppler_pass_offsets_hz .* (get_code_center_frequency_ratio(system) *
            samples_per_code / ustrip(Hz, get_code_frequency(system))) :
        zeros(Float64, num_doppler_passes)
    code_doppler_phasors = Union{Nothing,Matrix{ComplexF32}}[
        _code_doppler_phasors(fft_size, num_coherently_integrated_code_periods, shift)
        for shift in code_doppler_shifts_per_period]

    segment_length = num_coherently_integrated_code_periods * samples_per_code
    # Pre-allocated signal-block FFT cache: holds FFT(signal double-block k) for
    # every accumulation segment and Doppler pass (16 bytes per signal sample
    # per pass). Filled once per acquire! call before the PRN loop and read-only
    # shared across all PRNs — see the field docstring for why all N_nc
    # segments are cached.
    signal_block_ffts = zeros(ComplexF32, fft_size,
        num_coherently_integrated_code_periods * num_blocks * num_noncoherent_accumulations *
        num_doppler_passes)
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
    fftshift_perm = [mod(r - 1 + pass_fft_bins ÷ 2, pass_fft_bins) + 1 for r in 1:pass_fft_bins]
    result_buffers = Union{Nothing,Matrix{Float32}}[nothing for _ in prns]

    # Every search other than the simple path (num_data_bits == 1,
    # bit_edge_search_steps == 1, no rotation search) is a sign search, run per
    # tile by `_sign_search_tile_power!`.
    sign_search_path_active = num_data_bits > 1 || bit_edge_search_steps > 1 ||
                              rotation_search_active
    sign_layout = SignSearchLayout(num_data_bits, num_coherently_integrated_code_periods,
        bit_edge_search_steps, num_blocks)
    modulated_tile_size = sign_search_path_active ? (block_size, pass_fft_bins) : (0, 0)
    # Sign hypotheses per rotation; the max-reduced power tile is needed only
    # when there are several.
    hypotheses_per_rotation = _num_sign_hypotheses(sign_layout)
    hypothesis_power_size = sign_search_path_active && hypotheses_per_rotation > 1 ?
        (block_size, pass_fft_bins) : (0, 0)
    # Segment tiles for the alignments the kernel combines by Gray code.
    num_segment_tiles = _num_segment_tiles(sign_layout)

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
    code_drift_shifts_len = sequential_prn_mode ? 0 : doppler_bins_per_pass

    # Per-thread scratch pool. Sized to the most chunks the PRN loop ever runs at
    # once — `min(nthreads, num_cores, num_prns)` (the loop never spawns more
    # chunks than PRNs) — rather than `maxthreadid()`: a chunk claims a free slot
    # per PRN (see `_claim_scratch!`) instead of indexing by `threadid()`, so the
    # footprint is hard-capped at this many scratches no matter how `acquire!` is
    # scheduled (Issue #60). The pipeline is tiled, so
    # at N_nc == 1 each scratch's largest member is the (num_doppler_bins ×
    # block_size) tile — nothing in the pool scales with `samples_per_code`
    # (the L1C-P/16 MHz scratch that used to weigh ~296 MiB is now ~1.5 MiB).
    # At N_nc > 1 each slot additionally carries ONE accumulation surface (see
    # `noncoherent_accumulator`), bounding the multistep footprint by the
    # concurrency instead of the PRN count. All slots are built up front (the
    # size is known here), so the plan's construction footprint is its
    # steady-state footprint — no first-`acquire!` spike.
    nthreads = _num_prn_workers(length(prns))
    thread_scratch = map(1:nthreads) do _
        tile = zeros(ComplexF32, pass_fft_bins, block_size)
        tile_t = reshape(tile, block_size, pass_fft_bins)   # shares `tile`'s memory
        AcquisitionScratch(
            plan_bfft!(double_block_proto; flags = fft_flag, threads = 1),
            plan_fft!(col_proto; flags = fft_flag, threads = 1),
            # The full plan keeps two tile-sized work arrays per slot (~2.5× the
            # tile). `FFTL.plan_pass` needs ~0.5× the tile but measured ~4%
            # slower end-to-end (up to ~19% on long coherent integrations), so
            # speed wins here.
            plan_fft!(tile_t, 2; flags = fft_flag, threads = 1),
            zeros(ComplexF32, fft_size),
            zeros(ComplexF32, fft_size),
            tile,
            tile_t,
            zeros(Float32, block_size, num_secondary_rotations),
            ones(Int, block_size, num_secondary_rotations),
            zeros(ComplexF32, pass_fft_bins),
            zeros(Float32, accumulator_rows, accumulator_cols),
            zeros(Float32, row_sums_len),
            zeros(Float32, sequential_prn_mode ? pass_fft_bins : 0),
            zeros(ComplexF32, modulated_tile_size),
            zeros(Float32, hypothesis_power_size),
            zeros(Float32, sign_search_path_active ? pass_fft_bins : 0,
                sign_search_path_active ? 2 : 0),
            [zeros(ComplexF32, block_size, pass_fft_bins) for _ in 1:num_segment_tiles],
            ones(Float32, num_segment_tiles),
            zeros(Float32, col_sums_len),
            zeros(Int, code_drift_shifts_len),
        )
    end
    scratch_free = collect(1:nthreads)
    scratch_lock = Threads.SpinLock()

    # Single downconversion staging buffer — filled once per accumulation step
    # before the per-PRN parallel loop, so one instance serves every thread.
    sig_buf = zeros(ComplexF32, segment_length)

    avail_prns_vec = collect(Int, prns)

    # Pre-allocate concrete-typed results buffer to avoid boxing allocations in acquire!
    dummy_result = AcquisitionResults(
        system, 0, convert(typeof(1.0Hz), sampling_freq), 0.0Hz, 0.0, nothing, 0.0,
        0f0, 0f0, 0, nothing, doppler_freqs, num_blocks, block_size, num_secondary_rotations,
        nothing)
    acq_results_buf = Vector{typeof(dummy_result)}(undef, length(prns))

    # Pre-compute the ±1 sign-pattern matrix once per PRN. Patterns that don't
    # depend on PRN (no rotation, or shared secondary code) still get a per-PRN
    # entry — the underlying matrix object is shared to avoid duplicate storage.
    sign_patterns_by_prn = Dict{Int,Matrix{Float32}}()
    if sign_search_path_active
        secondary_code_obj = use_secondary_code ? get_secondary_code(system) : nothing
        # When patterns are PRN-independent, build once and alias under every key.
        prn_independent = !rotation_search_active ||
                          !(secondary_code_obj isa GNSSSignals.PerPRNSecondaryCode)
        if prn_independent
            shared = sign_patterns(secondary_code_obj, first(prns), 1,
                                   num_secondary_rotations,
                                   num_coherently_integrated_code_periods,
                                   use_secondary_code)
            for prn in prns
                sign_patterns_by_prn[prn] = shared
            end
        else
            for prn in prns
                sign_patterns_by_prn[prn] = sign_patterns(secondary_code_obj, prn,
                    1, num_secondary_rotations,
                    num_coherently_integrated_code_periods, use_secondary_code)
            end
        end
    end

    # Built from rotation 0 of the first PRN: a secondary code multiplies all
    # hypotheses of a rotation alike, so their correlation — all the null model
    # depends on — is the same for every rotation and PRN.
    sign_search_null = if hypotheses_per_rotation > 1 && sign_search_path_active
        _sign_search_null(sign_patterns_by_prn[first(prns)], sign_layout,
            num_noncoherent_accumulations)
    else
        nothing
    end

    plan = AcquisitionPlan(
        system, convert(typeof(1.0Hz), sampling_freq),
        samples_per_code, samples_per_code_eff, num_blocks, block_size, num_coherently_integrated_code_periods, num_data_bits, bit_edge_search_steps, num_noncoherent_accumulations,
        prn_conj_ffts,
        double_block_fft_plan,
        doppler_freqs,
        num_doppler_passes,
        doppler_bins_per_pass,
        pass_fft_bins,
        pass_row_offset,
        doppler_pass_offsets_hz,
        code_doppler_shifts_per_period,
        code_doppler_phasors,
        signal_block_ffts,
        sig_buf,
        fftshift_perm,
        result_buffers,
        avail_prns_vec,
        thread_scratch,
        scratch_free,
        scratch_lock,
        Threads.Atomic{Bool}(false),
        Threads.Atomic{Bool}(false),
        Ref(0),
        acq_results_buf,
        noise_estimator,
        use_secondary_code,
        max_secondary_code_rotations,
        num_secondary_rotations,
        sign_patterns_by_prn,
        sign_search_null,
    )

    _precompile_search(plan)
    _calibrate_preempt_stride!(plan, max_blocking_time)

    return plan
end

# Compile the plan-dependent part of the search now, so the first `acquire!` does
# not pay it after the samples have started flowing. The search is specialised on
# the plan's type, which carries the FFT plan types, and FFTL compiles its kernels
# per transform shape, so a plan shape the package image does not cover would
# otherwise compile ~1–2 s on its first scan.
#
# Only what is fully determined by the plan is compiled: the signal-block FFT
# precompute (it reads the plan's own downconverted `ComplexF32` buffer), the
# per-PRN tile kernels and the interpolation recompute. Nothing here depends on
# the caller's sample or PRN container types, so nothing is guessed. That is
# ~75–80% of the compile time; the rest (downconversion, the PRN loop and
# result extraction, ~0.4 s per sample type) compiles on the first call — or in
# the caller's own precompile workload (see the guide, "Avoiding first-call
# latency"). `precompile` compiles without running anything; for a plan type
# compiled before, it is a cache lookup.
#
# Skipped while a package image is being generated: a precompile workload runs
# `acquire!`, which compiles all of this anyway.
#
# Returns whether every `precompile` call found its method (`nothing` when
# skipped). `precompile` does not throw on a signature that no longer matches —
# it returns `false` — so the test suite pins this to `true`.
#
# `_PRECOMPILE_SEARCH[]` exists for the test suite only, which builds dozens of
# plan shapes it acquires with once or never. It is not an API.
const _PRECOMPILE_SEARCH = Ref(true)

function _precompile_search(plan::AcquisitionPlan)
    _PRECOMPILE_SEARCH[] || return nothing
    ccall(:jl_generating_output, Cint, ()) == 1 && return nothing
    P = typeof(plan)
    S = typeof(_default_scratch(plan))
    fwd = typeof(plan.double_block_fft_plan)
    step_ffts = typeof(_signal_block_ffts_for_step(plan, 1, 1))
    ok = true
    # Both search paths, whichever this plan takes: `_acquire!` branches on
    # `num_noncoherent_accumulations` at run time, so its first call compiles
    # the streamed (N_nc == 1) and the multistep (N_nc > 1) kernels alike.
    for ffts in (typeof(plan.signal_block_ffts), step_ffts)
        for phasors in (Nothing, Matrix{ComplexF32})
            ok &= precompile(_precompute_signal_block_ffts!, (ffts,
                Vector{ComplexF32}, Int, Int, Int, Int, Vector{ComplexF32}, fwd,
                Float64, phasors))
        end
    end
    # `nothing`: streamed statistics; a matrix: `store_power_bins = true`.
    ok &= precompile(_acquire_prn_streamed!, (P, S, Int, Nothing))
    ok &= precompile(_acquire_prn_streamed!, (P, S, Int, Matrix{Float32}))
    # The trailing `Int` is the Doppler pass.
    ok &= precompile(_accumulate_prn_step_tiled!, (Matrix{Float32}, step_ffts, P, S, Int, Int, Int))
    # Subsample interpolation without a stored surface (also per Doppler pass).
    ok &= precompile(_recompute_column_power!, (Vector{Float32}, P, S, Int, Int, Int))
    return ok
end

# Turn a requested `max_blocking_time` into `plan.preempt_block_stride` — the
# number of FM-DBZP code blocks the per-PRN kernels may run between `yield()`s.
#
# The conversion has to be measured rather than derived: one block iteration is
# `num_coh*num_blocks` double-block IFFTs plus a `block_size`-column reduction,
# so its wall time depends on the sampling rate, the Doppler grid, the search
# path AND the machine. So time a real PRN pass here — the same kernel
# `acquire!` will run, on the plan's own (still zeroed) buffers — and divide by
# `num_blocks`. `min` over a few passes; a pass costs one PRN's work, which is
# nothing next to the FFT planning this function has already done.
#
# What is being estimated is the CPU work between two yields, so `min` over
# several passes is the right statistic: it is the least perturbed by whatever
# else the machine is doing while `plan_acquire` runs. It is still an estimate —
# in practice the derived stride moves by up to ~50% between runs on a busy
# machine — so treat `max_blocking_time` as a target, not a guarantee, and
# verify against a real tracking loop if the deadline is hard. Erring high
# (measuring the pass as slower than it is) shortens the stride: a tighter bound
# and slightly less throughput, never a missed deadline.
function _calibrate_preempt_stride!(plan::AcquisitionPlan, max_blocking_time)
    max_blocking_time === nothing && return plan
    max_blocking_time isa Unitful.Time || throw(ArgumentError(
        "max_blocking_time must be a time quantity (e.g. `100u\"µs\"`) or `nothing`, " *
        "got $(repr(max_blocking_time))"))
    budget_ns = ustrip(Unitful.ns, max_blocking_time)
    budget_ns > 0 || throw(ArgumentError(
        "max_blocking_time must be positive, got $max_blocking_time"))

    # The streamed pass runs every Doppler pass; the multistep timing runs one
    # step of one pass (see `_time_prn_pass_ns`).
    blocks_timed = plan.num_noncoherent_accumulations == 1 ?
        plan.num_blocks * plan.num_doppler_passes : plan.num_blocks
    block_ns = _time_prn_pass_ns(plan) / blocks_timed
    # Clamp before narrowing to Int so neither a sub-nanosecond block time nor an
    # absurdly large budget can overflow the conversion. The bounds are the ones
    # the kernels actually offer: one block is the finest granularity, so a
    # budget below that simply gets the tightest bound available; and a budget of
    # a whole PRN pass or more needs no in-PRN yield at all, because the
    # between-PRN yield already covers that boundary.
    stride = clamp(budget_ns / block_ns, 1.0, Float64(plan.num_blocks * plan.num_doppler_passes))
    plan.preempt_block_stride[] = floor(Int, stride)
    return plan
end

# Wall time of one PRN pass over the plan's zeroed buffers, in nanoseconds.
# Runs whichever per-PRN kernel `acquire!` would run for this plan, so the
# measurement tracks the configured search path.
function _time_prn_pass_ns(plan::AcquisitionPlan)
    scratch = _default_scratch(plan)
    prn = first(plan.avail_prns)
    run_one() =
        if plan.num_noncoherent_accumulations == 1
            _acquire_prn_streamed!(plan, scratch, prn, nothing)
        else
            accumulator = scratch.noncoherent_accumulator
            fill!(accumulator, 0f0)
            _accumulate_prn_step_tiled!(accumulator,
                _signal_block_ffts_for_step(plan, 1, 1), plan, scratch, prn, 0, 1)
        end
    run_one()   # warm up — the first pass pays cold caches and branch predictors
    best = typemax(UInt64)
    for _ in 1:5
        t0 = time_ns()
        run_one()
        best = min(best, time_ns() - t0)
    end
    return best
end
