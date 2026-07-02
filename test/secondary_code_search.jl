# Test-only shim that re-presents GPSL5I as if its primary code were the full
# 102 300-chip primary×NH10 product (10 primary periods × NH10 length 10). With
# no secondary code on the wrapped system, `plan_acquire(..., N_coh=1)` runs the
# simple pilot path over a single 10 ms coherent integration — i.e. exactly the
# 10 ms column FFT the rotation kernel SHOULD be equivalent to. Used as the
# reference oracle in the algebraic-equivalence @test_broken below.
#
# Mirrors /workspace/Acquisition/claude_scratch/long_code.jl. GPSL5I's gen_code!
# naturally bakes NH10 across primary periods at the sample level, so forwarding
# `gen_code!` to the wrapped GPSL5I instance is sufficient.
const _Frequency = Union{Unitful.Quantity{T, Unitful.𝐓^-1, U},
                         Unitful.Level{L, S, Unitful.Quantity{T, Unitful.𝐓^-1, U}} where {L, S}} where {T, U}
struct _LongL5I <: GNSSSignals.AbstractGNSSSignal{Matrix{Int16}}
    inner::GPSL5I{Matrix{Int16}}
end
_LongL5I() = _LongL5I(GPSL5I())
GNSSSignals.get_code_length(::_LongL5I)             = 102_300
GNSSSignals.get_code_frequency(::_LongL5I)          = GNSSSignals.get_code_frequency(GPSL5I())
GNSSSignals.get_center_frequency(::_LongL5I)        = GNSSSignals.get_center_frequency(GPSL5I())
GNSSSignals.get_data_frequency(::_LongL5I)          = 0.0Hz
GNSSSignals.get_secondary_code(::_LongL5I)          = GNSSSignals.NoSecondaryCode()
GNSSSignals.get_secondary_code_length(::_LongL5I)   = 1
GNSSSignals.get_code_type(::_LongL5I)               = Int16
function GNSSSignals.gen_code!(buffer::AbstractVector, ::_LongL5I, prn::Integer,
                               sampling_frequency::_Frequency, code_frequency::_Frequency,
                               start_phase, start_index::Integer)
    GNSSSignals.gen_code!(buffer, GPSL5I(), prn, sampling_frequency,
                          code_frequency, start_phase, start_index)
end

@testset "secondary_code_search" begin
    @testset "L5I rotation discovery — detection succeeds at every NH10 starting phase" begin
        # For each r ∈ 0..9, build an L5I signal whose first integrated primary period
        # carries NH10[r], second carries NH10[(r+1) mod 10], etc. With use_secondary_code
        # = true, the rotation search must find the matching rotation and the resulting
        # peak must clear the opt-out baseline by a wide margin at every starting phase.
        system        = GPSL5I()
        sampling_freq = 10.24e6Hz
        prn           = 1
        fc            = get_code_frequency(system)
        samples_per_code = ceil(Int, get_code_length(system) / fc * sampling_freq)
        N = 10
        L = 10
        sec = get_secondary_code(system)
        primary = ComplexF32.(gen_code(samples_per_code, system, prn, sampling_freq, fc, 0.0))

        plan = plan_acquire(system, sampling_freq, [prn];
            min_doppler_coverage = 1_000Hz, num_coherently_integrated_code_periods = N,
            num_noncoherent_accumulations = 1)
        # Run an opt-out reference once so we have a baseline peak to compare against.
        ref_signal = ComplexF32[]
        for k in 0:N-1
            append!(ref_signal, Float32(GNSSSignals.secondary_value(sec, prn, k)) .* primary)
        end
        plan_off = plan_acquire(system, sampling_freq, [prn];
            min_doppler_coverage = 1_000Hz, num_coherently_integrated_code_periods = N,
            num_noncoherent_accumulations = 1, use_secondary_code = false)
        results_off = acquire!(plan_off, ref_signal, [prn]; store_power_bins = true)
        peak_off = maximum(results_off[1].power_bins)

        @testset "rotation r = $r" for r in 0:L-1
            signal = ComplexF32[]
            for k in 0:N-1
                nh10_chip = Float32(GNSSSignals.secondary_value(sec, prn, mod(k + r, L)))
                append!(signal, nh10_chip .* primary)
            end
            results_on = acquire!(plan, signal, [prn]; store_power_bins = true)
            peak_on = maximum(results_on[1].power_bins)
            # Rotation discovery: with the search active, the peak should clear the opt-out
            # baseline by a substantial margin, regardless of NH10 starting phase.
            @test peak_on > 5 * peak_off
            # The recovered secondary-code phase must equal the planted rotation r
            # (the NH10 chip the integration window started on). Locks the
            # `rotation_block → secondary_code_phase` decode convention.
            @test results_on[1].secondary_code_phase == r
        end

        # When no rotation search runs the field is `nothing`, not an integer:
        # use_secondary_code = false (opt-out)…
        @test acquire!(plan_off, ref_signal, [prn])[1].secondary_code_phase === nothing
        # …and a signal with no secondary code at all (L1 C/A, L = 1).
        let l1 = GPSL1CA()
            l1plan = plan_acquire(l1, 5e6Hz, [1]; num_coherently_integrated_code_periods = 1)
            l1sig = ComplexF32.(randn(ComplexF64, l1plan.samples_per_code))
            @test acquire!(l1plan, l1sig, [1])[1].secondary_code_phase === nothing
        end
    end

    @testset "Opt-out parity — use_secondary_code=false reproduces the pre-feature pilot fast-path output" begin
        # Pinned values captured by running this exact configuration with the pilot fast-path
        # before the rotation kernel was added. Since `use_secondary_code = false` makes the
        # dispatcher route through the unchanged pilot fast-path (no kernel restructuring),
        # output stays numerically equivalent to the pre-feature behaviour. Comparisons use
        # `≈` rather than `==` because the underlying FFT (FFTW) gives ULP-level
        # differences across Julia versions / FFTW builds — the parity guarantee is
        # "no behavioural drift in the opt-out path", not platform-exact bits. `size` and
        # `argmax` are integer-valued and stay exact.
        system        = GPSL5I()
        sampling_freq = 10.24e6Hz
        prn           = 1
        fc            = get_code_frequency(system)
        samples_per_code = ceil(Int, get_code_length(system) / fc * sampling_freq)
        N = 10
        sec = get_secondary_code(system)

        primary = ComplexF32.(gen_code(samples_per_code, system, prn, sampling_freq, fc, 0.0))
        signal = ComplexF32[]
        for k in 0:N-1
            nh10_chip = Float32(GNSSSignals.secondary_value(sec, prn, k))
            append!(signal, nh10_chip .* primary)
        end

        plan = plan_acquire(system, sampling_freq, [prn];
            min_doppler_coverage = 1_000Hz, num_coherently_integrated_code_periods = N,
            num_noncoherent_accumulations = 1, use_secondary_code = false)
        results = acquire!(plan, signal, [prn]; store_power_bins = true)
        nim = results[1].power_bins

        @test size(nim) == (40, 10240)
        @test maximum(nim)   ≈ 1.146687232f9   rtol = 1e-4
        @test sum(nim)       ≈ 5.2553412608f10 rtol = 1e-4
        @test nim[1, 100]    ≈ 8.2944015625f4  rtol = 1e-4
        @test nim[end, end]  ≈ 2.318815625f4   rtol = 1e-4
        # The NH10 signs make the opt-out's naive coherent sum sign-modulated,
        # so its Doppler spectrum is symmetric about 0 Hz: rows 17 and 25
        # (∓400 Hz) carry an EXACT mirror-pair peak (equal in Float32). Which
        # one argmax returns is decided by FFT last-bit noise and differs
        # across FFTW builds/platforms — accept either, and pin the tie itself.
        @test argmax(nim) in (CartesianIndex(17, 1), CartesianIndex(25, 1))
        @test nim[17, 1] ≈ nim[25, 1] rtol = 1e-3
    end

    @testset "L5I gain recovery — use_secondary_code=true recovers ~10× peak over the opt-out" begin
        # Synthesise a clean L5I signal with NH10 modulation applied across 10 primary periods
        # (= 1 NH10 period at the primary code rate). Without secondary-code rotation search,
        # summing 10 primary periods coherently against the bare primary code adds up the
        # NH10 = [+,+,+,+,-,-,+,-,+,-] signs → inner product = 2 instead of 10. With the
        # rotation search active, the correct alignment is found and the full coherent gain is
        # recovered. The theoretical ratio is (10/2)² = 25× in coherent power, but the search
        # also picks up the next-best off-alignment rotation in the opt-out's denominator, so
        # the realised on/off ratio is closer to ~9× (issue PRD documents "~10×"). Assert
        # ratio > 8 so a real regression of the rotation kernel — which would collapse the
        # ratio toward 1× — is caught, without flapping on FFT shape/windowing noise.
        system        = GPSL5I()
        sampling_freq = 10.24e6Hz
        prn           = 1
        fc            = get_code_frequency(system)
        samples_per_code = ceil(Int, get_code_length(system) / fc * sampling_freq)
        N = 10
        sec = get_secondary_code(system)

        primary = ComplexF32.(gen_code(samples_per_code, system, prn, sampling_freq, fc, 0.0))
        signal = ComplexF32[]
        for k in 0:N-1
            nh10_chip = Float32(GNSSSignals.secondary_value(sec, prn, k))
            append!(signal, nh10_chip .* primary)
        end

        # Acquire with use_secondary_code = true (default) and with use_secondary_code = false.
        plan_on = plan_acquire(system, sampling_freq, [prn];
            min_doppler_coverage = 5_000Hz, num_coherently_integrated_code_periods = N,
            num_noncoherent_accumulations = 1)
        plan_off = plan_acquire(system, sampling_freq, [prn];
            min_doppler_coverage = 5_000Hz, num_coherently_integrated_code_periods = N,
            num_noncoherent_accumulations = 1, use_secondary_code = false)

        results_on  = acquire!(plan_on,  signal, [prn]; store_power_bins = true)
        results_off = acquire!(plan_off, signal, [prn]; store_power_bins = true)

        peak_on  = maximum(results_on[1].power_bins)
        peak_off = maximum(results_off[1].power_bins)
        ratio = peak_on / peak_off
        @info "L5I 10ms peak ratio (on/off) = $ratio"
        @test ratio > 8
    end

    @testset "L5I rotation search reports the correct Doppler (no half-band fftshift offset)" begin
        # Regression for the double-fftshift bug shared by all sign-search kernels:
        # the per-column sub-block FFT fftshifts internally (circshift by N/2) AND
        # the result was scattered through fftshift_perm a second time. The two
        # compose to the identity, leaving the Doppler axis in raw FFT order — the
        # rotation search then recovered full peak *power* but at a Doppler bin
        # offset by exactly half the searched band. Detection and the gain-recovery
        # tests above pass either way (they don't pin the Doppler), so this asserts
        # the absolute Doppler value, which the bug got wrong.
        system        = GPSL5I()
        sampling_freq = 12e6Hz
        prn           = 1
        true_doppler  = 1000Hz        # on the 100 Hz Doppler grid at N_coh=10
        true_cp       = 1234.5

        (; signal) = generate_test_signal(system, prn;
            num_samples = 10 * 12000,
            doppler = true_doppler, code_phase = true_cp,
            sampling_freq, interm_freq = 0.0Hz, CN0 = 45)

        plan = plan_acquire(system, sampling_freq, [prn];
            num_coherently_integrated_code_periods = 10, num_noncoherent_accumulations = 1)
        @test plan.num_secondary_rotations > 1   # confirm the rotation search is active

        result = acquire!(plan, signal, prn; interm_freq = 0.0Hz)

        @test is_detected(result)
        @test result.code_phase ≈ true_cp atol = 1.0
        @test abs(result.carrier_doppler / 1Hz - ustrip(Hz, true_doppler)) < ustrip(Hz, step(plan.doppler_freqs))
    end

    @testset "L5I rotation search — multi-symbol N≥2L (data-bit × rotation; segfault regression)" begin
        # ──────────────────────────────────────────────────────────────────────
        # Regression for the buffer-overflow segfault when BOTH the data-bit
        # search and the rotation search are active (N ≥ 2·L). At N = 20 on L5I
        # (L = 10) the kernel enumerates num_data_combos(2) × num_secondary_
        # rotations(10) = 20 pattern columns, but the noncoherent buffer's cp
        # axis is only widened by num_secondary_rotations = 10. The kernel used
        # to write `dest_col = col + (q-1)*samples_per_code` for all 20 patterns,
        # overrunning the 10-block buffer → out-of-bounds write → segfault. No
        # prior test exercised this: every other rotation test uses N = 10
        # (num_data_bits = 1), so num_data_combos = 1 and pattern index == rotation
        # index. The fix maps each pattern back to its rotation block and collapses
        # the data-bit polarities sharing a rotation by a cell-wise max (the data
        # sign is a nuisance hypothesis, exactly as the non-rotation kernel treats
        # it). This test plants a genuine data-bit sign flip between the two 10 ms
        # symbols so the data-combo max must pick the correct polarity to detect.
        system        = GPSL5I()
        sampling_freq = 12e6Hz
        prn           = 1
        true_doppler  = 1000Hz        # on the 50 Hz grid at N = 20
        true_cp       = 1234.5
        N = 20; L = 10

        fc  = get_code_frequency(system)
        spc = ceil(Int, get_code_length(system) / fc * sampling_freq)
        sec = get_secondary_code(system)
        primary = ComplexF32.(gen_code(spc, system, prn, sampling_freq, fc, true_cp))

        signal = ComplexF32[]
        for k in 0:N-1
            nh10      = Float32(GNSSSignals.secondary_value(sec, prn, mod(k, L)))
            data_sign = k < L ? 1.0f0 : -1.0f0   # bit flip between the two symbols
            append!(signal, (nh10 * data_sign) .* primary)
        end
        ω = 2π * ustrip(Hz, true_doppler) / ustrip(Hz, sampling_freq)
        signal .*= ComplexF32.(cis.(ω .* (0:length(signal)-1)))

        plan = plan_acquire(system, sampling_freq, [prn];
            num_coherently_integrated_code_periods = N, num_noncoherent_accumulations = 1)
        @test plan.num_secondary_rotations == L   # rotation search active
        @test plan.num_data_bits == N ÷ L         # data-bit search active (= 2)

        result = acquire!(plan, signal, prn; interm_freq = 0.0Hz)   # must not segfault

        @test is_detected(result)
        @test result.code_phase ≈ true_cp atol = 1.0
        @test abs(result.carrier_doppler / 1Hz - ustrip(Hz, true_doppler)) < ustrip(Hz, step(plan.doppler_freqs))
        # NH10 starts at chip 0 here (mod(k, L)); the estimator must recover it even
        # across the data-bit boundary at the 10 ms mark.
        @test result.secondary_code_phase == 0
    end

    @testset "secondary_code_phase is exact across code phase & Doppler (estimator)" begin
        # `_estimate_secondary_code_phase` despreads the first L per-period prompt
        # correlations at the recovered (doppler, code_phase) — exact at every code
        # phase, unlike the raw FM-DBZP rotation index which is ±1 at worst-case code
        # phases (primary-code wrap mid-block). This sweep is the regression guard for
        # that fix: a non-zero code phase like 5115 (the documented worst case) and a
        # mid-grid Doppler used to mis-report the secondary phase by one chip.
        system        = GPSL5I()
        sampling_freq = 12e6Hz
        prn           = 1
        N = 10; L = 10
        fc  = get_code_frequency(system)
        spc = ceil(Int, get_code_length(system) / fc * sampling_freq)
        sec = get_secondary_code(system)
        plan = plan_acquire(system, sampling_freq, [prn];
            num_coherently_integrated_code_periods = N, num_noncoherent_accumulations = 1)

        @testset "r0=$r0, cp=$cp, dop=$dop" for r0 in (0, 3, 7, 9),
                                                 cp in (0.0, 2000.0, 5115.0, 9000.0),
                                                 dop in (0.0, 1300.0)
            primary = ComplexF32.(gen_code(spc, system, prn, sampling_freq, fc, cp))
            signal = ComplexF32[]
            for k in 0:N-1
                append!(signal, Float32(GNSSSignals.secondary_value(sec, prn, mod(k + r0, L))) .* primary)
            end
            if dop != 0.0
                ω = 2π * dop / ustrip(Hz, sampling_freq)
                signal .*= ComplexF32.(cis.(ω .* (0:length(signal)-1)))
            end
            result = acquire!(plan, signal, prn; interm_freq = 0.0Hz)
            @test result.secondary_code_phase == r0
        end
    end

    @testset "L5I rotation search — correct Doppler/code-phase with non-zero code drift (N_nc>1)" begin
        # Locks the sign-search path's drift correction. `_apply_code_drift!` is
        # currently called per non-coherent step with a sorted-row buf; any future
        # refactor that moves the buf to raw-row order (e.g. fusing the kernel's
        # internal fftshift into a single output shift) must keep the drift
        # mapping consistent or this test will catch the regression.
        #
        # Parameters chosen so the per-row drift `round(f_d·m·T_coh·fs/f_c)`
        # rounds to non-zero at later steps: at f_c=L5=1.176 GHz, T_coh=10 ms,
        # fs=12 MHz, f_d=5000 Hz, the step-1 drift is ≈0.51 samples → rounds to 1;
        # step-3 ≈1.53 → 2. So the drift path is genuinely exercised.
        system        = GPSL5I()
        sampling_freq = 12e6Hz
        prn           = 1
        true_doppler  = 5000Hz
        true_cp       = 1234.5
        Ncoh = 10; Nnc = 4

        (; signal) = generate_test_signal(system, prn;
            num_samples = Nnc * Ncoh * 12000,
            doppler = true_doppler, code_phase = true_cp,
            sampling_freq, interm_freq = 0.0Hz, CN0 = 45)

        plan = plan_acquire(system, sampling_freq, [prn];
            num_coherently_integrated_code_periods = Ncoh,
            num_noncoherent_accumulations = Nnc)
        @test plan.num_secondary_rotations > 1
        @test plan.num_noncoherent_accumulations > 1   # multistep path → drift active

        result = acquire!(plan, signal, prn; interm_freq = 0.0Hz)

        @test is_detected(result)
        @test result.code_phase ≈ true_cp atol = 1.5
        @test abs(result.carrier_doppler / 1Hz - ustrip(Hz, true_doppler)) < ustrip(Hz, step(plan.doppler_freqs))
    end

    @testset "L5I rotation search — worst-case code-phase (block / NH10 misalignment fixed)" begin
        # ──────────────────────────────────────────────────────────────────────
        # Regression lock for the "mid-primary-code secondary-code shift" fix.
        # Before the fix this test was @test_broken: the rotation search used a
        # FIXED sub-block partition (`row_offset` constant across columns), so
        # one NH10 chip mapped cleanly to one sub-block only when the primary-
        # code wrap landed at sample 0 (i.e. `code_phase == 0`). At any other
        # code phase the wrap drifted INSIDE every sub-block; each block then
        # carried a fractional mix of two consecutive NH10 chips, no discrete
        # ±1 rotation hypothesis matched the mix, ~6.5 dB of coherent gain
        # burned, the Doppler spectrum flattened, and `argmax` flipped on noise
        # → up to ±700 Hz Doppler bias at the worst code phase.
        #
        # The wrap sample for column `c` is exactly the FM-DBZP delay,
        #     W(c) = _fmdbzp_column_to_tau(c-1, num_blocks, block_size),
        # so `_sign_search_step_with_rotations!` now picks a per-column row
        # shift `s(c) = round(W(c) / block_size) mod num_blocks` that snaps the
        # sub-block partition to the nearest NH10 chip boundary. Residual
        # misalignment is at most half a block_size (≈ `1/(2*num_blocks)` of a
        # primary-code period) regardless of code phase — at L5I/12 MHz that's
        # 400 samples → 0.4 dB residual loss vs. ideal (vs. 6.5 dB before).
        # Cost is four integer ops per column iteration of the kernel.
        #
        # The Doppler axis stays correct because the shift is uniform across
        # the column's sub-blocks: it adds an overall phase factor
        # `exp(2πi · f_d · s · block_size / fs)` that vanishes in `|·|²`. The
        # cyclic wrap that the partition introduces for the last sub-block is
        # consistent because the divisibility check in `plan_acquire` requires
        # the integration window to span a whole number of secondary-code
        # periods, so NH10 has period equal to the buffer length and the
        # wrapped portion carries the same chip back to itself.
        #
        # `cp = 5115` chips picks the deterministic worst case across the cp
        # sweep (wrap at exactly 50% of the receive block); `dop = 1000 Hz`
        # lands on the 100 Hz Doppler grid so the assertion has no half-bin
        # slop.
        system        = GPSL5I()
        sampling_freq = 12e6Hz
        prn           = 1
        true_doppler  = 1000Hz
        true_cp = 5115.0

        (; signal) = generate_test_signal(system, prn;
            num_samples = 10 * 12000,
            doppler = true_doppler, code_phase = true_cp,
            sampling_freq, interm_freq = 0.0Hz, CN0 = 45)

        plan = plan_acquire(system, sampling_freq, [prn];
            num_coherently_integrated_code_periods = 10, num_noncoherent_accumulations = 1)
        @test plan.num_secondary_rotations > 1   # confirm the rotation search is active

        result = acquire!(plan, signal, prn; interm_freq = 0.0Hz)

        @test is_detected(result)
        @test result.code_phase ≈ true_cp atol = 1.0
        @test abs(result.carrier_doppler / 1Hz - ustrip(Hz, true_doppler)) < ustrip(Hz, step(plan.doppler_freqs))
    end

    @testset "L5I rotation search — matches LongL5I (full 10 ms FFT) on off-grid Doppler" begin
        # ──────────────────────────────────────────────────────────────────────
        # Algebraic-equivalence regression for the inter-sub-block phase ramp
        # combination. Planted as @test_broken: today the rotation kernel
        # combines sub-block FFTs with ±1 NH10 patterns, which is matched-filter
        # optimal ONLY when the true Doppler lands on the coarse 1-kHz sub-block
        # FFT grid (or the 500 Hz half-grid where alternating ±1 happens to
        # align). Between grid points, the inter-sub-block phase rotates by
        # δ ∈ (0, 2π) per code period — a fractional phase that ±1 cannot fit —
        # so the rotation sum loses several dB of coherent gain.
        #
        # LongL5I avoids this by treating NH10·primary as a single 102 300-chip
        # primary code and running one 10 ms column FFT. The full FFT decomposes
        # exactly as:
        #     FFT_total(x)[ω] = Σ_p exp(-2πi·p·s/N_coh) · sub_block_FFT_p[ω]
        # with s = ω mod N_coh, so the rotation kernel can match LongL5I by
        # replacing `±1 · NH10[(p+r)]` with the complex phasor
        # `NH10[(p+r)] · exp(-2πi·p·s/N_coh)`.
        #
        # This noiseless test compares maximum power-bin amplitude. ON-grid
        # (Doppler exactly on the 1 kHz sub-block grid) both kernels match
        # today — the on-grid baseline assertion stays as a live @test. OFF-grid
        # (true Doppler off the coarse grid but on the fine 100 Hz output grid)
        # is currently a structural ~5 dB gap → @test_broken. The fix promotes
        # this to a hard @test.
        #
        # The signal is constructed without `generate_test_signal`'s noise term
        # so the gap is purely structural (no statistical jitter on the ratio).
        system        = GPSL5I()
        long_system   = _LongL5I()
        sampling_freq = 12e6Hz
        prn           = 1
        true_cp       = 0.0       # isolates off-grid Doppler loss from any chip-boundary residual
        N_coh         = 10
        spc           = 12000     # samples_per_code for L5I at 12 MHz
        N_total       = N_coh * spc

        # Build a noiseless NH10·primary signal at amplitude 1.0 at the test Doppler.
        sec = get_secondary_code(system)
        fc  = get_code_frequency(system)
        code_full = ComplexF32.(gen_code(N_total, system, prn, sampling_freq,
                                         fc, true_cp))
        # The L5I sample-level code generation already bakes NH10 across primary
        # periods (same property `_LongL5I` exploits), so `code_full` *is* the
        # NH10·primary product over 10 periods.

        plan_rot = plan_acquire(system, sampling_freq, [prn];
            num_coherently_integrated_code_periods = N_coh, num_noncoherent_accumulations = 1)
        plan_long = plan_acquire(long_system, sampling_freq, [prn];
            num_coherently_integrated_code_periods = 1, num_noncoherent_accumulations = 1)

        # Helper: run both plans on the same noiseless signal at `doppler_hz`
        # and return (peak_rot_power, peak_long_power). Both plans share the
        # same Doppler grid (100 Hz spacing, 150 bins covering ±7.5 kHz).
        function peaks_at(doppler_hz)
            ω = 2π * doppler_hz / ustrip(Hz, sampling_freq)
            carrier = cis.(ω .* (0:N_total-1))
            signal  = ComplexF32.(carrier .* code_full)
            res_rot  = acquire!(plan_rot,  signal, prn; interm_freq = 0.0Hz, store_power_bins = true)
            res_long = acquire!(plan_long, signal, prn; interm_freq = 0.0Hz, store_power_bins = true)
            (maximum(res_rot.power_bins), maximum(res_long.power_bins))
        end

        # On-coarse-grid Doppler — both kernels agree today (live @test).
        peak_rot_on, peak_long_on = peaks_at(1000.0)
        @test peak_rot_on / peak_long_on > 0.95

        # Off-coarse-grid, ON the fine 100 Hz output grid. With the inter-sub-
        # block phase ramp combination active, the rotation kernel reaches
        # LongL5I-equivalent peak power within FFTW jitter (ratio ≈ 1.0).
        # Before the phase-ramp fix this lost ≈6 dB at 100 Hz and ≈8 dB at
        # 500 Hz (see /workspace/Acquisition/claude_scratch/probe_gap-style
        # measurements). The ratio > 0.95 bound below catches a regression
        # back to the ±1 combination without flapping on FFTW noise.
        peak_rot_off, peak_long_off = peaks_at(100.0)
        @test peak_rot_off / peak_long_off > 0.95

        # 500 Hz was the worst-case off-grid loss before the fix (≈8 dB:
        # neither the unshifted ±1 NH10 nor the alternating ±1 hypothesis
        # aligns with the half-coarse-grid Doppler). Asserting ratio > 0.95
        # at 500 Hz keeps the strongest single-point regression locked.
        peak_rot_500, peak_long_500 = peaks_at(500.0)
        @test peak_rot_500 / peak_long_500 > 0.95
    end

    @testset "get_num_cells / is_detected count the rotation expansion (issue #70)" begin
        # On the rotation-search path the peak is taken over an EXPANDED surface
        # num_doppler_bins × (samples_per_code × num_secondary_rotations) — L× more
        # independent cells than the bare Doppler × code-phase grid. `get_num_cells`
        # (and therefore `is_detected`) used to return only num_doppler_bins ×
        # samples_per_code, understating the count L-fold → the Bonferroni CFAR
        # correction was applied for too few cells → threshold too low → realised
        # false-alarm rate well above the requested pfa, and pure-noise peaks called
        # "detected". The fix carries `num_secondary_rotations` on the result and folds
        # it into `get_num_cells`.
        system        = GPSL5I()
        sampling_freq = 12e6Hz
        prn           = 1

        plan = plan_acquire(system, sampling_freq, [prn];
            num_coherently_integrated_code_periods = 10, num_noncoherent_accumulations = 1)
        @test plan.num_secondary_rotations == 10   # rotation search active

        (; signal) = generate_test_signal(system, prn;
            num_samples = 10 * 12000, doppler = 1000Hz, code_phase = 1234.5,
            sampling_freq, interm_freq = 0.0Hz, CN0 = 45)
        result = acquire!(plan, signal, prn; interm_freq = 0.0Hz, store_power_bins = true)

        # The expansion factor is carried on the result …
        @test result.num_secondary_rotations == plan.num_secondary_rotations
        # … and `get_num_cells` is the full expanded count, which equals the stored
        # power-surface size (num_doppler_bins × samples_per_code × num_rotations).
        num_dop = length(result.dopplers)
        @test get_num_cells(result) ==
              num_dop * plan.samples_per_code * plan.num_secondary_rotations
        @test get_num_cells(result) == length(result.power_bins)
        # It is exactly L× what the pre-fix formula (Doppler × code-phase only) gave.
        @test get_num_cells(result) ==
              10 * num_dop * plan.num_blocks * plan.block_size

        # `is_detected` now flips exactly at the expanded-count threshold, the same
        # threshold the `secondary_code_phase` gate uses — the two were inconsistent
        # before the fix (the gate already counted the expansion, `is_detected` did not).
        thr = cfar_threshold(0.01, get_num_cells(result);
            num_noncoherent_integrations = result.num_noncoherent_integrations)
        @test is_detected(result; pfa = 0.01) == (result.peak_to_noise_ratio > thr)
        # A detected rotation peak carries a secondary-code phase; both decisions agree.
        @test is_detected(result; pfa = 0.01)
        @test result.secondary_code_phase !== nothing
    end

    @testset "get_num_cells unchanged off the rotation path (issue #70 regression guard)" begin
        # The expansion must be a no-op everywhere the rotation search is inactive:
        # num_secondary_rotations == 1, so `get_num_cells` stays the bare
        # Doppler × code-phase grid. Guards against the fix changing CFAR statistics
        # on L1 C/A and the pilot/sign-search paths.
        system        = GPSL1CA()
        prn           = 1
        (; signal, sampling_freq, interm_freq) = generate_test_signal(system, prn)
        result = only(acquire(system, signal, sampling_freq, [prn]; interm_freq))

        @test result.num_secondary_rotations == 1
        @test get_num_cells(result) ==
              length(result.dopplers) * result.num_blocks * result.block_size
    end
end
