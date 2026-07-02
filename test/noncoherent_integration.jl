# test/noncoherent_integration.jl
# Tests for noncoherent integration: column FFT accumulation, data bit search, code drift correction

@testset "Fused FFT+|x|²+code-drift+fftshift kernel matches unfused pipeline" begin
    # Issue #62: extends the slice-5 fusion to the multistep simple path by
    # folding `_apply_code_drift!` into the per-column accumulator loop. Each
    # row r writes its (c) cell into nim[fftshift_perm[r], (c + shift_r) mod spc]
    # where shift_r is the existing _apply_code_drift! formula (which reads
    # plan.doppler_freqs[r] — we mirror that exact behaviour, including the
    # raw-FFT-bin vs sorted-doppler indexing convention, for bitwise parity).
    Random.seed!(20260526)
    sampling_freq_hz_test = 5.0e6
    carrier_freq_hz_test = 1.57542e9  # GPS L1
    for (num_doppler_bins, samples_per_code, accumulation_step) in [
        (64, 256, 5),       # batched-path size, non-zero step
        (321, 128, 3),      # per-column-path size (one above the threshold)
        (8, 5000, 7),       # L1CA 5MHz / N_coh=1 / N_nc=8 (the canonical issue-62 case)
        (32, 2048, 0),      # step 0: drift must be a no-op
    ]
        cim_template = randn(ComplexF32, num_doppler_bins, samples_per_code)
        col_proto = zeros(ComplexF32, num_doppler_bins)
        col_fft_plan = plan_fft!(col_proto)
        col_batch_fft_plan = if num_doppler_bins <= Acquisition.BATCH_FFT_THRESHOLD
            batch_proto = zeros(ComplexF32, num_doppler_bins, samples_per_code)
            plan_fft!(batch_proto, 1)
        else
            nothing
        end
        fftshift_perm = [mod(r - 1 + num_doppler_bins ÷ 2, num_doppler_bins) + 1
                         for r in 1:num_doppler_bins]

        # Synthetic doppler grid in sorted order (matches plan.doppler_freqs).
        doppler_coverage_hz = 32_000.0
        doppler_freqs = range(-doppler_coverage_hz / 2,
                              step = doppler_coverage_hz / num_doppler_bins,
                              length = num_doppler_bins)
        coherent_duration_s = samples_per_code / sampling_freq_hz_test
        # Raw shifts for the reference circshift (any integer is fine).
        raw_shifts = [round(Int, doppler_freqs[r] * accumulation_step *
                       coherent_duration_s * sampling_freq_hz_test / carrier_freq_hz_test)
                      for r in 1:num_doppler_bins]
        # Fused kernels expect shifts normalised to [0, samples_per_code) so
        # they can replace `mod` with a single conditional subtract — matches
        # the convention `_fill_code_drift_shifts!` writes.
        norm_shifts = [mod(s, samples_per_code) for s in raw_shifts]

        # Reference: unfused pipeline (pilot accumulator → row-wise circshift → fftshift scatter)
        ref_acc = zeros(Float32, num_doppler_bins, samples_per_code)
        ref_buf = zeros(Float32, num_doppler_bins, samples_per_code)
        col_buf = zeros(ComplexF32, num_doppler_bins)
        Acquisition._accumulate_noncoherent_integration_pilot!(ref_buf, copy(cim_template),
            col_buf, col_fft_plan, samples_per_code)
        # Apply the row-wise circular shift (mirrors _apply_code_drift! body exactly).
        if accumulation_step != 0
            row_buf = zeros(Float32, samples_per_code)
            row_shift_buf = zeros(Float32, samples_per_code)
            for r in 1:num_doppler_bins
                raw_shifts[r] == 0 && continue
                row_buf .= view(ref_buf, r, :)
                circshift!(row_shift_buf, row_buf, raw_shifts[r])
                ref_buf[r, :] .= row_shift_buf
            end
        end
        Acquisition._scatter_fftshift_accumulate!(ref_acc, ref_buf, fftshift_perm, samples_per_code)

        # Fused: per-column path
        fused_acc = zeros(Float32, num_doppler_bins, samples_per_code)
        Acquisition._accumulate_fftshifted_power_drift_pilot!(fused_acc, copy(cim_template),
            zeros(ComplexF32, num_doppler_bins), col_fft_plan,
            samples_per_code, num_doppler_bins, fftshift_perm, norm_shifts)
        @test fused_acc ≈ ref_acc

        # Fused: batched path (only valid below the threshold)
        if col_batch_fft_plan !== nothing
            fused_acc_batched = zeros(Float32, num_doppler_bins, samples_per_code)
            Acquisition._accumulate_fftshifted_power_drift_pilot_batched!(fused_acc_batched,
                copy(cim_template), col_batch_fft_plan, samples_per_code, num_doppler_bins,
                fftshift_perm, norm_shifts)
            @test fused_acc_batched ≈ ref_acc
        end
    end
end

@testset "_fill_code_drift_shifts! has_drift flag and non-drift dispatch" begin
    # Validate the early-exit that lets the multistep simple path skip the
    # fused-drift kernel and run the SIMD-friendly slice-5 kernel when every
    # row's rounded drift is zero. Two scenarios:
    #   (a) L1CA / 5 MHz / N_coh=1: doppler×step×T_coh×fs/fc rounds to 0 for
    #       every row up to N_nc≈8. has_drift must stay false → dispatcher
    #       must produce the same output as the non-drift kernel run direct.
    #   (b) L1CA / 2.048 MHz / N_coh=20: drift is large enough that some
    #       rows are non-zero at step=7. has_drift must flip to true and
    #       shifts must be normalised to [0, samples_per_code).
    Random.seed!(20260526)

    # (a) all-zero drift on the simple/batched path
    plan_a = plan_acquire(GPSL1CA(), 5.0e6Hz, [1];
        min_doppler_coverage = 2000Hz, num_noncoherent_accumulations = 8)
    n_dop_a = length(plan_a.doppler_freqs)
    shifts_a = zeros(Int, n_dop_a)

    # step 0: definitionally has_drift=false, shifts all 0
    @test Acquisition._fill_code_drift_shifts!(shifts_a, plan_a, 0) == false
    @test all(iszero, shifts_a)

    # step 7: doppler×step×T_coh×fs/fc < 0.5 for every row at this grid →
    # rounds to 0. has_drift must STILL be false.
    @test Acquisition._fill_code_drift_shifts!(shifts_a, plan_a, 7) == false
    @test all(iszero, shifts_a)

    # End-to-end: dispatcher at step 7 must produce the same nim as a direct
    # call to the non-drift slice-5 kernel on the same CIM.
    cim_a = randn(ComplexF32, n_dop_a, plan_a.samples_per_code)
    nim_via_dispatch = zeros(Float32, n_dop_a, plan_a.samples_per_code)
    Acquisition._accumulate_noncoherent_integration_step!(nim_via_dispatch, copy(cim_a), plan_a, 7)

    # The plan's batched FFT plan is tile-shaped (the pipeline consumes one
    # column-block tile at a time), so the full-matrix reference uses the
    # per-column kernel — the batched/per-column parity is asserted separately
    # in the fused-kernel testsets above.
    nim_direct = zeros(Float32, n_dop_a, plan_a.samples_per_code)
    Acquisition._accumulate_fftshifted_power_pilot!(
        nim_direct, copy(cim_a), Acquisition._default_scratch(plan_a).col_buf,
        plan_a.col_fft_plan, plan_a.samples_per_code, n_dop_a)
    @test nim_via_dispatch ≈ nim_direct

    # (b) non-zero drift on the per-column path
    plan_b = plan_acquire(GPSL1CA(), 2.048e6Hz, [1];
        min_doppler_coverage = 12_000Hz,
        num_coherently_integrated_code_periods = 20,
        num_noncoherent_accumulations = 8)
    n_dop_b = length(plan_b.doppler_freqs)
    shifts_b = zeros(Int, n_dop_b)

    # step 0 is still has_drift=false even on the larger grid.
    @test Acquisition._fill_code_drift_shifts!(shifts_b, plan_b, 0) == false
    @test all(iszero, shifts_b)

    # step 7 must flip has_drift to true and normalise shifts to [0, spc).
    @test Acquisition._fill_code_drift_shifts!(shifts_b, plan_b, 7) == true
    @test any(!iszero, shifts_b)
    @test all(s -> 0 <= s < plan_b.samples_per_code, shifts_b)
end

@testset "Fused FFT+|x|²+fftshift kernel matches unfused pipeline" begin
    # The slice-5 fused kernel must produce bit-identical output to the
    # unfused `column FFT -> |x|² -> _scatter_fftshift_accumulate!` chain it
    # replaces. Run both on the same synthetic CIM and assert equality. The
    # `_accumulate_fftshifted_power_pilot!` (per-column) path is otherwise
    # unreachable from the test suite because it only fires at
    # num_doppler_bins > BATCH_FFT_THRESHOLD with simple-path + N_nc=1,
    # which no acquire! test currently spans.
    Random.seed!(20260526)
    for (num_doppler_bins, samples_per_code) in [
        (64, 256),      # batched-path size
        (321, 128),     # per-column-path size (one above the threshold)
        (55, 64),       # odd num_doppler_bins (16.368 MHz / N_coh=5 maps here)
    ]
        cim_template = randn(ComplexF32, num_doppler_bins, samples_per_code)
        # Pre-build an FFTW plan matching how plan_acquire would
        col_proto = zeros(ComplexF32, num_doppler_bins)
        col_fft_plan = plan_fft!(col_proto)
        col_batch_fft_plan = if num_doppler_bins <= Acquisition.BATCH_FFT_THRESHOLD
            batch_proto = zeros(ComplexF32, num_doppler_bins, samples_per_code)
            plan_fft!(batch_proto, 1)
        else
            nothing
        end
        fftshift_perm = [mod(r - 1 + num_doppler_bins ÷ 2, num_doppler_bins) + 1
                         for r in 1:num_doppler_bins]

        # Reference: unfused pipeline on a copy of the CIM.
        ref_acc = zeros(Float32, num_doppler_bins, samples_per_code)
        ref_buf = zeros(Float32, num_doppler_bins, samples_per_code)
        col_buf = zeros(ComplexF32, num_doppler_bins)
        Acquisition._accumulate_noncoherent_integration_pilot!(ref_buf, copy(cim_template),
            col_buf, col_fft_plan, samples_per_code)
        Acquisition._scatter_fftshift_accumulate!(ref_acc, ref_buf, fftshift_perm, samples_per_code)

        # Fused: per-column path.
        fused_acc = zeros(Float32, num_doppler_bins, samples_per_code)
        Acquisition._accumulate_fftshifted_power_pilot!(fused_acc, copy(cim_template),
            zeros(ComplexF32, num_doppler_bins), col_fft_plan,
            samples_per_code, num_doppler_bins)
        @test fused_acc ≈ ref_acc

        # Fused: batched path (only valid below the threshold).
        if col_batch_fft_plan !== nothing
            fused_acc_batched = zeros(Float32, num_doppler_bins, samples_per_code)
            Acquisition._accumulate_fftshifted_power_pilot_batched!(fused_acc_batched,
                copy(cim_template), col_batch_fft_plan, samples_per_code, num_doppler_bins)
            @test fused_acc_batched ≈ ref_acc
        end
    end
end

@testset "Column FFT accumulation — Tier 2: code delay and Doppler alias check" begin
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    prn = 1
    # 1050 Hz: f_D*samples_per_code/sampling_freq = 1.05 (non-integer → non-degenerate), 1050 mod 1000 = 50 Hz
    true_doppler = 1050Hz
    code_phase = 973.0  # tau = 1023 - 973 = 50 chips → tau_samples = round(50*2048/1023) = 100

    # num_coherently_integrated_code_periods=20 = one GPS data bit period (num_data_bits=1, pilot path)
    plan = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 20)
    scratch = Acquisition._default_scratch(plan)
    # Full CIM materialised locally via the reference wrapper (production is tiled).
    cim = zeros(ComplexF32, plan.num_coherently_integrated_code_periods * plan.num_blocks, plan.samples_per_code)
    doppler_bin_spacing = step(plan.doppler_freqs)
    tau_samples = round(Int, (get_code_length(system) - code_phase) * plan.samples_per_code / get_code_length(system))

    (; signal) = generate_test_signal(system, prn;
        num_samples = 20 * plan.samples_per_code,
        doppler = true_doppler, code_phase = code_phase,
        sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = 45)

    signal_f32 = ComplexF32.(signal)
    Acquisition._precompute_signal_block_ffts!(plan.signal_block_ffts, signal_f32,
        plan.samples_per_code, plan.num_blocks, plan.block_size,
        plan.num_coherently_integrated_code_periods, scratch.double_block_buf,
        plan.double_block_fft_plan)
    Acquisition._build_coherent_integration_matrix!(cim, plan.signal_block_ffts, plan.prn_conj_ffts[prn],
        plan.samples_per_code, plan.num_blocks, plan.block_size, plan.num_coherently_integrated_code_periods,
        scratch.corr_buf, plan.double_block_bfft_plan)

    noncoherent_integration_matrix = zeros(Float32, plan.num_coherently_integrated_code_periods * plan.num_blocks, plan.samples_per_code)
    Acquisition._accumulate_noncoherent_integration_pilot!(noncoherent_integration_matrix, cim, scratch.col_buf,
        plan.col_fft_plan, plan.samples_per_code)

    peak_row, peak_col = Tuple(argmax(noncoherent_integration_matrix))
    detected_doppler = plan.doppler_freqs[plan.fftshift_perm[peak_row]]

    # 1. Only the one column block scrambled_block = (num_blocks - block_row) mod num_blocks
    # accumulates coherently; all other column blocks are noise-level (verified empirically).
    # block_row   = tau_samples ÷ block_size
    # fine_offset = tau_samples % block_size
    # Expected peak_col = scrambled_block * block_size + fine_offset + 1 (1-indexed).
    block_row        = tau_samples ÷ plan.block_size
    fine_offset      = tau_samples % plan.block_size
    scrambled_block  = mod(plan.num_blocks - block_row, plan.num_blocks)
    expected_peak_col = scrambled_block * plan.block_size + fine_offset + 1
    @test peak_col == expected_peak_col

    # 2. Detected Doppler must be a valid alias of true_doppler.
    # Alias period = sampling_freq / samples_per_code = 1000 Hz (spacing between indistinguishable
    # Doppler bins in a single-step FM-DBZP column FFT).
    alias_period = plan.sampling_freq / plan.samples_per_code  # 1000 Hz
    diff = mod((detected_doppler - true_doppler) / (1Hz), alias_period / (1Hz))
    diff_sym = min(diff, alias_period / (1Hz) - diff)
    @test diff_sym < doppler_bin_spacing / (1Hz)

    # 3. Verify += accumulation: a second call should double the peak value.
    peak_value_first = noncoherent_integration_matrix[peak_row, peak_col]
    Acquisition._accumulate_noncoherent_integration_pilot!(noncoherent_integration_matrix, cim, scratch.col_buf,
        plan.col_fft_plan, plan.samples_per_code)
    @test noncoherent_integration_matrix[peak_row, peak_col] ≈ 2 * peak_value_first
end

@testset "Data bit combination search — num_data_bits=2, no bit transitions in test signal" begin
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    prn = 1
    true_doppler = 1500Hz

    # num_coherently_integrated_code_periods=40 → num_data_bits=2 (two GPS data bit periods)
    plan = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 40, bit_edge_search_steps = 1, num_noncoherent_accumulations = 1)
    scratch = Acquisition._default_scratch(plan)
    # Full CIM materialised locally via the reference wrapper (production is tiled).
    cim = zeros(ComplexF32, plan.num_coherently_integrated_code_periods * plan.num_blocks, plan.samples_per_code)
    num_doppler_bins = plan.num_coherently_integrated_code_periods * plan.num_blocks  # 640

    code_phase = 100.0
    tau_samples = round(Int, (get_code_length(system) - code_phase) * plan.samples_per_code / get_code_length(system))

    (; signal) = generate_test_signal(system, prn;
        num_samples = plan.num_coherently_integrated_code_periods * plan.samples_per_code,
        doppler = true_doppler, code_phase = code_phase,
        sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = 60)

    signal_f32 = ComplexF32.(signal)
    Acquisition._precompute_signal_block_ffts!(plan.signal_block_ffts, signal_f32,
        plan.samples_per_code, plan.num_blocks, plan.block_size,
        plan.num_coherently_integrated_code_periods, scratch.double_block_buf,
        plan.double_block_fft_plan)
    Acquisition._build_coherent_integration_matrix!(cim, plan.signal_block_ffts, plan.prn_conj_ffts[prn],
        plan.samples_per_code, plan.num_blocks, plan.block_size, plan.num_coherently_integrated_code_periods,
        scratch.corr_buf, plan.double_block_bfft_plan)

    noncoherent_integration_matrix = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    noncoherent_integration_buf = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    sub_block_ffts = zeros(ComplexF32, num_doppler_bins, plan.num_data_bits)
    patterns = Acquisition.sign_patterns(nothing, 0, plan.num_data_bits, 1, plan.num_coherently_integrated_code_periods, false)
    Acquisition._sign_search_step!(noncoherent_integration_buf, cim, scratch.col_buf,
        plan.col_fft_plan, plan.samples_per_code, num_doppler_bins, plan.num_data_bits, 0, sub_block_ffts, patterns)
    noncoherent_integration_matrix .+= noncoherent_integration_buf

    peak_row, peak_col = Tuple(argmax(noncoherent_integration_matrix))

    # 1. Code delay: expected peak column from tau_samples
    block_row         = tau_samples ÷ plan.block_size
    fine_offset       = tau_samples % plan.block_size
    scrambled_block   = mod(plan.num_blocks - block_row, plan.num_blocks)
    expected_peak_col = scrambled_block * plan.block_size + fine_offset + 1
    @test peak_col == expected_peak_col

    # 2. Doppler: FM-DBZP has alias ambiguity at alias_period = sampling_freq/samples_per_code = 1000 Hz.
    # The detected bin must be a valid alias of the true Doppler.
    detected_doppler = plan.doppler_freqs[peak_row]
    doppler_bin_spacing = step(plan.doppler_freqs)
    alias_period = plan.sampling_freq / plan.samples_per_code
    diff = mod((detected_doppler - true_doppler) / (1Hz), alias_period / (1Hz))
    diff_sym = min(diff, alias_period / (1Hz) - diff)
    @test diff_sym < doppler_bin_spacing / (1Hz)
end

@testset "Data bit combination search — bit transition recovery" begin
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    prn = 1
    true_doppler = 1050Hz  # 1050 mod 1000 = 50 Hz: non-degenerate

    plan = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 40, bit_edge_search_steps = 1, num_noncoherent_accumulations = 1)
    scratch = Acquisition._default_scratch(plan)
    # Full CIM materialised locally via the reference wrapper (production is tiled).
    cim = zeros(ComplexF32, plan.num_coherently_integrated_code_periods * plan.num_blocks, plan.samples_per_code)
    num_doppler_bins = plan.num_coherently_integrated_code_periods * plan.num_blocks  # 640

    # Generate a clean signal, then negate the second half to simulate a bit
    # transition at code period 20.  With a perfect polarity flip, the pilot
    # path (naive sum over all 40 periods) cancels to ~0, whereas the data bit
    # search with pattern (+1,−1) recombines both halves coherently.
    (; signal) = generate_test_signal(system, prn;
        num_samples = plan.num_coherently_integrated_code_periods * plan.samples_per_code,
        doppler = true_doppler, code_phase = 100.0,
        sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = 60)
    signal_flipped = ComplexF32.(signal)
    half = (plan.num_coherently_integrated_code_periods ÷ plan.num_data_bits) * plan.samples_per_code  # = 20 * 2048 = 40960
    signal_flipped[half+1:end] .*= -1

    Acquisition._precompute_signal_block_ffts!(plan.signal_block_ffts, signal_flipped,
        plan.samples_per_code, plan.num_blocks, plan.block_size,
        plan.num_coherently_integrated_code_periods, scratch.double_block_buf,
        plan.double_block_fft_plan)
    Acquisition._build_coherent_integration_matrix!(cim, plan.signal_block_ffts, plan.prn_conj_ffts[prn],
        plan.samples_per_code, plan.num_blocks, plan.block_size, plan.num_coherently_integrated_code_periods,
        scratch.corr_buf, plan.double_block_bfft_plan)

    noncoherent_integration_data = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    sub_block_ffts = zeros(ComplexF32, num_doppler_bins, plan.num_data_bits)
    patterns = Acquisition.sign_patterns(nothing, 0, plan.num_data_bits, 1, plan.num_coherently_integrated_code_periods, false)
    Acquisition._sign_search_step!(noncoherent_integration_data, cim, scratch.col_buf,
        plan.col_fft_plan, plan.samples_per_code, num_doppler_bins, plan.num_data_bits, 0, sub_block_ffts, patterns)

    # Pilot path: no bit search → the two halves cancel → weaker peak
    noncoherent_integration_pilot = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    Acquisition._accumulate_noncoherent_integration_pilot!(noncoherent_integration_pilot, cim, scratch.col_buf,
        plan.col_fft_plan, plan.samples_per_code)

    # Data bit search should recover a much stronger peak than naive pilot
    @test maximum(noncoherent_integration_data) > maximum(noncoherent_integration_pilot)

    # The detected Doppler (from the corrected peak) must be a valid alias
    peak_row, _ = Tuple(argmax(noncoherent_integration_data))
    detected_doppler = plan.doppler_freqs[peak_row]
    doppler_bin_spacing = step(plan.doppler_freqs)
    alias_period = plan.sampling_freq / plan.samples_per_code
    diff = mod((detected_doppler - true_doppler) / (1Hz), alias_period / (1Hz))
    diff_sym = min(diff, alias_period / (1Hz) - diff)
    @test diff_sym < doppler_bin_spacing / (1Hz)
end

@testset "Bit edge search — bit_edge_search_steps=4 does not degrade strong signal" begin
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    prn = 1
    # 3050 Hz: 3050/1000 = 3.05 (non-degenerate alias), 3050 mod 1000 = 50 Hz
    true_doppler = 3050Hz

    plan_no_search = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 40, bit_edge_search_steps = 1, num_noncoherent_accumulations = 1)
    plan_with_search = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 40, bit_edge_search_steps = 4, num_noncoherent_accumulations = 1)

    (; signal) = generate_test_signal(system, prn;
        num_samples = 40 * 2048,
        doppler = true_doppler, code_phase = 100.0,
        sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = 60)
    signal_f32 = ComplexF32.(signal)

    alias_period = plan_no_search.sampling_freq / plan_no_search.samples_per_code  # 1000 Hz

    # Drive each plan end-to-end through acquire! and verify the detected
    # Doppler is a valid alias of the true Doppler. The simple-path N_nc=1
    # plan now goes through the fused FFT+|x|²+fftshift kernel, so we can't
    # poke at the unfused noncoherent matrix directly the way we used to.
    for plan in (plan_no_search, plan_with_search)
        result = only(acquire!(plan, signal_f32, [prn]; interm_freq = 0.0Hz))
        detected_doppler = result.carrier_doppler
        doppler_bin_spacing = step(plan.doppler_freqs)
        # FM-DBZP alias ambiguity: check detected Doppler is a valid alias
        diff = mod((detected_doppler - true_doppler) / (1Hz), alias_period / (1Hz))
        diff_sym = min(diff, alias_period / (1Hz) - diff)
        @test diff_sym < doppler_bin_spacing / (1Hz)
    end
end

@testset "Code drift correction — _apply_code_drift! shifts row by expected amount" begin
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    plan = plan_acquire(system, sampling_freq, [1];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 40, bit_edge_search_steps = 1, num_noncoherent_accumulations = 200)
    num_doppler_bins = plan.num_coherently_integrated_code_periods * plan.num_blocks  # 640

    # `_apply_code_drift!` now operates on a buffer in **sorted-Doppler** row order
    # (the sign-search kernels fold fftshift into their cell-write loops, so the
    # buf reaches `_apply_code_drift!` already in sorted order — see
    # `_sign_search_step_with_rotations!`). The per-row Doppler is therefore
    # `doppler_freqs[row]` directly, no `fftshift_perm` lookup needed. To test
    # "shift at 7000 Hz Doppler" plant the spike at the sorted-Doppler bin for
    # 7 kHz.
    sorted_bin_7khz = findfirst(f -> abs(f / (1Hz) - 7000) < 1, plan.doppler_freqs)

    doppler_hz_7khz       = plan.doppler_freqs[sorted_bin_7khz] / (1Hz)    # ≈ 7000 Hz
    sampling_freq_hz_val  = plan.sampling_freq / (1Hz)
    carrier_freq_hz_val   = get_center_frequency(plan.system) / (1Hz)
    coherent_duration_s   = plan.num_coherently_integrated_code_periods * plan.samples_per_code / sampling_freq_hz_val  # 0.04 s
    expected_shift = round(Int, doppler_hz_7khz * 100 * coherent_duration_s * sampling_freq_hz_val / carrier_freq_hz_val)  # ≈ 36

    # Spike at (sorted_bin_7khz, 500) — all other entries zero
    noncoherent_integration_step0   = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    noncoherent_integration_step100 = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    noncoherent_integration_step0[sorted_bin_7khz, 500] = 1.0f0
    noncoherent_integration_step100[sorted_bin_7khz, 500] = 1.0f0

    Acquisition._apply_code_drift!(noncoherent_integration_step0, plan, 0)    # m=0: no change
    Acquisition._apply_code_drift!(noncoherent_integration_step100, plan, 100) # m=100: shift ≈ 36 samples

    _, peak_col_no_drift   = Tuple(argmax(noncoherent_integration_step0))
    _, peak_col_after_drift = Tuple(argmax(noncoherent_integration_step100))

    @test peak_col_no_drift == 500  # accumulation_step_index=0: no drift applied
    expected_col_after_drift = mod(500 + expected_shift - 1, plan.samples_per_code) + 1  # circular 1-indexed
    @test peak_col_after_drift == expected_col_after_drift
end

@testset "Non-coherent integration — detection probability increases and false alarm rate stays controlled" begin
    # At CN0=30 dB-Hz with 10ms coherent integration, M=1 rarely detects (~5%) but
    # M=8 almost always detects (>95%).  The CFAR threshold automatically adjusts for
    # M via num_noncoherent_integrations stored in the result, keeping P_fa ≈ pfa.
    #
    # Seed the global RNG so the noise arrays used for the false-alarm measurement
    # (below) are reproducible. Signal generation already takes explicit `seed=`
    # arguments; the noise calls use randn() which reads the global RNG.
    Random.seed!(20260423)
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    prn = 1
    false_alarm_rate = 0.01
    num_trials = 100
    CN0 = 30.0
    num_coherent_periods = 10
    samples_per_code = 2048

    plan_ni1 = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 10_000Hz,
        num_coherently_integrated_code_periods = num_coherent_periods,
        num_noncoherent_accumulations = 1)
    plan_ni8 = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 10_000Hz,
        num_coherently_integrated_code_periods = num_coherent_periods,
        num_noncoherent_accumulations = 8)

    num_detections_ni1 = 0; num_detections_ni8 = 0; num_detections_ni8_wrong_threshold = 0
    num_false_alarms_ni1 = 0; num_false_alarms_ni8 = 0

    for seed in 1:num_trials
        (; signal_ni1) = (; signal_ni1 = generate_test_signal(system, prn;
            num_samples = 1 * num_coherent_periods * samples_per_code,
            doppler = 1500Hz, code_phase = 100.0,
            sampling_freq = sampling_freq, interm_freq = 0.0Hz,
            CN0 = CN0, seed = seed).signal)
        (; signal_ni8) = (; signal_ni8 = generate_test_signal(system, prn;
            num_samples = 8 * num_coherent_periods * samples_per_code,
            doppler = 1500Hz, code_phase = 100.0,
            sampling_freq = sampling_freq, interm_freq = 0.0Hz,
            CN0 = CN0, seed = seed).signal)

        result_ni1 = only(acquire!(plan_ni1, signal_ni1, [prn]))
        result_ni8 = only(acquire!(plan_ni8, signal_ni8, [prn]))

        num_detections_ni1 += is_detected(result_ni1; pfa = false_alarm_rate) ? 1 : 0
        num_detections_ni8 += is_detected(result_ni8; pfa = false_alarm_rate) ? 1 : 0
        threshold_ni1 = cfar_threshold(false_alarm_rate, get_num_cells(result_ni8); num_noncoherent_integrations = 1)
        num_detections_ni8_wrong_threshold += result_ni8.peak_to_noise_ratio > threshold_ni1 ? 1 : 0

        noise_ni1 = (randn(1 * num_coherent_periods * samples_per_code) .+ im .* randn(1 * num_coherent_periods * samples_per_code)) ./ sqrt(2)
        noise_ni8 = (randn(8 * num_coherent_periods * samples_per_code) .+ im .* randn(8 * num_coherent_periods * samples_per_code)) ./ sqrt(2)
        num_false_alarms_ni1 += is_detected(only(acquire!(plan_ni1, noise_ni1, [prn])); pfa = false_alarm_rate) ? 1 : 0
        num_false_alarms_ni8 += is_detected(only(acquire!(plan_ni8, noise_ni8, [prn])); pfa = false_alarm_rate) ? 1 : 0
    end

    prob_detect_ni1       = num_detections_ni1 / num_trials
    prob_detect_ni8       = num_detections_ni8 / num_trials
    prob_detect_ni8_wrong = num_detections_ni8_wrong_threshold / num_trials
    prob_false_alarm_ni1  = num_false_alarms_ni1 / num_trials
    prob_false_alarm_ni8  = num_false_alarms_ni8 / num_trials

    @test prob_detect_ni8 > prob_detect_ni1
    @test prob_detect_ni1 < 0.3
    @test prob_detect_ni8 > 0.7
    @test prob_false_alarm_ni1 < 3 * false_alarm_rate
    @test prob_false_alarm_ni8 < 3 * false_alarm_rate
    @test prob_detect_ni8_wrong < prob_detect_ni8
end

@testset "Bit transition at 0 Hz Doppler — data bit search recovers same peak as no-transition signal" begin
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    prn    = 1

    signal_no_transition = generate_test_signal(system, prn;
        num_samples = 40 * 2048, doppler = 0Hz, code_phase = 0.0,
        sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = 60, seed = 42).signal

    signal_with_transition = copy(signal_no_transition)
    signal_with_transition[20*2048+1:end] .*= -1

    plan = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 40, bit_edge_search_steps = 1, num_noncoherent_accumulations = 1)
    scratch = Acquisition._default_scratch(plan)
    # Full CIM materialised locally via the reference wrapper (production is tiled).
    cim = zeros(ComplexF32, plan.num_coherently_integrated_code_periods * plan.num_blocks, plan.samples_per_code)
    num_doppler_bins = plan.num_coherently_integrated_code_periods * plan.num_blocks

    sub_block_ffts = zeros(ComplexF32, num_doppler_bins, plan.num_data_bits)

    Acquisition._precompute_signal_block_ffts!(plan.signal_block_ffts, ComplexF32.(signal_no_transition),
        plan.samples_per_code, plan.num_blocks, plan.block_size,
        plan.num_coherently_integrated_code_periods, scratch.double_block_buf,
        plan.double_block_fft_plan)
    Acquisition._build_coherent_integration_matrix!(cim, plan.signal_block_ffts,
        plan.prn_conj_ffts[prn], plan.samples_per_code, plan.num_blocks, plan.block_size,
        plan.num_coherently_integrated_code_periods, scratch.corr_buf,
        plan.double_block_bfft_plan)
    noncoherent_integration_no_transition = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    Acquisition._accumulate_noncoherent_integration_pilot!(noncoherent_integration_no_transition, cim, scratch.col_buf,
        plan.col_fft_plan, plan.samples_per_code)
    peak_no_transition = maximum(noncoherent_integration_no_transition)

    Acquisition._precompute_signal_block_ffts!(plan.signal_block_ffts, ComplexF32.(signal_with_transition),
        plan.samples_per_code, plan.num_blocks, plan.block_size,
        plan.num_coherently_integrated_code_periods, scratch.double_block_buf,
        plan.double_block_fft_plan)
    Acquisition._build_coherent_integration_matrix!(cim, plan.signal_block_ffts,
        plan.prn_conj_ffts[prn], plan.samples_per_code, plan.num_blocks, plan.block_size,
        plan.num_coherently_integrated_code_periods, scratch.corr_buf,
        plan.double_block_bfft_plan)
    noncoherent_integration_with_transition = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    patterns = Acquisition.sign_patterns(nothing, 0, plan.num_data_bits, 1, plan.num_coherently_integrated_code_periods, false)
    Acquisition._sign_search_step!(noncoherent_integration_with_transition, cim, scratch.col_buf, plan.col_fft_plan, plan.samples_per_code, num_doppler_bins,
        plan.num_data_bits, 0, sub_block_ffts, patterns)
    peak_with_transition = maximum(noncoherent_integration_with_transition)

    @test peak_with_transition ≈ peak_no_transition rtol = 0.1
end

@testset "Multiple bit transitions at 0 Hz Doppler — 3 bits with pattern +1 −1 +1" begin
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    prn    = 1

    signal_no_transition = generate_test_signal(system, prn;
        num_samples = 60 * 2048, doppler = 0Hz, code_phase = 0.0,
        sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = 60, seed = 42).signal

    signal_with_transition = copy(signal_no_transition)
    signal_with_transition[20*2048+1:40*2048] .*= -1

    plan = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 60, bit_edge_search_steps = 1, num_noncoherent_accumulations = 1)
    scratch = Acquisition._default_scratch(plan)
    # Full CIM materialised locally via the reference wrapper (production is tiled).
    cim = zeros(ComplexF32, plan.num_coherently_integrated_code_periods * plan.num_blocks, plan.samples_per_code)
    num_doppler_bins = plan.num_coherently_integrated_code_periods * plan.num_blocks

    sub_block_ffts = zeros(ComplexF32, num_doppler_bins, plan.num_data_bits)

    Acquisition._precompute_signal_block_ffts!(plan.signal_block_ffts, ComplexF32.(signal_no_transition),
        plan.samples_per_code, plan.num_blocks, plan.block_size,
        plan.num_coherently_integrated_code_periods, scratch.double_block_buf,
        plan.double_block_fft_plan)
    Acquisition._build_coherent_integration_matrix!(cim, plan.signal_block_ffts,
        plan.prn_conj_ffts[prn], plan.samples_per_code, plan.num_blocks, plan.block_size,
        plan.num_coherently_integrated_code_periods, scratch.corr_buf,
        plan.double_block_bfft_plan)
    noncoherent_integration_no_transition = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    Acquisition._accumulate_noncoherent_integration_pilot!(noncoherent_integration_no_transition, cim, scratch.col_buf,
        plan.col_fft_plan, plan.samples_per_code)
    peak_no_transition = maximum(noncoherent_integration_no_transition)

    Acquisition._precompute_signal_block_ffts!(plan.signal_block_ffts, ComplexF32.(signal_with_transition),
        plan.samples_per_code, plan.num_blocks, plan.block_size,
        plan.num_coherently_integrated_code_periods, scratch.double_block_buf,
        plan.double_block_fft_plan)
    Acquisition._build_coherent_integration_matrix!(cim, plan.signal_block_ffts,
        plan.prn_conj_ffts[prn], plan.samples_per_code, plan.num_blocks, plan.block_size,
        plan.num_coherently_integrated_code_periods, scratch.corr_buf,
        plan.double_block_bfft_plan)
    noncoherent_integration_with_transition = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    patterns = Acquisition.sign_patterns(nothing, 0, plan.num_data_bits, 1, plan.num_coherently_integrated_code_periods, false)
    Acquisition._sign_search_step!(noncoherent_integration_with_transition, cim, scratch.col_buf, plan.col_fft_plan, plan.samples_per_code, num_doppler_bins,
        plan.num_data_bits, 0, sub_block_ffts, patterns)
    peak_with_transition = maximum(noncoherent_integration_with_transition)

    @test peak_with_transition ≈ peak_no_transition rtol = 0.1
end

@testset "Tiled production pipeline matches materialised reference dispatcher (N_nc>1)" begin
    # The production multistep driver `_accumulate_prn_step_tiled!` builds one
    # column-block tile at a time and folds code drift into the destination
    # scatter; `_accumulate_noncoherent_integration_step!` is the retained
    # reference that materialises the full coherent matrix and runs the
    # historical kernel → `_apply_code_drift!` → max/accumulate pipeline. Both
    # must produce the same per-PRN accumulation matrix on every path.
    prn = 1
    cases = [
        # (label, plan, accumulation_step_index)
        # Simple path with non-zero drift (plan_b geometry from the
        # _fill_code_drift_shifts! testset: drift rounds non-zero at step 7).
        ("pilot+drift", plan_acquire(GPSL1CA(), 2.048e6Hz, [prn];
            min_doppler_coverage = 12_000Hz,
            num_coherently_integrated_code_periods = 20,
            num_noncoherent_accumulations = 8, fft_flag = FFTW.ESTIMATE), 7),
        # Secondary-code rotation search with drift (L5I NH10).
        ("rotation+drift", plan_acquire(GPSL5I(), 10.24e6Hz, [prn];
            num_coherently_integrated_code_periods = 10,
            num_noncoherent_accumulations = 4, fft_flag = FFTW.ESTIMATE), 3),
        # Data-bit + bit-edge search (max across alignments), no drift at step 0.
        ("bitedge", plan_acquire(GPSL1CA(), 2.048e6Hz, [prn];
            min_doppler_coverage = 10_000Hz,
            num_coherently_integrated_code_periods = 40, bit_edge_search_steps = 4,
            num_noncoherent_accumulations = 2, fft_flag = FFTW.ESTIMATE), 0),
    ]
    for (label, plan, step_idx) in cases
        @testset "$label" begin
            scratch = Acquisition._default_scratch(plan)
            ndop = length(plan.doppler_freqs)
            (; signal) = generate_test_signal(plan.system, prn;
                num_samples = plan.num_coherently_integrated_code_periods * plan.samples_per_code,
                doppler = 1300Hz, code_phase = 210.7,
                sampling_freq = plan.sampling_freq, interm_freq = 0.0Hz, CN0 = 45, seed = 11)
            plan.sig_buf .= ComplexF32.(signal)
            Acquisition._precompute_signal_block_ffts!(plan.signal_block_ffts, plan.sig_buf,
                plan.samples_per_code, plan.num_blocks, plan.block_size,
                plan.num_coherently_integrated_code_periods, scratch.double_block_buf,
                plan.double_block_fft_plan)

            # Production: tiled driver.
            nim_tiled = zeros(Float32, ndop, plan.samples_per_code_eff)
            Acquisition._accumulate_prn_step_tiled!(nim_tiled, plan, scratch, prn, step_idx)

            # Reference: full CIM + materialised dispatcher.
            cim = zeros(ComplexF32, ndop, plan.samples_per_code)
            Acquisition._build_coherent_integration_matrix!(cim, plan.signal_block_ffts,
                plan.prn_conj_ffts[prn], plan.samples_per_code, plan.num_blocks,
                plan.block_size, plan.num_coherently_integrated_code_periods,
                scratch.corr_buf, plan.double_block_bfft_plan)
            nim_ref = zeros(Float32, ndop, plan.samples_per_code_eff)
            Acquisition._accumulate_noncoherent_integration_step!(nim_ref, cim, plan,
                scratch, prn, step_idx)

            @test nim_tiled ≈ nim_ref
            # Sanity: drift really fires where intended so the fold is exercised.
            if step_idx > 0
                shifts = zeros(Int, ndop)
                @test Acquisition._fill_code_drift_shifts!(shifts, plan, step_idx) ==
                    (label != "bitedge")
            end
        end
    end
end
