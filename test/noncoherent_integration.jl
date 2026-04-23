# test/noncoherent_integration.jl
# Tests for noncoherent integration: column FFT accumulation, data bit search, code drift correction

@testset "Column FFT accumulation — Tier 2: code delay and Doppler alias check" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz
    prn = 1
    # 1050 Hz: f_D*samples_per_code/sampling_freq = 1.05 (non-integer → non-degenerate), 1050 mod 1000 = 50 Hz
    true_doppler = 1050Hz
    code_phase = 973.0  # tau = 1023 - 973 = 50 chips → tau_samples = round(50*2048/1023) = 100

    # num_coherently_integrated_code_periods=20 = one GPS data bit period (num_data_bits=1, pilot path)
    plan = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 20)
    doppler_bin_spacing = step(plan.doppler_freqs)
    tau_samples = round(Int, (get_code_length(system) - code_phase) * plan.samples_per_code / get_code_length(system))

    (; signal) = generate_test_signal(system, prn;
        num_samples = 20 * plan.samples_per_code,
        doppler = true_doppler, code_phase = code_phase,
        sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = 45)

    signal_f32 = ComplexF32.(signal)
    fill!(plan.coherent_integration_matrix, zero(ComplexF32))
    Acquisition._build_coherent_integration_matrix!(plan.coherent_integration_matrix, signal_f32, plan.prn_conj_ffts[prn],
        plan.samples_per_code, plan.num_blocks, plan.block_size, plan.num_coherently_integrated_code_periods,
        plan.double_block_buf, plan.corr_buf, plan.double_block_fft_plan, plan.double_block_bfft_plan)

    noncoherent_integration_matrix = zeros(Float32, plan.num_coherently_integrated_code_periods * plan.num_blocks, plan.samples_per_code)
    Acquisition._accumulate_noncoherent_integration_pilot!(noncoherent_integration_matrix, plan.coherent_integration_matrix, plan.col_buf,
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
    Acquisition._accumulate_noncoherent_integration_pilot!(noncoherent_integration_matrix, plan.coherent_integration_matrix, plan.col_buf,
        plan.col_fft_plan, plan.samples_per_code)
    @test noncoherent_integration_matrix[peak_row, peak_col] ≈ 2 * peak_value_first
end

@testset "Data bit combination search — num_data_bits=2, no bit transitions in test signal" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz
    prn = 1
    true_doppler = 1500Hz

    # num_coherently_integrated_code_periods=40 → num_data_bits=2 (two GPS data bit periods)
    plan = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 40, bit_edge_search_steps = 1, num_noncoherent_accumulations = 1)
    num_doppler_bins = plan.num_coherently_integrated_code_periods * plan.num_blocks  # 640

    code_phase = 100.0
    tau_samples = round(Int, (get_code_length(system) - code_phase) * plan.samples_per_code / get_code_length(system))

    (; signal) = generate_test_signal(system, prn;
        num_samples = plan.num_coherently_integrated_code_periods * plan.samples_per_code,
        doppler = true_doppler, code_phase = code_phase,
        sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = 60)

    signal_f32 = ComplexF32.(signal)
    fill!(plan.coherent_integration_matrix, zero(ComplexF32))
    Acquisition._build_coherent_integration_matrix!(plan.coherent_integration_matrix, signal_f32, plan.prn_conj_ffts[prn],
        plan.samples_per_code, plan.num_blocks, plan.block_size, plan.num_coherently_integrated_code_periods,
        plan.double_block_buf, plan.corr_buf, plan.double_block_fft_plan, plan.double_block_bfft_plan)

    noncoherent_integration_matrix = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    noncoherent_integration_buf = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    sub_block_ffts = zeros(ComplexF32, num_doppler_bins, plan.num_data_bits)
    Acquisition._accumulate_noncoherent_integration_data_bits!(noncoherent_integration_buf, plan.coherent_integration_matrix, plan.col_buf, plan.col_fftshift_buf,
        plan.col_fft_plan, plan.samples_per_code, num_doppler_bins, plan.num_data_bits, 0, sub_block_ffts)
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
    system = GPSL1()
    sampling_freq = 2.048e6Hz
    prn = 1
    true_doppler = 1050Hz  # 1050 mod 1000 = 50 Hz: non-degenerate

    plan = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 40, bit_edge_search_steps = 1, num_noncoherent_accumulations = 1)
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

    fill!(plan.coherent_integration_matrix, zero(ComplexF32))
    Acquisition._build_coherent_integration_matrix!(plan.coherent_integration_matrix, signal_flipped, plan.prn_conj_ffts[prn],
        plan.samples_per_code, plan.num_blocks, plan.block_size, plan.num_coherently_integrated_code_periods,
        plan.double_block_buf, plan.corr_buf, plan.double_block_fft_plan, plan.double_block_bfft_plan)

    noncoherent_integration_data = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    sub_block_ffts = zeros(ComplexF32, num_doppler_bins, plan.num_data_bits)
    Acquisition._accumulate_noncoherent_integration_data_bits!(noncoherent_integration_data, plan.coherent_integration_matrix, plan.col_buf, plan.col_fftshift_buf,
        plan.col_fft_plan, plan.samples_per_code, num_doppler_bins, plan.num_data_bits, 0, sub_block_ffts)

    # Pilot path: no bit search → the two halves cancel → weaker peak
    noncoherent_integration_pilot = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    Acquisition._accumulate_noncoherent_integration_pilot!(noncoherent_integration_pilot, plan.coherent_integration_matrix, plan.col_buf,
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
    system = GPSL1()
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

    for plan in (plan_no_search, plan_with_search)
        num_doppler_bins = plan.num_coherently_integrated_code_periods * plan.num_blocks
        fill!(plan.coherent_integration_matrix, zero(ComplexF32))
        Acquisition._build_coherent_integration_matrix!(plan.coherent_integration_matrix, signal_f32, plan.prn_conj_ffts[prn],
            plan.samples_per_code, plan.num_blocks, plan.block_size, plan.num_coherently_integrated_code_periods,
            plan.double_block_buf, plan.corr_buf, plan.double_block_fft_plan, plan.double_block_bfft_plan)
        noncoherent_integration_matrix = zeros(Float32, num_doppler_bins, plan.samples_per_code)
        Acquisition._accumulate_noncoherent_integration_step!(noncoherent_integration_matrix, plan.coherent_integration_matrix, plan, 0)

        peak_row, _ = Tuple(argmax(noncoherent_integration_matrix))
        detected_doppler = plan.doppler_freqs[peak_row]
        doppler_bin_spacing = step(plan.doppler_freqs)
        # FM-DBZP alias ambiguity: check detected Doppler is a valid alias
        diff = mod((detected_doppler - true_doppler) / (1Hz), alias_period / (1Hz))
        diff_sym = min(diff, alias_period / (1Hz) - diff)
        @test diff_sym < doppler_bin_spacing / (1Hz)
    end
end

@testset "Code drift correction — _apply_code_drift! shifts row by expected amount" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz
    plan = plan_acquire(system, sampling_freq, [1];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 40, bit_edge_search_steps = 1, num_noncoherent_accumulations = 200)
    num_doppler_bins = plan.num_coherently_integrated_code_periods * plan.num_blocks  # 640

    # Row corresponding to doppler_hz ≈ 7000 Hz
    doppler_row_7khz = findfirst(f -> abs(f / (1Hz) - 7000) < 1, plan.doppler_freqs)

    doppler_hz_7khz       = plan.doppler_freqs[doppler_row_7khz] / (1Hz)   # ≈ 7000 Hz
    sampling_freq_hz_val  = plan.sampling_freq / (1Hz)
    carrier_freq_hz_val   = get_center_frequency(plan.system) / (1Hz)
    coherent_duration_s   = plan.num_coherently_integrated_code_periods * plan.samples_per_code / sampling_freq_hz_val  # 0.04 s
    expected_shift = round(Int, doppler_hz_7khz * 100 * coherent_duration_s * sampling_freq_hz_val / carrier_freq_hz_val)  # ≈ 36

    # Spike at (doppler_row_7khz, 500) — all other entries zero
    noncoherent_integration_step0   = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    noncoherent_integration_step100 = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    noncoherent_integration_step0[doppler_row_7khz, 500] = 1.0f0
    noncoherent_integration_step100[doppler_row_7khz, 500] = 1.0f0

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
    system = GPSL1()
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
    system = GPSL1()
    sampling_freq = 2.048e6Hz
    prn    = 1

    signal_no_transition = generate_test_signal(system, prn;
        num_samples = 40 * 2048, doppler = 0Hz, code_phase = 0.0,
        sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = 60, seed = 42).signal

    signal_with_transition = copy(signal_no_transition)
    signal_with_transition[20*2048+1:end] .*= -1

    plan = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 40, bit_edge_search_steps = 1, num_noncoherent_accumulations = 1)
    num_doppler_bins = plan.num_coherently_integrated_code_periods * plan.num_blocks

    sub_block_ffts = zeros(ComplexF32, num_doppler_bins, plan.num_data_bits)

    fill!(plan.coherent_integration_matrix, zero(ComplexF32))
    Acquisition._build_coherent_integration_matrix!(plan.coherent_integration_matrix, ComplexF32.(signal_no_transition),
        plan.prn_conj_ffts[prn], plan.samples_per_code, plan.num_blocks, plan.block_size,
        plan.num_coherently_integrated_code_periods, plan.double_block_buf, plan.corr_buf,
        plan.double_block_fft_plan, plan.double_block_bfft_plan)
    noncoherent_integration_no_transition = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    Acquisition._accumulate_noncoherent_integration_pilot!(noncoherent_integration_no_transition, plan.coherent_integration_matrix, plan.col_buf,
        plan.col_fft_plan, plan.samples_per_code)
    peak_no_transition = maximum(noncoherent_integration_no_transition)

    fill!(plan.coherent_integration_matrix, zero(ComplexF32))
    Acquisition._build_coherent_integration_matrix!(plan.coherent_integration_matrix, ComplexF32.(signal_with_transition),
        plan.prn_conj_ffts[prn], plan.samples_per_code, plan.num_blocks, plan.block_size,
        plan.num_coherently_integrated_code_periods, plan.double_block_buf, plan.corr_buf,
        plan.double_block_fft_plan, plan.double_block_bfft_plan)
    noncoherent_integration_with_transition = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    Acquisition._accumulate_noncoherent_integration_data_bits!(noncoherent_integration_with_transition, plan.coherent_integration_matrix, plan.col_buf,
        plan.col_fftshift_buf, plan.col_fft_plan, plan.samples_per_code, num_doppler_bins,
        plan.num_data_bits, 0, sub_block_ffts)
    peak_with_transition = maximum(noncoherent_integration_with_transition)

    @test peak_with_transition ≈ peak_no_transition rtol = 0.1
end

@testset "Multiple bit transitions at 0 Hz Doppler — 3 bits with pattern +1 −1 +1" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz
    prn    = 1

    signal_no_transition = generate_test_signal(system, prn;
        num_samples = 60 * 2048, doppler = 0Hz, code_phase = 0.0,
        sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = 60, seed = 42).signal

    signal_with_transition = copy(signal_no_transition)
    signal_with_transition[20*2048+1:40*2048] .*= -1

    plan = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 10_000Hz, num_coherently_integrated_code_periods = 60, bit_edge_search_steps = 1, num_noncoherent_accumulations = 1)
    num_doppler_bins = plan.num_coherently_integrated_code_periods * plan.num_blocks

    sub_block_ffts = zeros(ComplexF32, num_doppler_bins, plan.num_data_bits)

    fill!(plan.coherent_integration_matrix, zero(ComplexF32))
    Acquisition._build_coherent_integration_matrix!(plan.coherent_integration_matrix, ComplexF32.(signal_no_transition),
        plan.prn_conj_ffts[prn], plan.samples_per_code, plan.num_blocks, plan.block_size,
        plan.num_coherently_integrated_code_periods, plan.double_block_buf, plan.corr_buf,
        plan.double_block_fft_plan, plan.double_block_bfft_plan)
    noncoherent_integration_no_transition = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    Acquisition._accumulate_noncoherent_integration_pilot!(noncoherent_integration_no_transition, plan.coherent_integration_matrix, plan.col_buf,
        plan.col_fft_plan, plan.samples_per_code)
    peak_no_transition = maximum(noncoherent_integration_no_transition)

    fill!(plan.coherent_integration_matrix, zero(ComplexF32))
    Acquisition._build_coherent_integration_matrix!(plan.coherent_integration_matrix, ComplexF32.(signal_with_transition),
        plan.prn_conj_ffts[prn], plan.samples_per_code, plan.num_blocks, plan.block_size,
        plan.num_coherently_integrated_code_periods, plan.double_block_buf, plan.corr_buf,
        plan.double_block_fft_plan, plan.double_block_bfft_plan)
    noncoherent_integration_with_transition = zeros(Float32, num_doppler_bins, plan.samples_per_code)
    Acquisition._accumulate_noncoherent_integration_data_bits!(noncoherent_integration_with_transition, plan.coherent_integration_matrix, plan.col_buf,
        plan.col_fftshift_buf, plan.col_fft_plan, plan.samples_per_code, num_doppler_bins,
        plan.num_data_bits, 0, sub_block_ffts)
    peak_with_transition = maximum(noncoherent_integration_with_transition)

    @test peak_with_transition ≈ peak_no_transition rtol = 0.1
end
