# test/coherent_integration.jl
# Tests for _build_coherent_integration_matrix!

@testset "Coherent integration matrix assembly — Tier 1: peak at known code delay" begin
    system = GPSL1()
    sampling_freq = 2.048e6Hz
    prn = 1

    plan = plan_acquire(system, sampling_freq, [prn];
        min_doppler_coverage = 5_000Hz, num_coherently_integrated_code_periods = 1)
    # min_doppler_coverage=5000 → span 10000 Hz → num_blocks=16, block_size=128

    # code_phase=991 chips → tau_samples = round((1023-991)*2048/1023) = 64 samples
    # (must fit within one block, i.e. < block_size=128)
    code_phase_1 = 991.0
    tau_samples_1 = round(Int, (get_code_length(system) - code_phase_1) * plan.samples_per_code / get_code_length(system))

    # Strong signal, zero doppler, zero interm_freq for clean coherent integration matrix test
    (; signal) = generate_test_signal(system, prn;
        num_samples = plan.samples_per_code,
        doppler = 0Hz, code_phase = code_phase_1,
        sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = 100)

    signal_f32 = ComplexF32.(signal)
    fill!(plan.coherent_integration_matrix, zero(ComplexF32))
    Acquisition._build_coherent_integration_matrix!(plan.coherent_integration_matrix, signal_f32, plan.prn_conj_ffts[prn],
        plan.samples_per_code, plan.num_blocks, plan.block_size, plan.num_coherently_integrated_code_periods,
        plan.double_block_buf, plan.corr_buf, plan.double_block_fft_plan, plan.double_block_bfft_plan)

    # In the full coherent integration matrix structure ALL rows peak at the same column:
    #   pc (0-indexed) = mod(num_blocks - tau÷block_size, num_blocks) * block_size + (tau%block_size)
    # For tau_samples=64 (< block_size=128): pc = mod(16-0,16)*128 + 64 = 64, peak_col = 65 = tau+1.
    k_peak_1 = tau_samples_1 ÷ plan.block_size  # = 0
    expected_pc_1 = mod(plan.num_blocks - k_peak_1, plan.num_blocks) * plan.block_size + (tau_samples_1 % plan.block_size)
    expected_peak_col_1 = expected_pc_1 + 1
    for code_period in 0:plan.num_coherently_integrated_code_periods-1
        row = code_period * plan.num_blocks + 1  # check row 1 per code period
        row_vals = abs.(plan.coherent_integration_matrix[row, :])
        _, peak_col = findmax(row_vals)
        @test peak_col == expected_peak_col_1
    end

    # Second case: code_phase=923.097656 chips → tau_samples = round((1023-923.097656)*2048/1023) = 200 (k=1, fine=72).
    # pc = mod(16-1,16)*128 + 72 = 15*128+72 = 1992, so peak is in column block 15 (columns 1921..2048).
    code_phase_2 = 923.097656
    tau_samples_2 = round(Int, (get_code_length(system) - code_phase_2) * plan.samples_per_code / get_code_length(system))
    result_2 = generate_test_signal(system, prn;
        num_samples = plan.samples_per_code,
        doppler = 0Hz, code_phase = code_phase_2,
        sampling_freq = sampling_freq, interm_freq = 0.0Hz, CN0 = 100)
    signal_f32_2 = ComplexF32.(result_2.signal)
    fill!(plan.coherent_integration_matrix, zero(ComplexF32))
    Acquisition._build_coherent_integration_matrix!(plan.coherent_integration_matrix, signal_f32_2, plan.prn_conj_ffts[prn],
        plan.samples_per_code, plan.num_blocks, plan.block_size, plan.num_coherently_integrated_code_periods,
        plan.double_block_buf, plan.corr_buf, plan.double_block_fft_plan, plan.double_block_bfft_plan)
    block_row_2 = tau_samples_2 ÷ plan.block_size
    @test block_row_2 == 1  # Verify we're testing block_row > 0 (non-trivial column block)
    scrambled_block_2    = mod(plan.num_blocks - block_row_2, plan.num_blocks)  # = 15
    scrambled_col_2      = scrambled_block_2 * plan.block_size + (tau_samples_2 % plan.block_size)
    expected_peak_col_2  = scrambled_col_2 + 1
    for code_period_2 in 0:plan.num_coherently_integrated_code_periods-1
        row_2 = code_period_2 * plan.num_blocks + 1
        row_vals_2 = abs.(plan.coherent_integration_matrix[row_2, :])
        _, peak_col_2 = findmax(row_vals_2)
        @test peak_col_2 == expected_peak_col_2
    end
end
