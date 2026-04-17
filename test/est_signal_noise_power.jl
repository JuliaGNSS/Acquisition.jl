@testset "Estimate signal / noise power" begin
    # power_bins layout: (num_doppler_bins, samples_per_code) — rows=Doppler, cols=code phase

    # Peak in the middle
    power_bins = abs2.(1 / sqrt(2) * complex.(ones(29, 1023), ones(29, 1023)))
    power_bins[11, 489] = power_bins[11, 489] + 10^(15 / 10) # 15 dB SNR
    col_sums = zeros(eltype(power_bins), size(power_bins, 2))
    signal_power, noise_power, code_index, doppler_index =
        @inferred Acquisition.est_signal_noise_power(power_bins, 4e6, 1e6, col_sums)

    @test noise_power ≈ 1
    @test signal_power ≈ 10^(15 / 10)
    @test code_index == 489
    @test doppler_index == 11

    # Peak at doppler and code border
    power_bins = abs2.(1 / sqrt(2) * complex.(ones(29, 1023), ones(29, 1023)))
    power_bins[1, 1] = power_bins[1, 1] + 10^(15 / 10) # 15 dB SNR
    col_sums = zeros(eltype(power_bins), size(power_bins, 2))
    signal_power, noise_power, code_index, doppler_index =
        @inferred Acquisition.est_signal_noise_power(power_bins, 4e6, 1e6, col_sums)

    @test noise_power ≈ 1
    @test signal_power ≈ 10^(15 / 10)
    @test code_index == 1
    @test doppler_index == 1

    power_bins = abs2.(1 / sqrt(2) * complex.(ones(29, 1023), ones(29, 1023)))
    power_bins[29, 1023] = power_bins[29, 1023] + 10^(15 / 10) # 15 dB SNR
    col_sums = zeros(eltype(power_bins), size(power_bins, 2))
    signal_power, noise_power, code_index, doppler_index =
        @inferred Acquisition.est_signal_noise_power(power_bins, 4e6, 1e6, col_sums)

    @test noise_power ≈ 1
    @test signal_power ≈ 10^(15 / 10)
    @test code_index == 1023
    @test doppler_index == 29
end
