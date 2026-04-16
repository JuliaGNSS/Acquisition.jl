@testset "est_signal_noise_power — known noise_power_in overrides estimation" begin
    # Exercise the else-branch: noise_power_in != nothing
    power_bins = abs2.(1 / sqrt(2) * complex.(ones(5, 100), ones(5, 100)))
    power_bins[3, 50] += 10^(20 / 10)  # strong peak
    known_noise = 1.0f0

    signal_power, noise_power, code_index, doppler_index =
        @inferred Acquisition.est_signal_noise_power(power_bins, 4e6, 1e6, known_noise)

    @test noise_power == known_noise
    @test signal_power ≈ power_bins[3, 50] - known_noise
    @test code_index == 50
    @test doppler_index == 3
end

@testset "Estimate signal / noise power" begin
    # power_bins layout: (num_doppler_bins, samples_per_code) — rows=Doppler, cols=code phase

    # Peak in the middle
    power_bins = abs2.(1 / sqrt(2) * complex.(ones(29, 1023), ones(29, 1023)))
    power_bins[11, 489] = power_bins[11, 489] + 10^(15 / 10) # 15 dB SNR
    signal_power, noise_power, code_index, doppler_index =
        @inferred Acquisition.est_signal_noise_power(power_bins, 4e6, 1e6, nothing)

    @test noise_power ≈ 1
    @test signal_power ≈ 10^(15 / 10)
    @test code_index == 489
    @test doppler_index == 11

    # Peak at doppler and code border
    power_bins = abs2.(1 / sqrt(2) * complex.(ones(29, 1023), ones(29, 1023)))
    power_bins[1, 1] = power_bins[1, 1] + 10^(15 / 10) # 15 dB SNR
    signal_power, noise_power, code_index, doppler_index =
        @inferred Acquisition.est_signal_noise_power(power_bins, 4e6, 1e6, nothing)

    @test noise_power ≈ 1
    @test signal_power ≈ 10^(15 / 10)
    @test code_index == 1
    @test doppler_index == 1

    power_bins = abs2.(1 / sqrt(2) * complex.(ones(29, 1023), ones(29, 1023)))
    power_bins[29, 1023] = power_bins[29, 1023] + 10^(15 / 10) # 15 dB SNR
    signal_power, noise_power, code_index, doppler_index =
        @inferred Acquisition.est_signal_noise_power(power_bins, 4e6, 1e6, nothing)

    @test noise_power ≈ 1
    @test signal_power ≈ 10^(15 / 10)
    @test code_index == 1023
    @test doppler_index == 29
end
