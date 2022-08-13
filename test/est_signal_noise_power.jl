@testset "Estimate signal / noise power" begin
    # Peak in the middle
    power_bins = abs2.(1 / sqrt(2) * complex.(ones(1023,29), ones(1023,29)))
    power_bins[489,11] = power_bins[489,11] + 10^(15 / 10) # 15 dB SNR
    signal_power, noise_power, code_index, doppler_index = Acquisition.est_signal_noise_power(power_bins, 4e6, 1e6)

    @test noise_power ≈ 1
    @test signal_power ≈ 10^(15 / 10)
    @test code_index == 489
    @test doppler_index == 11

    # Peak at doppler and code border
    power_bins = abs2.(1 / sqrt(2) * complex.(ones(1023,29), ones(1023,29)))
    power_bins[1,1] = power_bins[1,1] + 10^(15 / 10) # 15 dB SNR
    signal_power, noise_power, code_index, doppler_index = Acquisition.est_signal_noise_power(power_bins, 4e6, 1e6)

    @test noise_power ≈ 1
    @test signal_power ≈ 10^(15 / 10)
    @test code_index == 1
    @test doppler_index == 1

    power_bins = abs2.(1 / sqrt(2) * complex.(ones(1023,29), ones(1023,29)))
    power_bins[1023,29] = power_bins[1023,29] + 10^(15 / 10) # 15 dB SNR
    signal_power, noise_power, code_index, doppler_index = Acquisition.est_signal_noise_power(power_bins, 4e6, 1e6)

    @test noise_power ≈ 1
    @test signal_power ≈ 10^(15 / 10)
    @test code_index == 1023
    @test doppler_index == 29

    #plotlyjs()
    #surface(-7000:500:7000, 1:1023, power_bins)
    #gui()
end