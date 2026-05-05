@testset "Estimate signal / noise power" begin
    # power_bins layout: (num_doppler_bins, samples_per_code) — rows=Doppler, cols=code phase

    @testset "OppositeRowNoiseEstimator (default)" begin
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

        # Doppler-conditional banding: opposite row should track the noise floor
        # of that row (not the inflated peak row). Build a matrix where most
        # rows are at noise level 1, but row 11 is artificially inflated to 5.
        # Peak at row 11 (which sits in the inflated band). The opposite row
        # (11 + 29÷2 = 25) should have noise level 1.
        power_bins = ones(Float64, 29, 1023)
        power_bins[11, :] .= 5.0  # inflated band
        power_bins[11, 489] = 5.0 + 10^(15 / 10)
        col_sums = zeros(eltype(power_bins), size(power_bins, 2))
        signal_power, noise_power, code_index, doppler_index =
            Acquisition.est_signal_noise_power(power_bins, 4e6, 1e6, col_sums,
                Acquisition.OppositeRowNoiseEstimator())

        @test noise_power ≈ 1   # opposite row, not contaminated by the band
        @test signal_power ≈ 5.0 + 10^(15 / 10) - 1
        @test code_index == 489
        @test doppler_index == 11
    end

    @testset "GlobalMeanNoiseEstimator" begin
        # Peak in the middle
        power_bins = abs2.(1 / sqrt(2) * complex.(ones(29, 1023), ones(29, 1023)))
        power_bins[11, 489] = power_bins[11, 489] + 10^(15 / 10) # 15 dB SNR
        col_sums = zeros(eltype(power_bins), size(power_bins, 2))
        signal_power, noise_power, code_index, doppler_index =
            @inferred Acquisition.est_signal_noise_power(power_bins, 4e6, 1e6, col_sums,
                Acquisition.GlobalMeanNoiseEstimator())

        @test noise_power ≈ 1
        @test signal_power ≈ 10^(15 / 10)
        @test code_index == 489
        @test doppler_index == 11

        # Peak at doppler and code border
        power_bins = abs2.(1 / sqrt(2) * complex.(ones(29, 1023), ones(29, 1023)))
        power_bins[1, 1] = power_bins[1, 1] + 10^(15 / 10) # 15 dB SNR
        col_sums = zeros(eltype(power_bins), size(power_bins, 2))
        signal_power, noise_power, code_index, doppler_index =
            @inferred Acquisition.est_signal_noise_power(power_bins, 4e6, 1e6, col_sums,
                Acquisition.GlobalMeanNoiseEstimator())

        @test noise_power ≈ 1
        @test signal_power ≈ 10^(15 / 10)
        @test code_index == 1
        @test doppler_index == 1

        power_bins = abs2.(1 / sqrt(2) * complex.(ones(29, 1023), ones(29, 1023)))
        power_bins[29, 1023] = power_bins[29, 1023] + 10^(15 / 10) # 15 dB SNR
        col_sums = zeros(eltype(power_bins), size(power_bins, 2))
        signal_power, noise_power, code_index, doppler_index =
            @inferred Acquisition.est_signal_noise_power(power_bins, 4e6, 1e6, col_sums,
                Acquisition.GlobalMeanNoiseEstimator())

        @test noise_power ≈ 1
        @test signal_power ≈ 10^(15 / 10)
        @test code_index == 1023
        @test doppler_index == 29
    end
end
