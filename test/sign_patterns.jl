@testset "sign_patterns" begin
    @testset "L5I N=4, num_secondary_rotations=1, num_data_bits=1 → 1 column of all +1" begin
        result = Acquisition.sign_patterns(nothing, 1, 1, 1, 4, false)
        @test eltype(result) == Float32
        @test size(result) == (4, 1)
        @test result == ones(Float32, 4, 1)
    end

    @testset "L5I N=10, num_secondary_rotations=1, num_data_bits=1 → 1 column of all +1" begin
        result = Acquisition.sign_patterns(nothing, 1, 1, 1, 10, false)
        @test size(result) == (10, 1)
        @test result == ones(Float32, 10, 1)
    end

    @testset "L5I N=20, num_secondary_rotations=1, num_data_bits=2 → 2 columns of data-bit polarities" begin
        result = Acquisition.sign_patterns(nothing, 1, 2, 1, 20, false)
        @test eltype(result) == Float32
        @test size(result) == (20, 2)
        # column 1: all +1 (both data bits = +1)
        @test result[:, 1] == ones(Float32, 20)
        # column 2: +1 for first 10 rows, -1 for next 10 (d[0]=+1, d[1]=-1)
        @test result[1:10,  2] == ones(Float32, 10)
        @test result[11:20, 2] == -ones(Float32, 10)
    end

    @testset "L1 C/A N=20, num_secondary_rotations=1, num_data_bits=1 → 1 column of all +1" begin
        result = Acquisition.sign_patterns(nothing, 1, 1, 1, 20, false)
        @test size(result) == (20, 1)
        @test result == ones(Float32, 20, 1)
    end

    @testset "L5I NH10, N=10, num_secondary_rotations=10, num_data_bits=1, use_secondary_code=true → 10 columns of cyclic rotations" begin
        sec = get_secondary_code(GPSL5I())
        L = 10
        N = 10
        result = Acquisition.sign_patterns(sec, 1, 1, L, N, true)
        @test eltype(result) == Float32
        @test size(result) == (N, L)
        # Rotation r: column r+1 has sign at row k+1 = secondary_value(sec, prn, (k + r) mod L)
        for r in 0:L-1
            expected_col = Float32[GNSSSignals.secondary_value(sec, 1, mod(k + r, L)) for k in 0:N-1]
            @test result[:, r + 1] == expected_col
        end
        # Pin a specific rotation: r=0 column should equal NH10 itself.
        # NH10 = (1, 1, 1, 1, -1, -1, 1, -1, 1, -1)
        @test result[:, 1] == Float32[1, 1, 1, 1, -1, -1, 1, -1, 1, -1]
    end

    @testset "L5I NH10, N=10, num_secondary_rotations=10 but use_secondary_code=false → 1 column of all +1 (opt-out)" begin
        sec = get_secondary_code(GPSL5I())
        result = Acquisition.sign_patterns(sec, 1, 1, 10, 10, false)
        @test size(result) == (10, 1)
        @test result == ones(Float32, 10, 1)
    end

    @testset "combined_phase_patterns" begin
        @testset "shape and element type" begin
            patterns = Acquisition.sign_patterns(nothing, 1, 1, 1, 10, false)
            ramp = Acquisition.combined_phase_patterns(patterns, 10)
            @test eltype(ramp) == ComplexF32
            @test size(ramp) == (10, 1, 10)
        end

        @testset "s = 0 column equals the ±1 pattern (no phase ramp at DC offset)" begin
            sec = get_secondary_code(GPSL5I())
            patterns = Acquisition.sign_patterns(sec, 1, 1, 10, 10, true)   # 10 NH10 rotations
            ramp = Acquisition.combined_phase_patterns(patterns, 10)
            # At s = 0 the phase factor exp(-2πi·p·0/N) ≡ 1, so the complex
            # combined pattern degenerates to the real ±1 pattern.
            for q in 1:size(patterns, 2)
                @test ramp[:, q, 1] ≈ ComplexF32.(patterns[:, q])
            end
        end

        @testset "DFT matrix structure when patterns ≡ +1" begin
            # All-+1 pattern → combined_phase_patterns[:, 1, s+1] is exactly the
            # s-th column of the N×N DFT matrix (forward sign), i.e. the
            # length-N inverse-DFT basis vector for the s-th coarse-bin offset.
            patterns = Acquisition.sign_patterns(nothing, 1, 1, 1, 10, false)
            ramp = Acquisition.combined_phase_patterns(patterns, 10)
            for s in 0:9, p in 0:9
                expected = cispi(-2.0f0 * p * s / 10)
                @test ramp[p + 1, 1, s + 1] ≈ expected rtol = 1e-6
            end
        end

        @testset "tile_phase_patterns reproduces compact phasor at every ω" begin
            # Tiling expands the (N_coh, num_patterns, N_coh) phasor table
            # into a (num_doppler_bins, N_coh, num_patterns) view indexed by
            # the kernel's (ω, p, q). For each ω, the tile must equal
            # combined_phase_patterns[p+1, q, ((ω-1) mod N_coh)+1].
            sec = get_secondary_code(GPSL5I())
            num_coh = 10
            patterns = Acquisition.sign_patterns(sec, 1, 1, num_coh, num_coh, true)
            combined = Acquisition.combined_phase_patterns(patterns, num_coh)
            num_doppler_bins = 150       # = num_coh × num_blocks (L5I @ 12 MHz)
            tiled = Acquisition.tile_phase_patterns(combined, num_doppler_bins)
            @test eltype(tiled) == ComplexF32
            @test size(tiled) == (num_doppler_bins, num_coh, num_coh)
            for q in 1:num_coh, p in 1:num_coh, ω in 1:num_doppler_bins
                s = (ω - 1) % num_coh
                @test tiled[ω, p, q] === combined[p, q, s + 1]
            end
        end

        @testset "phase ramp factors uniformly per pattern column" begin
            # combined_phase_patterns[p, q, s] / patterns[p, q] must equal the
            # phase ramp exp(-2πi · p · s / N) for every column q — i.e. the
            # ramp does not depend on q. (Mismatch here would indicate a per-
            # column bug in the factory.)
            sec = get_secondary_code(GPSL5I())
            patterns = Acquisition.sign_patterns(sec, 1, 1, 10, 10, true)
            ramp = Acquisition.combined_phase_patterns(patterns, 10)
            for s in 0:9, p in 0:9
                expected_ramp = cispi(-2.0f0 * p * s / 10)
                for q in 1:size(patterns, 2)
                    @test ramp[p + 1, q, s + 1] / ComplexF32(patterns[p + 1, q]) ≈ expected_ramp rtol = 1e-6
                end
            end
        end
    end
end
