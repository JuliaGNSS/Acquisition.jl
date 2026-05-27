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
end
