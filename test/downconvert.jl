@testset "Downconvert" begin
    signal = rand(Complex{Int16}, 20000)
    downconverted_signal = Vector{ComplexF32}(undef, length(signal))

    Acquisition.downconvert!(downconverted_signal, signal, 2000Hz, 5e6Hz)
    @test downconverted_signal ≈ signal .* cis.(-2π * (0:length(signal) - 1) * 2000Hz / 5e6Hz)
end