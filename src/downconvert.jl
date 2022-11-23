function downconvert!(downconverted_signal, signal, frequency, sampling_freq)
    downconverted_signal .=
        signal .*
        cis.(
            Float32(-2π) .* (0:length(signal)-1) .* Float32(frequency) ./
            Float32(sampling_freq)
        )
end

function downconvert!(
    downconverted_signal::Vector{Complex{T}},
    signal::AbstractVector{Complex{TS}},
    frequency,
    sampling_freq,
) where {T,TS}
    signal_real = reinterpret(reshape, TS, signal)
    downconverted_signal_real = reinterpret(reshape, T, downconverted_signal)
    @turbo for i = 1:length(signal)
        c_im, c_re = sincos(T(2π) * (i - 1) * T(frequency / sampling_freq))
        downconverted_signal_real[1, i] =
            signal_real[1, i] * c_re + signal_real[2, i] * c_im
        downconverted_signal_real[2, i] =
            signal_real[2, i] * c_re - signal_real[1, i] * c_im
    end
    downconverted_signal
end