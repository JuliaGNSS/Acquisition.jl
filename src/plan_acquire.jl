struct AcquisitionPlan{S,DS,P,PS}
    system::S
    signal_length::Int
    sampling_freq::typeof(1.0Hz)
    dopplers::DS
    code_freq_domain::Vector{ComplexF32}
    signal_baseband::Vector{ComplexF32}
    signal_baseband_freq_domain::Vector{ComplexF32}
    code_freq_baseband_freq_domain::Vector{ComplexF32}
    code_baseband::Vector{ComplexF32}
    signal_powers::Vector{Matrix{Float32}}
    fft_plan::P
    avail_prn_channels::PS
end

function AcquisitionPlan(
    system,
    signal_length,
    sampling_freq;
    max_doppler = 7000Hz,
    min_doppler = -max_doppler,
    dopplers = min_doppler:250Hz:max_doppler,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
)
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    code_freq_domain,
    fft_plan = common_buffers(signal_length, fft_flag)
    Δt = signal_length / sampling_freq
    code_interval = get_code_length(system) / get_code_frequency(system)
    signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(dopplers),
        ) for _ in prns
    ]
    AcquisitionPlan(
        system,
        signal_length,
        sampling_freq,
        dopplers,
        code_freq_domain,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        signal_powers,
        fft_plan,
        prns,
    )
end

struct CoarseFineAcquisitionPlan{C<:AcquisitionPlan,F<:AcquisitionPlan}
    coarse_plan::C
    fine_plan::F
end

function CoarseFineAcquisitionPlan(
    system,
    signal_length,
    sampling_freq;
    max_doppler = 7000Hz,
    min_doppler = -max_doppler,
    coarse_step = 250Hz,
    fine_step = 25Hz,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
)
    coarse_dopplers = min_doppler:coarse_step:max_doppler
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    code_freq_domain,
    fft_plan = common_buffers(signal_length, fft_flag)
    Δt = signal_length / sampling_freq
    code_interval = get_code_length(system) / get_code_frequency(system)
    coarse_signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(coarse_dopplers),
        ) for _ in prns
    ]
    fine_doppler_range = -2*coarse_step:fine_step:2*coarse_step
    fine_signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(fine_doppler_range),
        ) for _ in prns
    ]
    coarse_plan = AcquisitionPlan(
        system,
        signal_length,
        sampling_freq,
        coarse_dopplers,
        code_freq_domain,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        coarse_signal_powers,
        fft_plan,
        prns,
    )
    fine_plan = AcquisitionPlan(
        system,
        signal_length,
        sampling_freq,
        fine_doppler_range,
        code_freq_domain,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        fine_signal_powers,
        fft_plan,
        prns,
    )
    CoarseFineAcquisitionPlan(coarse_plan, fine_plan)
end

function common_buffers(signal_length, fft_flag)
    signal_baseband = Vector{ComplexF32}(undef, signal_length)
    signal_baseband_freq_domain = similar(signal_baseband)
    code_freq_baseband_freq_domain = similar(signal_baseband)
    code_baseband = similar(signal_baseband)
    fft_plan = plan_fft(signal_baseband; flags = fft_flag)
    code_freq_domain = similar(signal_baseband)
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    code_freq_domain,
    fft_plan
end