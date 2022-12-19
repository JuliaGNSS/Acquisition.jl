struct AcquisitionPlan{S,DS,CS,P,PS}
    system::S
    signal_length::Int
    sampling_freq::typeof(1.0Hz)
    dopplers::DS
    codes_freq_domain::CS
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
    dopplers = -max_doppler:1/3/(signal_length/sampling_freq):max_doppler,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
)
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan = common_buffers(system, signal_length, sampling_freq, prns, fft_flag)
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
        codes_freq_domain,
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
    coarse_step = 1 / 3 / (signal_length / sampling_freq),
    fine_step = 1 / 12 / (signal_length / sampling_freq),
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
)
    coarse_dopplers = -max_doppler:coarse_step:max_doppler
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan = common_buffers(system, signal_length, sampling_freq, prns, fft_flag)
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
        codes_freq_domain,
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
        codes_freq_domain,
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

function common_buffers(system, signal_length, sampling_freq, prns, fft_flag)
    codes = [gen_code(signal_length, system, sat_prn, sampling_freq) for sat_prn in prns]
    signal_baseband = Vector{ComplexF32}(undef, signal_length)
    signal_baseband_freq_domain = similar(signal_baseband)
    code_freq_baseband_freq_domain = similar(signal_baseband)
    code_baseband = similar(signal_baseband)
    fft_plan = plan_fft(signal_baseband; flags = fft_flag)
    codes_freq_domain = map(code -> fft_plan * code, codes)
    signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan
end