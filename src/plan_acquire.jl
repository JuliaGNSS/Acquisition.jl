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
    complex_signal::Vector{Matrix{ComplexF32}}
    fft_plan::P
    avail_prn_channels::PS
    compensate_doppler_code::Symbol
    noncoherent_rounds::Int
end

function AcquisitionPlan(
    system,
    signal_length,
    sampling_freq;
    max_doppler = 7000Hz,
    dopplers = -max_doppler:1/3/(signal_length/sampling_freq):max_doppler,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
    compensate_doppler_code = :disabled,
    noncoherent_rounds = 1
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
    complex_signal = [
        Matrix{ComplexF32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(dopplers),
        ) for _ in prns
    ]
    @assert length(signal_powers) == length(complex_signal)
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
        complex_signal,
        fft_plan,
        prns,
        compensate_doppler_code,
        noncoherent_rounds
    )
end

function Base.show(io::IO, ::MIME"text/plain", plan::AcquisitionPlan)
    println(io,"AcquisitionPlan for GPS L1:\n")
    println(io,"Sample rate: $(plan.sampling_freq)")
    println(io,"Doppler taps: $(plan.dopplers)")
    println(io,"Coherent integration length: $(plan.signal_length) samples ($(upreferred(plan.signal_length/plan.sampling_freq)))")
    println(io,"Noncoherent integration round: $(plan.noncoherent_rounds) ($(upreferred(plan.signal_length*plan.noncoherent_rounds/plan.sampling_freq)))")
    println(io, "Doppler compensation: $(plan.compensate_doppler_code)")
    println(io,"FFTW plan: $(plan.fft_plan)")
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
    noncoherent_rounds=1
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
    coarse_signal_powers_complex = [
        Matrix{ComplexF32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(coarse_dopplers),
        ) for _ in prns
    ]
    fine_signal_powers_complex = [
        Matrix{ComplexF32}(
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
        coarse_signal_powers_complex,
        fft_plan,
        prns,
        :disabled,
        noncoherent_rounds
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
        fine_signal_powers_complex,
        fft_plan,
        prns,
        :disabled,
        noncoherent_rounds
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

function preallocate_thread_local_buffer(signal_length,doppler_taps)
    #TODO: align buffers to fftw friendly offsets
    #DO NOT merge this to upstream JuliaGNSS until this is stopped hard coded
    signal_length = 16368
    signal_baseband_buffer = Array{ComplexF32}(undef, signal_length,doppler_taps)
    signal_baseband_freq_domain_buffer = similar(signal_baseband_buffer)

    return signal_baseband_buffer,signal_baseband_freq_domain_buffer

end