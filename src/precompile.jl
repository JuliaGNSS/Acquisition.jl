# Precompile workload (PrecompileTools). A first `acquire` on a fresh session
# costs seconds of compilation — 6.8 s for `Complex{Int16}` samples plus 1.6 s
# for a second sample type on a workstation, three to four times that on an
# embedded ARM host — while the search itself takes tens of milliseconds. A
# live receiver pays that on its first scan, after the samples have started
# flowing (GNSSReceiver.jl#107). Running every signal `GNSSSignals` defines
# here, for both sample element types a front end produces (`Complex{Int16}`
# and `ComplexF32`), plus a multi-period coherent plan and the `acquire!`
# entry point a receiver drives its own plan through, moves the cost into
# `Pkg.precompile`. The search is specialised on the signal type, so a signal
# left out here would pay its own first call live. `FFTW.ESTIMATE` keeps the
# plans cheap to build — FFTW wisdom does not survive precompilation, only the
# compiled Julia code does. Every signal costs ~3 s of compilation here.
using PrecompileTools: @setup_workload, @compile_workload

# Galileo E5b, E6 and the E5a pilot's QP component only exist from GNSSSignals 4
# on, while this package supports `GNSSSignals = "3, 4"`. Naming one that the
# resolved version does not define would make this workload — and with it the
# whole package — fail to precompile, so they are included only when present.
_precompile_optional(name) =
    isdefined(GNSSSignals, name) ? (getfield(GNSSSignals, name)(),) : ()

const _PRECOMPILE_SIGNALS = (
    GPSL1CA(),
    GPSL1C_D(),
    GPSL1C_P(),
    GPSL2CM(),
    # GPS L2CL is left out: its code is 767 250 chips (1.5 s) long, so one code
    # period is a multi-million-sample search that takes minutes — and a receiver
    # acquires L2 on the 20 ms L2CM code, never on the pilot.
    GPSL5I(),
    GPSL5Q(),
    GalileoE1B(),
    GalileoE1B_BOC11(),
    GalileoE1C(),
    GalileoE1C_BOC11(),
    GalileoE5aI(),
    GalileoE5aQ(),
    _precompile_optional(:GalileoE5aQP)...,
    _precompile_optional(:GalileoE5bI)...,
    _precompile_optional(:GalileoE5bQ)...,
    _precompile_optional(:GalileoE6B)...,
    _precompile_optional(:GalileoE6C)...,
    BeiDouB1I(),
    BeiDouB1C_D(),
    BeiDouB1C_P(),
    BeiDouB2aI(),
    BeiDouB2aQ(),
    BeiDouB2bI(),
    BeiDouB3I(),
)

# Sampling frequency for one signal's workload. The BOC-modulated signals (GPS
# L1C, Galileo E1, BeiDou B1C) need at least their sub-chip rate — twelve times
# the chip rate for TMBOC/CBOC/QMBOC — so every signal at the 1.023 MHz chip
# rate is sampled at 16 samples per chip; the 5.115 and 10.23 MHz BPSK codes get
# two samples per chip.
# Float64-valued, like the `5e6Hz` a receiver passes: the search specialises on
# the frequency's element type, and an `Int`-valued rate would compile a
# specialisation nobody calls.
function _precompile_sampling_frequency(system)
    code_frequency = get_code_frequency(system)
    code_frequency <= 1.1e6Hz ? 16.0 * code_frequency : 2.0 * code_frequency
end

# One code period at four samples per chip: the shortest signal `acquire`
# accepts for the system, with the same kernel shapes as any longer one.
function _precompile_signal(system)
    sampling_freq = _precompile_sampling_frequency(system)
    num_samples =
        round(Int, get_code_length(system) * sampling_freq / get_code_frequency(system))
    (; signal) = generate_test_signal(
        system,
        1;
        num_samples,
        sampling_freq,
        interm_freq = 0.0Hz,
        CN0 = 45,
        unit_noise_power = true,
    )
    ComplexF32.(signal), Complex{Int16}.(round.(signal .* 64)), sampling_freq
end

@setup_workload begin
    signals = map(_precompile_signal, _PRECOMPILE_SIGNALS)
    @compile_workload begin
        for (system, (signal_f32, signal_i16, sampling_freq)) in
            zip(_PRECOMPILE_SIGNALS, signals)
            acquire(system, signal_i16, sampling_freq, [1, 2]; fft_flag = FFTW.ESTIMATE)
            acquire(system, signal_f32, sampling_freq, 1; fft_flag = FFTW.ESTIMATE)
        end
        # A receiver builds its plan once and drives `acquire!` with it; a
        # multi-period coherent plan is the common shape for that.
        system = GPSL1CA()
        signal_f32, signal_i16, sampling_freq = signals[1]
        plan = plan_acquire(
            system,
            sampling_freq,
            [1, 2];
            num_coherently_integrated_code_periods = 2,
            fft_flag = FFTW.ESTIMATE,
        )
        results = acquire!(plan, vcat(signal_i16, signal_i16), [1, 2])
        acquire!(plan, vcat(signal_f32, signal_f32), [1])
        acquire!(plan, view(vcat(signal_f32, signal_f32), :), [1])
        is_detected(results[1])
    end
end
