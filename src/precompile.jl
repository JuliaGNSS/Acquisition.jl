# Precompile workload (PrecompileTools). A first `acquire` on a fresh session
# costs seconds of compilation — 6.8 s for `Complex{Int16}` samples plus 1.6 s
# for a second sample type on a workstation, three to four times that on an
# embedded ARM host — while the search itself takes tens of milliseconds. A
# live receiver pays that on its first scan, after the samples have started
# flowing (GNSSReceiver.jl#107). Running the common shapes here moves the cost
# into `Pkg.precompile`: the sample element types a front end produces
# (`Complex{Int16}` and `ComplexF32`), a single-period and a multi-period
# coherent plan, and the `acquire!` entry point a receiver drives its own plan
# through. `FFTW.ESTIMATE` keeps the plans cheap to build — FFTW wisdom does not
# survive precompilation, only the compiled Julia code does.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    system = GPSL1CA()
    sampling_freq = 2.048e6Hz
    one_period = 2048
    (; signal) = generate_test_signal(
        system, 1;
        num_samples = 2one_period, sampling_freq, interm_freq = 0.0Hz, CN0 = 45,
        unit_noise_power = true,
    )
    signal_f32 = ComplexF32.(signal)
    signal_i16 = Complex{Int16}.(round.(signal .* 64))
    @compile_workload begin
        acquire(system, signal_i16[1:one_period], sampling_freq, [1, 2]; fft_flag = FFTW.ESTIMATE)
        acquire(system, signal_f32[1:one_period], sampling_freq, 1; fft_flag = FFTW.ESTIMATE)
        acquire(system, view(signal_f32, 1:one_period), sampling_freq, 1; fft_flag = FFTW.ESTIMATE)
        plan = plan_acquire(
            system, sampling_freq, [1, 2];
            num_coherently_integrated_code_periods = 2, fft_flag = FFTW.ESTIMATE,
        )
        results = acquire!(plan, signal_i16, [1, 2])
        acquire!(plan, signal_f32, [1])
        is_detected(results[1])
    end
end
