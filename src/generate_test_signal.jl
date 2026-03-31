"""
    generate_test_signal(system, prn; kwargs...) -> NamedTuple

Generate a synthetic noisy GNSS signal. Useful for testing acquisition and
demonstrating the API.

Returns a NamedTuple with fields: `signal`, `code`, `carrier`, `num_samples`,
`doppler`, `code_phase`, `sampling_freq`, `interm_freq`, `CN0`, `prn`.

# Keywords
- `seed=2345`: Random seed for reproducibility
- `num_samples=60000`: Number of signal samples
- `doppler=1234Hz`: Carrier Doppler frequency
- `code_phase=110.613261`: True code phase in chips
- `sampling_freq=15e6Hz - 1Hz`: Sampling frequency
- `interm_freq=243.0Hz`: Intermediate frequency
- `CN0=45`: Carrier-to-noise density ratio (dB-Hz)
- `phase_offset=π/8`: Carrier phase offset
- `unit_noise_power=false`: When true, scale so noise power ≈ 1;
  when false, scale so signal power corresponds directly to CN0

# Example

```julia
using Acquisition, GNSSSignals

(; signal, prn, sampling_freq, interm_freq) = generate_test_signal(GPSL1(), 1)
result = acquire(GPSL1(), signal, sampling_freq, prn; interm_freq)
is_detected(result)  # true
```
"""
function generate_test_signal(
    system,
    prn;
    seed = 2345,
    num_samples = 60000,
    doppler = 1234Hz,
    code_phase = 110.613261,
    sampling_freq = 15e6Hz - 1Hz,
    interm_freq = 243.0Hz,
    CN0 = 45,
    phase_offset = π / 8,
    unit_noise_power = false,
)
    Random.seed!(seed)

    code = gen_code(
        num_samples,
        system,
        prn,
        sampling_freq,
        get_code_frequency(system) + doppler * get_code_center_frequency_ratio(system),
        code_phase,
    )

    carrier =
        cis.(2π * (0:(num_samples-1)) * (interm_freq + doppler) / sampling_freq .+ phase_offset)

    if unit_noise_power
        noise_power = 1
        signal_power = CN0 - 10 * log10(sampling_freq / 1.0Hz)
    else
        noise_power = 10 * log10(sampling_freq / 1.0Hz)
        signal_power = CN0
    end

    noise = randn(ComplexF64, num_samples)
    signal = (carrier .* code) * 10^(signal_power / 20) + noise * 10^(noise_power / 20)

    return (;
        signal,
        code,
        carrier,
        num_samples,
        doppler,
        code_phase,
        sampling_freq,
        interm_freq,
        CN0,
        prn,
    )
end
