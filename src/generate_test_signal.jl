"""
    generate_test_signal(signal, prn; kwargs...) -> NamedTuple

Generate a synthetic noisy GNSS signal. Useful for testing acquisition and
demonstrating the API. Not exported, as it is test tooling rather than part of a
receiver: write `Acquisition.generate_test_signal`.

Returns a NamedTuple with fields: `samples`, `code`, `carrier`, `num_samples`,
`carrier_doppler`, `code_phase`, `sampling_freq`, `interm_freq`, `CN0`, `prn`.

# Keywords
- `seed=2345`: Random seed for reproducibility
- `num_samples=60000`: Number of signal samples
- `carrier_doppler=1234Hz`: Carrier Doppler frequency
- `code_phase=110.613261`: True code phase in chips
- `sampling_freq=15e6Hz - 1Hz`: Sampling frequency
- `interm_freq=243.0Hz`: Intermediate frequency
- `CN0=45`: Carrier-to-noise density ratio (dB-Hz)
- `carrier_phase=1/16`: Carrier phase in cycles, as GNSSSignals and Tracking count it
- `unit_noise_power=false`: When true, scale so noise power ≈ 1;
  when false, scale so signal power corresponds directly to CN0

# Example

```julia
using Acquisition, GNSSSignals

(; samples, prn, sampling_freq, interm_freq) = Acquisition.generate_test_signal(GPSL1CA(), 1)
result = acquire(GPSL1CA(), samples, sampling_freq, prn; interm_freq)
is_detected(result)  # true
```
"""
function generate_test_signal(
    signal,
    prn;
    seed = 2345,
    num_samples = 120000,
    carrier_doppler = 1234Hz,
    code_phase = 110.613261,
    sampling_freq = 15e6Hz - 1Hz,
    interm_freq = 243.0Hz,
    CN0 = 45,
    carrier_phase = 1 / 16,
    unit_noise_power = false,
)
    Random.seed!(seed)

    code_frequency = get_chip_rate(signal) +
        carrier_doppler * get_chip_rate_to_center_frequency_ratio(signal)
    code = gen_code(
        num_samples,
        signal,
        prn,
        sampling_freq,
        code_frequency,
        code_phase,
    )

    carrier =
        cis.(2π * ((0:(num_samples-1)) * (interm_freq + carrier_doppler) / sampling_freq .+
            carrier_phase))

    if unit_noise_power
        noise_power = 1
        signal_power = CN0 - 10 * log10(sampling_freq / 1.0Hz)
    else
        noise_power = 10 * log10(sampling_freq / 1.0Hz)
        signal_power = CN0
    end

    noise = randn(ComplexF64, num_samples)
    samples = (carrier .* code) * 10^(signal_power / 20) + noise * 10^(noise_power / 20)

    return (;
        samples,
        code,
        carrier,
        num_samples,
        carrier_doppler,
        code_phase,
        sampling_freq,
        interm_freq,
        CN0,
        prn,
    )
end
