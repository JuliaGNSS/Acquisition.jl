@recipe function f(acq_res::AcquisitionResults, log_scale::Bool = false;)
    seriestype := :surface
    seriescolor --> :viridis
    yguide --> "Code phase"
    xguide --> "Dopplers [Hz]"
    z = log_scale ? 10 * log10.(acq_res.power_bins) : acq_res.power_bins
    zguide --> (log_scale ? "Magnitude [dB]" : "Magnitude")
    noise = log_scale ? 10 * log10(acq_res.noise_power) : acq_res.noise_power
    clims --> (noise, maximum(z))
    y =
        (1:size(acq_res.power_bins, 1)) ./ acq_res.sampling_frequency .*
        get_code_frequency(acq_res.system)
    acq_res.dopplers, y, z
end