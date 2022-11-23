@recipe function f(acq_res::AcquisitionResults, log_scale::Bool = false;)
    seriestype := :surface
    seriescolor --> :viridis
    yguide --> "Code phase"
    xguide --> "Dopplers [Hz]"
    zguide --> (log_scale ? "Magnitude [dB]" : "Magnitude")
    y =
        (1:size(acq_res.power_bins, 1)) ./ acq_res.sampling_frequency .*
        get_code_frequency(acq_res.system)
    acq_res.dopplers, y, log_scale ? 10 * log10.(acq_res.power_bins) : acq_res.power_bins
end