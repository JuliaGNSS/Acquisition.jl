"""
$(SIGNATURES)

Plot the result of the acquisition 'power_bins' over the 'max_doppler' frequency into the figure 'fig' onto 'position' or default position (1,1,1). 
"""
function plot_acquisition_results(acquisition_results, gnss_system::T, sample_freq, fig = figure(), position = (1,1,1)) where T <: AbstractGNSSSystem
    PyPlot.pyimport("mpl_toolkits.mplot3d.axes3d")
    ax = fig[:add_subplot](position..., projection = "3d")
    num_dopplers = size(acquisition_results.power_bins, 2)
    num_code_phases = size(acquisition_results.power_bins, 1)
    X = acquisition_results.doppler_steps
    Y = collect(1:num_code_phases) .* ones(num_dopplers, 1)' ./ sample_freq .* gnss_system.code_freq
    xlabel("Doppler in Hz")
    ylabel("Code-Phase in Chips")
    zlabel("Relative Leistung")
    ax[:plot_surface](X, Y, acquisition_results.power_bins, rstride=1, cstride=1000, cmap="viridis")
end