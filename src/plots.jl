"""
$(SIGNATURES)

Plot the result of the acquisition 'power_bins' over the 'max_doppler' frequency into the figure 'fig' onto 'position' or default position (1,1,1). 
"""
function plot_acquisition_results(power_bins, max_doppler, fig, position = (1,1,1))
    ax = fig[:add_subplot](position..., projection = "3d")
    X = linspace(-max_doppler, max_doppler, size(power_bins, 2)) .* ones(1, size(power_bins, 1))
    Y = (1:size(power_bins, 1))' .* ones(size(power_bins, 2), 1)
    ax[:plot_surface](X', Y', power_bins)
end