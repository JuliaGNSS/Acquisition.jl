# src/plot.jl
#
# Plot recipe for AcquisitionResults from FM-DBZP acquisition.
#
# power_bins is stored in FM-DBZP column order: (num_doppler_bins, samples_per_code).
# Columns are in block-permuted order — column c corresponds to delay tau, which maps
# to code phase chip = mod(-tau * code_freq_hz / sampling_freq_hz, code_length).  At
# typical GNSS sampling rates (2× oversampled) this mapping is a bijection, so columns
# can be sorted into monotone chip order for plotting.

"""
    _fmdbzp_column_to_tau(scrambled_col_idx, num_blocks, block_size) -> Int

Convert a 0-indexed FM-DBZP scrambled column index to the delay in samples.
"""
function _fmdbzp_column_to_tau(scrambled_col_idx::Int, num_blocks::Int, block_size::Int)
    block_row        = scrambled_col_idx ÷ block_size
    within_block_idx = scrambled_col_idx % block_size
    mod(num_blocks - block_row, num_blocks) * block_size + within_block_idx
end

"""
    _fmdbzp_sort_by_chip(power_bins, num_blocks, block_size, code_freq_hz, sampling_freq_hz)
        -> (chip_axis, chip_sorted_power_bins)

Return `(chip_axis, chip_sorted_power_bins)` where columns of `power_bins` are permuted
into monotone code-phase order.  `chip_axis[i]` is the code phase in chips for column
`i` of `chip_sorted_power_bins`.

Each FM-DBZP scrambled column `scrambled_col_idx` corresponds to delay `tau` samples,
and chip phase `mod(-tau * code_freq_hz / sampling_freq_hz, code_length)`.  At GNSS
sampling rates (≥ 2× oversampled) this is a bijection, so `sortperm` on the chip values
gives a valid reordering.
"""
function _fmdbzp_sort_by_chip(
    power_bins::Matrix,
    num_blocks::Int,
    block_size::Int,
    code_freq_hz::Float64,
    sampling_freq_hz::Float64,
)
    _, samples_per_code = size(power_bins)
    code_length = samples_per_code * code_freq_hz / sampling_freq_hz  # chips (≈ 1023 for GPS L1)

    # Compute chip value for each scrambled column
    chip_phases = Vector{Float64}(undef, samples_per_code)
    for scrambled_col_idx in 0:samples_per_code-1
        tau = _fmdbzp_column_to_tau(scrambled_col_idx, num_blocks, block_size)
        chip_phases[scrambled_col_idx + 1] =
            mod(-tau * code_freq_hz / sampling_freq_hz, code_length)
    end

    # Sort columns into chip order
    chip_order             = sortperm(chip_phases)
    chip_sorted_power_bins = power_bins[:, chip_order]
    chip_phases[chip_order], chip_sorted_power_bins
end

@recipe function f(result::AcquisitionResults, log_scale::Bool = false)
    result.power_bins !== nothing ||
        throw(ArgumentError("power_bins is nothing — call acquire! with store_power_bins=true to enable plotting"))

    num_blocks = result.num_blocks
    block_size = result.block_size

    code_freq_hz     = Float64(ustrip(Hz, get_code_frequency(result.system)))
    sampling_freq_hz = Float64(ustrip(Hz, result.sampling_frequency))

    chip_axis, chip_sorted_power_bins =
        _fmdbzp_sort_by_chip(result.power_bins, num_blocks, block_size, code_freq_hz, sampling_freq_hz)

    seriestype  := :surface
    seriescolor --> :viridis
    xguide      --> "Doppler [Hz]"
    yguide      --> "Code phase [chips]"
    zguide      --> (log_scale ? "Magnitude [dB]" : "Magnitude")

    z           = log_scale ? 10 .* log10.(max.(chip_sorted_power_bins, 1f-30)) : chip_sorted_power_bins
    noise_floor = log_scale ? 10 * log10(Float64(result.noise_power)) : Float64(result.noise_power)
    clims --> (noise_floor, maximum(z))

    doppler_hz = ustrip.(Hz, result.dopplers)

    # surface: x=dopplers, y=chip_axis, z matrix must be (length(y), length(x))
    # chip_sorted_power_bins is (num_doppler_bins, samples_per_code)
    # → transpose to (samples_per_code, num_doppler_bins)
    doppler_hz, chip_axis, collect(z')
end
