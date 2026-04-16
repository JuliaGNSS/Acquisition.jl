# src/plot.jl
#
# Plot recipe for AcquisitionResults from FM-DBZP acquisition.
#
# power_bins is stored in FM-DBZP column order: (num_doppler_bins, samples_per_code).
# Columns are in block-permuted order — column c corresponds to delay tau, which maps
# to code phase chip = mod(-tau * f_code / fs, code_length).  At typical GNSS sampling
# rates (2× oversampled) this mapping is a bijection, so columns can be sorted into
# monotone chip order for plotting.

"""
    _fmdbzp_column_to_tau(c, num_blocks, block_size) -> Int

Convert a 0-indexed FM-DBZP scrambled column index `c` to the delay in samples.
"""
function _fmdbzp_column_to_tau(c::Int, num_blocks::Int, block_size::Int)
    r    = c ÷ block_size
    fine = c % block_size
    mod(num_blocks - r, num_blocks) * block_size + fine
end

"""
    _fmdbzp_sort_by_chip(power_bins, num_blocks, block_size, f_code, fs) -> (chip_axis, ordered)

Return `(chip_axis, ordered)` where columns of `power_bins` are permuted into monotone
code-phase order.  `chip_axis[i]` is the code phase in chips for column `i` of `ordered`.

Each FM-DBZP scrambled column `c` corresponds to delay `tau` samples, and chip phase
`mod(-tau * f_code / fs, code_length)`.  At GNSS sampling rates (≥ 2× oversampled)
this is a bijection, so `sortperm` on the chip values gives a valid reordering.
"""
function _fmdbzp_sort_by_chip(
    power_bins::Matrix,
    num_blocks::Int,
    block_size::Int,
    f_code::Float64,   # Hz
    fs::Float64,       # Hz
)
    _, samples_per_code = size(power_bins)
    code_length = samples_per_code * f_code / fs  # chips (≈ 1023 for GPS L1)

    # Compute chip value for each scrambled column
    chips = Vector{Float64}(undef, samples_per_code)
    for c in 0:samples_per_code-1
        tau        = _fmdbzp_column_to_tau(c, num_blocks, block_size)
        chips[c+1] = mod(-tau * f_code / fs, code_length)
    end

    # Sort columns into chip order
    perm    = sortperm(chips)
    ordered = power_bins[:, perm]
    chips[perm], ordered
end

@recipe function f(result::AcquisitionResults, log_scale::Bool = false)
    result.power_bins !== nothing ||
        throw(ArgumentError("power_bins is nothing — call acquire! with store_power_bins=true to enable plotting"))

    num_blocks = result.num_blocks
    block_size = result.block_size

    f_code = Float64(ustrip(Hz, get_code_frequency(result.system)))
    fs     = Float64(ustrip(Hz, result.sampling_frequency))

    chip_axis, ordered = _fmdbzp_sort_by_chip(result.power_bins, num_blocks, block_size, f_code, fs)

    seriestype  := :surface
    seriescolor --> :viridis
    xguide      --> "Doppler [Hz]"
    yguide      --> "Code phase [chips]"
    zguide      --> (log_scale ? "Magnitude [dB]" : "Magnitude")

    z     = log_scale ? 10 .* log10.(max.(ordered, 1f-30)) : ordered
    noise = log_scale ? 10 * log10(Float64(result.noise_power)) : Float64(result.noise_power)
    clims --> (noise, maximum(z))

    doppler_hz = ustrip.(Hz, result.dopplers)

    # surface: x=dopplers, y=chip_axis, z matrix must be (length(y), length(x))
    # ordered is (num_doppler_bins, samples_per_code) → transpose to (samples_per_code, num_doppler_bins)
    doppler_hz, chip_axis, collect(z')
end
