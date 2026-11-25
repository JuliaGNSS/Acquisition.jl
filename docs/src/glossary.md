# Glossary

Acquisition.jl uses the terms of the [TrackingLoops
glossary](https://juliagnss.github.io/TrackingLoops.jl/stable/glossary/), which ties
them to the textbooks (Kaplan & Hegarty, *Understanding GPS*; Misra & Enge, *Global
Positioning System*; Spilker et al., *Global Positioning System: Theory and
Applications*): an **integration** is a coherent integration (Kaplan and Hegarty's
predetection integration) over whole primary code periods. Acquisition also sums
integrations non-coherently, so it says *coherent integration* and *non-coherent
integration* throughout, as that glossary asks where both kinds occur. A **bit** is a
navigation bit as tracked, i.e. a channel symbol before error-correction decoding, as
GNSSSignals counts it with `get_bit_rate` and `get_code_periods_per_bit`; a **pilot** is
a dataless component; the **secondary code** is the Neuman-Hoffman code overlaid on the
primary code. The terms below are this package's own.

- **Signal** and **samples**: the GNSS signal searched for, a GNSSSignals type such as
  `GPSL1CA()`, and the received complex baseband or IF samples searched in. The
  [`AcquisitionResults`](@ref) field `signal` is the former.
- **Chip rate** and **code frequency**: the nominal rate of the primary code
  (`get_chip_rate`), and the rate a replica runs at, shifted by the code Doppler.
- **Coherent integration**: `num_coherently_integrated_code_periods` code periods whose
  correlations are summed before the power is taken. It sets the Doppler bin spacing,
  `1 / T`, with `T` the coherent integration time.
- **Non-coherent integration**: `num_noncoherent_integrations` successive coherent
  integrations whose powers are summed, Kaplan and Hegarty's and Misra and Enge's
  non-coherent (post-detection) integration.
- **Search space**: the code phases × Doppler bins the search correlates, Kaplan and
  Hegarty's two-dimensional search of code and Doppler bins. A **Doppler bin** is one
  cell of the searched Doppler grid (`doppler_bins`); the code phase is in chips.
- **Doppler pass**: one FM-DBZP grid of narrow Doppler coverage, downconverted to its own
  centre. On sampling frequencies whose `samples_per_code` has only sparse divisors the
  search covers the Doppler range in several passes (see
  [Doppler Passes](@ref)).
- **Sign search**: the search over the unknown signs within an integration — the bit
  polarities, the **bit-edge alignment** (where in the window the first bit edge falls,
  tried at `bit_edge_search_steps` positions) and the **secondary-code rotation** (which
  secondary-code chip the window starts on). Each combination is a *hypothesis*; a cell's
  power is the maximum over them, which [`cfar_threshold`](@ref) accounts for.
- **Peak-to-noise ratio**: the peak power over the estimated noise power, the CFAR test
  statistic [`is_detected`](@ref) compares against the threshold. C/N₀ (`cn0`, dB-Hz) is
  derived from it and the coherent integration time.

## FM-DBZP symbols

The search implements Heckler and Garrison's (2009) FM-DBZP algorithm, and the plan's
fields name the paper's symbols in full:

| Paper | Field | Meaning |
|---|---|---|
| `N_τ` | `samples_per_code` | samples per primary code period |
| `N_step` | `num_blocks` | blocks per code period |
| `B_size` | `block_size` | samples per block, `samples_per_code ÷ num_blocks` |
| `N` | `num_coherently_integrated_code_periods` | code periods per coherent integration |
| `N_db` | `num_bits` | bits per coherent integration (1 for a pilot or a window of at most one bit) |
| `N_be` | `bit_edge_search_steps` | bit-edge alignments searched |
| `C_fd` | `pass_fft_bins` | Doppler FFT length of one pass, `N · N_step` |
