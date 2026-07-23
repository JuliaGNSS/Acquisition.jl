# Changelog

## [2.6.1](https://github.com/JuliaGNSS/Acquisition.jl/compare/v2.6.0...v2.6.1) (2026-07-23)


### Bug Fixes

* rank recommend_sampling_freqs by the column FFT, not the inner FFT ([0347706](https://github.com/JuliaGNSS/Acquisition.jl/commit/0347706df21759a4cd42b8766c844d1c195cda6b))


### Performance Improvements

* pad inner FFT to a smooth size to remove sampling-frequency regression ([1b10d9c](https://github.com/JuliaGNSS/Acquisition.jl/commit/1b10d9cd4698137065fb3474c621ef8186313409))

# [2.6.0](https://github.com/JuliaGNSS/Acquisition.jl/compare/v2.5.3...v2.6.0) (2026-07-02)


### Features

* support GNSSSignals 3 ([8710840](https://github.com/JuliaGNSS/Acquisition.jl/commit/87108405db435c0c4ee1f1219f53fcf27399162e))

## [2.5.3](https://github.com/JuliaGNSS/Acquisition.jl/compare/v2.5.2...v2.5.3) (2026-07-02)


### Bug Fixes

* index multistep pilot drift by destination Doppler row ([#74](https://github.com/JuliaGNSS/Acquisition.jl/issues/74)) ([455f5b6](https://github.com/JuliaGNSS/Acquisition.jl/commit/455f5b65f7dcc94fde1fe478d12908dc0deceabf)), closes [#62](https://github.com/JuliaGNSS/Acquisition.jl/issues/62)

## [2.5.2](https://github.com/JuliaGNSS/Acquisition.jl/compare/v2.5.1...v2.5.2) (2026-07-02)


### Performance Improvements

* PRN-outer multistep — one accumulation surface per slot, not per PRN ([f5feade](https://github.com/JuliaGNSS/Acquisition.jl/commit/f5feade1c1393b7c46391b4693acd75defb9c9f0))
* tile FM-DBZP build+reduce, stream result stats at N_nc=1 ([a134aff](https://github.com/JuliaGNSS/Acquisition.jl/commit/a134aff1f57040f2417e3daadec7a3f6ffa32457))

## [2.5.1](https://github.com/JuliaGNSS/Acquisition.jl/compare/v2.5.0...v2.5.1) (2026-06-09)


### Bug Fixes

* count secondary-code rotation expansion in get_num_cells ([#70](https://github.com/JuliaGNSS/Acquisition.jl/issues/70)) ([508a861](https://github.com/JuliaGNSS/Acquisition.jl/commit/508a861e7d9eacab68686ef8cffaa7ae0711787e)), closes [#69](https://github.com/JuliaGNSS/Acquisition.jl/issues/69)

# [2.5.0](https://github.com/JuliaGNSS/Acquisition.jl/compare/v2.4.1...v2.5.0) (2026-06-09)


### Bug Fixes

* reject partial secondary-code periods (Doppler sign ambiguity, [#68](https://github.com/JuliaGNSS/Acquisition.jl/issues/68)) ([9d152bd](https://github.com/JuliaGNSS/Acquisition.jl/commit/9d152bde23a6cdb181a242af5a93342c8c23fad9))
* rotation kernel buffer overflow at N >= 2L (data-bit × rotation) ([2201261](https://github.com/JuliaGNSS/Acquisition.jl/commit/2201261b6e8551b1742253eeff67f4ebaefc4a90))


### Features

* expose exact secondary_code_phase in AcquisitionResults ([4466401](https://github.com/JuliaGNSS/Acquisition.jl/commit/4466401baf051c43ad1c23c9ef16cc9449d45c68))


### Performance Improvements

* keep L5I rotation path at baseline speed (PR [#69](https://github.com/JuliaGNSS/Acquisition.jl/issues/69) CI regression) ([0eb991f](https://github.com/JuliaGNSS/Acquisition.jl/commit/0eb991f6c9eb2caa022d510fde7563e4c5dfc89c))

## [2.4.1](https://github.com/JuliaGNSS/Acquisition.jl/compare/v2.4.0...v2.4.1) (2026-06-03)


### Performance Improvements

* hard-cap per-thread acquisition scratch with a slot pool (Issue [#60](https://github.com/JuliaGNSS/Acquisition.jl/issues/60)) ([52c2e36](https://github.com/JuliaGNSS/Acquisition.jl/commit/52c2e364c2ddc9a2dbc79eb7217aa96afe71bf37))

# [2.4.0](https://github.com/JuliaGNSS/Acquisition.jl/compare/v2.3.3...v2.4.0) (2026-06-02)


### Bug Fixes

* correct half-band Doppler offset on the secondary-code rotation path ([18e143d](https://github.com/JuliaGNSS/Acquisition.jl/commit/18e143d06936c6e657aa790ae797b42f6937cb28)), closes [#65](https://github.com/JuliaGNSS/Acquisition.jl/issues/65)


### Features

* close ~5 dB GPSL5I-rotation sensitivity gap to LongL5I ([5a4578d](https://github.com/JuliaGNSS/Acquisition.jl/commit/5a4578daffd5277cfc3e07ee4e87fe4d05eb54fb))
* secondary-code rotation search with use_secondary_code opt-out and cap ([b58d237](https://github.com/JuliaGNSS/Acquisition.jl/commit/b58d2373858cfa839dc2b3f89c0aa8148702e02f)), closes [hi#level](https://github.com/hi/issues/level) [JuliaGNSS/Acquisition.jl#58](https://github.com/JuliaGNSS/Acquisition.jl/issues/58)


### Performance Improvements

* fold fftshift into sign-search kernels + drop max_buf when bit_edge_search_steps==1 ([b2d524e](https://github.com/JuliaGNSS/Acquisition.jl/commit/b2d524e04abd1931e30b4860aa00d3997b18cd55))
* parallelise rotation-kernel col_idx loop with Polyester [@batch](https://github.com/batch) ([99c0279](https://github.com/JuliaGNSS/Acquisition.jl/commit/99c02794cf62df17f7b19c91d4e1bd5e30011573))


### Reverts

* Revert "perf: parallelise rotation-kernel col_idx loop with Polyester [@batch](https://github.com/batch)" ([0cce813](https://github.com/JuliaGNSS/Acquisition.jl/commit/0cce81389a5104875d651ca71e201cdd5c0fbc1a))

## [2.3.3](https://github.com/JuliaGNSS/Acquisition.jl/compare/v2.3.2...v2.3.3) (2026-05-27)


### Bug Fixes

* correct half-band Doppler offset in the sign-search path ([29269f9](https://github.com/JuliaGNSS/Acquisition.jl/commit/29269f92753a8c38aa29410d7ea44828d4ace5cf))

## [2.3.2](https://github.com/JuliaGNSS/Acquisition.jl/compare/v2.3.1...v2.3.2) (2026-05-27)


### Performance Improvements

* fuse |x|² + code-drift + fftshift on the multistep simple path ([de0522e](https://github.com/JuliaGNSS/Acquisition.jl/commit/de0522ef35cdf0ffdc786841b8efe79585afc350)), closes [#62](https://github.com/JuliaGNSS/Acquisition.jl/issues/62)

## [2.3.1](https://github.com/JuliaGNSS/Acquisition.jl/compare/v2.3.0...v2.3.1) (2026-05-26)


### Performance Improvements

* allocate result_buffers lazily when store_power_bins is set ([9d8d9b2](https://github.com/JuliaGNSS/Acquisition.jl/commit/9d8d9b22dfe1d8a6af90922b12a292e19fc49fc4))
* fuse |x|² + fftshift into accumulator on the N_nc=1 simple path ([2b53632](https://github.com/JuliaGNSS/Acquisition.jl/commit/2b536321fe87150841b51f52ae8aa36bc12203c0)), closes [#60](https://github.com/JuliaGNSS/Acquisition.jl/issues/60)
* process PRNs sequentially when num_noncoherent_accumulations == 1 ([7f15f84](https://github.com/JuliaGNSS/Acquisition.jl/commit/7f15f8487c919ee1bd30785ffca40dfab233cc8b))
* skip per-thread sign-search buffers when the simple path is taken ([eaa2b58](https://github.com/JuliaGNSS/Acquisition.jl/commit/eaa2b585082736ff253aed05274064f786636f3c))

# [2.3.0](https://github.com/JuliaGNSS/Acquisition.jl/compare/v2.2.0...v2.3.0) (2026-05-15)


### Features

* **deps:** bump to GNSSSignals v2 and update for renamed API ([2cae67b](https://github.com/JuliaGNSS/Acquisition.jl/commit/2cae67bf8f4e4e7d24d00b0483ae2c453248c6bd))

# [2.2.0](https://github.com/JuliaGNSS/Acquisition.jl/compare/v2.1.1...v2.2.0) (2026-05-15)


### Features

* opposite-row noise estimator as CFAR default ([5deabfc](https://github.com/JuliaGNSS/Acquisition.jl/commit/5deabfc84c29528a2bd2e385c6bf8f87418d9978))

## [2.1.1](https://github.com/JuliaGNSS/Acquisition.jl/compare/v2.1.0...v2.1.1) (2026-05-02)


### Performance Improvements

* hoist signal-block FFTs out of per-PRN loop and drop dead fill! ([a90e9af](https://github.com/JuliaGNSS/Acquisition.jl/commit/a90e9af5383d7e511ee387c206c8047ffb6b8911))

# [2.1.0](https://github.com/JuliaGNSS/Acquisition.jl/compare/v2.0.0...v2.1.0) (2026-04-30)


### Bug Fixes

* min_doppler_coverage now guarantees both grid ends ([cc10032](https://github.com/JuliaGNSS/Acquisition.jl/commit/cc1003282febd63762c152b4b9c62cbbfc781d7b))


### Features

* add recommend_sampling_freqs with SDR-aware filtering ([dd1219b](https://github.com/JuliaGNSS/Acquisition.jl/commit/dd1219bf20f3b3411a221ad98ef99260d793da3c))

# [2.0.0](https://github.com/JuliaGNSS/Acquisition.jl/compare/v1.7.1...v2.0.0) (2026-04-27)


* feat!: replace acquisition backend with FM-DBZP algorithm ([36f5717](https://github.com/JuliaGNSS/Acquisition.jl/commit/36f571763f94641a7bea5cf8cc949e181f6ae540))


### Bug Fixes

* **docs:** declare Plotly-hit counters global inside for loop ([9c176af](https://github.com/JuliaGNSS/Acquisition.jl/commit/9c176af0eb88aaef183a0d73aa8647f78cb59cf1))


### Performance Improvements

* batched column FFT for pilot path + enable CI multi-threading ([9f05f2c](https://github.com/JuliaGNSS/Acquisition.jl/commit/9f05f2c26cc3d3514242aa7e6be75bae84aab715))
* chunk PRNs across threads to reduce scheduling overhead ([7f4136d](https://github.com/JuliaGNSS/Acquisition.jl/commit/7f4136dd2cbdf0fb7ef91b14e197af4adb19df7b))
* eliminate all allocations in acquire! ([820ffdc](https://github.com/JuliaGNSS/Acquisition.jl/commit/820ffdcd1a150e2a63cf9f6ed1d17a34e0b984b4))
* SIMD-vectorize est_signal_noise_power, fftshift scatter, parallelize results build ([b31d617](https://github.com/JuliaGNSS/Acquisition.jl/commit/b31d617902fd7603234e4645e00587ab326e20a0))
* use Polyester.[@batch](https://github.com/batch) for zero-overhead multi-PRN threading ([42935a6](https://github.com/JuliaGNSS/Acquisition.jl/commit/42935a68df8d9a01d93cee73ad4dbdfb95e82570))


### BREAKING CHANGES

* The public API has been redesigned around the FM-DBZP
(Frequency-domain Modified Double Block Zero Padding) algorithm by
Heckler & Garrison (2009). The old serial Doppler search functions
(AcquisitionPlan, CoarseFineAcquisitionPlan, coarse_fine_acquire) are
removed. Users should migrate to plan_acquire / acquire! / acquire.

## New API

```julia
# Simple one-shot acquisition
results = acquire(system, signal, sampling_freq, 1:32; interm_freq)

# Pre-planned acquisition (reuses FFT plans and allocations)
plan = plan_acquire(system, sampling_freq, collect(1:32))
results = acquire!(plan, signal, 1:32; interm_freq)
```

## Algorithm

FM-DBZP performs a joint 2D search over code phase and Doppler in the
frequency domain. The signal is divided into num_blocks sub-blocks;
a 2D correlation matrix is built via double-block FFTs, then a
column-wise FFT yields the Doppler dimension. This eliminates the
explicit per-Doppler-bin loop of the classical approach.

Key constraints (documented in the guide):
- Doppler bin spacing = 1 / T_coh (finer resolution requires longer
  coherent integration, unlike classical search)
- num_blocks must divide samples_per_code exactly; plan_acquire finds
  the smallest valid divisor automatically
- Data bit handling for GPS L1 C/A: bit_edge_search_steps for sub-bit
  integration, automatic sign-flip search for multi-bit integration
- Pilot channels (GPS L5, Galileo E1): no data bit constraints

## Features

**Multi-threaded PRN loop**: acquire! uses Threads.@threads over PRNs
automatically. Per-thread scratch buffers are allocated at plan_acquire
time (sized by Threads.maxthreadid()). No code changes required — just
start Julia with -t N.

Benchmarks at 1 ms / 4 MHz / 32 PRNs:
- 1 thread:  ~40 ms (same as old implementation)
- 4 threads: ~13 ms (6.6× speedup over 1 thread)
- 8 threads: ~8.5 ms (9.8× speedup over 1 thread)

Old implementation showed no threading benefit (no parallel PRN loop).

**Non-coherent integration**: accumulate power across multiple signal
segments for sensitivity at low CN0:

```julia
plan = plan_acquire(system, sampling_freq, prns;
    num_coherently_integrated_code_periods = 10,
    num_noncoherent_accumulations = 8)
```

**Parabolic sub-sample interpolation** for code phase and Doppler,
with a noise guard that skips interpolation when neighbouring bins are
below sqrt(noise_power) — preventing the estimator from chasing noise:

```julia
results = acquire!(plan, signal, prns; subsample_interpolation = true)
```

**Interactive PlotlyJS surface plots** via Plots.jl recipe:

```julia
result = acquire!(plan, signal, prns; store_power_bins = true)
plot(result[1])          # linear scale
plot(result[1], true)    # dB scale
```

## Performance vs old implementation (same Doppler grid, 1 thread)

| Integration | Speedup |
|-------------|---------|
| 1 ms        | ~1×     |
| 5 ms        | ~10-20× |
| 10 ms       | ~20-37× |
| 20-40 ms    | ~37-52× |

FM-DBZP's advantage grows with integration length because the 2D FFT
cost scales much better than the classical per-Doppler-bin IFFT loop.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

## [1.7.1](https://github.com/JuliaGNSS/Acquisition.jl/compare/v1.7.0...v1.7.1) (2026-04-01)


### Bug Fixes

* add DBZP validation, update tests for 2N signal requirement ([48dedf3](https://github.com/JuliaGNSS/Acquisition.jl/commit/48dedf303e0e8d5ec951c8156993cac0671a9fc3))
* add default_coherent_samples and prepare_signal_for_dbzp helpers ([f5f06d5](https://github.com/JuliaGNSS/Acquisition.jl/commit/f5f06d523e608dc65551702605f9c4fc94e569b1))
* implement true DBZP with overlapping 2N-sample windows ([5693564](https://github.com/JuliaGNSS/Acquisition.jl/commit/56935644398533c86a2898967d621ee22e1e4c58))
* make benchmarks backward-compatible with pre-DBZP versions ([bfcf0d5](https://github.com/JuliaGNSS/Acquisition.jl/commit/bfcf0d58311376d2f91796895a30a4721bcd5376))
* only repeat signal for DBZP when chunk_samples is one code period ([1a5ef87](https://github.com/JuliaGNSS/Acquisition.jl/commit/1a5ef87cddcd8e425192de23ef07baf251022d01))
* revert test workarounds in calc_powers and ka_acquire tests ([78dc8fb](https://github.com/JuliaGNSS/Acquisition.jl/commit/78dc8fb1087976c8d6b727fbe0d08d2a7151650a))
* revert test workarounds, update multi-period tests for DBZP, add validation test ([5b4246a](https://github.com/JuliaGNSS/Acquisition.jl/commit/5b4246adef768deb9ddd5aea609eddcdc7312e89))
* update benchmarks and docs for DBZP 2N signal requirement ([2031416](https://github.com/JuliaGNSS/Acquisition.jl/commit/20314166e04b638024041bff86992f7e63b0b7b4))
* use default_coherent_samples in convenience functions, pass directly to plan ([dedd7dd](https://github.com/JuliaGNSS/Acquisition.jl/commit/dedd7dd08a484f35f1cf9fd408627b05cf003a52))
* use prepare_signal_for_dbzp in acquire!(::AcquisitionPlan) ([364c11e](https://github.com/JuliaGNSS/Acquisition.jl/commit/364c11eca2adc5c9c7df560aa0e23dc739879d3d))
* use prepare_signal_for_dbzp in acquire!(::CoarseFineAcquisitionPlan) ([aa0985e](https://github.com/JuliaGNSS/Acquisition.jl/commit/aa0985ed23cea31bd3670e050681df2742fd71be))
* use prepare_signal_for_dbzp in acquire!(::KAAcquisitionPlan) ([235ef91](https://github.com/JuliaGNSS/Acquisition.jl/commit/235ef91e1036ef70b31a16465c365924d62f8884))

# [1.7.0](https://github.com/JuliaGNSS/Acquisition.jl/compare/v1.6.0...v1.7.0) (2026-03-31)


### Bug Fixes

* **docs:** add Unitful to docs dependencies for [@example](https://github.com/example) blocks ([e855bde](https://github.com/JuliaGNSS/Acquisition.jl/commit/e855bde552fad75b404f455fff8e3c7a634c35c8))
* normalize peak_to_noise_ratio to χ²(2M) scale for correct CFAR threshold ([99081a4](https://github.com/JuliaGNSS/Acquisition.jl/commit/99081a4b12fde1aaf65522e6ea47b7b8fe2a8ab2))


### Features

* add CFAR detection with peak_to_noise_ratio metric ([7847482](https://github.com/JuliaGNSS/Acquisition.jl/commit/7847482a4810ae4615e0ad29fa4804f125ab4f01))

# [1.6.0](https://github.com/JuliaGNSS/Acquisition.jl/compare/v1.5.0...v1.6.0) (2026-02-27)


### Features

* Doppler batching and fused GPU kernels for KAAcquisitionPlan ([7d27be7](https://github.com/JuliaGNSS/Acquisition.jl/commit/7d27be7cb320f0e498ae3733fb42bccf02b731b2))

# [1.5.0](https://github.com/JuliaGNSS/Acquisition.jl/compare/v1.4.0...v1.5.0) (2026-02-26)


### Bug Fixes

* eliminate benchmark regressions and add DBZP to KA path ([318964f](https://github.com/JuliaGNSS/Acquisition.jl/commit/318964fdcb79aca31be44babc88b091afa44799f))
* include code Doppler in code phase computation ([a61ab93](https://github.com/JuliaGNSS/Acquisition.jl/commit/a61ab93be9b4e94cec5a6cc5eba6c7fe356a8492))


### Features

* add code Doppler compensation to CPU acquisition ([9c2502a](https://github.com/JuliaGNSS/Acquisition.jl/commit/9c2502ac8df0ff1c12e40d633080ec59ab6d9eca)), closes [hi#Doppler](https://github.com/hi/issues/Doppler)
* add code Doppler compensation to KAAcquisitionPlan ([57012f3](https://github.com/JuliaGNSS/Acquisition.jl/commit/57012f3905cab5ae3ee65a89eab068a5efb91cc7))
* add DBZP for linear correlation and fix acquire() code Doppler grid ([eb3bd96](https://github.com/JuliaGNSS/Acquisition.jl/commit/eb3bd9675fcd2fdab3ecafca5db7ad106688b6a0))
* add frequency-domain zero-padding for improved code phase resolution ([b0748dc](https://github.com/JuliaGNSS/Acquisition.jl/commit/b0748dc054a586b6a0917644b11cedd4234511a8))
* pass code_doppler_tolerance through convenience functions ([c52ca4a](https://github.com/JuliaGNSS/Acquisition.jl/commit/c52ca4a4bd7903a4b4575e444793c1495835dd65))


### Performance Improvements

* optimize KA acquire! and tighten test tolerances ([4b0879e](https://github.com/JuliaGNSS/Acquisition.jl/commit/4b0879e272b97a910168bcf2825c0fd9fd9e1a18))

# [1.4.0](https://github.com/JuliaGNSS/Acquisition.jl/compare/v1.3.1...v1.4.0) (2026-02-17)


### Features

* compute Doppler step dynamically from coherent integration time ([60a5b19](https://github.com/JuliaGNSS/Acquisition.jl/commit/60a5b191aa98cf8d07ee89554ba398d84dbe4635))

## [1.3.1](https://github.com/JuliaGNSS/Acquisition.jl/compare/v1.3.0...v1.3.1) (2026-02-17)


### Performance Improvements

* cache FFTW wisdom with Scratch.jl for faster FFT planning ([25a7b06](https://github.com/JuliaGNSS/Acquisition.jl/commit/25a7b0608237632badbce99748353f2ba9d44d60))

# [1.3.0](https://github.com/JuliaGNSS/Acquisition.jl/compare/v1.2.0...v1.3.0) (2026-02-10)


### Features

* add non-coherent integration for signals longer than bit period ([a0dcbc8](https://github.com/JuliaGNSS/Acquisition.jl/commit/a0dcbc808e40f98d013eecf14e431f697b691378))

# [1.2.0](https://github.com/JuliaGNSS/Acquisition.jl/compare/v1.1.0...v1.2.0) (2026-01-26)


### Features

* make acquire! allocation-free for result handling ([55c84b5](https://github.com/JuliaGNSS/Acquisition.jl/commit/55c84b54c7f122cc9881b23e71fca712a1bf2a10))

# [1.1.0](https://github.com/JuliaGNSS/Acquisition.jl/compare/v1.0.0...v1.1.0) (2026-01-14)


### Features

* add KernelAbstractions GPU-accelerated acquisition ([3063bf7](https://github.com/JuliaGNSS/Acquisition.jl/commit/3063bf7ab08c37346469069478bd4eb633a6c7ef))

# [0.3.0](https://github.com/JuliaGNSS/Acquisition.jl/compare/v0.2.2...v0.3.0) (2026-01-12)


### Features

* add Base.show methods for AcquisitionResults ([cfa2441](https://github.com/JuliaGNSS/Acquisition.jl/commit/cfa2441bc6f17cb135496802cecd7b8c6126c3e4))

## [0.2.2](https://github.com/JuliaGNSS/Acquisition.jl/compare/v0.2.1...v0.2.2) (2026-01-10)


### Performance Improvements

* reduce allocations in acquire! from 136 to 20 ([752da10](https://github.com/JuliaGNSS/Acquisition.jl/commit/752da10316b51dffbcd9472ac247e7782c3abbdd))

## [0.2.1](https://github.com/JuliaGNSS/Acquisition.jl/compare/v0.2.0...v0.2.1) (2025-12-11)


### Bug Fixes

* update PrettyTables API for v3 compatibility ([#27](https://github.com/JuliaGNSS/Acquisition.jl/issues/27)) ([07ace33](https://github.com/JuliaGNSS/Acquisition.jl/commit/07ace332538ef697d888d35de90dee32a524afe0))

# [0.2.0](https://github.com/JuliaGNSS/Acquisition.jl/compare/v0.1.2...v0.2.0) (2025-12-09)


### Features

* allow to set minimum doppler and have a better default step ([#26](https://github.com/JuliaGNSS/Acquisition.jl/issues/26)) ([67a48c5](https://github.com/JuliaGNSS/Acquisition.jl/commit/67a48c5853f98eee020544b4fab8dc5d0ab7f95f))
