# Changelog

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
