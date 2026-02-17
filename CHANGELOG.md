# Changelog

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
