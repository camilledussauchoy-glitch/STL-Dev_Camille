# Changelog

## [v1.0.0] - 2026-01-27
### Added
- Calculation of scattering and cross-channel statistics
  - Available in real space via the `STL_2D_Kernel_Torch` class
  - Available in Fourier space via the `STL_2D_FFT_Torch` class
  - Support for both periodic and non-periodic images (`Kernel` & `Torch`)
  - Handling of NaN values in real space (`Kernel`)
- Image synthesis 
- User example notebook demonstrating main features (only scattering stats computation, no synthesis)

### Fixed
- N/A (first stable release)

### Breaking Changes
- N/A (first stable release)
