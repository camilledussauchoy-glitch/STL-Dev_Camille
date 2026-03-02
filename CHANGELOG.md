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


## [v1.1.0] - 2026-02-13
### Added
- Reduction of scattering cov coefficients (thanks Sebastien!)
- Power spectrum computation for the FFT class (works with both `pbc=True` and `pbc=False`)
- Refactor: factorization of `DataClass` into an abstract base data class
- Steerable bump wavelets in Fourier space

### Fixed
- Normalization fix in Fourier downgrade: ensures real-space means match downsampled maps

### Known limitations
- FFT handling for `pbc=False` is not fully supported yet


## [v1.2.0] - 2026-02-22
### Added
- Sihao reduced scattering statistics (one fewer convolution for S3 and S4). Enable this feature by setting the `has_fewer_convolutions` argument in the `st_operator` constructor.
- Support for `pbc=False` in FFT, regardless of the defined wavelet type (`gaussian`, `bump_steerable`, etc.).
- Power spectrum computation for the Kernel class (works with both `pbc=True` and `pbc=False`)
- Cleaned up user notebook for computing and comparing scattering coefficients.