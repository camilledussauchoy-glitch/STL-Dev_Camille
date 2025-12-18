#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday Nov 26 2025

Example methods for a test data type.

2D planar maps with convolution using kernel.

This class makes all computations in torch.

Characteristics:
    - in pytorch
    - assume real maps
    - N0 gives x and y sizes for array shaped (..., Nx, Ny).
    - masks are supported in convolutions
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from STL_main.torch_backend import (
    _DEFAULT_DEVICE,
    _DEFAULT_DTYPE,
    _get_device,
    _get_dtype,
    maskmean,
    nan,
    to_torch_tensor,
)


###############################################################################
###############################################################################
class STL_2D_Kernel_Torch:
    """
    Class which contain the different types of data used in STL.
    Store important parameters, such as DT, N0, and the Fourier type.
    Also allow to convert from numpy to pytorch (or other type).
    Allow to transfer internally these parameters.

    Has different standard functions as methods (
    modulus, mean, cov, downsample)

    The initial resolution N0 is fixed, but the maps can be downgraded. The
    downgrading factor is the power of 2 that is used. A map of initial
    resolution N0=256 and with dg = 3 is thus at resolution 256/2^3 = 32.
    The downgraded resolutions are called N0, N1, N2, ...

    Can store array at a given downgradind dg:
        - attribute MR is False
        - attribute N0 gives the initial resolution
        - attribute dg gives the downgrading level
        - array is an array of size (..., N) with N = N0 // 2^dg
    Or at multi-resolution (MR):
        - attribute MR is True
        - attribute N0 gives the initial resolution
        - attribute dg is None
        - array is a list of array of sizes (..., N1), (..., N2), etc.,
        with the same dimensions excepts N.

    Method usages if MR=True.
        - mean, cov give a single vector or last dim len(list_N)
        - downsample gives an output of size (..., len(list_N), Nout). Only
          possible if all resolution are downsampled this way.

    The class initialization is the frontend one, which can work from DT and
    data only. It enforces MR=False and dg=0. Two backend init functions for
    MR=False and MR=True also exist.

    Attributes
    ----------
    - DT : str
        Type of data (1d, 2d planar, HealPix, 3d)
    - N0 : tuple of int
        Initial size of array (can be multiple dimensions)
    - dg : int
        2^dg is the downgrading level w.r.t. N0.
    - array : array (..., N)
          array(s) to store

    """

    ###########################################################################
    def __init__(self, array, dg=None, N0=None, history=[]):
        """
        Constructor, see details above. Frontend version, which assume the
        array is at N0 resolution with dg=0.
        """

        # Check that a signle array is given (not a list of multiple resolutions)
        if isinstance(array, list):
            raise ValueError("Only single resolution array are accepted.")

        # Main
        self.DT = "Planar2D_kernel_torch"
        if dg is None:
            self.dg = 0
            self.N0 = array.shape[-2:]
        else:
            self.dg = dg
            if N0 is None:
                raise ValueError("dg is given, N0 should not be None")
            self.N0 = N0

        self.array = self.to_array(array)

        self.device = self.array.device
        self.dtype = self.array.dtype

        self.history = history

    ###########################################################################
    def to_array(self, array):
        """
        Transform input array (NumPy or PyTorch) into a PyTorch tensor.
        Should return None if None.

        Parameters
        ----------
        array : np.ndarray or torch.Tensor
            Input array to be converted.

        Returns
        -------
        torch.Tensor
            Converted PyTorch tensor.
        """

        if array is None:
            return None
        elif isinstance(array, list):
            return array
        else:
            # Choose device: use GPU if available, otherwise CPU
            # matches the input tensor dtype to the device
            return to_torch_tensor(array)

    ###########################################################################
    def copy(self, empty=False):
        """
        Copy a STL_2D_Kernel_Torch instance.
        Array is put to None if empty==True.

        Parameters
        ----------
        - empty : bool
            If True, set array to None.

        Output
        ----------
        - STL_2D_Kernel_Torch
           copy of self
        """
        new = object.__new__(STL_2D_Kernel_Torch)

        # Copy metadata
        new.N0 = self.N0
        new.dg = self.dg
        new.device = self.device
        new.dtype = self.dtype
        new.history = self.history

        # Copy array
        if empty:
            new.array = None
        else:
            new.array = (
                self.array.clone() if isinstance(self.array, torch.Tensor) else None
            )

        return new

    ###########################################################################
    def __getitem__(self, key):
        """
        To slice directly the array attribute. Produce a view of array, to
        match with usual practices, allowing to conveniently pass only part
        of an instance.
        """
        new = self.copy(empty=True)
        new.array = self.array[key]

        return new

    ###########################################################################
    def modulus(self, inplace=False):
        """
        Compute the modulus (absolute value) of the data.
        """
        data = self.copy(empty=False) if not inplace else self

        data.array = torch.abs(data.array)

        data.dtype = data.array.dtype

        return data

    def get_wavelet_op(
        self, J=None, L=None, kernel_size=None, mask_full_res=None, *args, **kwargs
    ):
        if L is None:
            L = 4
        if kernel_size is None:
            kernel_size = 5
        if J is None:
            J = np.min([int(np.log2(self.N0[0])), int(np.log2(self.N0[1]))]) - 2
        if mask_full_res is None:
            if torch.any(self.array.isnan()):
                mask_full_res = self.array.isnan()
        return WavelateOperator2Dkernel_torch(
            kernel_size,
            L,
            J,
            device=self.array.device,
            dtype=self.array.dtype,
            mask_full_res=mask_full_res,
            *args,
            **kwargs,
        )


class WavelateOperator2Dkernel_torch:
    @staticmethod
    def _conv2d_circular(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Backend-style 2D convolution mirroring FoCUS/BkTorch strategy.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [..., Nx, Ny].
        w : torch.Tensor
            Kernel tensor of shape [O_c, wx, wy].

        Returns
        -------
        torch.Tensor
            Convolved tensor with shape [..., O_c, Nx, Ny].
        """

        *leading_dims, Nx, Ny = x.shape
        O_c, wx, wy = w.shape

        B = int(torch.prod(torch.tensor(leading_dims))) if leading_dims else 1
        x4d = x.reshape(B, 1, Nx, Ny)

        weight = w[:, None, :, :]
        pad_x = wx // 2
        pad_y = wy // 2

        x_padded = F.pad(x4d, (pad_y, pad_y, pad_x, pad_x), mode="circular")
        y = F.conv2d(x_padded, weight)

        return y.reshape(*leading_dims, O_c, Nx, Ny)

    @classmethod
    def _complex_conv2d_circular(cls, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Complex-aware wrapper around ``_conv2d_circular``."""

        xr = torch.real(x) if torch.is_complex(x) else x
        xi = torch.imag(x) if torch.is_complex(x) else torch.zeros_like(xr)

        wr = torch.real(w) if torch.is_complex(w) else w
        wi = torch.imag(w) if torch.is_complex(w) else torch.zeros_like(wr)

        real_part = cls._conv2d_circular(xr, wr) - cls._conv2d_circular(xi, wi)
        imag_part = cls._conv2d_circular(xr, wi) + cls._conv2d_circular(xi, wr)

        if torch.is_complex(x) or torch.is_complex(w):
            return torch.complex(real_part, imag_part)
        else:
            return real_part

    def __init__(
        self,
        kernel_size: int,
        L: int,
        J: int,
        device=_DEFAULT_DEVICE,
        mask_full_res=None,
        dtype=_DEFAULT_DTYPE,
        sigma_smooth=1.0,
        downsample_nan_weight_threshold=0.33,
    ):
        self.KERNELSZ = kernel_size
        self.L = L
        self.J = J
        self.device = _get_device(torch.device(device))
        self.dtype = _get_dtype(dtype=dtype, device=self.device)

        self._wav_kernel = self._build_wavelet_kernel()
        self.sigma_smooth = (
            sigma_smooth  # to build smoothing kernel used in downsampling
        )
        # raise
        # build low pass kernel?
        self.WType = "simple"

        self.mask_full_res = (
            STL_2D_Kernel_Torch(
                array=mask_full_res.to(device=self.device, dtype=torch.bool)
            )
            if mask_full_res is not None
            else None
        )  # None if no NaN in the data. Is True where the data is NaN. TODO: add a more flexible constructor later..!!!
        self.downsample_nan_weight_threshold = downsample_nan_weight_threshold
        (
            self._reweighting_maps_smooth,
            self._reweighting_maps_wav,
            self._masks_wav_conv,
        ) = self._build_reweighting_maps()

        self.layer1_mask, self.layer2_mask = self._build_scattering_layer_masks()

    def _build_reweighting_maps(self):
        if self.mask_full_res is None:
            return None, None, None
        else:
            # 1) reweighting maps needed in downsampling of data mask smoothed
            local_nan_weight_maps_smooth = {}
            smooth_kernel = self._gaussian_kernel_5x5(
                device=self.mask_full_res.array.device, dtype=self.dtype
            )
            assert torch.isclose(
                smooth_kernel.sum(), torch.tensor(1.0, dtype=smooth_kernel.dtype)
            )

            # no need for reweighting at resolution dg=0.
            for dg in range(1, self.J):
                parent_array = (
                    local_nan_weight_maps_smooth[dg - 1]
                    .array.isnan()
                    .to(dtype=self.dtype)
                    if dg > 1
                    else self.mask_full_res.array.to(dtype=self.dtype)
                )
                local_nan_weight_maps_smooth[dg] = STL_2D_Kernel_Torch(
                    array=self._downsample_tensor(
                        x=parent_array,
                        smooth_kernel=smooth_kernel,
                        dg_inc=1,
                    ),
                    dg=dg,
                    N0=self.mask_full_res.N0,
                )  # local nan fraction

                local_nan_weight_maps_smooth[dg].array = torch.where(
                    condition=local_nan_weight_maps_smooth[dg].array
                    <= self.downsample_nan_weight_threshold,
                    input=local_nan_weight_maps_smooth[dg].array,
                    other=nan,
                )  # replace with nan where above threshold

            # 2) reweighting maps needed in downsampling of data mask convolved with wavelets
            wav_kernels_envelope = torch.ones(
                self._wav_kernel.shape[-2:], dtype=self.dtype, device=self.device
            ).unsqueeze(
                0
            )  # (1,K,K) assumes identical wavelet support for all angles
            local_nan_weight_maps_wav = {}

            # Need to know NaNs position of a mask at any dg convolved with wavelets.
            masks_wav_conv = {
                dg: STL_2D_Kernel_Torch(
                    array=torch.abs(
                        self.__class__._conv2d_circular(
                            x=(
                                self.mask_full_res.array
                                if dg == 0
                                else local_nan_weight_maps_smooth[dg].array.isnan()
                            ).to(dtype=self.dtype),
                            w=wav_kernels_envelope,  # assumes identical wavelet support for all angles
                        ).squeeze(0)
                    )
                    > 0.0,
                    dg=dg,
                    N0=self.mask_full_res.N0,
                )
                for dg in range(self.J)
            }

            # no need for reweighting at resolution dg=0
            local_nan_weight_maps_wav = {
                dg: {} for dg in range(1, self.J)
            }  # j in range(dg-1)
            for j in range(self.J - 1):  # level at which the map was convolved
                for dg in range(j + 1, self.J):  # target level of downsampling
                    if (
                        dg == j + 1
                    ):  # needs to convolve with wavelets' support before downsampling
                        parent_array = masks_wav_conv[j].array.to(dtype=self.dtype)
                    else:  # dg > j+1, needs only to downsample with a smoothing from previous level
                        parent_array = (
                            local_nan_weight_maps_wav[dg - 1][j]
                            .array.isnan()
                            .to(dtype=self.dtype)
                        )

                    local_nan_weight_maps_wav[dg][j] = STL_2D_Kernel_Torch(
                        array=self._downsample_tensor(
                            x=parent_array,
                            smooth_kernel=smooth_kernel,
                            dg_inc=1,
                        ),
                        dg=dg,
                        N0=self.mask_full_res.N0,
                    )  # (Ndg,Ndg) local nan fraction

                    local_nan_weight_maps_wav[dg][j].array = torch.where(
                        condition=local_nan_weight_maps_wav[dg][j].array
                        <= self.downsample_nan_weight_threshold,
                        input=local_nan_weight_maps_wav[dg][j].array,
                        other=nan,
                    )  # (Ndg,Ndg) replace with nan where above threshold

            reweighting_maps_smooth = local_nan_weight_maps_smooth
            reweighting_maps_wav = local_nan_weight_maps_wav
            for dg in range(1, self.J):
                reweighting_maps_smooth[dg].array = 1.0 / (
                    1.0 - reweighting_maps_smooth[dg].array
                )
                for j in range(dg):
                    reweighting_maps_wav[dg][j].array = 1.0 / (
                        1.0 - reweighting_maps_wav[dg][j].array
                    )

            return reweighting_maps_smooth, reweighting_maps_wav, masks_wav_conv

    def _build_scattering_layer_masks(self):
        if self.mask_full_res is None:
            return None, None
        else:
            wav_kernels_envelope = torch.ones(
                self._wav_kernel.shape[-2:], dtype=self.dtype, device=self.device
            ).unsqueeze(
                0
            )  # (1,K,K) assumes identical wavelet support for all angles

            layer1_mask = {
                j3: STL_2D_Kernel_Torch(
                    array=self.__class__._conv2d_circular(
                        x=(
                            self._reweighting_maps_smooth[j3].array.isnan()
                            if j3 > 0
                            else self.mask_full_res.array
                        ).to(
                            dtype=self.dtype
                        ),  # can also be computed as a downsampling of mask_full_res at dg=j3 then convolved with wav_kernels_envelope
                        w=wav_kernels_envelope,
                    ).squeeze(
                        0
                    ),  # will be set to True iff > 0.0 before return
                    dg=j3,
                    N0=self.mask_full_res.N0,
                    history=[j3],
                )
                for j3 in range(self.J)
            }  # {J: (N3)} one mask per scale j3 at resolution dg=j3, same for all angles

            layer2_mask = {
                j3: {
                    j2: self.downsample(
                        data=layer1_mask[j2].copy(),
                        dg_out=j3,
                        inplace=True,
                        replace_nan_value=nan,
                    )
                    for j2 in range(j3 + 1)
                }
                for j3 in range(self.J)
            }

            for j3 in range(self.J):
                layer1_mask[j3].array = layer1_mask[j3].array > 0.0
                for j2 in range(j3 + 1):
                    layer2_mask[j3][j2].array = (
                        self.__class__._conv2d_circular(
                            x=(
                                layer2_mask[j3][j2].array.isnan()
                                if j2 < j3
                                else layer2_mask[j3][j2].array
                            ).to(dtype=self.dtype),
                            w=wav_kernels_envelope,
                        )
                        .squeeze(0)
                        .squeeze(0)
                        > 0.0
                    )
            return layer1_mask, layer2_mask

    def _find_mask(self, data):
        if self.mask_full_res is None:
            return None
        else:
            layer = len(data.history)
            if layer == 0:
                raise NotImplementedError(
                    "So far, data mask should not be called for data at layer 0."
                )
            assert data.dg == data.history[-1]
            if layer == 1:
                return self.layer1_mask[data.history[-1]].array
            elif layer == 2:
                return self.layer2_mask[data.history[-1]][data.history[0]].array

    def _build_wavelet_kernel(self, sigma=1):
        """Create a 2D Wavelet kernel."""

        # Morlay wavelet
        coords = (
            torch.arange(self.KERNELSZ, device=self.device, dtype=self.dtype)
            - (self.KERNELSZ - 1) / 2.0
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        # Gaussian envelope
        gaussian_envelope = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        # Orientations
        angles = (
            torch.arange(self.L, device=self.device, dtype=self.dtype)
            / self.L
            * torch.pi
        )

        # Morlet wavelet: exp(i*k0*x_rot) * gaussian_envelope
        # x_rot is the coordinate along the orientation direction
        x_rot = xx[None, :, :] * torch.cos(angles[:, None, None]) + yy[
            None, :, :
        ] * torch.sin(angles[:, None, None])

        # Complex Morlet wavelet
        kernel = torch.exp(1j * 0.75 * np.pi * x_rot) * gaussian_envelope[None, :, :]

        # Remove DC component (admissibility condition)
        kernel = kernel - torch.mean(kernel, dim=(1, 2))[:, None, None]

        # L2 normalization
        kernel = (
            kernel
            / torch.sqrt(torch.sum(torch.abs(kernel) ** 2, dim=(1, 2)))[:, None, None]
        )

        return kernel.reshape(1, self.L, self.KERNELSZ, self.KERNELSZ)

    def mean(self, data, square=False, dim=(-2, -1)):
        """
        Compute the mean on the last two dimensions (Nx, Ny).
        """
        return maskmean(
            data.array,
            square=square,
            dim=dim,
            mask=self._find_mask(data),
        )

    def cov(self, data1, data2=None, remove_mean=False, dim=(-2, -1)):
        """
        Compute the covariance between data1=self and data2 on the last two
        dimensions (Nx, Ny).
        """
        if data2 is None:
            data2 = data1
            mask = self._find_mask(data1)
        else:
            assert (
                data1.dg == data2.dg
            ), "data1 and data2 must have the same resolution."

            # finding the appropriate mask
            if self.mask_full_res is None:
                mask = None
            else:
                if len(data1.history) > len(
                    data2.history
                ):  # mask for |I*psi2|*psi3 contains the one for I*psi3
                    mask = self._find_mask(data1)
                elif len(data1.history) < len(
                    data2.history
                ):  # mask for |I*psi2|*psi3 contains the one for I*psi3
                    mask = self._find_mask(data2)
                else:  # mask for |I*psi2|*psi3 does not necessarily contains the one for |I*psi1|*psi3, and vice-versa
                    mask = self._find_mask(data1) + self._find_mask(data2)

        x = data1.array
        y = data2.array

        if remove_mean:
            raise NotImplementedError(
                "remove_mean is not yet implemented. think about giving the right mask when doing it"
            )
            # x_c = x - x.mean(dim=dim, keepdim=True)
            # y_c = y - y.mean(dim=dim, keepdim=True)
        else:
            x_c = x
            y_c = y

        cov = maskmean(
            x_c * y_c.conj(),
            square=False,
            dim=dim,
            mask=mask,
        )

        return cov

    def apply(self, data, j):
        """
        Apply the convolution kernel to data.array [..., Nx, Ny]
        and return cdata [..., L, Nx, Ny].

        Parameters
        ----------
        data : object
            Object with an attribute `array` storing the data as a tensor
            or numpy array with shape [..., Nx, Ny].

        Returns
        -------
        torch.Tensor
            Convolved data with shape [..., L, Nx, Ny].
        """
        if j != data.dg:
            raise ValueError("j is not equal to dg, convolution not possible")

        x = data.array  # [..., Nx, Ny]

        # Ensure x is a torch tensor on the same device / dtype as the _wav_kernel
        x = torch.as_tensor(
            x, device=self._wav_kernel.device, dtype=self._wav_kernel.dtype
        )

        weight = self._wav_kernel.squeeze(0)  # [L, K, K]

        convolved = self.__class__._complex_conv2d_circular(x, weight)

        return STL_2D_Kernel_Torch(
            convolved, dg=data.dg, N0=data.N0, history=data.history + [j]
        )

    @staticmethod
    def _downsample_tensor(
        x: torch.Tensor, smooth_kernel: torch.Tensor, dg_inc: int
    ) -> torch.Tensor:
        """
        Downsample a tensor by a factor 2**dg_inc along the last two
        dimensions using (successive iterations of, if dg_inc > 1) torch.conv2d with stride=2.

        Requires that both spatial dimensions be divisible by 2**dg_inc.
        """
        if dg_inc < 0:
            raise ValueError("dg_inc must be non-negative")
        if dg_inc == 0:
            return x

        scale = 2**dg_inc
        H, W = x.shape[-2:]
        if H % scale != 0 or W % scale != 0:
            raise ValueError(
                f"Cannot downsample from ({H},{W}) by 2^{dg_inc}: "
                "dimensions must be divisible."
            )
        if len(smooth_kernel.shape) != 2:
            raise ValueError("Smooth kernel must be of dimension 2.")
        if smooth_kernel.shape[0] != smooth_kernel.shape[1]:
            raise ValueError("Smooth kernel must be a square.")
        if smooth_kernel.shape[-1] % 2 == 0:
            raise ValueError("Smooth kernel side length must be odd.")

        leading_dims = x.shape[:-2]
        B = int(torch.prod(torch.tensor(leading_dims))) if leading_dims else 1
        y = x.reshape(B, 1, H, W)

        for _ in range(dg_inc):
            h, w = y.shape[-2:]
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError(
                    "Downsampling requires even spatial dimensions at each step."
                )
            # Add circular padding for periodic boundaries
            pad = smooth_kernel.shape[-1] // 2
            y_padded = F.pad(
                y, (pad, pad, pad, pad), mode="circular"
            )  # TODO: check if really necessary to pad with PBC.
            y = F.conv2d(
                input=y_padded, weight=smooth_kernel.unsqueeze(0).unsqueeze(0), stride=2
            )

        H2, W2 = y.shape[-2:]
        return y.reshape(*leading_dims, H2, W2)

    ###########################################################################
    def downsample(self, data, dg_out, inplace=True, replace_nan_value=nan):
        """
        Downsample the data to the dg_out resolution.
        Downsampling is done in real space along the last two dimensions using (successive iterations of, if dg_out - dg > 1) torch.conv2d with stride=2.
        If a mask is provided at full resolution, the downsampling is nan-aware, and sufficiently isolated NaNs can be removed through local averaging.
        """
        if dg_out < 0:
            raise ValueError("dg_out must be non-negative.")
        if dg_out == data.dg and inplace:
            return data
        if dg_out < data.dg:
            raise ValueError(
                "Requested dg_out < current dg; upsampling not supported by downsampling method."
            )

        data = data.copy(empty=False) if not inplace else data
        dg_inc = dg_out - data.dg

        if dg_inc > 0:
            smooth_kernel = self._gaussian_kernel_5x5(
                device=data.array.device, dtype=data.array.dtype
            )
            if self.mask_full_res is None:  # no mask
                data.array = self._downsample_tensor(
                    x=data.array, smooth_kernel=smooth_kernel, dg_inc=dg_inc
                )
                data.dg = dg_out
            else:  # mask
                if len(data.history) == 0:
                    convolved_at = None
                else:
                    assert (
                        len(data.history) < 2
                    ), "data must be at layer 0 or 1 to be downsampled."
                    convolved_at = data.history[0]

                if convolved_at is None:
                    if data.dg == 0:
                        input_data_mask = self.mask_full_res.array
                    else:
                        input_data_mask = self._reweighting_maps_smooth[
                            data.dg
                        ].array.isnan()
                else:
                    if data.dg < convolved_at:
                        raise ValueError(
                            "convolved_at level must be greater than or equal to input data resolution."
                        )
                    if data.dg == convolved_at:
                        input_data_mask = self._masks_wav_conv[data.dg].array
                    else:
                        input_data_mask = self._reweighting_maps_wav[data.dg][
                            convolved_at
                        ].array.isnan()

                data.array = torch.where(
                    condition=~input_data_mask,
                    input=data.array,
                    other=0.0,
                )

                for _ in range(
                    dg_inc
                ):  # downsampling is done step by step to apply reweighting at each step
                    data.array = self._downsample_tensor(
                        x=data.array, smooth_kernel=smooth_kernel, dg_inc=1
                    )
                    data.dg += 1

                    reweighting_map = (
                        self._reweighting_maps_smooth[data.dg]
                        if convolved_at is None
                        else self._reweighting_maps_wav[data.dg][convolved_at]
                    )

                    data.array *= torch.where(
                        condition=~reweighting_map.array.isnan(),
                        input=reweighting_map.array,
                        other=0.0,
                    )  # reweighting while avoiding to thrwow NaNs into data.attay for backprop

                if replace_nan_value is not None:
                    data.array = torch.where(
                        condition=~reweighting_map.array.isnan(),
                        input=data.array,
                        other=replace_nan_value,
                    )  # put a large value instead of NaNs WARNING: if applied, this breaks the backprop!!!
        return data

    def _gaussian_kernel_5x5(self, device, dtype):
        """
        Build and cache a normalized 5x5 Gaussian kernel on (device, dtype)
        for antialiasing 2D filter used in downsampling.

        Returns
        -------
        kernel : torch.Tensor
            Shape (5, 5)
        """
        if (
            not hasattr(self, "_smooth_kernel_5x5")
            or self._smooth_kernel_5x5.device != device
            or self._smooth_kernel_5x5.dtype != dtype
        ):
            size = 5
            coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
            kernel = torch.exp(-(xx**2 + yy**2) / (2 * self.sigma_smooth**2))
            kernel = kernel / kernel.sum()
            # _conv2d_circular expects w shape (O_c, wx, wy)
            self._smooth_kernel_5x5 = kernel
        return self._smooth_kernel_5x5

    def _apply_crop(self, data):
        return data[
            ...,
            self.KERNELSZ // 2 : -self.KERNELSZ // 2,
            self.KERNELSZ // 2 : -self.KERNELSZ // 2,
        ]
