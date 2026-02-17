import time

import numpy as np
import torch
from torch import nn
from torch.optim import LBFGS

# Suppose:
# - DataClass is your data wrapper class (e.g. STL_2D_Kernel_Torch or STL_2D_FFT_Torch)
# - st_op is an operator such that st_op.apply(DC).to_flatten()
#   returns a 1D tensor of scattering coefficients.
# - target.array is your reference map of shape (1,1,128,128)
#   and target_stats is the corresponding scattering vector.


class ScatteringMatchModel(nn.Module):
    def __init__(
        self, st_op, DataClass, pbc, init_shape, compute_cross_matrix, device, dtype
    ):
        super().__init__()
        self.st_op = st_op
        self.DataClass = DataClass
        self.pbc = pbc
        self.init_shape = init_shape
        self.mask_full_res = st_op.wavelet_op.mask_full_res
        self.compute_cross_matrix = compute_cross_matrix

        # Learnable field u
        self.u = torch.randn(
            init_shape, device=device, dtype=dtype
        )  # WARNING: if filtered to match PS constraints in the future, think about sampling non PBC fields when asked!!!

        # for security put large values which should raise abberant values if actually used
        if self.mask_full_res is not None:
            self.u[..., self.mask_full_res.array] = (
                1e10  ##################################### IMPORTANT TO KEEP IN MIND
            )

        self.u.requires_grad_()

        if False:  # self.mask_full_res is not None:

            def freeze_hook(grad):
                return grad * (~self.mask_full_res.array)

            self.u.register_hook(freeze_hook)
            print(
                "NaN detected in the running synthesis mask, the synthesis takes it into account"
            )

    def forward(self):
        DC_u = self.DataClass(self.u, pbc=self.pbc)
        st_u = self.st_op.apply(
            DC_u, compute_cross_matrix=self.compute_cross_matrix, norm="load_ref"
        )
        s_flat_u = st_u.to_flatten(mean_along_batch=True, keepnans=True)
        return s_flat_u


def optimize_scattering_LBFGS(
    target,
    st_op_target,
    st_op_running,
    pbc_running=True,
    compute_cross_matrix=None,
    max_iter=100,
    nbatch=1,
    lr=1.0,
    history_size=50,
    print_iter=10,
    verbose=True,
    seed=None,
):
    device = st_op_running.wavelet_op.device
    dtype = st_op_running.wavelet_op.dtype
    print("Running synthesis on device", device, "dtype", dtype)

    torch.manual_seed(seed) if seed is not None else None

    if target.array.isnan().any():
        print("NaN detected in the target, the synthesis takes it into account")

    target.array = target.array.to(
        device=device, dtype=dtype
    )  # dtype is now float64 on cuda but was previously float32, if it slows down your synthesis raise us this issue !!!!!

    input_dim = target.array.ndim

    if input_dim == 2:
        init_shape = (nbatch, 1, *target.array.shape)
    elif input_dim == 3:
        init_shape = (nbatch, *target.array.shape)
    else:
        raise ValueError("target.array must be 2D or 3D tensor")
    print("Initial shape for u:", init_shape)

    # Reference scattering
    with torch.no_grad():
        target_stats = st_op_target.apply(
            target, norm="store_ref", compute_cross_matrix=compute_cross_matrix
        ).to_flatten(keepnans=True)
    target_stats = target_stats.detach()
    target_coeffs_mask = ~target_stats.isnan()
    target_stats = target_stats[target_coeffs_mask]
    print("Synthesis on {:} ST coefficients".format(target_coeffs_mask.sum().item()))

    # reference to running normalization
    st_op_running.S2_ref_sqrt_chan_diag = st_op_target.S2_ref_sqrt_chan_diag
    st_op_running.mean_ref = st_op_target.mean_ref
    st_op_running.var_ref = st_op_target.var_ref

    # Model with learnable u
    model = ScatteringMatchModel(
        st_op=st_op_running,
        DataClass=target.__class__,
        pbc=pbc_running,
        init_shape=init_shape,
        compute_cross_matrix=compute_cross_matrix,
        device=device,
        dtype=dtype,
    )

    optimizer = LBFGS(
        [model.u],
        lr=lr,
        max_iter=max_iter,  # <-- le nombre d'itérations internes LBFGS
        history_size=history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
    )

    loss_history = []

    def closure():
        optimizer.zero_grad()
        s_flat_u = model()
        # assert not torch.any(s_flat_u[target_coeffs_mask].isnan()) ####################### sanity check that can be removed
        loss = ((s_flat_u[target_coeffs_mask] - target_stats).abs() ** 2).sum()
        loss.backward()

        if len(loss_history) < 2:
            if False:  ####################### set to True to debug backprop
                import matplotlib.pyplot as plt

                plt.imshow(model.u[0, 0].detach().cpu().numpy()), plt.title(
                    "running u"
                ), plt.colorbar(), plt.show()
                plt.imshow(model.u.grad[0, 0].cpu().numpy()), plt.title(
                    "running u grad"
                ), plt.colorbar(), plt.show()
                plt.imshow(model.u.grad[0, 0].cpu().numpy() == 0), plt.title(
                    "running u grad == 0"
                ), plt.colorbar(), plt.show()

        # assert model.u.grad.isnan().sum().cpu().item() == 0 ####################### sanity check that can be removed

        # Log à chaque appel interne
        loss_val = loss.item()
        loss_history.append(loss_val)
        if verbose:
            if len(loss_history) % print_iter == 0:
                print(f"[LBFGS] inner iter {len(loss_history)}, loss = {loss_val:.6e}")

        return loss

    start = time.perf_counter()
    # Un seul appel : toutes les itérations LBFGS internes sont faites ici
    optimizer.step(closure)
    end = time.perf_counter()

    print(
        "{:} iterations of synthesis done with nbatch={:} and {:} ST coefficients".format(
            len(loss_history), nbatch, target_coeffs_mask.sum().item()
        )
    )
    print(f"Execution time: {end - start:.3f} s")

    u_opt = model.u.detach()
    if st_op_running.wavelet_op.mask_full_res is not None:
        u_opt[..., st_op_running.wavelet_op.mask_full_res.array] = torch.nan

    if input_dim == 2:
        u_opt = u_opt[:, 0, ...]  # remove channel dim
        target.array = target.array[0, 0, ...]  # remove batch and channel dim
    if nbatch == 1:
        u_opt = u_opt[0]  # remove batch dim

    return u_opt, loss_history
