#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import STL_main.torch_backend as bk
from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch as DataClass


def configure_backend_defaults(device: torch.device, dtype: torch.dtype) -> None:
    if hasattr(bk, "set_default_device"):
        bk.set_default_device(device)
    else:
        bk._DEFAULT_DEVICE = device
    bk._DEFAULT_DTYPE = dtype
    bk._DEFAULT_COMPLEX_DTYPE = (
        torch.complex64 if dtype == torch.float32 else torch.complex128
    )


def nyquist_mask(shape: tuple[int, int], *, device, dtype) -> torch.Tensor:
    n, m = shape
    fy = n * torch.fft.fftfreq(n, d=1.0, device=device)
    fx = m * torch.fft.fftfreq(m, d=1.0, device=device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    return (radius <= min(n, m) / 2).to(dtype=dtype)


def apply_nyquist_filter_torch(x: torch.Tensor) -> torch.Tensor:
    mask = nyquist_mask(x.shape[-2:], device=x.device, dtype=torch.bool)
    x_fft = torch.fft.fft2(x, norm="ortho")
    x_fft = x_fft.masked_fill(~mask, 0)
    return torch.fft.ifft2(x_fft, norm="ortho").real


def apply_nyquist_filter_numpy(x: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(x)
    filtered = apply_nyquist_filter_torch(tensor).numpy()
    return filtered.astype(x.dtype, copy=False)


def imshow2d(ax, x: np.ndarray, title: str, cmap: str = "inferno") -> None:
    im = ax.imshow(x, origin="lower", cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def parse_bool(name: str, value: str) -> bool:
    value_normalized = value.strip().lower()
    if value_normalized in {"1", "true", "yes", "y"}:
        return True
    if value_normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def env_bool(*names: str, default: bool) -> bool:
    parsed: list[tuple[str, bool]] = []
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            parsed.append((name, parse_bool(name, value)))
    if not parsed:
        return default
    first_name, first_value = parsed[0]
    for name, value in parsed[1:]:
        if value != first_value:
            raise ValueError(
                f"Conflicting boolean environment variables: "
                f"{first_name}={first_value} but {name}={value}"
            )
    return first_value


def main() -> None:
    seed = int(os.environ.get("COMPSEP_SEED", "0"))
    n_noise = int(os.environ.get("COMPSEP_N_NOISE", "500"))
    n_batch = int(os.environ.get("COMPSEP_BATCH", "25"))
    n_iter = int(
        os.environ.get("COMPSEP_ITERS", os.environ.get("COMPSEP_LBFGS_MAX_ITER", "100"))
    )
    wtype = os.environ.get("COMPSEP_WTYPE", "Bump-Steerable")
    pbc = env_bool("COMPSEP_PBC", "PBC", default=False)
    compute_ps = env_bool("COMPSEP_ST_COMPUTE_PS", "ST_COMPUTE_PS", default=False)
    apply_nyquist_after = env_bool(
        "COMPSEP_APPLY_NYQUIST_AFTER", "APPLY_NYQUIST_AFTER", default=True
    )
    optimizer_lr = float(os.environ.get("COMPSEP_OPTIMIZER_LR", os.environ.get("OPTIMIZER_LR", "1")))
    out_dir_env = os.environ.get("COMPSEP_OUTDIR", os.environ.get("OUTDIR", "")).strip()

    device_override = os.environ.get("COMPSEP_DEVICE")
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dtype_str = os.environ.get("COMPSEP_DTYPE", os.environ.get("DTYPE", "float64")).strip().lower()
    if dtype_str in {"float64", "fp64", "double"}:
        dtype = torch.float64
    elif dtype_str in {"float32", "fp32", "single"}:
        dtype = torch.float32
    else:
        raise ValueError("COMPSEP_DTYPE/DTYPE must be float64 or float32")
    configure_backend_defaults(device, dtype)
    torch.manual_seed(seed)

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    print(f"Iterations: {n_iter} | noise samples: {n_noise} | batch: {n_batch}")
    print(f"PBC: {pbc} | compute_PS: {compute_ps}")
    print(f"Optimizer lr: {optimizer_lr} | apply Nyquist after optimization: {apply_nyquist_after}")

    rng = np.random.default_rng(seed)

    signal = np.load(ROOT / "data/test/patch_129_main.npy")[0].astype(np.float64)
    H, W = signal.shape
    noise_std = float(np.std(signal))

    noise_obs = rng.standard_normal(size=(H, W)).astype(np.float64) * noise_std
    noise_samples = (
        rng.standard_normal(size=(n_noise, H, W)).astype(np.float64) * noise_std
    )
    data = signal + noise_obs

    cross_matrix = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool, device=device)

    n_ref = noise_samples[int(rng.integers(0, n_noise))]
    ref_tensor = torch.from_numpy(np.stack([data, n_ref], axis=0)).to(
        device, dtype=dtype
    )
    ref_dc = DataClass(ref_tensor[None, ...], pbc=pbc)

    st_op = ref_dc.get_ST_op(compute_PS=compute_ps)
    try:
        st_op.wavelet_op = ref_dc.get_wavelet_op(J=st_op.J, L=st_op.L, WType=wtype)
    except TypeError:
        st_op.wavelet_op = ref_dc.get_wavelet_op(J=st_op.J, L=st_op.L)
    st_op.WType = getattr(st_op.wavelet_op, "WType", wtype)

    with torch.no_grad():
        st_op.apply(ref_dc, norm="store_ref", compute_cross_matrix=cross_matrix)

    data_t = torch.from_numpy(data).to(device, dtype=dtype)
    noise_t = torch.from_numpy(noise_samples).to(device, dtype=dtype)

    def make_target_batch(indices: np.ndarray) -> torch.Tensor:
        nb = int(indices.shape[0])
        batch_noise = noise_t[indices]
        batch_data = data_t[None, :, :].expand(nb, -1, -1)
        return torch.stack([batch_data, batch_noise], dim=1)

    def make_running_batch(running_signal: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
        nb = int(indices.shape[0])
        batch_noise = noise_t[indices]
        term1 = running_signal[None, :, :] + batch_noise
        term2 = (data_t - running_signal)[None, :, :].expand(nb, -1, -1)
        return torch.stack([term1, term2], dim=1)

    def stats_flat(dc: DataClass) -> torch.Tensor:
        return st_op.apply(
            dc, norm="load_ref", compute_cross_matrix=cross_matrix
        ).to_flatten(mean_along_batch=True, keepnans=False)

    def squared_l2(diff: torch.Tensor) -> torch.Tensor:
        return diff.abs().square().sum()

    running_signal = data_t.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [running_signal],
        lr=optimizer_lr,
        max_iter=1,
        tolerance_grad=-1,
        tolerance_change=-1,
        history_size=100,
        line_search_fn=None,
    )

    loss_calls: list[float] = []

    for iteration in range(n_iter):
        idx = rng.choice(n_noise, size=min(n_batch, n_noise), replace=False)

        with torch.no_grad():
            target_batch = make_target_batch(idx)
            target_flat = stats_flat(DataClass(target_batch, pbc=pbc))

        def closure():
            optimizer.zero_grad()
            running_batch = make_running_batch(running_signal, idx)
            running_flat = stats_flat(DataClass(running_batch, pbc=pbc))
            if running_flat.numel() != target_flat.numel():
                raise RuntimeError(
                    f"Flattened statistic length mismatch: "
                    f"running={running_flat.numel()} target={target_flat.numel()}."
                )
            loss = squared_l2(running_flat - target_flat)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        loss_value = float(loss.detach().cpu())
        loss_calls.append(loss_value)
        print(f"iteration {iteration + 1}/{n_iter} | loss={loss_value:.4e}")

    if apply_nyquist_after:
        with torch.no_grad():
            running_signal.copy_(apply_nyquist_filter_torch(running_signal))

    recovered = running_signal.detach().cpu().numpy()
    residual = data - recovered
    ppc = recovered + noise_obs

    out_dir = Path(out_dir_env).expanduser() if out_dir_env else ROOT / "results" / "compsep_notebook_gpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "recovered_signal.npy", recovered)
    np.save(out_dir / "loss_calls.npy", np.asarray(loss_calls, dtype=np.float64))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    imshow2d(axes[0, 0], signal, "True signal s")
    imshow2d(axes[0, 1], data, "Observed data d")
    imshow2d(axes[0, 2], recovered, "Recovered signal u")
    imshow2d(axes[1, 0], noise_obs, "One noise sample n")
    imshow2d(axes[1, 1], residual, "Residual d - u")
    imshow2d(axes[1, 2], ppc, "PPC s + n")
    fig.tight_layout()
    fig.savefig(out_dir / "mapsTest_Heal_LSS.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(loss_calls, linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("optimizer iteration")
    ax.set_ylabel("loss")
    ax.set_title("Loss vs optimizer iteration")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curveTest_Heal_LSS.png", dpi=200)
    plt.close(fig)

    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
