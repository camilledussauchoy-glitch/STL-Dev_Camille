# -----------------------------
# Imports
# -----------------------------
import os
import sys
from contextlib import contextmanager
from datetime import datetime

import google_benchmark as benchmark
import numpy as np

# -----------------------------
# Setup
# -----------------------------
# Add parent directory to sys.path to import STL package modules
PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.append(PARENT_DIR)
print("Parent directory added to sys.path:", ".../" + os.path.basename(PARENT_DIR))

# Path to test data
DATA_TEST_PATH = PARENT_DIR + "/data" + "/test"
print(
    "Dataset directory used:",
    ".../"
    + os.path.basename(PARENT_DIR)
    + DATA_TEST_PATH.split(os.path.basename(PARENT_DIR))[-1],
)

from STL_main.STL_2D_FFT_Torch import STL_2D_FFT_Torch

# Import STL modules
from STL_main.STL_2D_Kernel_Torch import STL_2D_Kernel_Torch
from STL_main.Synthesis import synthesize_from_maps
from STL_main.torch_backend import _DEFAULT_DEVICE


# Run benchmarks in silent mode to avoid cluttering benchmark output
@contextmanager
def silent():
    """Suppress stdout and stderr temporarily."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = open(os.devnull, "w")
    sys.stderr = sys.stdout
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# Setup benchmark output directory and file naming
benchmark_name = os.path.splitext(os.path.basename(__file__))[0]
base_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", benchmark_name
)
os.makedirs(base_dir, exist_ok=True)

sys.argv += [
    f"--benchmark_out={os.path.join(base_dir, f'run_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv')}",
    "--benchmark_out_format=csv",
    "--benchmark_time_unit=s",
]

# Load target image
im = np.load(DATA_TEST_PATH + "/" + "Turb_6.npy")[:, None, :, :]  # [Nb, Nc, Nx, Ny]
im_target = im[0, 0, :, :]  # Mono channel image

# Instantiate data classes
data_target_kernel = STL_2D_Kernel_Torch(im_target, pbc=True)
data_target_fft = STL_2D_FFT_Torch(im_target, pbc=True)


# -----------------------------
# Benchmark registration
# -----------------------------
@benchmark.register
def Kernel_synthesis(state):
    while state:
        with silent():
            # Call user-friendly synthesis wrapper
            u_kernel = synthesize_from_maps(
                data_target_kernel, pbc_running=True, nbatch=1
            )


@benchmark.register
def FFT_synthesis(state):
    while state:
        with silent():
            # Call user-friendly synthesis wrapper
            u_fft = synthesize_from_maps(data_target_fft, pbc_running=True, nbatch=1)


if __name__ == "__main__":
    # benchmark.add_custom_context("Working on device:", _DEFAULT_DEVICE) # TODO: Add custom context to benchmark output (currently not working as expected)
    benchmark.main()
