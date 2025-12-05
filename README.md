# Graph-Root-Finding-Simulation

trials_system.ipynb builds a file system containing randomly generated recursive trees stored in compressed number arrays. Then trials_plots.ipynb generates statistics on these trees and plots resulting information. All of this is designed to run in a CUDA-accelerated environment using cupy (a GPU-accelerated numpy analog for NVIDIA GPUS), greatly improving the runtime of the generator and algorithms.
