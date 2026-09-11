# Lewiscrypt HUB

Universal Windows face-recognition build by Lewis.

## Runtime behavior

Lewis checks the machine before choosing an inference backend:

1. NVIDIA GPU + modern CUDA stack that passes a real inference test -> NVIDIA GPU.
2. NVIDIA GPU where the modern stack fails but the legacy-compatible stack works -> NVIDIA GPU.
3. No usable NVIDIA CUDA backend -> CPU.

Provider visibility alone is not trusted: Lewis executes the face detector during the GPU smoke test so unsupported GPU kernels are caught before normal use.

## First run / offline behavior

The first run can install missing Python components and therefore normally needs Internet access. Once `.venv` and the required packages exist, later launches reuse them and do not reinstall them.

If a required component is missing while offline, Lewis stops with a clear message rather than repeatedly trying to install packages.

## CLI

The launcher uses Rich for the Lewis banner, status messages and human-readable setup output.
