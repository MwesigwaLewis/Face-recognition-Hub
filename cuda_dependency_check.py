import os
import sys
import ctypes
from pathlib import Path

site = Path(sys.prefix) / "Lib" / "site-packages"

dirs = [
    site / "nvidia" / "cuda_runtime" / "bin",
    site / "nvidia" / "cuda_nvrtc" / "bin",
    site / "nvidia" / "cublas" / "bin",
    site / "nvidia" / "cudnn" / "bin",
    site / "onnxruntime" / "capi",
]

print("=" * 70)
print("LEWISCRYPT - CUDA DEPENDENCY CHECK")
print("=" * 70)

# Add every NVIDIA DLL directory to the Windows DLL search path
handles = []
for d in dirs:
    if d.exists():
        try:
            handles.append(os.add_dll_directory(str(d)))
            print("DLL PATH:", d)
        except Exception as e:
            print("PATH ERROR:", d, e)

print("\n[DIRECT DEPENDENCIES]")

dlls = [
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn64_8.dll",
    "nvrtc64_120_0.dll",
    "nvrtc-builtins64_129.dll",
]

for name in dlls:
    found = None
    for d in dirs:
        candidate = d / name
        if candidate.exists():
            found = candidate
            break

    print(f"\n{name}")
    if not found:
        print("  NOT FOUND")
        continue

    try:
        ctypes.WinDLL(str(found))
        print("  LOAD OK")
    except OSError as e:
        print("  LOAD FAILED")
        print("  ERROR:", e)
        print("  WINERROR:", getattr(e, "winerror", None))

print("\n[ONNXRUNTIME CUDA PROVIDER]")

provider = site / "onnxruntime" / "capi" / "onnxruntime_providers_cuda.dll"

print(provider)

try:
    ctypes.WinDLL(str(provider))
    print("LOAD OK")
except OSError as e:
    print("LOAD FAILED")
    print("ERROR:", e)
    print("WINERROR:", getattr(e, "winerror", None))

print("\n[ENVIRONMENT]")
print("CUDA_PATH =", os.environ.get("CUDA_PATH"))
print("CUDA_PATH_V12_1 =", os.environ.get("CUDA_PATH_V12_1"))

print("\n" + "=" * 70)
input("Press Enter to exit...")