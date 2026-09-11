"""Lewiscrypt HUB first-run/runtime setup and launcher."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PYTHON = VENV / "Scripts" / "python.exe"
BACKEND_FILE = ROOT / ".lewis_backend"
GPU_FILE = ROOT / ".lewis_gpu"


def rich_console():
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        return Console(), Panel, Table, Text
    except Exception:
        return None, None, None, None


def banner():
    console, Panel, _, Text = rich_console()
    art = r"""
██╗     ███████╗██╗    ██╗██╗███████╗
██║     ██╔════╝██║    ██║██║██╔════╝
██║     █████╗  ██║ █╗ ██║██║███████╗
██║     ██╔══╝  ██║███╗██║██║╚════██║
███████╗███████╗╚███╔███╔╝██║███████║
╚══════╝╚══════╝ ╚══╝╚══╝ ╚═╝╚══════╝

              LEWISCRYPT HUB
       Intelligent Vision • by Lewis
"""
    if console:
        console.print(Panel(Text(art, style="bold cyan"), border_style="cyan", expand=False))
    else:
        print(art)


def say(message, style=""):
    console, _, _, _ = rich_console()
    if console:
        console.print(f"[bold cyan]LEWIS >[/bold cyan] {message}")
    else:
        print(f"LEWIS > {message}")


def status(label, ok=True, detail=""):
    console, _, _, _ = rich_console()
    mark = "[bold green]✓[/bold green]" if ok else "[bold red]✗[/bold red]"
    if console:
        console.print(f"  {mark} {label}" + (f" — {detail}" if detail else ""))
    else:
        print(("  [OK] " if ok else "  [FAIL] ") + label + (f" — {detail}" if detail else ""))


def run(cmd, check=False, quiet=False):
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE if quiet else None,
                          stderr=subprocess.STDOUT if quiet else None, check=check)


def python_ok():
    return sys.version_info >= (3, 11) and sys.version_info < (3, 14) and sys.maxsize > 2**32


def nvidia_name():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return ""


def ensure_venv():
    if not PYTHON.exists():
        say("This is your first run here. I am creating my local Python environment.")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], cwd=ROOT, check=True)
    return PYTHON


def has_module(module):
    return subprocess.run([str(PYTHON), "-c", f"import {module}"],
                          cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def pip_install(requirements):
    say(f"I need to install a few components from {requirements}.")
    return subprocess.run([str(PYTHON), "-m", "pip", "install", "-r", str(ROOT / requirements)], cwd=ROOT).returncode == 0


def uninstall_ort():
    subprocess.run([str(PYTHON), "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu",
                    "nvidia-cudnn-cu12", "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12"],
                   cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ort_version():
    p = subprocess.run([str(PYTHON), "-c", "import onnxruntime as o; print(o.__version__)"],
                       cwd=ROOT, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def cuda_smoke_test():
    code = "from core.runtime import test_cuda_inference; raise SystemExit(0 if test_cuda_inference() else 1)"
    return subprocess.run([str(PYTHON), "-c", code], cwd=ROOT).returncode == 0


def cpu_smoke_test():
    code = "import onnxruntime as o; assert 'CPUExecutionProvider' in o.get_available_providers()"
    return subprocess.run([str(PYTHON), "-c", code], cwd=ROOT).returncode == 0


def setup_backend(gpu):
    current_gpu = nvidia_name()
    old_gpu = GPU_FILE.read_text(encoding="utf-8").strip() if GPU_FILE.exists() else ""
    backend = BACKEND_FILE.read_text(encoding="utf-8").strip() if BACKEND_FILE.exists() else ""

    # If the same machine already has a working backend, do not reinstall anything.
    if backend and current_gpu == old_gpu and has_module("onnxruntime"):
        if backend in ("modern-gpu", "legacy-gpu"):
            if "CUDAExecutionProvider" in get_providers() and cuda_smoke_test():
                status("NVIDIA acceleration", True, backend)
                return True
        elif backend == "cpu" and cpu_smoke_test():
            status("CPU inference", True)
            return True

    if gpu:
        say(f"I found an NVIDIA GPU: {current_gpu}")
        say("I am testing GPU acceleration instead of assuming it works.")
        # Modern stack first. It supports current CUDA/cuDNN combinations.
        if ort_version() != "1.26.0" or "CUDAExecutionProvider" not in get_providers():
            uninstall_ort()
            if not pip_install("requirements-modern-gpu.txt"):
                say("The modern GPU stack could not be installed. I will try my legacy-compatible stack.")
        if ort_version() == "1.26.0" and "CUDAExecutionProvider" in get_providers() and cuda_smoke_test():
            BACKEND_FILE.write_text("modern-gpu", encoding="utf-8")
            GPU_FILE.write_text(current_gpu, encoding="utf-8")
            status("NVIDIA acceleration", True, "modern CUDA stack")
            return True

        say("This GPU did not pass the modern CUDA test. I am trying my older-GPU stack.")
        uninstall_ort()
        if pip_install("requirements-legacy-gpu.txt") and cuda_smoke_test():
            BACKEND_FILE.write_text("legacy-gpu", encoding="utf-8")
            GPU_FILE.write_text(current_gpu, encoding="utf-8")
            status("NVIDIA acceleration", True, "legacy-compatible CUDA stack")
            return True

        say("CUDA is not usable on this machine. That is okay — I can still run on the CPU.")

    # CPU fallback.
    if ort_version() != "1.26.0" or "CUDAExecutionProvider" in get_providers():
        uninstall_ort()
        if not pip_install("requirements-cpu.txt"):
            return False
    if not cpu_smoke_test():
        return False
    BACKEND_FILE.write_text("cpu", encoding="utf-8")
    GPU_FILE.write_text(current_gpu, encoding="utf-8")
    status("CPU inference", True, "universal fallback")
    return True


def get_providers():
    p = subprocess.run([str(PYTHON), "-c", "import onnxruntime as o; print(','.join(o.get_available_providers()))"],
                       cwd=ROOT, capture_output=True, text=True)
    return p.stdout.strip().split(",") if p.returncode == 0 else []


def main():
    banner()
    if not python_ok():
        say("I need 64-bit Python 3.11, 3.12, or 3.13 on Windows.")
        return 1
    status(f"Python {sys.version_info.major}.{sys.version_info.minor} (64-bit)")
    ensure_venv()

    missing = [m for m in ("PySide6", "cv2", "numpy", "insightface", "rich") if not has_module(m)]
    if missing:
        say("Some of my core components are missing.")
        if not pip_install("requirements-base.txt"):
            say("I cannot finish first-time setup without downloading the missing components.")
            say("Connect to the Internet once, then run start.bat again. I will work offline afterwards.")
            return 1
    else:
        status("Core components", True, "already installed — no reinstall")

    gpu = bool(nvidia_name())
    if not setup_backend(gpu):
        say("I could not prepare an inference backend. No files were removed from your project.")
        return 1

    say("System ready. Starting Lewiscrypt HUB.")
    return subprocess.run([str(PYTHON), "main.py"], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
