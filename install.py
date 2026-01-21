#!/usr/bin/env python3
"""
Installation script for ComfyUI TripoSR Simple.

This script helps set up the required dependencies and model files.
Run this script from the ComfyUI/custom_nodes/comfyui-triposr-simple directory.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def get_comfyui_dir():
    """Find the ComfyUI root directory."""
    current = Path(__file__).parent
    # Go up to find ComfyUI root (should be 2 levels up from custom_nodes/this_package)
    for _ in range(3):
        if (current / "main.py").exists() and (current / "custom_nodes").exists():
            return current
        current = current.parent
    return None


def get_pip_command():
    """Determine whether to use uv pip or regular pip."""
    # Check if uv is available
    if shutil.which("uv"):
        return ["uv", "pip", "install"]
    else:
        return [sys.executable, "-m", "pip", "install"]


def install_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")

    pip_cmd = get_pip_command()
    requirements_file = Path(__file__).parent / "requirements.txt"

    print(f"Using: {' '.join(pip_cmd)}")

    # Install requirements
    subprocess.check_call(pip_cmd + ["-r", str(requirements_file)])

    print("\nDependencies installed successfully!")


def install_triposr():
    """Install TripoSR from source."""
    print("\nInstalling TripoSR...")

    pip_cmd = get_pip_command()

    # Clone TripoSR repo and install dependencies manually
    # TripoSR doesn't have setup.py, so we clone and add to path
    triposr_dir = Path(__file__).parent / "TripoSR"

    if triposr_dir.exists():
        print(f"TripoSR already cloned at {triposr_dir}")
    else:
        print("Cloning TripoSR repository...")
        subprocess.check_call([
            "git", "clone", "https://github.com/VAST-AI-Research/TripoSR.git",
            str(triposr_dir)
        ])

    # Install TripoSR's requirements
    triposr_requirements = triposr_dir / "requirements.txt"
    if triposr_requirements.exists():
        print("Installing TripoSR requirements...")
        subprocess.check_call(pip_cmd + ["-r", str(triposr_requirements)])

    print("\nTripoSR installed successfully!")
    print(f"TripoSR path: {triposr_dir}")


def setup_model_directory():
    """Create the triposr models directory."""
    comfyui_dir = get_comfyui_dir()
    if comfyui_dir is None:
        print("Warning: Could not find ComfyUI root directory.")
        print("Please manually create: ComfyUI/models/triposr/")
        return None

    triposr_models_dir = comfyui_dir / "models" / "triposr"
    triposr_models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTripoSR models directory: {triposr_models_dir}")
    return triposr_models_dir


def download_model(models_dir: Path):
    """Download TripoSR model files from HuggingFace."""
    if models_dir is None:
        print("Skipping model download (models directory not found).")
        return

    print("\nDownloading TripoSR model from HuggingFace...")
    print("This will download ~1GB of model files.")

    try:
        from huggingface_hub import hf_hub_download

        # Download model checkpoint
        checkpoint_path = hf_hub_download(
            repo_id="stabilityai/TripoSR",
            filename="model.ckpt",
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        print(f"Downloaded: {checkpoint_path}")

        # Download config
        config_path = hf_hub_download(
            repo_id="stabilityai/TripoSR",
            filename="config.yaml",
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        print(f"Downloaded: {config_path}")

        print("\nModel downloaded successfully!")

    except ImportError:
        print("huggingface_hub not installed. Installing...")
        pip_cmd = get_pip_command()
        subprocess.check_call(pip_cmd + ["huggingface_hub"])
        download_model(models_dir)

    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nPlease manually download the model files from:")
        print("https://huggingface.co/stabilityai/TripoSR")
        print(f"\nPlace the files in: {models_dir}")
        print("Required files: model.ckpt, config.yaml")


def setup_headless_rendering():
    """Check and setup headless rendering environment."""
    print("\nChecking headless rendering setup...")

    # Check for EGL or OSMesa
    egl_paths = [
        "/usr/lib/x86_64-linux-gnu/libEGL.so",
        "/usr/lib/libEGL.so",
        "/usr/lib/x86_64-linux-gnu/libEGL.so.1"
    ]
    osmesa_paths = [
        "/usr/lib/x86_64-linux-gnu/libOSMesa.so",
        "/usr/lib/libOSMesa.so"
    ]

    has_egl = any(os.path.exists(p) for p in egl_paths)
    has_osmesa = any(os.path.exists(p) for p in osmesa_paths)

    if has_egl:
        print("EGL found - headless rendering should work.")
        print("The node automatically sets PYOPENGL_PLATFORM=egl")
    elif has_osmesa:
        print("OSMesa found - headless rendering should work.")
        print("You may need to set: export PYOPENGL_PLATFORM=osmesa")
    else:
        print("Warning: Neither EGL nor OSMesa found.")
        print("Headless rendering may not work on this system.")
        print("\nTo install on Ubuntu/Debian:")
        print("  apt-get install libegl1-mesa-dev libosmesa6-dev freeglut3-dev")


def main():
    """Run the full installation process."""
    print("=" * 60)
    print("ComfyUI TripoSR Simple - Installation")
    print("=" * 60)

    # Step 1: Install dependencies
    try:
        install_dependencies()
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

    # Step 2: Install TripoSR
    try:
        install_triposr()
    except Exception as e:
        print(f"Error installing TripoSR: {e}")
        print("You may need to install it manually.")

    # Step 3: Setup model directory
    models_dir = setup_model_directory()

    # Step 4: Ask about model download
    if models_dir:
        print("\n" + "-" * 60)
        response = input("Download TripoSR model from HuggingFace? (~1GB) [y/N]: ")
        if response.lower() in ('y', 'yes'):
            download_model(models_dir)
        else:
            print("\nSkipping model download.")
            print(f"Please manually place model.ckpt and config.yaml in: {models_dir}")

    # Step 5: Check headless rendering
    setup_headless_rendering()

    print("\n" + "=" * 60)
    print("Installation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Ensure model files are in ComfyUI/models/triposr/")
    print("2. Restart ComfyUI")
    print("3. Find new nodes under 'TripoSR' category")


if __name__ == "__main__":
    main()
