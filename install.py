#!/usr/bin/env python3
"""
Installation script for ComfyUI TripoSR Simple.

This script helps set up the required dependencies and model files.
Run this script from the ComfyUI/custom_nodes/comfyui-triposr-simple directory.
"""

import os
import sys
import subprocess
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


def install_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")

    requirements_file = Path(__file__).parent / "requirements.txt"

    # Install requirements
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
    ])

    # Install TripoSR
    print("\nInstalling TripoSR...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/VAST-AI-Research/TripoSR.git"
    ])

    print("\nDependencies installed successfully!")


def setup_model_directory():
    """Create the triposr models directory."""
    comfyui_dir = get_comfyui_dir()
    if comfyui_dir is None:
        print("Warning: Could not find ComfyUI root directory.")
        print("Please manually create: ComfyUI/models/triposr/")
        return None

    triposr_models_dir = comfyui_dir / "models" / "triposr"
    triposr_models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTripoSR models directory created at: {triposr_models_dir}")
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
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "huggingface_hub"
        ])
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
    has_egl = os.path.exists("/usr/lib/x86_64-linux-gnu/libEGL.so") or \
              os.path.exists("/usr/lib/libEGL.so")
    has_osmesa = os.path.exists("/usr/lib/x86_64-linux-gnu/libOSMesa.so") or \
                 os.path.exists("/usr/lib/libOSMesa.so")

    if has_egl:
        print("EGL found - headless rendering should work.")
        print("Set PYOPENGL_PLATFORM=egl if you encounter issues.")
    elif has_osmesa:
        print("OSMesa found - headless rendering should work.")
        print("Set PYOPENGL_PLATFORM=osmesa if you encounter issues.")
    else:
        print("Warning: Neither EGL nor OSMesa found.")
        print("Headless rendering may not work on this system.")
        print("\nTo install on Ubuntu/Debian:")
        print("  sudo apt-get install libegl1-mesa-dev libosmesa6-dev")
        print("\nAlternatively, run with a display (X11) available.")


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

    # Step 2: Setup model directory
    models_dir = setup_model_directory()

    # Step 3: Ask about model download
    if models_dir:
        response = input("\nDownload TripoSR model from HuggingFace? (~1GB) [y/N]: ")
        if response.lower() in ('y', 'yes'):
            download_model(models_dir)
        else:
            print("\nSkipping model download.")
            print(f"Please manually place model.ckpt and config.yaml in: {models_dir}")

    # Step 4: Check headless rendering
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
