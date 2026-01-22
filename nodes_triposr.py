"""
TripoSR nodes for image-to-3D mesh generation.
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from .nodes_common import resize_foreground, rgba_to_rgb_gray_background


# Add TripoSR to path if cloned locally
_triposr_local = Path(__file__).parent / "TripoSR"
if _triposr_local.exists() and str(_triposr_local) not in sys.path:
    sys.path.insert(0, str(_triposr_local))


# ============================================================================
# Patch missing modules before importing TripoSR
# ============================================================================
import types

def _patch_rembg():
    """Create a fake rembg module."""
    try:
        import rembg
        return
    except ImportError:
        pass

    print("[TripoSR] rembg not found, creating stub")

    def remove(image, session=None, **kwargs):
        print("[TripoSR] Warning: rembg.remove() called but rembg not installed.")
        return image

    def new_session(*args, **kwargs):
        return None

    fake_rembg = types.ModuleType('rembg')
    fake_rembg.remove = remove
    fake_rembg.new_session = new_session
    sys.modules['rembg'] = fake_rembg

def _check_torchmcubes():
    """Check that native torchmcubes is available."""
    try:
        import torchmcubes
        print("[TripoSR] torchmcubes found and available")
    except ImportError:
        print("[TripoSR] ERROR: torchmcubes not installed!")
        raise ImportError(
            "torchmcubes is required for TripoSR mesh extraction. "
            "Install with: pip install git+https://github.com/tatsy/torchmcubes.git"
        )

# Apply patches
_patch_rembg()
_check_torchmcubes()
# ============================================================================


# Global model cache
_triposr_model_cache = {}


def get_triposr_checkpoints():
    """Get list of available TripoSR checkpoints."""
    import folder_paths

    if "triposr" not in folder_paths.folder_names_and_paths:
        triposr_path = os.path.join(folder_paths.models_dir, "triposr")
        os.makedirs(triposr_path, exist_ok=True)
        folder_paths.folder_names_and_paths["triposr"] = ([triposr_path], {".ckpt", ".safetensors", ".pt", ".pth"})

    checkpoints = folder_paths.get_filename_list("triposr")
    if not checkpoints:
        checkpoints = ["model.ckpt"]
    return checkpoints


class LoadTripoSRModel:
    """
    Loads the TripoSR model from a local checkpoint file.
    Model is cached for reuse across multiple generations.

    Place checkpoint files in: ComfyUI/models/triposr/
    """

    CATEGORY = "TripoSR"
    RETURN_TYPES = ("TRIPOSR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (get_triposr_checkpoints(),),
                "device": (["cuda:0", "cuda:1", "cpu"],),
            }
        }

    def load(self, checkpoint: str, device: str):
        import folder_paths
        global _triposr_model_cache

        cache_key = f"{checkpoint}_{device}"

        if cache_key in _triposr_model_cache:
            print(f"[TripoSR] Using cached model: {checkpoint} on {device}")
            return (_triposr_model_cache[cache_key],)

        print(f"[TripoSR] Loading model from checkpoint: {checkpoint} on {device}...")

        triposr_path = Path(__file__).parent / "TripoSR"
        if not triposr_path.exists():
            raise ImportError(
                f"TripoSR not found at {triposr_path}\n"
                "Please clone it with:\n"
                f"  cd {Path(__file__).parent}\n"
                "  git clone https://github.com/VAST-AI-Research/TripoSR.git"
            )

        if str(triposr_path) not in sys.path:
            sys.path.insert(0, str(triposr_path))

        try:
            from tsr.system import TSR
        except ImportError as e:
            raise ImportError(f"Failed to import TripoSR: {e}")

        checkpoint_path = folder_paths.get_full_path("triposr", checkpoint)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint}' not found. "
                f"Place your TripoSR checkpoint in: ComfyUI/models/triposr/"
            )

        model_dir = os.path.dirname(checkpoint_path)
        checkpoint_name = os.path.basename(checkpoint_path)

        model = TSR.from_pretrained(
            model_dir,
            config_name="config.yaml",
            weight_name=checkpoint_name
        )
        model.to(device)
        model.renderer.set_chunk_size(8192)

        _triposr_model_cache[cache_key] = model
        print(f"[TripoSR] Model loaded successfully from {checkpoint_path} on {device}")

        return (model,)


class ImageTo3DMesh:
    """
    Converts an input image to a 3D mesh using TripoSR.

    For best results, provide an image with a mask for transparent background.
    Connect MASK output from RMBG node to the mask input.
    """

    CATEGORY = "TripoSR"
    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TRIPOSR_MODEL",),
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 256, "min": 64, "max": 512, "step": 32}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    def generate(self, model, image: torch.Tensor, resolution: int, seed: int, unload_model: bool, mask: torch.Tensor = None):
        global _triposr_model_cache

        # Set seed for reproducibility
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"[TripoSR] Using seed: {seed}")

        # Free up VRAM
        try:
            import comfy.model_management as mm
            print("[TripoSR] Unloading ComfyUI models to free VRAM...")
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception as e:
            print(f"[TripoSR] Could not unload ComfyUI models: {e}")

        # Convert to PIL
        img_np = image[0].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        # If mask is provided, combine RGB image with mask to create RGBA
        if mask is not None:
            mask_np = mask[0].cpu().numpy()
            mask_np = (mask_np * 255).astype(np.uint8)

            # Ensure mask matches image size
            if mask_np.shape[:2] != img_np.shape[:2]:
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray(mask_np)
                mask_pil = mask_pil.resize((img_np.shape[1], img_np.shape[0]), PILImage.Resampling.LANCZOS)
                mask_np = np.array(mask_pil)

            # Create RGBA by combining RGB + mask as alpha
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                rgba_np = np.dstack([img_np, mask_np])
                pil_image = Image.fromarray(rgba_np, 'RGBA')
                print(f"[TripoSR] Input: RGB image + MASK -> RGBA {pil_image.size}")
            else:
                pil_image = Image.fromarray(img_np[:, :, :4], 'RGBA')
                print(f"[TripoSR] Input: RGBA image {pil_image.size}")

            pil_image = resize_foreground(pil_image, ratio=0.85)
            pil_image = rgba_to_rgb_gray_background(pil_image)
        elif img_np.shape[2] == 4:
            pil_image = Image.fromarray(img_np, 'RGBA')
            print(f"[TripoSR] Input: RGBA image {pil_image.size}")

            pil_image = resize_foreground(pil_image, ratio=0.85)
            pil_image = rgba_to_rgb_gray_background(pil_image)
        else:
            pil_image = Image.fromarray(img_np, 'RGB')
            print(f"[TripoSR] Input: RGB image {pil_image.size}")
            print("[TripoSR] Warning: For best results, connect MASK from RMBG node")

        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        device = next(model.parameters()).device

        print(f"[TripoSR] Generating 3D mesh at resolution {resolution}...")

        with torch.no_grad():
            scene_codes = model([pil_image], device=device)

            meshes = model.extract_mesh(
                scene_codes,
                has_vertex_color=True,
                resolution=resolution
            )
            mesh = meshes[0]

        print(f"[TripoSR] Mesh generated: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        if len(mesh.vertices) > 0:
            verts = mesh.vertices
            print(f"[TripoSR] Mesh bounds: X=[{verts[:,0].min():.3f}, {verts[:,0].max():.3f}], "
                  f"Y=[{verts[:,1].min():.3f}, {verts[:,1].max():.3f}], "
                  f"Z=[{verts[:,2].min():.3f}, {verts[:,2].max():.3f}]")

        if unload_model:
            print("[TripoSR] Unloading model to free VRAM...")
            model.to("cpu")
            _triposr_model_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[TripoSR] Model unloaded, VRAM freed")

        return (mesh,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "LoadTripoSRModel": LoadTripoSRModel,
    "ImageTo3DMesh": ImageTo3DMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTripoSRModel": "Load TripoSR Model",
    "ImageTo3DMesh": "Image to 3D Mesh (TripoSR)",
}
